import os
import sys
import io
import argparse
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import (
    get_optimal_device, set_deterministic_behavior, get_dataset,
    clamp_imagenet_normalized, denorm_to_01, chw_to_numpy_img01,
    make_subdataset, estimate_average_distance,
    estimate_per_pixel_variance_global_mean, compute_c_factor,
    mf_from_tau, add_noise_with_l2_norm_batch,
)

# ---------------------------
# Optional LPIPS (pip install lpips)
# ---------------------------
try:
    import lpips  # type: ignore
    HAS_LPIPS = True
except ImportError:
    lpips = None
    HAS_LPIPS = False

import warnings
warnings.simplefilter("ignore", UserWarning)


# ---------------------------
# Simple Tee to mirror stdout/stderr to a file (multi-stream variant)
# ---------------------------
class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return any(getattr(st, "isatty", lambda: False)() for st in self.streams)


# ---------------------------
# Visualization / formatting helpers
# ---------------------------
def tau_to_latex(tau: float) -> str:
    if tau <= 0:
        return r"$\tau=0$"
    exp = int(np.round(np.log10(tau)))
    if np.isclose(tau, 10.0**exp):
        return rf"$\tau=10^{{{exp}}}$"
    mant = tau / (10.0**exp)
    return rf"$\tau={mant:.1f}\times 10^{{{exp}}}$"


def recovery_snr_db(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    """Per-image SNR in dB: 10 log10( ||x||^2 / ||x - xhat||^2 )"""
    B = x.size(0)
    xv = x.view(B, -1)
    yv = xhat.view(B, -1)
    sig = torch.sum(xv**2, dim=1)
    err = torch.sum((xv - yv) ** 2, dim=1) + 1e-20
    snr = sig / err
    return 10.0 * torch.log10(snr)


# ---------------------------
# Build a mixup graph + observed mixtures
# ---------------------------
@torch.no_grad()
def build_mixup_observations(
    X: torch.Tensor, alpha: float, r: float, mf: float, device="cpu"
):
    """
    X: [N,C,H,W] normalized clean images of the subset

    Returns:
      Y: [N,C,H,W] mixed observations
      partner_idx: LongTensor [N] partner index for each i

    Construction:
      Y[i] = alpha * X[i] + (1-alpha) * (X[j] + noise_ij),
      ||noise_ij|| = mf * r
    """
    N = X.size(0)
    perm = torch.randperm(N, device=device)
    if (perm == torch.arange(N, device=device)).any():
        perm = torch.roll(perm, shifts=1)
    partner = perm.clone()

    target_norm = float(mf) * float(r)
    noisy_partner = add_noise_with_l2_norm_batch(
        X[partner], target_norm=target_norm)
    Y = alpha * X + (1.0 - alpha) * noisy_partner
    return Y, partner


# ---------------------------
# TV loss (anisotropic)
# ---------------------------
def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic total variation:
      sum |x[:, :, :, 1:] - x[:, :, :, :-1]| +
          |x[:, :, 1:, :] - x[:, :, :-1, :]|
    Return mean over batch.
    """
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return dh.mean() + dw.mean()


# ---------------------------
# Linear attack optimizer
# ---------------------------
def run_linear_attack(
    X_clean: torch.Tensor,
    Y_obs: torch.Tensor,
    partner_idx: torch.Tensor,
    alpha: float,
    steps: int = 200,
    lr: float = 0.05,
    lambda_tv: float = 1e-3,
    lambda_l2: float = 1e-4,
) -> torch.Tensor:
    """
    Recover variables V (clean images) minimizing:

      MSE(Y, alpha V + (1-alpha) V[partner])
      + lambda_tv * TV(V)
      + lambda_l2 * ||V||^2

    Clamp to plausible range after each optimizer step.
    """
    V = Y_obs.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)
    mse = nn.MSELoss()

    for t in range(steps):
        opt.zero_grad()
        V_partner = V[partner_idx]
        recon = alpha * V + (1.0 - alpha) * V_partner
        loss_recon = mse(recon, Y_obs)
        loss_tv = lambda_tv * tv_loss(V)
        loss_l2 = lambda_l2 * (V.pow(2).mean())
        loss = loss_recon + loss_tv + loss_l2
        loss.backward()
        opt.step()
        with torch.no_grad():
            V.copy_(clamp_imagenet_normalized(V))
        if (t + 1) % max(steps // 5, 1) == 0:
            print(
                f" [attack] step {t+1:4d} "
                f"L={loss.item():.6f} mse={loss_recon.item():.6f} "
                f"tv={loss_tv.item():.6f} l2={loss_l2.item():.6f}"
            )
    return V.detach()


# ---------------------------
# SSIM (per-image) and LPIPS
# ---------------------------
def _gaussian_kernel(
    window_size: int, sigma: float, channels: int, device: torch.device
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - window_size // 2
    coords2 = coords**2
    sigma2 = sigma**2
    g = torch.exp(-(coords2) / (2 * sigma2))
    g = g / g.sum()
    g2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel = g2d.view(1, 1, window_size, window_size)
    kernel = kernel.repeat(channels, 1, 1, 1)  # [C,1,w,w] for depthwise conv
    return kernel


@torch.no_grad()
def ssim_per_image(
    x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """
    Compute SSIM per image between x and y.
    x, y: [B,C,H,W] in [0,1].

    Returns: [B] tensor of SSIM values.
    """
    assert x.shape == y.shape, "SSIM inputs must have same shape"
    B, C, H, W = x.shape
    device = x.device
    kernel = _gaussian_kernel(window_size, sigma, C, device=device)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=C)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=C) - mu_xy

    C1 = 0.012
    C2 = 0.032

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )

    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def compute_privacy_metrics(
    X_clean: torch.Tensor,
    X_rec: torch.Tensor,
    lpips_model: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 64,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Compute privacy metrics between clean and reconstructed images:

    - SSIM per image (on [0,1] space),
    - LPIPS per image (if lpips_model is provided, on [-1,1] space).
    """
    N = X_clean.size(0)

    ssim_vals_list: List[torch.Tensor] = []
    lpips_vals_list: List[torch.Tensor] = []

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        x_c = X_clean[start:end].to(device)
        x_r = X_rec[start:end].to(device)

        # Convert to [0,1] for SSIM/LPIPS
        x_c01 = denorm_to_01(x_c)
        x_r01 = denorm_to_01(x_r)

        # SSIM
        ssim_vals_list.append(ssim_per_image(x_c01, x_r01).cpu())

        # LPIPS (if available)
        if lpips_model is not None:
            # LPIPS expects [-1,1]
            x_c_lp = x_c01 * 2.0 - 1.0
            x_r_lp = x_r01 * 2.0 - 1.0
            lp = lpips_model(x_c_lp, x_r_lp)  # shape [B,1,1,1] or [B,1]
            lpips_vals_list.append(lp.view(-1).detach().cpu())

    ssim_vals = torch.cat(ssim_vals_list, dim=0)
    lpips_vals = (
        torch.cat(lpips_vals_list, dim=0) if (
            lpips_model is not None) else None
    )

    return {
        "ssim": ssim_vals,  # [N]
        "lpips": lpips_vals,  # [N] or None
    }


# ---------------------------
# Visualization (single multi-tau grid: Original / Mixup / Recovered)
# ---------------------------
def save_multi_tau_grid(
    datasets_in_order: List[str],
    taus: List[float],
    per_dataset_results: Dict[str, Dict],
    out_dir: str,
    fname: str = "multi_dataset_multi_tau.png",
):
    """
    Build one big grid figure.
    For each dataset -> three rows (Original, Mixup, Recovered) across tau columns.
    Only tau is shown above columns; no other text/labels.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_cols = len(taus)
    n_rows = 3 * len(datasets_in_order)

    fig_w = max(3 * n_cols, 6)
    fig_h = max(2.5 * n_rows, 6)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Put tau titles ONLY on the top row
    for j, tau in enumerate(taus):
        axes[0, j].set_title(tau_to_latex(tau), fontsize=30)

    for dsi, ds_name in enumerate(datasets_in_order):
        ds_res = per_dataset_results[ds_name]
        chosen_idx = ds_res["best_index_at_max_tau"]

        top_row = 3 * dsi  # original
        mid_row = top_row + 1  # mixup
        bot_row = top_row + 2  # recovered

        # Prepare the original once (same for all tau)
        orig_img_np = chw_to_numpy_img01(ds_res["X_clean"][chosen_idx])

        for j, tau in enumerate(taus):
            # Mixup and Recovered for this tau
            Yobs = ds_res["per_tau"][tau]["Y_obs"][chosen_idx]
            Xrec = ds_res["per_tau"][tau]["X_rec"][chosen_idx]
            mix_np = chw_to_numpy_img01(Yobs)
            rec_np = chw_to_numpy_img01(Xrec)

            # ORIGINAL
            ax = axes[top_row, j]
            ax.imshow(orig_img_np)
            ax.axis("off")

            # MIXUP
            ax = axes[mid_row, j]
            ax.imshow(mix_np)
            ax.axis("off")

            # RECOVERED
            ax = axes[bot_row, j]
            ax.imshow(rec_np)
            ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved grid] {out_path}")


# ---------------------------
# Main experiment
# ---------------------------
def run_experiment(
    datasets: List[str],
    taus: List[float],
    alpha: float,
    sub_size: int,
    device: torch.device,
    seed: int,
    max_images_for_stats: int,
    batch_size_stats: int,
    attack_steps: int,
    attack_lr: float,
    lambda_tv: float,
    lambda_l2: float,
    out_dir: str,
):
    if seed is not None:
        set_deterministic_behavior(seed)

    print("\n=== Linear attack on mixup (known graph) ===")
    print(f"alpha : {alpha}")
    print(f"taus : {taus}")
    print(f"subset : {sub_size} images per dataset")
    print(f"device : {device}")
    print(
        f"attack : steps={attack_steps}, lr={attack_lr}, "
        f"tv={lambda_tv}, l2={lambda_l2}\n"
    )

    # Load LPIPS model once (feature-based perceptual metric)
    lpips_model = None
    if HAS_LPIPS:
        print("[models] Loading LPIPS (VGG) model for perceptual distance...")
        lpips_model = lpips.LPIPS(net="vgg").to(device)
        lpips_model.eval()
    else:
        print(
            "[warn] 'lpips' package not found. LPIPS metric will be skipped. "
            "Install with: pip install lpips"
        )

    per_dataset_results: Dict[str, Dict] = {}

    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        _, test_ds, num_classes = get_dataset(ds_name)
        sub_ds = make_subdataset(test_ds, max_images=sub_size, seed=seed or 0)

        # Load the subset into one tensor X_clean & labels y_clean
        loader = DataLoader(
            sub_ds,
            batch_size=batch_size_stats,
            shuffle=False)
        X_list: List[torch.Tensor] = []
        y_list: List[torch.Tensor] = []
        for imgs, labels in loader:
            X_list.append(imgs.to(device))
            y_list.append(labels.cpu())
        X_clean = torch.cat(X_list, dim=0)  # [N,C,H,W]
        y_clean = torch.cat(y_list, dim=0)  # [N]
        N, C, H, W = X_clean.shape
        print(f"[subset] N={N}, C={C}, H={H}, W={W}")

        # Dataset stats on the same subset
        r = estimate_average_distance(
            sub_ds,
            device=device,
            max_images=max_images_for_stats,
            batch_size=batch_size_stats,
        )
        v_hat, d = estimate_per_pixel_variance_global_mean(
            sub_ds,
            device=device,
            max_images=max_images_for_stats,
            batch_size=batch_size_stats,
        )
        c = compute_c_factor(r, v_hat, d)
        print(f"[stats] r={r:.6f} | d={d} | v_hat={v_hat:.4e} | c={c:.4f}")

        ds_store = {
            "X_clean": X_clean.detach(),
            "y_clean": y_clean.detach(),
            "per_tau": {},
            "max_tau": max(taus) if len(taus) > 0 else None,
            "best_index_at_max_tau": None,
        }

        # Per-tau experiments
        for tau in taus:
            mf = mf_from_tau(alpha=alpha, tau=tau, c=c)
            print(
                f"\n  tau={tau:.6g} -> mf(theory)={mf:.6f} "
                f"(target ||n|| = {mf*r:.2f})"
            )
            with torch.no_grad():
                Y_obs, partner = build_mixup_observations(
                    X_clean, alpha=alpha, r=r, mf=mf, device=device
                )

            X_rec = run_linear_attack(
                X_clean,
                Y_obs,
                partner,
                alpha=alpha,
                steps=attack_steps,
                lr=attack_lr,
                lambda_tv=lambda_tv,
                lambda_l2=lambda_l2,
            )

            # SNR
            snr_db = recovery_snr_db(X_clean, X_rec)
            mean_db = float(snr_db.mean().item())
            std_db = float(snr_db.std().item())
            print(
                f"  [result] avg recovery SNR = {mean_db:.2f} +/- {std_db:.2f} dB "
                f"over {N} images"
            )

            # Additional privacy metrics (SSIM, LPIPS)
            privacy = compute_privacy_metrics(
                X_clean=X_clean,
                X_rec=X_rec,
                lpips_model=lpips_model,
                device=device,
                batch_size=batch_size_stats,
            )
            ssim_vals = privacy["ssim"]
            lpips_vals = privacy["lpips"]

            mean_ssim = float(ssim_vals.mean().item())
            std_ssim = float(ssim_vals.std().item())

            if lpips_vals is not None:
                mean_lpips = float(lpips_vals.mean().item())
                std_lpips = float(lpips_vals.std().item())
                print(
                    f"  [privacy] SSIM = {mean_ssim:.4f} +/- {std_ssim:.4f} | "
                    f"LPIPS = {mean_lpips:.4f} +/- {std_lpips:.4f}"
                )
            else:
                print(
                    f"  [privacy] SSIM = {mean_ssim:.4f} +/- {std_ssim:.4f} | "
                    f"LPIPS = (skipped)"
                )

            # Move large tensors to CPU immediately to free GPU memory.
            ds_store["per_tau"][tau] = {
                "X_rec": X_rec.detach().cpu(),
                "Y_obs": Y_obs.detach().cpu(),
                "snr_db": snr_db.detach().cpu(),
                "ssim": ssim_vals.detach().cpu(),
                "lpips": lpips_vals.detach().cpu()
                if lpips_vals is not None
                else None,
            }
            del X_rec, Y_obs, partner, snr_db, privacy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Choose best index at the largest tau
        if len(taus) > 0:
            tau_max = ds_store["max_tau"]
            snr_max_tau = ds_store["per_tau"][tau_max]["snr_db"]
            best_idx = int(torch.argmax(snr_max_tau).item())
            ds_store["best_index_at_max_tau"] = best_idx
            print(
                f"  [select] best image index at max tau={tau_max:g}: {best_idx} "
                f"(SNR={snr_max_tau[best_idx].item():.2f} dB)"
            )

        # Move X_clean to CPU now that the tau loop is done.
        ds_store["X_clean"] = ds_store["X_clean"].cpu()
        ds_store["y_clean"] = ds_store["y_clean"].cpu()
        per_dataset_results[ds_name] = ds_store

        # Free GPU memory before processing the next dataset.
        del X_clean, y_clean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final single figure
    desired_order = ["mnist", "cifar5m",
                     "cifar10", "cifar100", "tiny-imagenet"]
    datasets_in_order = [
        ds for ds in desired_order if ds in per_dataset_results]

    if len(datasets_in_order) == 0 or len(taus) == 0:
        print("[warn] No datasets or no taus provided; skipping figure.")
        return

    save_multi_tau_grid(
        datasets_in_order=datasets_in_order,
        taus=taus,
        per_dataset_results=per_dataset_results,
        out_dir=out_dir,
        fname="multi_dataset_multi_tau.png",
    )


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Simulate linear reconstruction attack on mixup "
            "(known mixing graph, TV+L2 priors), with MPS support, "
            "SSIM/LPIPS privacy metrics."
        )
    )
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mnist", "cifar5m", "cifar10", "cifar100", "tiny-imagenet"],
    )
    p.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[1e00, 1e-01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06],
    )
    p.add_argument("--alpha", type=float, default=0.7)

    p.add_argument(
        "--sub_size",
        type=int,
        default=512,
        help="Subset size per dataset (opt scales with this).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: 'cpu', 'cuda', or 'mps'. Default: 'cpu'.",
    )
    p.add_argument("--seed", type=int, default=0)

    # Stats
    p.add_argument("--max_images_for_stats", type=int, default=1024)
    p.add_argument("--batch_size_stats", type=int, default=64)

    # Attack optimizer
    p.add_argument("--attack_steps", type=int, default=200)
    p.add_argument("--attack_lr", type=float, default=0.05)
    p.add_argument("--lambda_tv", type=float, default=1e-3)
    p.add_argument("--lambda_l2", type=float, default=1e-4)

    p.add_argument("--out_dir", type=str, default="attack_figs")

    # Logging
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help=(
            "If set, write logs to this file (in addition to stdout). "
            "Default: attack_figs/run_YYYYmmdd_HHMMSS.log"
        ),
    )
    return p.parse_args()


# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    args = parse_args()

    # Ensure output dir exists to place logs/figures
    os.makedirs(args.out_dir, exist_ok=True)

    # Choose log file path
    if args.log_file is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.out_dir, f"run_{stamp}.log")
    else:
        log_path = args.log_file
    os.makedirs(os.path.dirname(
        os.path.abspath(log_path) or "."), exist_ok=True)

    # Install Tee for stdout + stderr
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    _log_fh = open(log_path, mode="w", buffering=1,
                   encoding="utf-8")  # line-buffered
    sys.stdout = Tee(_orig_stdout, _log_fh)
    sys.stderr = Tee(_orig_stderr, _log_fh)

    try:
        device = get_optimal_device()

        run_experiment(
            datasets=args.datasets,
            taus=args.taus,
            alpha=args.alpha,
            sub_size=args.sub_size,
            device=device,
            seed=args.seed,
            max_images_for_stats=args.max_images_for_stats,
            batch_size_stats=args.batch_size_stats,
            attack_steps=args.attack_steps,
            attack_lr=args.attack_lr,
            lambda_tv=args.lambda_tv,
            lambda_l2=args.lambda_l2,
            out_dir=args.out_dir,
        )

        print(
            f"[done] Finished. Full log saved at: {os.path.abspath(log_path)}")
    finally:
        # Restore std streams and close file handle
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        try:
            _log_fh.close()
        except Exception:
            pass
