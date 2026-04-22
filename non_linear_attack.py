import os
import argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchmetrics
import lpips as lpips_lib

import warnings
warnings.simplefilter("ignore", UserWarning)

from utils import (
    DEFAULT_SEED,
    get_optimal_device, set_deterministic_behavior,
    Tee, get_transforms, get_dataset,
    clamp_imagenet_normalized, chw_to_numpy_img01,
    denorm_to_01, denorm_to_lpips,
    make_subdataset, estimate_average_distance,
    estimate_per_pixel_variance_global_mean,
    compute_c_factor, mf_from_tau,
)


# ---------------------------
# Dataset for tiny-imagenet-cifar10-matched (kept local -- not in utils)
# ---------------------------
import torchvision

def _get_curated_tiny_imagenet(root_dir: str = "./data"):
    """Load curated Tiny-ImageNet subset matching CIFAR-10 semantic classes."""
    tfm = get_transforms("cifar10")  # same Resize(224) + Normalize transform
    train_dir = os.path.join(root_dir, "tiny-imagenet-cifar10-matched", "train")
    val_dir = os.path.join(root_dir, "tiny-imagenet-cifar10-matched", "val")

    # Check if val_dir has actual image files (not just empty directories)
    has_val_images = False
    if os.path.isdir(val_dir):
        for class_dir in os.listdir(val_dir):
            class_path = os.path.join(val_dir, class_dir)
            if os.path.isdir(class_path) and len(os.listdir(class_path)) > 0:
                has_val_images = True
                break

    eval_dir = val_dir if has_val_images else train_dir
    train = torchvision.datasets.ImageFolder(root=train_dir, transform=tfm)
    test = torchvision.datasets.ImageFolder(root=eval_dir, transform=tfm)
    num_classes = 9  # Only 9 CIFAR-10 classes available (no airplane)
    return train, test, num_classes


def get_dataset_with_curated(dataset_type: str, root_dir: str = "./data"):
    """Wrapper that supports 'tiny-imagenet-cifar10-matched' in addition to standard datasets."""
    if dataset_type.lower() == "tiny-imagenet-cifar10-matched":
        return _get_curated_tiny_imagenet(root_dir)
    return get_dataset(dataset_type, root_dir)


# ---------------------------
# Mixup pair dataset (on-the-fly, deterministic per index)
# ---------------------------

class MixupPairDataset(torch.utils.data.Dataset):
    """
    For each index i:
      input  = alpha * x_i + (1-alpha) * (x_j + n), with ||n|| = mf * r
      target = x_i

    IMPORTANT: partner j is sampled deterministically per index using (seed + idx),
    so that the same base target index can be reproduced across different taus.
    """

    def __init__(self, base_subset, alpha: float, r: float, mf: float, seed: int = 0):
        self.base = base_subset
        self.alpha = alpha
        self.r = r
        self.mf = mf
        self.seed = int(seed)

    def __len__(self):
        return len(self.base)

    @torch.no_grad()
    def _noise(self, x: torch.Tensor, target_norm: float) -> torch.Tensor:
        B = x.size(0)
        noise = torch.randn_like(x)
        norms = torch.norm(noise.view(B, -1), p=2, dim=1)
        scales = (target_norm / (norms + 1e-12)).view(B, 1, 1, 1)
        return noise * scales

    def __getitem__(self, idx: int):
        x_i, _ = self.base[idx]

        # deterministic RNG per (seed, idx) so we can reproduce "corresponding image" across taus
        rng = np.random.default_rng(self.seed + int(idx))
        j = int(rng.integers(0, len(self.base)))
        if j == idx:
            j = (j + 1) % len(self.base)
        x_j, _ = self.base[j]

        # build single-element batch for the noise util
        x_j_b = x_j.unsqueeze(0)
        n = self._noise(x_j_b, target_norm=self.mf * self.r)[0]

        mix = self.alpha * x_i + (1.0 - self.alpha) * (x_j + n)
        mix = clamp_imagenet_normalized(mix.unsqueeze(0))[0]
        return mix, x_i

# ---------------------------
# U-Net (lightweight)
# ---------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.bott = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        bt = self.bott(p3)

        u3 = self.up3(bt)
        x3 = torch.cat([u3, d3], dim=1)
        x3 = self.dec3(x3)

        u2 = self.up2(x3)
        x2 = torch.cat([u2, d2], dim=1)
        x2 = self.dec2(x2)

        u1 = self.up1(x2)
        x1 = torch.cat([u1, d1], dim=1)
        x1 = self.dec1(x1)

        y = self.outc(x1)
        y = clamp_imagenet_normalized(y)
        return y

# ---------------------------
# Training / evaluation
# ---------------------------


def snr_db(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    B = x.size(0)
    xv = x.view(B, -1)
    yv = xhat.view(B, -1)
    sig = torch.sum(xv**2, dim=1)
    err = torch.sum((xv - yv)**2, dim=1) + 1e-20
    return 10.0 * torch.log10(sig / err)


def train_unet(model, loader, epochs: int, device, lr: float = 1e-3, tv_weight: float = 1e-4, l2_weight: float = 0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    l1 = nn.L1Loss()
    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for mix, tgt in loader:
            mix = mix.to(device)
            tgt = tgt.to(device)
            pred = model(mix)
            loss_l1 = l1(pred, tgt)
            dh = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
            dw = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()
            loss_tv = tv_weight * (dh + dw)
            loss_l2 = l2_weight * (pred.pow(2).mean())
            loss = loss_l1 + loss_tv + loss_l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[train] epoch {ep:02d}/{epochs}  loss={np.mean(losses):.5f}")


@torch.no_grad()
def evaluate_unet(model, loader, device, ssim_metric=None, lpips_metric=None,
                  track_best: bool = True, max_batches=None):
    """
    Evaluate U-Net reconstruction quality with SNR, SSIM, and LPIPS.

    Returns a dict:
        {
            "snr":   {"mean": float, "std": float},
            "ssim":  {"mean": float, "std": float},   # if ssim_metric provided
            "lpips": {"mean": float, "std": float},    # if lpips_metric provided
            "best":  {"snr": float, "ssim": float, "lpips": float,
                      "gt": Tensor, "in": Tensor, "out": Tensor, "global_index": int}
        }
    """
    model.eval()
    snrs = []
    ssims = []
    lpipss = []
    best = {"snr": -1e9, "ssim": float("nan"), "lpips": float("nan"),
            "gt": None, "in": None, "out": None, "global_index": None}
    seen = 0
    with torch.no_grad():
        for b, (mix, tgt) in enumerate(loader):
            B = mix.size(0)
            mix = mix.to(device)
            tgt = tgt.to(device)
            pred = model(mix)

            # --- SNR (existing) ---
            s = snr_db(tgt, pred)
            snrs.append(s.cpu())

            # --- SSIM (on [0,1] range) ---
            batch_ssim = None
            if ssim_metric is not None:
                tgt_01 = denorm_to_01(tgt)
                pred_01 = denorm_to_01(pred)
                # Compute per-image SSIM by iterating (torchmetrics SSIM
                # returns a scalar for the whole batch by default)
                batch_ssim = []
                for i in range(B):
                    ssim_val = ssim_metric(
                        pred_01[i:i+1], tgt_01[i:i+1])
                    batch_ssim.append(float(ssim_val.item()))
                ssims.extend(batch_ssim)

            # --- LPIPS (on [-1,1] range) ---
            batch_lpips = None
            if lpips_metric is not None:
                tgt_lp = denorm_to_lpips(tgt)
                pred_lp = denorm_to_lpips(pred)
                # lpips returns [B,1,1,1] tensor of distances per image
                lp_vals = lpips_metric(pred_lp, tgt_lp).view(-1).cpu()
                batch_lpips = [float(v) for v in lp_vals]
                lpipss.append(lp_vals)

            if track_best:
                k = int(torch.argmax(s).item())
                if s[k].item() > best["snr"]:
                    best["snr"] = s[k].item()
                    best["gt"] = tgt[k].detach().cpu()
                    best["in"] = mix[k].detach().cpu()
                    best["out"] = pred[k].detach().cpu()
                    best["global_index"] = seen + k
                    # Store per-image SSIM/LPIPS for best example
                    if batch_ssim is not None:
                        best["ssim"] = batch_ssim[k]
                    if batch_lpips is not None:
                        best["lpips"] = batch_lpips[k]
            seen += B
            if (max_batches is not None) and (b+1 >= max_batches):
                break

    # Aggregate metrics
    snrs_t = torch.cat(snrs, dim=0)
    results = {
        "snr": {"mean": float(snrs_t.mean().item()),
                "std": float(snrs_t.std().item())},
        "best": best if track_best else None,
    }

    if ssim_metric is not None and len(ssims) > 0:
        ssims_t = torch.tensor(ssims)
        results["ssim"] = {"mean": float(ssims_t.mean().item()),
                           "std": float(ssims_t.std().item())}
    else:
        results["ssim"] = {"mean": float("nan"), "std": float("nan")}

    if lpips_metric is not None and len(lpipss) > 0:
        lpipss_t = torch.cat(lpipss, dim=0)
        results["lpips"] = {"mean": float(lpipss_t.mean().item()),
                            "std": float(lpipss_t.std().item())}
    else:
        results["lpips"] = {"mean": float("nan"), "std": float("nan")}

    return results

# ---------------------------
# Helpers for figure & tau formatting
# ---------------------------


def tau_to_tex_power10(tau: float) -> str:
    if tau <= 0:
        return r"$\tau=0$"
    exp = int(np.round(np.log10(tau)))
    if np.isclose(tau, 10.0**exp):
        return rf"$\tau=10^{{{exp}}}$"
    # fallback for non-exact powers
    mant = tau / (10.0**exp)
    return rf"$\tau={mant:.1f}\times 10^{{{exp}}}$"


def save_best_triplet(figpath: str, best: dict, title: str):
    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    fig = plt.figure(figsize=(10, 4))
    for i, (lbl, img) in enumerate([("GT", best["gt"]), ("Mix", best["in"]), ("UNet", best["out"])]):
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(chw_to_numpy_img01(img))
        ax.set_title(lbl)
        ax.axis("off")
    plt.suptitle(title, fontsize=9, family='monospace')
    plt.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {figpath}")


def main():
    ap = argparse.ArgumentParser(
        "Non-linear (U-Net) attack across multiple taus.")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--taus", type=str, default="1e-00, 1e-01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06",
                    help="Comma-separated list of tau values, e.g. '1e-1,1e-2,1e-3'")
    ap.add_argument("--train_size", type=int, default=4000,
                    help="Tiny-ImageNet subset size")
    ap.add_argument("--test_size", type=int, default=2000,
                    help="CIFAR-10 subset size")
    ap.add_argument("--public-dataset", type=str, default="tiny-imagenet",
                    help="The dataset that the attacker will use to train the UNET (tiny-imagenet, cifar10, tiny-imagenet-cifar10-matched, etc.)")
    ap.add_argument("--curated", action="store_true",
                    help="Use curated Tiny-ImageNet subset (9 CIFAR-10 semantic classes). Equivalent to --public-dataset tiny-imagenet-cifar10-matched")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tv", type=float, default=1e-4)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--stats_max", type=int, default=2048)
    ap.add_argument("--outdir", type=str, default="unet_attack_figs")
    ap.add_argument("--logfile", type=str, default=None,
                    help="Optional log filename (defaults to timestamped).")
    args = ap.parse_args()

    # Handle --curated flag: override public-dataset if curated is set
    if args.curated:
        args.public_dataset = "tiny-imagenet-cifar10-matched"

    # Prepare seed-based output subdirectory
    seed_outdir = os.path.join(args.outdir, f"seed_{args.seed}")
    os.makedirs(seed_outdir, exist_ok=True)

    # Prepare Tee logger
    log_name = args.logfile or f"log_multitau_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(seed_outdir, log_name)
    tee = Tee(log_path)

    # Device and reproducibility
    device = get_optimal_device()
    set_deterministic_behavior(args.seed)

    # Initialize perceptual metrics (once, reused across all taus)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(
        data_range=1.0).to(device)
    lpips_metric = lpips_lib.LPIPS(net='alex').to(device).eval()

    # Log header with run metadata
    print(f"[LOG] Logging to: {log_path}")
    print(f"[RUN] Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[RUN] Device: {device}")
    print(f"[RUN] Seed: {args.seed}")
    print(f"[RUN] Args: {vars(args)}")
    print()

    # Parse taus
    taus: List[float] = []
    for t in args.taus.split(","):
        t = t.strip()
        if t:
            taus.append(float(t))
    if len(taus) == 0:
        raise ValueError(
            "No tau values provided. Use --taus like '1e-1,1e-2,1e-3'.")

    taus_sorted = sorted(taus, reverse=True)
    tau_max = taus_sorted[0]
    print(f"[CFG] taus (sorted desc): {taus_sorted}  | tau_max={tau_max}")

    # --- Datasets ---
    # PUBLIC -> attacker trains UNET
    # PRIVATE -> cifar10
    print(f'Public dataset: {args.public_dataset}')
    private_dataset = "cifar10"
    public_train, public_eval, _ = get_dataset_with_curated(args.public_dataset)
    _, private_eval, _ = get_dataset(private_dataset)

    public_subset = make_subdataset(
        public_train, args.train_size, seed=args.seed)
    private_subset = make_subdataset(
        private_eval,  args.test_size,  seed=args.seed)

    # --- Shared stats baseline (d is same after transforms) ---
    print("\n[STATS] Estimating dataset statistics (independent of tau)...")
    r_public = estimate_average_distance(
        public_subset, device=device, max_images=args.stats_max, batch_size=args.batch)
    v_public, d_public = estimate_per_pixel_variance_global_mean(
        public_subset, device=device, max_images=args.stats_max, batch_size=args.batch)
    r_private = estimate_average_distance(
        private_subset,  device=device, max_images=args.stats_max, batch_size=args.batch)
    v_private, d_private = estimate_per_pixel_variance_global_mean(
        private_subset,  device=device, max_images=args.stats_max, batch_size=args.batch)
    print(
        f"[PUBLIC-DATASET] | {args.public_dataset}: r={r_public:.4f}  v_hat={v_public:.4e}  d={d_public}")
    print(
        f"[PRIVATE-DATASET] | {private_dataset}: r={r_private:.4f}  v_hat={v_private:.4e}  d={d_private}")
    assert d_public == d_private, f"Dimension mismatch: {d_public, d_private}"

    # Storage for per-tau results (for final panel)
    per_tau: Dict[float, Dict[str, Any]] = {}

    # --------- Run experiment for each tau ----------
    for tau in taus_sorted:
        print(f"\n===== [TAU {tau}] ({tau_to_tex_power10(tau)}) =====")
        # Compute c & mf for train and test at this tau
        c_public = compute_c_factor(r_public, v_public, d_public)
        mf_public = mf_from_tau(args.alpha, tau, c_public)
        c_private = compute_c_factor(r_private, v_private, d_private)
        mf_private = mf_from_tau(args.alpha, tau, c_private)
        print(
            f"[tau={tau:g}] c_public={c_public:.4f} -> mf_public={mf_public:.4f} | c_private={c_private:.4f} -> mf_private={mf_private:.4f}")

        # Make mixup datasets (deterministic per index using seed for cross-tau correspondence)
        train_mix = MixupPairDataset(
            public_subset, alpha=args.alpha, r=r_public, mf=mf_public, seed=args.seed)
        test_mix = MixupPairDataset(
            private_subset,  alpha=args.alpha, r=r_private,  mf=mf_private,  seed=args.seed+1)

        train_loader = DataLoader(
            train_mix, batch_size=args.batch, shuffle=True)
        test_loader = DataLoader(
            test_mix,  batch_size=args.batch, shuffle=False)

        # Model
        model = UNetSmall(base=48).to(device)

        # Train
        print(
            f"[Training U-Net on {args.public_dataset} mixups] - DEVICE : {device} | SEED: {torch.random.initial_seed()}")
        train_unet(model, train_loader, epochs=args.epochs, device=device,
                   lr=args.lr, tv_weight=args.tv, l2_weight=args.l2)

        # Save model checkpoint
        model_path = os.path.join(
            seed_outdir, f"unet_tau{tau:g}_seed{args.seed}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"[saved] model checkpoint: {model_path}")

        # Evaluate on CIFAR-10 (track best) with all 3 metrics
        print("[Evaluating on CIFAR-10 mixups]")
        eval_results = evaluate_unet(
            model, test_loader, device=device,
            ssim_metric=ssim_metric, lpips_metric=lpips_metric,
            track_best=True)

        snr_m = eval_results["snr"]["mean"]
        snr_s = eval_results["snr"]["std"]
        ssim_m = eval_results["ssim"]["mean"]
        ssim_s = eval_results["ssim"]["std"]
        lpips_m = eval_results["lpips"]["mean"]
        lpips_s = eval_results["lpips"]["std"]
        best = eval_results["best"]

        print(f"[RESULT] tau={tau:g}"
              f" | SNR: {snr_m:.2f} +/- {snr_s:.2f} dB"
              f" | SSIM: {ssim_m:.4f} +/- {ssim_s:.4f}"
              f" | LPIPS: {lpips_m:.4f} +/- {lpips_s:.4f}"
              f" | best_snr={best['snr']:.2f} dB"
              f" | best_ssim={best['ssim']:.4f}"
              f" | best_lpips={best['lpips']:.4f}"
              f" | best_index={best['global_index']}")

        # Save individual triplet
        indiv_path = os.path.join(
            seed_outdir, f"best_unet_recovery_cifar10_tau{tau:g}.png")
        triplet_title = (
            f"CIFAR-10 best (tau={tau:g}, alpha={args.alpha})\n"
            f"Best -- SNR={max(best['snr'], 0):.2f}dB | "
            f"SSIM={best['ssim']:.4f} | LPIPS={best['lpips']:.4f}\n"
            f"Avg  -- SNR={snr_m:.2f}dB | "
            f"SSIM={ssim_m:.4f} | LPIPS={lpips_m:.4f}"
        )
        save_best_triplet(indiv_path, best, title=triplet_title)

        per_tau[tau] = {
            "snr": eval_results["snr"],
            "ssim": eval_results["ssim"],
            "lpips": eval_results["lpips"],
            "best": best,
            "model": model, "mf_public": mf_public, "mf_private": mf_private,
            "test_mix": test_mix
        }

    # --------- Build the single summary figure (ONLY tau text allowed) ---------
    print("\n[FIG] Building multi-tau summary figure (3 rows x N columns)...")
    N = len(taus_sorted)

    # Use the "best global index" from the largest tau
    idx_best = per_tau[tau_max]["best"]["global_index"]
    gt_from_tau_max = per_tau[tau_max]["best"]["gt"]

    # Collect per-column images
    rows = {1: [], 2: [], 3: []}
    for tau in taus_sorted:
        rows[1].append(gt_from_tau_max)  # repeat GT
        mix_i, _ = per_tau[tau]["test_mix"][idx_best]
        rows[2].append(mix_i.detach().cpu())
        with torch.no_grad():
            out = per_tau[tau]["model"](mix_i.unsqueeze(0).to(device))[
                0].detach().cpu()
        rows[3].append(out)

    # Plot with ONLY tau as column titles
    fig_h = 3 * 3.2
    fig_w = N * 3.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    col_titles = [tau_to_tex_power10(t) for t in taus_sorted]

    for r in range(1, 4):
        for c, img in enumerate(rows[r], start=1):
            ax = plt.subplot(3, N, (r-1)*N + c)
            ax.imshow(chw_to_numpy_img01(img))
            ax.axis("off")
            if r == 1:
                ax.set_title(col_titles[c-1], fontsize=20)

    out_panel = os.path.join(
        seed_outdir, f"panel_multitau_{len(taus_sorted)}cols.png")
    plt.savefig(out_panel, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_panel}")

    print("\n[SUMMARY]")
    for tau in taus_sorted:
        snr = per_tau[tau]["snr"]
        ssim = per_tau[tau]["ssim"]
        lp = per_tau[tau]["lpips"]
        b = per_tau[tau]["best"]["snr"]
        print(
            f"  tau={tau:g} ({tau_to_tex_power10(tau)})"
            f"  |  SNR={snr['mean']:.2f}+/-{snr['std']:.2f} dB"
            f"  |  SSIM={ssim['mean']:.4f}+/-{ssim['std']:.4f}"
            f"  |  LPIPS={lp['mean']:.4f}+/-{lp['std']:.4f}"
            f"  |  best_snr={b:.2f} dB")

    # Restore stdout and close tee file
    tee.close()
    print(f"Logs saved to: {log_path}")
    print(f"Figure saved to: {out_panel}")


if __name__ == "__main__":
    main()
