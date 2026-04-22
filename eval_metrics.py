import os
import re
import sys
import argparse
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Shared utilities
from utils import (
    get_dataset, make_subdataset,
    estimate_average_distance, estimate_per_pixel_variance_global_mean,
    compute_c_factor, mf_from_tau,
    get_optimal_device, set_deterministic_behavior,
    Tee, DEFAULT_SEED,
)

# Non-linear attack specific (model, dataset, evaluation)
from non_linear_attack import (
    UNetSmall,
    MixupPairDataset,
    evaluate_unet,
    tau_to_tex_power10,
)

import torchmetrics
import lpips as lpips_lib

import warnings
warnings.simplefilter("ignore", UserWarning)


def discover_checkpoints(checkpoint_dir: str) -> List[dict]:
    """
    Scan a directory for saved U-Net checkpoints and extract tau/seed
    from filenames matching the pattern: unet_tau{tau}_seed{seed}.pt

    Returns a list of dicts: [{"path": ..., "tau": ..., "seed": ...}, ...]
    sorted by tau descending (largest tau first, consistent with main script).
    """
    pattern = re.compile(r"unet_tau([0-9eE.+\-]+)_seed(\d+)\.pt$")
    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        m = pattern.match(fname)
        if m:
            tau = float(m.group(1))
            seed = int(m.group(2))
            checkpoints.append({
                "path": os.path.join(checkpoint_dir, fname),
                "tau": tau,
                "seed": seed,
            })
    checkpoints.sort(key=lambda x: x["tau"], reverse=True)
    return checkpoints


def evaluate_checkpoint(
    checkpoint_path: str,
    tau: float,
    seed: int,
    alpha: float,
    public_dataset: str,
    private_dataset: str,
    test_size: int,
    batch_size: int,
    stats_max: int,
    device: torch.device,
    ssim_metric,
    lpips_metric,
    unet_base: int = 48,
) -> dict:
    """
    Load a single U-Net checkpoint and compute all 3 metrics on the
    deterministic mixup test set.

    Returns the results dict from evaluate_unet().
    """
    # Load model
    model = UNetSmall(base=unet_base).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  [loaded] {checkpoint_path}")

    # Load private dataset (always CIFAR-10 test split for evaluation)
    _, private_eval, _ = get_dataset(private_dataset)
    private_subset = make_subdataset(private_eval, test_size, seed=seed)

    # Compute dataset statistics for the private set to derive mf
    # (we need these to recreate the same MixupPairDataset as the original run)
    r_private = estimate_average_distance(
        private_subset, device=device, max_images=stats_max, batch_size=batch_size)
    v_private, d_private = estimate_per_pixel_variance_global_mean(
        private_subset, device=device, max_images=stats_max, batch_size=batch_size)
    c_private = compute_c_factor(r_private, v_private, d_private)
    mf_private = mf_from_tau(alpha, tau, c_private)

    print(f"  [stats] r={r_private:.4f}  v_hat={v_private:.4e}  d={d_private}"
          f"  c={c_private:.4f}  mf={mf_private:.4f}")

    # Recreate the exact same mixup test set (deterministic via seed+1,
    # matching the convention in non_linear_attack.py main())
    test_mix = MixupPairDataset(
        private_subset, alpha=alpha, r=r_private, mf=mf_private, seed=seed + 1)
    test_loader = DataLoader(test_mix, batch_size=batch_size, shuffle=False)

    # Evaluate
    results = evaluate_unet(
        model, test_loader, device=device,
        ssim_metric=ssim_metric, lpips_metric=lpips_metric,
        track_best=True)

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate U-Net attack checkpoints with SNR, SSIM, and LPIPS.")

    # Checkpoint specification (one of these is required)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, default=None,
                       help="Path to a single U-Net checkpoint (.pt file)")
    group.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Directory containing U-Net checkpoints; "
                       "all matching unet_tau*_seed*.pt files will be evaluated")

    # Experiment parameters (must match the original training run)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed (must match the seed used during training)")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="Mixup alpha parameter (must match training)")
    ap.add_argument("--tau", type=float, default=None,
                    help="Tau value (required when using --checkpoint; "
                    "auto-detected when using --checkpoint-dir)")
    ap.add_argument("--public-dataset", type=str, default="tiny-imagenet",
                    help="Public dataset used by the attacker "
                    "(tiny-imagenet, cifar10, tiny-imagenet-cifar10-matched, etc.)")
    ap.add_argument("--curated", action="store_true",
                    help="Use curated Tiny-ImageNet subset "
                    "(equivalent to --public-dataset tiny-imagenet-cifar10-matched)")
    ap.add_argument("--private-dataset", type=str, default="cifar10",
                    help="Private dataset to evaluate on (default: cifar10)")
    ap.add_argument("--test_size", type=int, default=2000,
                    help="Number of test images to evaluate on")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stats_max", type=int, default=2048,
                    help="Max images for dataset statistics estimation")
    ap.add_argument("--unet_base", type=int, default=48,
                    help="U-Net base channel width (must match training)")
    ap.add_argument("--logfile", type=str, default=None,
                    help="Optional log file path (prints to stdout if not set)")

    args = ap.parse_args()

    # Handle --curated flag
    if args.curated:
        args.public_dataset = "tiny-imagenet-cifar10-matched"

    # Validate: --checkpoint requires --tau
    if args.checkpoint is not None and args.tau is None:
        ap.error("--tau is required when using --checkpoint")

    # Setup logging
    tee = None
    if args.logfile:
        os.makedirs(os.path.dirname(args.logfile) or ".", exist_ok=True)
        tee = Tee(args.logfile)

    # Device and reproducibility
    device = get_optimal_device()
    set_deterministic_behavior(args.seed)

    print(f"[eval_metrics] Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[eval_metrics] Device: {device}")
    print(f"[eval_metrics] Seed: {args.seed}")
    print(f"[eval_metrics] Alpha: {args.alpha}")
    print(f"[eval_metrics] Public dataset: {args.public_dataset}")
    print(f"[eval_metrics] Private dataset: {args.private_dataset}")
    print(f"[eval_metrics] Test size: {args.test_size}")
    print(f"[eval_metrics] U-Net base: {args.unet_base}")
    print()

    # Initialize metrics once
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(
        data_range=1.0).to(device)
    lpips_metric = lpips_lib.LPIPS(net='alex').to(device).eval()

    # Build list of checkpoints to evaluate
    if args.checkpoint is not None:
        checkpoints = [{"path": args.checkpoint, "tau": args.tau, "seed": args.seed}]
    else:
        checkpoints = discover_checkpoints(args.checkpoint_dir)
        if len(checkpoints) == 0:
            print(f"[ERROR] No checkpoints found in {args.checkpoint_dir}")
            sys.exit(1)
        # Filter by seed if specified
        checkpoints = [c for c in checkpoints if c["seed"] == args.seed]
        if len(checkpoints) == 0:
            print(f"[ERROR] No checkpoints found for seed={args.seed}")
            sys.exit(1)
        print(f"[eval_metrics] Found {len(checkpoints)} checkpoint(s):")
        for c in checkpoints:
            print(f"  tau={c['tau']:g}  ->  {c['path']}")
        print()

    # Evaluate each checkpoint
    all_results = {}
    for ckpt in checkpoints:
        tau = ckpt["tau"]
        print(f"===== [TAU {tau:g}] ({tau_to_tex_power10(tau)}) =====")
        results = evaluate_checkpoint(
            checkpoint_path=ckpt["path"],
            tau=tau,
            seed=args.seed,
            alpha=args.alpha,
            public_dataset=args.public_dataset,
            private_dataset=args.private_dataset,
            test_size=args.test_size,
            batch_size=args.batch,
            stats_max=args.stats_max,
            device=device,
            ssim_metric=ssim_metric,
            lpips_metric=lpips_metric,
            unet_base=args.unet_base,
        )
        all_results[tau] = results

        snr = results["snr"]
        ssim = results["ssim"]
        lp = results["lpips"]
        best_snr = results["best"]["snr"] if results["best"] else float("nan")

        print(f"  [RESULT] tau={tau:g}"
              f" | SNR: {snr['mean']:.2f} +/- {snr['std']:.2f} dB"
              f" | SSIM: {ssim['mean']:.4f} +/- {ssim['std']:.4f}"
              f" | LPIPS: {lp['mean']:.4f} +/- {lp['std']:.4f}"
              f" | best_snr={best_snr:.2f} dB")
        print()

    # Print summary table
    print("=" * 100)
    print("[SUMMARY TABLE]")
    print(f"{'tau':>12s}  {'SNR (dB)':>18s}  {'SSIM':>18s}  {'LPIPS':>18s}  {'best_snr':>10s}")
    print("-" * 100)
    for tau in sorted(all_results.keys(), reverse=True):
        r = all_results[tau]
        snr = r["snr"]
        ssim = r["ssim"]
        lp = r["lpips"]
        best_snr = r["best"]["snr"] if r["best"] else float("nan")
        print(f"  {tau:>10g}  {snr['mean']:>7.2f} +/- {snr['std']:<6.2f}  "
              f"{ssim['mean']:>7.4f} +/- {ssim['std']:<6.4f}  "
              f"{lp['mean']:>7.4f} +/- {lp['std']:<6.4f}  "
              f"{best_snr:>8.2f}")
    print("=" * 100)

    # Cleanup
    if tee is not None:
        tee.close()
        print(f"Logs saved to: {args.logfile}")


if __name__ == "__main__":
    main()
