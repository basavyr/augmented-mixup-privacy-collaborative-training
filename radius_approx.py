from typing import List
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# local imports
from utils import (
    DEFAULT_SEED,
    set_deterministic_behavior, get_dataset, get_optimal_device,
    build_resnet_feature_extractor,
    clamp_imagenet_normalized, make_subdataset,
    estimate_average_distance, estimate_per_pixel_variance_global_mean,
    compute_c_factor, mf_from_tau, add_noise_with_l2_norm_batch,
)


# ---------------------------
# Core measurement: feature-space radius
# ---------------------------

@torch.no_grad()
def flatten_features(feature_extractor: nn.Module, x_norm_nchw: torch.Tensor) -> torch.Tensor:
    fmap = feature_extractor(x_norm_nchw)  # [B, C, H', W']
    feats = fmap.flatten(1)                # [B, C*H'*W']
    return feats


@torch.no_grad()
def estimate_feature_space_radius(
    dataset,
    feature_extractor: nn.Module,
    image_avg_dist_r: float,
    mf: float,
    num_samples: int = 1024,
    batch_size: int = 64,
    device: torch.types.Device = torch.device("cpu")
) -> float:
    """
    Average L2 distance between features(feat(clean)) and features(feat(noisy)),
    where ||noise||_2 = mf * r per image, and clamped to valid pixel range.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    target_norm = float(mf) * float(image_avg_dist_r)

    total, count = 0.0, 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        noisy = add_noise_with_l2_norm_batch(imgs, target_norm=target_norm)
        noisy = clamp_imagenet_normalized(noisy)

        f_clean = flatten_features(feature_extractor, imgs)
        f_noisy = flatten_features(feature_extractor, noisy)

        dists = torch.norm(f_clean - f_noisy, p=2, dim=1)
        take = min(num_samples - count, dists.numel())
        total += float(dists[:take].sum())
        count += take
        if count >= num_samples:
            break
    return total / max(count, 1)


def run_experiment(
    backbone: str,
    datasets: List[str],
    taus: List[float],
    alpha: float,
    sub_size: int,
    seed: int,
    max_images_for_stats: int,
    batch_size_stats: int,
    num_samples_radius: int,
    batch_size_radius: int,
):
    device = get_optimal_device()

    print("\n=== Feature-space radius vs. dataset-calibrated noise ===")
    print(f"alpha    : {alpha}")
    print(f"taus     : {taus}")
    print(f"subset   : {sub_size} samples / dataset (for stats & radius)")
    print(f"device   : {device}\n")

    for ds_name in datasets:
        set_deterministic_behavior(seed)
        print(f"\n--- Dataset: {ds_name} ---")
        _, test_ds, _ = get_dataset(ds_name)
        sub_ds = make_subdataset(test_ds, max_images=sub_size, seed=seed or 0)

        r = estimate_average_distance(
            sub_ds,
            device=device,
            max_images=max_images_for_stats,
            batch_size=batch_size_stats)
        v_hat, d = estimate_per_pixel_variance_global_mean(
            sub_ds,
            device=device,
            max_images=max_images_for_stats,
            batch_size=batch_size_stats)
        c = compute_c_factor(r, v_hat, d)

        feat_extractor = build_resnet_feature_extractor(
            backbone, device=device)
        print(
            f"[stats] r={r:.6f} | v_hat={v_hat:.4e} | d={d} | c={c:.3f} | backbone={backbone}")
        print(" tau   |     mf     |       r       |    mf*r     |  avg_feat_radius")
        print("-------+------------+---------------+-------------+------------------")

        for tau in taus:
            mf = mf_from_tau(alpha=alpha, tau=tau, c=c)
            feat_radius = estimate_feature_space_radius(
                sub_ds, feat_extractor, image_avg_dist_r=r, mf=mf,
                num_samples=num_samples_radius, batch_size=batch_size_radius, device=device)
            print(
                f" {tau:>5.3f} | {mf:>10.4f} | {r:>13.4f} | {mf*r:>11.4f} | {feat_radius:>16.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate feature-space displacement (radius) for dataset-calibrated noise levels."
    )
    p.add_argument("--backbone", type=str, required=True,
                   help="The model for approximating the universal radius")
    p.add_argument("--datasets", type=str, nargs="+",
                   default=["mnist", "cifar5m", "cifar10", "cifar100", "tiny-imagenet", "imagenet"])
    p.add_argument("--taus", type=float, nargs="+",
                   default=[1e-00, 1e-01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06])
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--sub_size", type=int, default=2048,
                   help="Subset size per dataset for stats and radius estimation.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--max_images_for_stats", type=int, default=2048)
    p.add_argument("--batch_size_stats", type=int, default=64)

    p.add_argument("--num_samples_radius", type=int, default=1024,
                   help="How many images to average for the feature radius.")
    p.add_argument("--batch_size_radius", type=int, default=64)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        backbone=args.backbone,
        datasets=args.datasets,
        taus=args.taus,
        alpha=args.alpha,
        sub_size=args.sub_size,
        seed=args.seed,
        max_images_for_stats=args.max_images_for_stats,
        batch_size_stats=args.batch_size_stats,
        num_samples_radius=args.num_samples_radius,
        batch_size_radius=args.batch_size_radius,
    )
