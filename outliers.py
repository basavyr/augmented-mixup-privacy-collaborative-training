import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

import lpips
from matplotlib import pyplot as plt

from utils import (
    DEFAULT_SEED, set_deterministic_behavior, get_optimal_device,
    estimate_average_distance, estimate_per_pixel_variance_global_mean,
    compute_c_factor, mf_from_tau, add_noise_with_l2_norm_batch,
)

import warnings
warnings.simplefilter("ignore", UserWarning)


CIFAR_5M_FULL_PATH = os.getenv("CIFAR_5M_FULL_PATH", None)
SUPPORTED_DATASETS = ["mnist", "cifar5m",
                      "cifar10", "cifar100", "tiny-imagenet"]
INSPECT_THRESHOLDS = [round(i / 10.0, 1)
                      for i in range(1, 11)]  # 0.1, 0.2, ..., 1.0
ANALYSIS_TAUS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


# ---------------------------
# Transforms (LPIPS expects inputs later in [-1, 1])
# NOTE: These are intentionally different from utils.get_transforms()
# because LPIPS expects [0,1] inputs, not ImageNet-normalized.
# ---------------------------

def get_transforms(dataset_type: str):
    """
    Transforms for LPIPS:
    - Resize to 224x224
    - Ensure 3 channels
    - Convert to tensor in [0, 1], NO normalization
    """
    dataset_type = dataset_type.lower()

    if dataset_type in ["cifar5m", "cifar10", "cifar100", "tiny-imagenet"]:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),  # -> [0, 1]
        ])
    elif dataset_type == "mnist":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # -> [0, 1]
        ])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


# ---------------------------
# Dataset loading (train split)
# ---------------------------

def get_dataset(dataset_type: str):
    """
    Return the TRAIN dataset of the requested type.
    """
    dataset_type = dataset_type.lower()
    transform = get_transforms(dataset_type)

    if dataset_type == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_type == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_type == "cifar5m":
        from utils import CIFAR5m
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor()])
        assert CIFAR_5M_FULL_PATH is not None, "Environment variable <CIFAR_5M_FULL_PATH> is not set."
        train_dataset = CIFAR5m(
            CIFAR_5M_FULL_PATH, transform=transform, train=True)
    elif dataset_type == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_type == "tiny-imagenet":
        train_dir = os.path.join("./data", "tiny-imagenet-200", "train")
        train_dataset = torchvision.datasets.ImageFolder(
            root=train_dir, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return train_dataset


# ---------------------------
# Indexed subset (returns original dataset index alongside data)
# ---------------------------

class IndexedSubset(Dataset):
    """Wraps a dataset + index list and returns (img, label, orig_idx)."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        img, target = self.dataset[orig_idx]
        return img, target, orig_idx


# ---------------------------
# Inspection PNG helpers
# ---------------------------

def save_inspect_pngs(
    samples_by_threshold: dict,
    dataset_name: str,
    tau: float,
    alpha: float,
    output_dir: str = ".",
):
    """
    For each LPIPS threshold in INSPECT_THRESHOLDS, save one PNG (dpi=300)
    showing up to 6 samples where LPIPS < threshold.
    Layout: 6 rows x 2 cols  (left = original, right = mixup).
    Each subplot title shows the dataset index and LPIPS value.
    """
    os.makedirs(output_dir, exist_ok=True)

    for th in INSPECT_THRESHOLDS:
        samples = samples_by_threshold[th]
        n_show = min(len(samples), 6)
        n_rows = 6
        fig, axs = plt.subplots(n_rows, 2, figsize=(10, 2.6 * n_rows), dpi=300)

        for row in range(n_rows):
            ax_l = axs[row, 0]
            ax_r = axs[row, 1]

            if row < n_show:
                sample = samples[row]
                image_idx = sample["index"]
                lpips_val = sample["lpips"]
                orig_img = sample["orig"]   # [3, H, W] in [0, 1]
                mix_img = sample["mix"]     # [3, H, W] in [0, 1]

                ax_l.imshow(orig_img.permute(1, 2, 0).numpy())
                ax_r.imshow(mix_img.permute(1, 2, 0).numpy())
                ax_l.set_title(
                    f"Original | idx={image_idx} | LPIPS={lpips_val:.4f}")
                ax_r.set_title(
                    f"Mixup | idx={image_idx} | LPIPS={lpips_val:.4f}")
            else:
                ax_l.text(0.5, 0.5, "No sample", ha="center", va="center")
                ax_r.text(0.5, 0.5, "No sample", ha="center", va="center")

            ax_l.axis("off")
            ax_r.axis("off")

        fig.suptitle(
            f"LPIPS < {th:.1f} | {dataset_name} | tau={tau} | alpha={alpha} | "
            f"{n_show}/6 collected",
            fontsize=11,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        output_path = os.path.join(
            output_dir, f"inspect_lpips_th_{th:.1f}.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        print(f"Saved: {output_path}  ({n_show} samples)")


# ---------------------------
# Analysis grid PNG helper
# ---------------------------

def _save_analysis_grid(
    example_pairs: list,
    results: list,
    analysis_lpips_th: float,
    dataset_name: str,
    alpha: float,
    label: str = "Dropped",
    output_path: str = "analysis_tau_sweep.png",
):
    """
    Save a single PNG (dpi=300) with one row per tau value.
    Each row: left = original image, right = mixup image.
    ``label`` appears in the suptitle and placeholder text
    (e.g. "Dropped" or "Kept").
    """
    n_rows = len(ANALYSIS_TAUS)
    label_lower = label.lower()
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 2.8 * n_rows), dpi=300)

    for row_idx in range(n_rows):
        ax_l = axs[row_idx, 0]
        ax_r = axs[row_idx, 1]
        tau, mf, noise_norm, dropped, total = results[row_idx]
        kept = total - dropped
        pct = 100.0 * dropped / max(total, 1)
        example = example_pairs[row_idx]

        if example is not None:
            orig_img = example["orig"]   # [3, H, W] in [0, 1]
            mix_img = example["mix"]
            lpips_val = example["lpips"]

            ax_l.imshow(orig_img.permute(1, 2, 0).numpy())
            ax_r.imshow(mix_img.permute(1, 2, 0).numpy())
            ax_l.set_title(
                f"Original | tau={tau:.0e} | LPIPS={lpips_val:.4f}",
                fontsize=9,
            )
            ax_r.set_title(
                f"Mixup | dropped={dropped}/{total} ({pct:.1f}%) | kept={kept}",
                fontsize=9,
            )
        else:
            ax_l.text(
                0.5, 0.5, f"tau={tau:.0e}\nNo {label_lower} sample",
                ha="center", va="center", fontsize=10,
            )
            ax_r.text(
                0.5, 0.5, f"dropped={dropped}/{total} ({pct:.1f}%) | kept={kept}",
                ha="center", va="center", fontsize=10,
            )

        ax_l.axis("off")
        ax_r.axis("off")

    fig.suptitle(
        f"{label} samples | tau sweep | {dataset_name} | alpha={alpha} | "
        f"LPIPS {'<' if label == 'Dropped' else '>='} {analysis_lpips_th}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {label_lower} grid: {output_path}")


# ---------------------------
# Analysis mode: sweep tau, fixed LPIPS threshold
# ---------------------------

def run_analysis(
    dataset_name: str,
    alpha: float,
    analysis_lpips_th: float,
    subset_size: int,
    max_images_for_stats: int,
    batch_size_stats: int,
):
    """
    Sweep tau over ANALYSIS_TAUS at a fixed LPIPS threshold.
    For each tau, construct mixup with the corresponding noise level,
    compute LPIPS, and count dropped samples. Print a summary table.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset type not supported. Supported: {SUPPORTED_DATASETS}"
        )

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0.0, 1.0].")

    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")

    device = get_optimal_device()
    set_deterministic_behavior(DEFAULT_SEED)
    print(f"Using device: {device} | Seed: {DEFAULT_SEED}")
    print(f"Dataset: {dataset_name}")
    print(f"alpha (mixup coefficient): {alpha}")
    print(f"Analysis LPIPS threshold: {analysis_lpips_th}")
    print(f"Subset size: {subset_size}")
    print(f"Tau sweep: {ANALYSIS_TAUS}")

    # 1) Load dataset and choose a fixed subset
    full_dataset = get_dataset(dataset_name)
    full_size = len(full_dataset)
    actual_subset_size = min(subset_size, full_size)

    subset_indices = torch.randperm(full_size)[:actual_subset_size].tolist()
    subset_for_stats = Subset(full_dataset, subset_indices)
    subset_dataset = IndexedSubset(full_dataset, subset_indices)

    # 2) Compute dataset statistics once (r, v_hat, c are tau-independent)
    print("\nEstimating dataset statistics (r, v_hat, c)...")
    stats_max_images = min(actual_subset_size, max_images_for_stats)
    r = estimate_average_distance(
        subset_for_stats, device=device,
        max_images=stats_max_images, batch_size=batch_size_stats,
    )
    v_hat, d = estimate_per_pixel_variance_global_mean(
        subset_for_stats, device=device,
        max_images=stats_max_images, batch_size=batch_size_stats,
    )
    c = compute_c_factor(r, v_hat, d)

    print(f"[stats] r     = {r:.6f}")
    print(f"[stats] d     = {d}")
    print(f"[stats] v_hat = {v_hat:.4e}")
    print(f"[stats] c     = {c:.6f}")

    # 3) Load LPIPS model once
    print("Initializing LPIPS (VGG)...")
    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    # 4) Build DataLoader once (same subset for every tau)
    loader = DataLoader(
        subset_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    num_batches = len(loader)

    # 5) Sweep tau values
    results = []        # (tau, mf, noise_norm, dropped, total)
    dropped_pairs = []  # one dropped sample per tau (or None)
    kept_pairs = []     # one kept sample per tau (or None)
    used_dropped_idx: set = set()  # dataset indices already picked for dropped grid
    used_kept_idx: set = set()     # dataset indices already picked for kept grid

    for tau in ANALYSIS_TAUS:
        mf = mf_from_tau(alpha=alpha, tau=tau, c=c)
        target_norm = mf * r

        total_samples = 0
        dropped_samples = 0
        # Dropped sample picking (unique-first, then fallback)
        picked_dropped = None
        fallback_dropped = None
        # Kept sample picking (unique-first, then fallback)
        picked_kept = None
        fallback_kept = None

        with torch.no_grad():
            for batch_idx, (imgs, _, orig_indices) in enumerate(loader):
                imgs = imgs.to(device)
                B = imgs.size(0)
                if B < 2:
                    total_samples += B
                    continue

                # Partner permutation (no self-pairs)
                perm = torch.randperm(B, device=device)
                if (perm == torch.arange(B, device=device)).any():
                    perm = torch.roll(perm, shifts=1)

                # Noisy partner + mixup
                noisy_partner = add_noise_with_l2_norm_batch(
                    imgs[perm], target_norm=target_norm
                )
                mixed_imgs = alpha * imgs + (1.0 - alpha) * noisy_partner

                # LPIPS
                orig_lpips_in = imgs * 2.0 - 1.0
                mix_lpips_in = mixed_imgs * 2.0 - 1.0
                dists = loss_fn(orig_lpips_in, mix_lpips_in).view(-1)

                mask = dists < analysis_lpips_th
                dropped_samples += int(mask.sum().item())
                total_samples += B

                # --- Pick a DROPPED sample (LPIPS < threshold) ---
                if mask.any():
                    for idx in torch.where(mask)[0].tolist():
                        if fallback_dropped is None:
                            fallback_dropped = {
                                "tau": tau,
                                "lpips": float(dists[idx].item()),
                                "orig": imgs[idx].detach().cpu().clamp(0.0, 1.0),
                                "mix": mixed_imgs[idx].detach().cpu().clamp(0.0, 1.0),
                            }
                        if picked_dropped is None:
                            ds_idx = int(orig_indices[idx].item())
                            if ds_idx not in used_dropped_idx:
                                picked_dropped = {
                                    "tau": tau,
                                    "lpips": float(dists[idx].item()),
                                    "orig": imgs[idx].detach().cpu().clamp(0.0, 1.0),
                                    "mix": mixed_imgs[idx].detach().cpu().clamp(0.0, 1.0),
                                }
                                used_dropped_idx.add(ds_idx)
                                break

                # --- Pick a KEPT sample (LPIPS >= threshold) ---
                kept_mask = ~mask
                if kept_mask.any():
                    for idx in torch.where(kept_mask)[0].tolist():
                        if fallback_kept is None:
                            fallback_kept = {
                                "tau": tau,
                                "lpips": float(dists[idx].item()),
                                "orig": imgs[idx].detach().cpu().clamp(0.0, 1.0),
                                "mix": mixed_imgs[idx].detach().cpu().clamp(0.0, 1.0),
                            }
                        if picked_kept is None:
                            ds_idx = int(orig_indices[idx].item())
                            if ds_idx not in used_kept_idx:
                                picked_kept = {
                                    "tau": tau,
                                    "lpips": float(dists[idx].item()),
                                    "orig": imgs[idx].detach().cpu().clamp(0.0, 1.0),
                                    "mix": mixed_imgs[idx].detach().cpu().clamp(0.0, 1.0),
                                }
                                used_kept_idx.add(ds_idx)
                                break

        # Unique sample if found, otherwise fallback
        chosen_dropped = picked_dropped if picked_dropped is not None else fallback_dropped
        chosen_kept = picked_kept if picked_kept is not None else fallback_kept

        results.append((tau, mf, target_norm, dropped_samples, total_samples))
        dropped_pairs.append(chosen_dropped)
        kept_pairs.append(chosen_kept)
        pct = 100.0 * dropped_samples / max(total_samples, 1)
        print(f"  tau={tau:.1e}  mf={mf:.4f}  noise_L2={target_norm:.4f}  "
              f"dropped={dropped_samples}/{total_samples} ({pct:.2f}%)")

    # 6) Summary table
    print(f"\n{'='*78}")
    print(
        f"  Analysis: drop rate vs tau  (LPIPS < {analysis_lpips_th}, alpha={alpha})")
    print(f"{'='*78}")
    print(f"  {'tau':>11s} | {'mf':>10s} | {'noise L2':>12s} | "
          f"{'Dropped':>8s} / {'Total':>6s} | {'Dropped %':>10s}")
    print(f"  {'-'*11}-+-{'-'*10}-+-{'-'*12}-+-{'-'*17}-+-{'-'*10}")
    for tau, mf, noise_norm, dropped, total in results:
        pct = 100.0 * dropped / max(total, 1)
        print(f"  {tau:>11.1e} | {mf:>10.4f} | {noise_norm:>12.4f} | "
              f"{dropped:>8d} / {total:>6d} | {pct:>9.2f}%")
    print(f"{'='*78}")

    # 7) Save analysis grids (dropped + kept)
    print()
    _save_analysis_grid(
        example_pairs=dropped_pairs,
        results=results,
        analysis_lpips_th=analysis_lpips_th,
        dataset_name=dataset_name,
        alpha=alpha,
        label="Dropped",
        output_path=f"analysis_tau_sweep_{dataset_name}_dropped.png",
    )
    _save_analysis_grid(
        example_pairs=kept_pairs,
        results=results,
        analysis_lpips_th=analysis_lpips_th,
        dataset_name=dataset_name,
        alpha=alpha,
        label="Kept",
        output_path=f"analysis_tau_sweep_{dataset_name}_kept.png",
    )


# ---------------------------
# Main experiment: mixup-with-noisy-partner + LPIPS filter
# ---------------------------

def run_mixup_lpips_experiment(
    dataset_name: str,
    alpha: float,
    tau: float,
    lpips_th: float,
    subset_size: int,
    max_images_for_stats: int,
    batch_size_stats: int,
    inspect: bool = False,
):
    """
    Main experiment:
    - Load dataset.
    - Randomly select a subset of size subset_size.
    - Estimate dataset stats r, v_hat, c on that subset.
    - From (alpha, tau, c), compute mf and the target noise norm = mf * r.
    - For each batch in that subset, create mixup-with-noisy-partner images:
        Y[i] = alpha * X[i] + (1 - alpha) * (X[partner] + noise_ij),
        ||noise_ij||_2 = mf * r.
    - Compute LPIPS (VGG) between original and mixed images.
    - Count how many are below lpips_th.
    - Print percentage of dropped images.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset type not supported. Supported: {SUPPORTED_DATASETS}"
        )

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0.0, 1.0].")

    if subset_size < 0:
        raise ValueError("subset_size must be non-negative.")

    device = get_optimal_device()
    set_deterministic_behavior(DEFAULT_SEED)
    print(f"Using device: {device} | Seed: {DEFAULT_SEED}")
    print(f"Dataset: {dataset_name}")
    print(f"alpha (mixup coefficient): {alpha}")
    print(f"tau (noise/privacy parameter): {tau}")
    print(f"LPIPS threshold: {lpips_th}")
    print(f"Requested subset size: {subset_size}")
    print(f"max_images_for_stats: {max_images_for_stats}")
    print(f"batch_size_stats   : {batch_size_stats}")

    # 1) Dataset and random subset
    full_dataset = get_dataset(dataset_name)
    full_size = len(full_dataset)

    if subset_size == 0:
        print("subset_size is 0; nothing to do.")
        return

    actual_subset_size = min(subset_size, full_size)
    if actual_subset_size < subset_size:
        print(
            f"Requested subset_size {subset_size} exceeds dataset size {full_size}.")
        print(f"Using subset_size = {actual_subset_size} instead.")

    # Randomly choose subset indices
    subset_indices = torch.randperm(full_size)[:actual_subset_size].tolist()
    # Plain Subset for stats estimation (expects (img, label) tuples)
    subset_for_stats = Subset(full_dataset, subset_indices)
    # IndexedSubset for the main loop (returns (img, label, orig_idx))
    subset_dataset = IndexedSubset(full_dataset, subset_indices)

    # 1a) Compute dataset statistics on the same subset (for noise level)
    print("\nEstimating dataset statistics (r, v_hat, c) on the chosen subset...")
    stats_max_images = min(actual_subset_size, max_images_for_stats)
    r = estimate_average_distance(
        subset_for_stats,
        device=device,
        max_images=stats_max_images,
        batch_size=batch_size_stats,
    )
    v_hat, d = estimate_per_pixel_variance_global_mean(
        subset_for_stats,
        device=device,
        max_images=stats_max_images,
        batch_size=batch_size_stats,
    )
    c = compute_c_factor(r, v_hat, d)

    print(f"[stats] r      = {r:.6f}")
    print(f"[stats] d      = {d}")
    print(f"[stats] v_hat  = {v_hat:.4e}")
    print(f"[stats] c      = {c:.6f}")

    # 1b) Compute mf and target noise norm based on (alpha, tau, c)
    mf = mf_from_tau(alpha=alpha, tau=tau, c=c)
    target_noise_norm = mf * r
    print(f"\nComputed mf from tau and c: mf = {mf:.6f}")
    print(
        f"Target noise L2 norm per image: ||noise|| = mf * r = {target_noise_norm:.6f}\n")

    # DataLoader for the subset (for LPIPS measurement)
    loader = DataLoader(
        subset_dataset,
        batch_size=128,
        shuffle=False,   # randomness is already in subset_indices
        num_workers=2,
        pin_memory=True,
    )

    num_batches = len(loader)
    print(f"Number of samples in full dataset: {full_size}")
    print(f"Number of samples in subset      : {actual_subset_size}")
    print(f"Number of batches (batch_size=128): {num_batches}")
    print("Initializing LPIPS (VGG)... this may download weights on first run.")

    # 2) LPIPS model (VGG-based)
    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    print("Starting mixup-with-noisy-partner + LPIPS loop on subset...\n")

    total_samples = 0
    dropped_samples = 0
    # Per-threshold drop counts (always tracked for the summary table)
    dropped_per_th: dict = {th: 0 for th in INSPECT_THRESHOLDS}
    # For --inspect: collect up to 6 samples per LPIPS threshold
    inspect_samples: dict = {th: [] for th in INSPECT_THRESHOLDS}

    with torch.no_grad():
        for batch_idx, (imgs, _, orig_indices) in enumerate(loader):
            imgs = imgs.to(device)           # [B, 3, H, W] in [0, 1]
            B = imgs.size(0)
            if B < 2:
                # Can't do meaningful mixup with a single image
                total_samples += B
                continue

            # 3) Build random partner indices within the batch (no self-pairs if possible)
            perm = torch.randperm(B, device=device)
            if (perm == torch.arange(B, device=device)).any():
                # fallback: roll permutation to avoid fixed points
                perm = torch.roll(perm, shifts=1)
            partner = perm

            # 4) Add L2-normalized noise to the partner images
            noisy_partner = add_noise_with_l2_norm_batch(
                imgs[partner], target_norm=target_noise_norm
            )

            # 5) Mixup-with-noisy-partner:
            #    Y = alpha * X + (1 - alpha) * (X_partner + noise)
            mixed_imgs = alpha * imgs + (1.0 - alpha) * noisy_partner

            # 6) Prepare for LPIPS: expected input in [-1, 1]
            orig_lpips_in = imgs * 2.0 - 1.0
            mix_lpips_in = mixed_imgs * 2.0 - 1.0

            # 7) Compute LPIPS distances (VGG)
            # shape [B, 1, 1, 1] or [B, 1]
            dists = loss_fn(orig_lpips_in, mix_lpips_in)
            dists = dists.view(-1)                        # shape [B]

            # 8) Count dropped samples (per --lpips_th)
            mask_dropped = dists < lpips_th
            dropped = mask_dropped.sum().item()

            dropped_samples += dropped
            total_samples += B

            # 8b) Count drops for every threshold interval (for summary table)
            for th in INSPECT_THRESHOLDS:
                dropped_per_th[th] += int((dists < th).sum().item())

            # 9) Collect samples for --inspect (up to 6 per threshold)
            if inspect:
                for th in INSPECT_THRESHOLDS:
                    if len(inspect_samples[th]) >= 6:
                        continue
                    valid_idx = torch.where(dists < th)[0]
                    for idx_in_batch in valid_idx.tolist():
                        if len(inspect_samples[th]) >= 6:
                            break
                        inspect_samples[th].append({
                            "index": int(orig_indices[idx_in_batch].item()),
                            "lpips": float(dists[idx_in_batch].item()),
                            "orig": imgs[idx_in_batch].detach().cpu().clamp(0.0, 1.0),
                            "mix": mixed_imgs[idx_in_batch].detach().cpu().clamp(0.0, 1.0),
                        })

            # Progress print every 50 batches or on last batch
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                current_pct = 100.0 * (batch_idx + 1) / max(num_batches, 1)
                print(
                    f"Processed batch {batch_idx + 1}/{num_batches} "
                    f"({current_pct:.1f}%). "
                    f"Current dropped: {dropped_samples} / {total_samples}"
                )

    if total_samples == 0:
        print("\nNo samples processed (total_samples == 0).")
        return

    drop_percentage = 100.0 * dropped_samples / total_samples

    print("\n===== LPIPS Mixup-with-Noisy-Partner Filtering Results (VGG, subset) =====")
    print(f"Total samples processed (subset): {total_samples}")
    print(f"Dropped (LPIPS < {lpips_th})    : {dropped_samples}")
    print(f"Dropped percentage              : {drop_percentage:.2f}%")
    print("==========================================================================")

    # Summary table: drops per LPIPS threshold
    print(f"\n{'='*62}")
    print(f"  Drop statistics per LPIPS threshold  (tau={tau}, alpha={alpha})")
    print(f"{'='*62}")
    print(
        f"  {'LPIPS threshold':>16s} | {'Dropped':>8s} / {'Total':>6s} | {'Dropped %':>10s}")
    print(f"  {'-'*16}-+-{'-'*17}-+-{'-'*10}")
    for th in INSPECT_THRESHOLDS:
        d = dropped_per_th[th]
        pct = 100.0 * d / total_samples
        print(
            f"  {'< ' + f'{th:.1f}':>16s} | {d:>8d} / {total_samples:>6d} | {pct:>9.2f}%")
    print(f"{'='*62}")

    if inspect:
        print("\nSaving inspect-mode PNGs (one per LPIPS threshold)...")
        save_inspect_pngs(
            samples_by_threshold=inspect_samples,
            dataset_name=dataset_name,
            tau=tau,
            alpha=alpha,
        )


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="LPIPS Mixup-with-Noisy-Partner Filter Experiment (VGG, subset)",
        description=(
            "Compute percentage of mixup-with-noisy-partner images dropped "
            "based on LPIPS (VGG) similarity on a random subset. Mixup and "
            "noise model are aligned with the linear-attack script."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset to use: mnist, cifar10, cifar100, tiny-imagenet.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Mixup coefficient alpha (Y = alpha*X + (1-alpha)*(X_partner+noise)). Default: 0.7",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help=(
            "Noise/privacy parameter used in mf_from_tau; "
            "controls the L2 norm of the partner noise via mf. "
            "Required unless --analysis is used."
        ),
    )
    parser.add_argument(
        "--lpips_th",
        type=float,
        default=None,
        help=(
            "LPIPS threshold: mixup images with LPIPS < lpips_th are dropped. "
            "Required unless --analysis is used."
        ),
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        required=True,
        help="Number of samples to randomly select from the dataset for the experiment.",
    )
    parser.add_argument(
        "--max_images_for_stats",
        type=int,
        default=2048,
        help="Max images from the subset to estimate dataset stats (r, v_hat, c).",
    )
    parser.add_argument(
        "--batch_size_stats",
        type=int,
        default=64,
        help="Batch size for estimating dataset stats.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "For each LPIPS threshold from 0.1 to 1.0 (step 0.1), save a "
            "separate PNG with up to 6 samples where LPIPS < threshold."
        ),
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help=(
            "Sweep tau from 1.0 down to 1e-6 (each step 10x smaller) at a "
            "fixed LPIPS threshold and report drop rate vs tau. "
            "When set, --tau and --lpips_th are ignored."
        ),
    )
    parser.add_argument(
        "--analysis_lpips_th",
        type=float,
        default=0.7,
        help="LPIPS threshold used in --analysis mode. Default: 0.7.",
    )
    return parser.parse_args()


# ---------------------------
# Entry
# ---------------------------
# Analysis mode (sweep tau, fixed LPIPS < 0.7)
# python3 outliers.py --data cifar100 --subset_size 512 --analysis

# Analysis mode with custom LPIPS threshold
# python3 outliers.py --data cifar100 --subset_size 512 --analysis --analysis_lpips_th 0.5

# Normal single-run mode (unchanged)
# python3 outliers.py --data cifar100 --tau 1e-6 --lpips_th 0.6 --subset_size 512

# Normal mode with inspect PNGs (unchanged)
# python3 outliers.py --data cifar100 --tau 1e-6 --lpips_th 0.6 --subset_size 512 --inspect

if __name__ == "__main__":
    args = parse_args()

    if args.analysis:
        # --analysis mode: sweep tau, fixed LPIPS threshold
        run_analysis(
            dataset_name=args.data,
            alpha=args.alpha,
            analysis_lpips_th=args.analysis_lpips_th,
            subset_size=args.subset_size,
            max_images_for_stats=args.max_images_for_stats,
            batch_size_stats=args.batch_size_stats,
        )
    else:
        # Normal single-run mode: --tau and --lpips_th are required
        if args.tau is None:
            raise SystemExit(
                "Error: --tau is required when not using --analysis.")
        if args.lpips_th is None:
            raise SystemExit(
                "Error: --lpips_th is required when not using --analysis.")

        run_mixup_lpips_experiment(
            dataset_name=args.data,
            alpha=args.alpha,
            tau=args.tau,
            lpips_th=args.lpips_th,
            subset_size=args.subset_size,
            max_images_for_stats=args.max_images_for_stats,
            batch_size_stats=args.batch_size_stats,
            inspect=args.inspect,
        )
