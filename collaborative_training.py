import os
import time
import datetime
import argparse
import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils import (
    LOG_DIR, CHECKPOINTS_DIR, DEFAULT_SEED,
    get_optimal_device, set_deterministic_behavior,
    Tee, get_dataset, build_resnet_feature_extractor,
    DenseClassifier, mixup_batch_in_feature_space,
    clamp_imagenet_normalized, add_noise_with_l2_norm_batch,
    get_feature_maps_for_first_n, save_featuremap_grid_and_mixup_pairs,
    cross_entropy_with_soft_targets, evaluate_classifier,
    estimate_average_distance, estimate_per_pixel_variance_global_mean,
    compute_c_factor, mf_from_tau,
)


# --------------------------- Local helpers (specific to collaborative experiments) --- #

def generate_unified_log_filename(tag: str, model: str, dataset_name: str, num_epochs: int) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    log_id = int(time.time())
    log_filename = f"{LOG_DIR}/federated_instahide_{tag}_{model}_{dataset_name}_epochs{num_epochs}-ID-{log_id}.log"
    return log_filename


def checkpoint_path(prefix: str, model: str, dataset_name: str, num_epochs: int, num_parties: int) -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return os.path.join(
        CHECKPOINTS_DIR,
        f"{prefix}_{model}_{dataset_name}_{num_epochs}epochs_{num_parties}parties.pth"
    )


# --------------------------- Dirichlet non-iid split (specific to this script) --- #

def _get_targets_from_cifar(dataset):
    """
    Helper to get label targets from CIFAR10 dataset in a version-robust way.
    """
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "train_labels"):
        return np.array(dataset.train_labels)
    raise AttributeError("Cannot find targets in CIFAR10 dataset object.")


def _ensure_min_party_size(
    party_indices: List[List[int]],
    min_size: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """
    Ensure each party has at least ``min_size`` samples by moving samples from
    the largest parties to the smallest parties.

    This does NOT preserve class proportions exactly, but prevents empty/tiny parties
    that break downstream steps (radius estimation, batchnorm, etc.).
    """
    if min_size <= 0:
        return party_indices

    num_parties = len(party_indices)
    total = sum(len(x) for x in party_indices)
    if total < num_parties * min_size:
        raise ValueError(
            f"Cannot enforce min_size={min_size}: total samples={total} < "
            f"{num_parties} * {min_size}."
        )

    # Shuffle each party so "moving from the end" is roughly random
    for p in range(num_parties):
        rng.shuffle(party_indices[p])

    # Iteratively fix deficits
    while True:
        sizes = np.array([len(x) for x in party_indices], dtype=int)
        deficits = np.where(sizes < min_size)[0]
        if deficits.size == 0:
            break

        # donor: the party with maximum size (must be > min_size)
        donors = np.where(sizes > min_size)[0]
        if donors.size == 0:
            raise RuntimeError(
                "No donors available to satisfy min party size constraint. "
                f"Sizes={sizes.tolist()}, min_size={min_size}"
            )

        # Process one deficit at a time
        d = int(deficits[0])
        need = int(min_size - sizes[d])

        # take from biggest donors first
        donors_sorted = donors[np.argsort(-sizes[donors])]

        moved = 0
        for donor in donors_sorted:
            donor = int(donor)
            available = len(party_indices[donor]) - min_size
            if available <= 0:
                continue
            take = min(available, need - moved)
            if take <= 0:
                break
            # move ``take`` samples
            moved_samples = party_indices[donor][-take:]
            party_indices[donor] = party_indices[donor][:-take]
            party_indices[d].extend(moved_samples)
            moved += take
            if moved >= need:
                break

        if moved < need:
            raise RuntimeError(
                f"Failed to move enough samples to party {d}. "
                f"Need {need}, moved {moved}."
            )

        rng.shuffle(party_indices[d])

    return party_indices


def split_dataset_dirichlet(
    dataset,
    num_parties: int,
    alpha: float,
    seed: int = 0,
    min_size_per_party: int = 10,
) -> List[Subset]:
    """
    Dirichlet non-iid split for CIFAR-10.

    For each class c, draw proportions p_c ~ Dir(alpha, ..., alpha) over ``num_parties``,
    then assign examples of class c to parties according to p_c.

    Additional safety:
      Enforces each party has at least ``min_size_per_party`` samples by rebalancing.

    - Smaller alpha -> more skewed distributions (more non-iid).
    - Larger alpha  -> more uniform distributions (closer to iid).
    """
    assert num_parties >= 1, "num_parties must be >= 1."
    assert alpha > 0, "Dirichlet alpha must be > 0."

    targets = _get_targets_from_cifar(dataset)
    num_classes = len(np.unique(targets))

    rng = np.random.default_rng(seed)
    party_indices = [[] for _ in range(num_parties)]

    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)

        # Dirichlet draw for class c over parties
        proportions = rng.dirichlet(alpha * np.ones(num_parties))
        counts = rng.multinomial(len(idx_c), proportions)

        start = 0
        for p in range(num_parties):
            cnt = counts[p]
            if cnt == 0:
                continue
            part_idx = idx_c[start:start + cnt]
            party_indices[p].extend(part_idx.tolist())
            start += cnt

    # Shuffle indices within each party
    for p in range(num_parties):
        rng.shuffle(party_indices[p])

    # Enforce minimum party size
    party_indices = _ensure_min_party_size(party_indices, min_size=min_size_per_party, rng=rng)

    subsets = [Subset(dataset, idxs) for idxs in party_indices]
    return subsets


def split_dataset_dirichlet_size_skew(
    dataset,
    num_parties: int,
    alpha: float,
    rho: float,
    poor_party: int = 0,
    seed: int = 0,
    fix_total: bool = True,
    min_size_per_party: int = 10,
) -> List[Subset]:
    """
    Dirichlet non-iid split for CIFAR-10 with one "data-poor" party + min-size safeguard.

    Label skew:
        For each class c, draw proportions ~ Dir(alpha*1) across parties.

    Size skew:
        poor_party has smaller expected allocation via weights:
            weights[p] = rho if p==poor_party else 1

    fix_total=True:
        target sizes sum exactly to N=len(dataset), and we enforce target sizes via balancing.

    Additional safety:
        Enforces each party has at least ``min_size_per_party`` samples by rebalancing.
    """
    assert num_parties >= 1, "num_parties must be >= 1."
    assert alpha > 0, "Dirichlet alpha must be > 0."
    assert rho > 0, "rho must be > 0."
    assert 0 <= poor_party < num_parties, "poor_party index out of range."

    targets = _get_targets_from_cifar(dataset)
    num_classes = len(np.unique(targets))
    N = len(dataset)

    rng = np.random.default_rng(seed)

    # --- target sizes ---
    weights = np.ones(num_parties, dtype=np.float64)
    weights[poor_party] = float(rho)

    if fix_total:
        raw = weights / weights.sum() * N
        target_sizes = np.floor(raw).astype(int)
        remainder = N - target_sizes.sum()
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for i in range(remainder):
            target_sizes[order[i % num_parties]] += 1
    else:
        raw = weights / weights.sum() * N
        target_sizes = np.round(raw).astype(int)
        diff = N - target_sizes.sum()
        if diff != 0:
            frac = raw - np.floor(raw)
            order = np.argsort(-frac)
            for i in range(abs(diff)):
                target_sizes[order[i % num_parties]] += 1 if diff > 0 else -1

    # --- first pass: class-wise allocation with size-biased dirichlet ---
    party_indices = [[] for _ in range(num_parties)]
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)

        proportions = rng.dirichlet(alpha * np.ones(num_parties))

        biased = proportions * weights
        s = biased.sum()
        if s <= 0:
            biased = np.ones(num_parties) / num_parties
        else:
            biased = biased / s

        counts = rng.multinomial(len(idx_c), biased)

        start = 0
        for p in range(num_parties):
            cnt = counts[p]
            if cnt <= 0:
                continue
            part_idx = idx_c[start:start + cnt]
            party_indices[p].extend(part_idx.tolist())
            start += cnt

    for p in range(num_parties):
        rng.shuffle(party_indices[p])

    # --- second pass: enforce exact target_sizes ---
    buffer = []
    for p in range(num_parties):
        excess = len(party_indices[p]) - target_sizes[p]
        if excess > 0:
            buffer.extend(party_indices[p][-excess:])
            party_indices[p] = party_indices[p][:-excess]

    rng.shuffle(buffer)
    buf_ptr = 0
    for p in range(num_parties):
        deficit = target_sizes[p] - len(party_indices[p])
        if deficit > 0:
            take = buffer[buf_ptr:buf_ptr + deficit]
            party_indices[p].extend(take)
            buf_ptr += deficit

    # --- final consistency fix (rare edge-cases) ---
    assigned_total = sum(len(idxs) for idxs in party_indices)
    if assigned_total != N:
        all_assigned = set()
        for p in range(num_parties):
            all_assigned.update(party_indices[p])

        if assigned_total < N:
            missing = [i for i in range(N) if i not in all_assigned]
            rng.shuffle(missing)
            gaps = [(p, target_sizes[p] - len(party_indices[p])) for p in range(num_parties)]
            gaps.sort(key=lambda x: x[1], reverse=True)
            mptr = 0
            for p, gap in gaps:
                if gap <= 0:
                    continue
                take_n = min(gap, len(missing) - mptr)
                if take_n <= 0:
                    break
                party_indices[p].extend(missing[mptr:mptr + take_n])
                mptr += take_n

        assigned_total = sum(len(idxs) for idxs in party_indices)
        if assigned_total > N:
            overs = [(p, len(party_indices[p]) - target_sizes[p]) for p in range(num_parties)]
            overs.sort(key=lambda x: x[1], reverse=True)
            extra = assigned_total - N
            for p, over in overs:
                if extra <= 0:
                    break
                if over <= 0:
                    continue
                drop = min(over, extra)
                party_indices[p] = party_indices[p][:-drop]
                extra -= drop

    for p in range(num_parties):
        rng.shuffle(party_indices[p])

    # Enforce minimum party size (important for radius estimation)
    party_indices = _ensure_min_party_size(party_indices, min_size=min_size_per_party, rng=rng)

    subsets = [Subset(dataset, idxs) for idxs in party_indices]
    return subsets


# --------------------------- Radius estimation from tau (per-party) ------------- #

@torch.no_grad()
def estimate_feature_space_radius(
    dataset,
    feature_extractor: nn.Module,
    image_avg_dist_r: float,
    mf: float,
    num_samples: int = 1024,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Average L2 distance between features(clean) and features(noisy),
    where ||noise||_2 = mf * r per image, and clamped to valid pixel range.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=("cuda" in str(device) or "mps" in str(device)),
    )
    target_norm = float(mf) * float(image_avg_dist_r)

    total, count = 0.0, 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        noisy = add_noise_with_l2_norm_batch(imgs, target_norm=target_norm)
        noisy = clamp_imagenet_normalized(noisy)

        fmap_clean = feature_extractor(imgs)
        fmap_noisy = feature_extractor(noisy)

        f_clean = fmap_clean.view(fmap_clean.size(0), -1)
        f_noisy = fmap_noisy.view(fmap_noisy.size(0), -1)

        dists = torch.norm(f_clean - f_noisy, p=2, dim=1)
        take = min(num_samples - count, dists.numel())
        total += float(dists[:take].sum())
        count += take
        if count >= num_samples:
            break
    return total / max(count, 1)


def estimate_party_feature_radius(
    party_dataset,
    feature_extractor: nn.Module,
    device: torch.device,
    tau: float,
    alpha: float = 0.7,
    max_images_for_stats: int = 2048,
    batch_size_stats: int = 64,
    num_samples_radius: int = 1024,
    batch_size_radius: int = 64,
) -> Tuple[float, dict]:
    """
    For a given party dataset, approximate the feature-space radius corresponding to tau.
    Returns (radius, stats_dict).
    """
    # If dataset is tiny, reduce max_images_for_stats to avoid wasted loader work
    max_images_for_stats = min(max_images_for_stats, len(party_dataset))

    r = estimate_average_distance(
        party_dataset,
        device=device,
        max_images=max_images_for_stats,
        batch_size=batch_size_stats,
    )
    v_hat, d = estimate_per_pixel_variance_global_mean(
        party_dataset,
        device=device,
        max_images=max_images_for_stats,
        batch_size=batch_size_stats,
    )
    c = compute_c_factor(r, v_hat, d)
    mf = mf_from_tau(alpha=alpha, tau=tau, c=c)

    feat_radius = estimate_feature_space_radius(
        party_dataset,
        feature_extractor=feature_extractor,
        image_avg_dist_r=r,
        mf=mf,
        num_samples=min(num_samples_radius, len(party_dataset)),
        batch_size=batch_size_radius,
        device=device,
    )

    stats = {"r": r, "v_hat": v_hat, "d": d, "c": c, "mf": mf}
    return feat_radius, stats


# --------------------------- Training (collaborative-specific) ------------------- #

def train_classifier_mixup_union(
    device: torch.device,
    feature_extractor,
    classifier,
    optimizer,
    scheduler,
    party_loaders: List[DataLoader],
    party_radii: List[float],
    testloader: DataLoader,
    num_epochs: int,
    num_classes: int,
) -> float:
    """
    Train on the UNION of party-local mixup batches (mix within each party only),
    iterating over all parties each epoch. Uses per-party feature-space radius.
    Returns best test accuracy.
    """
    assert len(party_loaders) == len(
        party_radii), "party_loaders and party_radii must align"

    best_acc = 0.0
    classifier.to(device)
    feature_extractor.eval()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        classifier.train()
        running_loss = 0.0
        processed = 0

        for pidx, loader in enumerate(party_loaders):
            radius_p = party_radii[pidx]
            for imgs, labels in loader:
                optimizer.zero_grad()
                mix_feats, mix_soft_labels = mixup_batch_in_feature_space(
                    feature_extractor, imgs, labels, device, radius_p, num_classes
                )
                outputs = classifier(mix_feats)
                loss = cross_entropy_with_soft_targets(
                    outputs, mix_soft_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * mix_feats.size(0)
                processed += mix_feats.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, processed)
        test_acc = evaluate_classifier(
            classifier, feature_extractor, device, testloader)
        best_acc = max(best_acc, test_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"[Mixup-Union] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

    return best_acc


def train_classifier_fedprox(
    device: torch.device,
    feature_extractor,
    global_classifier: nn.Module,
    party_loaders: List[DataLoader],
    testloader: DataLoader,
    num_epochs: int,
    lr: float,
    mu: float,
) -> float:
    """
    FedProx training over parties (no mixup, feature extractor fixed).

    Each epoch is one global round; each party performs one local epoch per round.
    The local objective at party k is:

      L_k(w) + (mu / 2) * ||w - w_t||^2

    where w_t is the current global model.
    Aggregation is weighted by the local dataset sizes (like FedAvg).

    Important: Only floating-point parameters are averaged.
    Non-floating tensors (e.g., BatchNorm counters) are kept from the global model.
    """
    feature_extractor.eval()
    global_classifier.to(device)
    best_acc = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        global_state = global_classifier.state_dict()

        # Prepare proximal reference parameters (w_t)
        global_params = {
            name: p.detach().clone().to(device)
            for name, p in global_classifier.named_parameters()
        }

        # Initialize aggregation state
        agg_state = {}
        for k, v in global_state.items():
            if v.is_floating_point():
                agg_state[k] = torch.zeros_like(v)
            else:
                agg_state[k] = v.clone()

        total_samples = 0
        running_loss = 0.0

        for loader in party_loaders:
            n_k = len(loader.dataset)
            total_samples += n_k

            # Local model & optimizer
            local_model = copy.deepcopy(global_classifier).to(device)
            local_model.train()
            optimizer_local = optim.SGD(
                local_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
            )

            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    fmap = feature_extractor(imgs)
                    feats = fmap.view(fmap.size(0), -1)

                optimizer_local.zero_grad()
                outputs = local_model(feats)
                loss_ce = ce_loss_fn(outputs, labels)

                # Proximal term: (mu / 2) * ||w - w_t||^2
                prox_term = 0.0
                for name, param in local_model.named_parameters():
                    if not param.requires_grad:
                        continue
                    prox_term = prox_term + \
                        torch.sum((param - global_params[name]) ** 2)
                loss = loss_ce + (mu / 2.0) * prox_term

                loss.backward()
                optimizer_local.step()

                running_loss += loss.item() * labels.size(0)

            # Accumulate weighted params for floating-point tensors
            local_state = local_model.state_dict()
            for k, v in local_state.items():
                if not agg_state[k].is_floating_point():
                    continue
                agg_state[k] += n_k * v

        # Normalize aggregated parameters (only floats)
        for k, v in agg_state.items():
            if v.is_floating_point():
                agg_state[k] = v / float(total_samples)

        global_classifier.load_state_dict(agg_state)

        train_loss = running_loss / float(total_samples)
        test_acc = evaluate_classifier(
            global_classifier, feature_extractor, device, testloader)
        best_acc = max(best_acc, test_acc)

        print(
            f"[FedProx] Epoch (round) {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

    return best_acc


# --------------------------- Main experiment runner --------------------------- #

def augment_workflow(
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    tau: float,
    lr: float,
    mu: float,
    bench_mode: bool,
    num_parties: int,
    dirichlet_alpha: float,
    rho: float = 1.0,
    poor_party: int = 0,
    size_skew_fix_total: bool = True,
    min_party_size: int = 10,
    finish_with_model_eval: bool = True,
    tee: Optional[Tee] = None,
) -> None:
    """
    Runs a single experiment for a given num_parties with Dirichlet non-iid split.
    Each party estimates its own feature-space radius from tau using its local data.
    Compares FedProx (no mixup) vs Mixup-Union.
    New: optional rho size skew + minimum party size guarantee.
    """
    print(f"\n============ Experiment START (num_parties={num_parties}) ============")
    print(f"Started on: {datetime.datetime.now().ctime()}")
    print(f"Torch seed: {torch.random.initial_seed()}")
    print(f"Deterministic behavior: {bench_mode}")
    print(f"Using device: {device}")
    print(f"Model: resnet18 [cut after layer3]")
    print(f"Dataset: cifar10")
    print(f"Epochs (global rounds): {num_epochs}")
    print(f"tau: {tau}")
    print(f"mu (FedProx proximal coeff): {mu}")
    print(f"num_parties (Dirichlet non-iid): {num_parties}")
    print(f"Dirichlet alpha: {dirichlet_alpha}")
    print(f"rho (size skew poor/rich): {rho}")
    print(f"poor_party: {poor_party}")
    print(f"size_skew_fix_total: {size_skew_fix_total}")
    print(f"min_party_size: {min_party_size}")
    print("----------------------------------------------------")

    workflow_start_time = time.time()

    # 1) Dataset
    train_dataset, test_dataset, num_classes = get_dataset("cifar10")

    # 2) Feature extractor (shared) -- cut_layers=3 removes layer4, avgpool, fc
    feature_extractor = build_resnet_feature_extractor("resnet18", device, cut_layers=3)

    # 3) Dirichlet non-iid federated split (+ optional rho size-skew)
    seed_for_split = DEFAULT_SEED if bench_mode else int(time.time()) % (2**31 - 1)

    if abs(rho - 1.0) < 1e-12:
        party_subsets = split_dataset_dirichlet(
            train_dataset,
            num_parties=num_parties,
            alpha=dirichlet_alpha,
            seed=seed_for_split,
            min_size_per_party=min_party_size,
        )
    else:
        party_subsets = split_dataset_dirichlet_size_skew(
            train_dataset,
            num_parties=num_parties,
            alpha=dirichlet_alpha,
            rho=rho,
            poor_party=poor_party,
            seed=seed_for_split,
            fix_total=size_skew_fix_total,
            min_size_per_party=min_party_size,
        )

    sizes = [len(s) for s in party_subsets]
    print(f"\nParty dataset sizes: {sizes}")
    print(f"Total assigned: {sum(sizes)} / {len(train_dataset)}")
    print(f"Min size: {min(sizes)} | Max size: {max(sizes)}")
    if abs(rho - 1.0) >= 1e-12:
        print(
            f"Size-skew summary: poor_party={poor_party}, rho={rho}, "
            f"ratio(min/max)={min(sizes)/max(sizes):.4f}"
        )

    # 4) Per-party radius estimation from tau
    print("\nEstimating per-party feature-space radii from tau on local data...")
    party_radii: List[float] = []
    for pidx, subset in enumerate(party_subsets):
        print(f"  [Party {pidx}] Estimating radius... (n={len(subset)})")
        radius_p, stats_p = estimate_party_feature_radius(
            party_dataset=subset,
            feature_extractor=feature_extractor,
            device=device,
            tau=tau,
            alpha=0.7,
            max_images_for_stats=2048,
            batch_size_stats=64,
            num_samples_radius=1024,
            batch_size_radius=64,
        )
        party_radii.append(radius_p)
        print(
            f"    r={stats_p['r']:.6f} | v_hat={stats_p['v_hat']:.4e} | d={stats_p['d']} | "
            f"c={stats_p['c']:.4f} | mf={stats_p['mf']:.4f} | feat_radius={radius_p:.6f}"
        )

    avg_radius = float(np.mean(party_radii))
    print(f"\nAverage estimated feature-space radius across parties: {avg_radius:.6f}")

    # 5) Optional visualization (small, uses average radius)
    print("Saving original/mixup feature-map grid pairs before training begins...")
    small_feature_maps = get_feature_maps_for_first_n(
        feature_extractor, train_dataset, device, num_samples=16
    )
    vis_out_dir = f"./mixup_pairs_features_np{num_parties}_rho{rho}_poor{poor_party}"
    save_featuremap_grid_and_mixup_pairs(
        original_feature_maps=small_feature_maps,
        radius=avg_radius,
        output_dir=vis_out_dir,
        num_pairs=5,
    )

    # 6) DataLoaders
    party_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True,
                   num_workers=2, pin_memory=True)
        for subset in party_subsets
    ]

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Discover embedding dimension once
    with torch.no_grad():
        sample_img, _ = train_dataset[0]
        sample_img = sample_img.unsqueeze(0).to(device)
        fmap = feature_extractor(sample_img)
        embedding_dim = fmap.view(1, -1).size(1)

    # -------------------- Baseline: FedProx over all parties (no mixup) -------------------- #
    print("\n==================== Baseline: FedProx (no mixup, Dirichlet non-iid) ====================")
    classifier_fedprox = DenseClassifier(embedding_dim, num_classes).to(device)

    best_acc_fedprox = train_classifier_fedprox(
        device=device,
        feature_extractor=feature_extractor,
        global_classifier=classifier_fedprox,
        party_loaders=party_loaders,
        testloader=test_loader,
        num_epochs=num_epochs,
        lr=lr,
        mu=mu,
    )

    fedprox_ckpt = checkpoint_path(
        prefix="fedprox",
        model="resnet18",
        dataset_name="cifar10",
        num_epochs=num_epochs,
        num_parties=num_parties,
    )
    torch.save(classifier_fedprox.state_dict(), fedprox_ckpt)
    print(f"[FedProx] Best test accuracy: {best_acc_fedprox:.2f}%")
    print(f"[FedProx] Saved checkpoint: {fedprox_ckpt}")

    # -------------------- Mixup-Union: train on UNION of parties (feature-space) ----------- #
    print("\n==================== Mixup-Union: all parties (feature-space) ====================")
    classifier_union = DenseClassifier(embedding_dim, num_classes).to(device)
    optimizer_union = optim.SGD(
        classifier_union.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler_union = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_union, T_max=num_epochs, eta_min=1e-5
    )

    best_acc_union = train_classifier_mixup_union(
        device=device,
        feature_extractor=feature_extractor,
        classifier=classifier_union,
        optimizer=optimizer_union,
        scheduler=scheduler_union,
        party_loaders=party_loaders,
        party_radii=party_radii,
        testloader=test_loader,
        num_epochs=num_epochs,
        num_classes=num_classes,
    )

    union_ckpt = checkpoint_path(
        prefix="mixup_union",
        model="resnet18",
        dataset_name="cifar10",
        num_epochs=num_epochs,
        num_parties=num_parties,
    )
    torch.save(classifier_union.state_dict(), union_ckpt)
    print(f"[Mixup-Union] Best test accuracy: {best_acc_union:.2f}%")
    print(f"[Mixup-Union] Saved checkpoint: {union_ckpt}")

    # Summary for this experiment
    print("\n==================== Experiment Summary ====================")
    print(f"num_parties: {num_parties}")
    print(f"Dirichlet alpha: {dirichlet_alpha}")
    print(f"rho: {rho} | poor_party={poor_party} | fix_total={size_skew_fix_total}")
    print(f"min_party_size: {min_party_size}")
    print(f"Party sizes: {sizes}")
    print(f"tau: {tau}")
    print(f"mu (FedProx): {mu}")
    print(f"Average feature-space radius: {avg_radius:.6f}")
    print(f"FedProx best test acc:        {best_acc_fedprox:.2f}%")
    print(f"Mixup-Union best test acc:    {best_acc_union:.2f}%")
    print("============================================================")

    workflow_duration = time.time() - workflow_start_time
    print(f"Experiment execution time: {workflow_duration:.2f} s")

    # Optional final eval (uses already-trained models in memory)
    if finish_with_model_eval:
        print("\nRe-evaluating both models once more (no checkpoint reload)...")
        acc_fedprox_final = evaluate_classifier(
            classifier_fedprox, feature_extractor, device, test_loader)
        acc_union_final = evaluate_classifier(
            classifier_union, feature_extractor, device, test_loader)
        print(f"FedProx final test accuracy:     {acc_fedprox_final:.2f}%")
        print(f"Mixup-Union final test accuracy: {acc_union_final:.2f}%")

    print(f"============ Experiment END (num_parties={num_parties}) ============\n")


# --------------------------- CLI --------------------------- #

if __name__ == "__main__":
    batch_size = 128
    device = get_optimal_device()

    parser = argparse.ArgumentParser(
        prog="Augmented Mixup vs FedProx (tau-based radii, Dirichlet non-iid, optional rho size-skew + min-size guard)",
        description=(
            "Runs a single experiment for a chosen number of parties with "
            "Dirichlet non-iid label splits; each party estimates its own feature-space "
            "radius from tau, and compares FedProx vs Mixup-Union. "
            "Optionally enables 'one data-poor party' split via --rho and guarantees "
            "each party has at least --min-party-size samples."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1e-6,
        help="Privacy/utility parameter tau used to calibrate feature-space radius per party.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs / global rounds.",
    )
    parser.add_argument(
        "-b",
        "--bench",
        action="store_true",
        help="Set a seed and deterministic PyTorch settings for reproducibility.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the classifier (FedProx locals and Mixup-Union).",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.001,
        help="FedProx proximal regularization coefficient mu.",
    )
    parser.add_argument(
        "--num-parties",
        "--num_parties",
        type=int,
        default=2,
        help="Number of parties for Dirichlet non-iid CIFAR-10 splits.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.5,
        help=(
            "Concentration parameter alpha for Dirichlet-based non-iid "
            "label partitioning (>0). Smaller = more skewed (more non-iid)."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help=(
            "Size-skew ratio for one data-poor party: n_poor = rho * n_rich, others equal n_rich. "
            "Default 1.0 reproduces the original equal-size setting."
        ),
    )
    parser.add_argument(
        "--poor-party",
        type=int,
        default=0,
        help="Index of the data-poor party (0..num_parties-1). Default: 0.",
    )
    parser.add_argument(
        "--no-size-skew-fix-total",
        action="store_true",
        help=(
            "Disable exact party-size balancing to keep total fixed. "
            "By default, we enforce exact target sizes summing to N=len(train_dataset)."
        ),
    )
    parser.add_argument(
        "--min-party-size",
        type=int,
        default=10,
        help="Guarantee each party has at least this many samples by rebalancing after the split.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory to store log files.",
    )

    args = parser.parse_args()

    if args.num_parties < 1:
        raise ValueError("--num-parties must be >= 1.")
    if args.dirichlet_alpha <= 0:
        raise ValueError("--dirichlet-alpha must be > 0.")
    if args.rho <= 0:
        raise ValueError("--rho must be > 0.")
    if args.min_party_size < 0:
        raise ValueError("--min-party-size must be >= 0.")
    if not (0 <= args.poor_party < args.num_parties):
        raise ValueError("--poor-party must be in [0, num_parties-1].")

    if args.bench:
        set_deterministic_behavior(DEFAULT_SEED)

    size_skew_fix_total = not args.no_size_skew_fix_total

    LOG_DIR = args.log_dir

    # Create a single unified log file and redirect all stdout
    unified_log = generate_unified_log_filename(
        tag="federated_multi_dirichlet",
        model="resnet18",
        dataset_name="cifar10",
        num_epochs=args.epochs,
    )
    tee = Tee(unified_log)
    print(f"Saving experiment log to: {unified_log}")
    print(f"Run started on: {datetime.datetime.now().ctime()}")
    print(
        f"Planned experiment with num_parties={args.num_parties}, "
        f"tau={args.tau}, mu={args.mu}, dirichlet_alpha={args.dirichlet_alpha}, "
        f"rho={args.rho}, poor_party={args.poor_party}, fix_total={size_skew_fix_total}, "
        f"min_party_size={args.min_party_size}"
    )

    try:
        augment_workflow(
            device=device,
            num_epochs=args.epochs,
            batch_size=batch_size,
            tau=args.tau,
            lr=args.lr,
            mu=args.mu,
            bench_mode=args.bench,
            num_parties=args.num_parties,
            dirichlet_alpha=args.dirichlet_alpha,
            rho=args.rho,
            poor_party=args.poor_party,
            size_skew_fix_total=size_skew_fix_total,
            min_party_size=args.min_party_size,
            finish_with_model_eval=True,
            tee=tee,
        )

        print("\n==================== EXPERIMENT COMPLETED ====================")
        print(f"Run finished on: {datetime.datetime.now().ctime()}")
        print("==============================================================\n")
    finally:
        tee.close()
