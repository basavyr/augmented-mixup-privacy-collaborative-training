import os
import time
import datetime
import argparse
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import (
    LOG_DIR, CHECKPOINTS_DIR, DEFAULT_SEED,
    get_optimal_device, set_deterministic_behavior,
    Tee, get_dataset, build_resnet_feature_extractor,
    DenseClassifier, mixup_batch_in_feature_space,
    get_feature_maps_for_first_n, save_featuremap_grid_and_mixup_pairs,
    split_dataset_equally, cross_entropy_with_soft_targets, evaluate_classifier,
)


# --------------------------- Local helpers (specific to federated experiments) --- #

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


# --------------------------- Training (federated-specific) --------------------------- #

def train_classifier_standard(
    device,
    feature_extractor,
    classifier,
    optimizer,
    scheduler,
    trainloader,
    testloader,
    num_epochs,
) -> float:
    """
    Standard training (NO mixup), features computed on-the-fly.
    Returns best test accuracy.
    """
    best_acc = 0.0
    classifier.to(device)
    feature_extractor.eval()

    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        classifier.train()
        train_loss = 0.0

        for imgs, labels in trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                fmap = feature_extractor(imgs)
                feats = fmap.view(fmap.size(0), -1)

            optimizer.zero_grad()
            outputs = classifier(feats)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_loss /= len(trainloader.dataset)
        test_acc = evaluate_classifier(classifier, feature_extractor, device, testloader)
        best_acc = max(best_acc, test_acc)
        print(
            f"[Std] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.8f}"
        )
    return best_acc


def train_classifier_mixup_union(
    device: torch.device,
    feature_extractor,
    classifier,
    optimizer,
    scheduler,
    party_loaders: List[DataLoader],
    testloader: DataLoader,
    num_epochs: int,
    radius: float,
    num_classes: int,
) -> float:
    """
    Train on the UNION of party-local mixup batches (mix within each party only),
    iterating over all parties each epoch. Returns best test accuracy.
    """
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
            for imgs, labels in loader:
                optimizer.zero_grad()
                mix_feats, mix_soft_labels = mixup_batch_in_feature_space(
                    feature_extractor, imgs, labels, device, radius, num_classes
                )
                outputs = classifier(mix_feats)
                loss = cross_entropy_with_soft_targets(outputs, mix_soft_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * mix_feats.size(0)
                processed += mix_feats.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, processed)
        test_acc = evaluate_classifier(classifier, feature_extractor, device, testloader)
        best_acc = max(best_acc, test_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"[Mixup-Union] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

    return best_acc


# --------------------------- Main experiment runner --------------------------- #

def augment_workflow(
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    radius: float,
    lr: float,
    bench_mode: bool,
    num_parties: int,
    finish_with_model_eval: bool = True,
    tee: Optional[Tee] = None,
) -> None:
    """
    Runs a single experiment for a given num_parties.
    Logging is assumed to be already redirected by a Tee created outside.
    """
    print(f"\n============ Experiment START (num_parties={num_parties}) ============")
    print(f"Started on: {datetime.datetime.now().ctime()}")
    print(f"Torch seed: {torch.random.initial_seed()}")
    print(f"Deterministic behavior: {bench_mode}")
    print(f"Using device: {device}")
    print(f"Model: resnet18 [cut after layer4]")
    print(f"Dataset: cifar10")
    print(f"Epochs: {num_epochs}")
    print(f"radius (r): {radius}")
    print(f"num_parties: {num_parties}")
    print("----------------------------------------------------")

    workflow_start_time = time.time()

    # 1) Dataset
    train_dataset, test_dataset, num_classes = get_dataset("cifar10")

    # 2) Feature extractor
    feature_extractor = build_resnet_feature_extractor("resnet18", device)

    # 3) Optional visualization (small)
    print("Saving original/mixup feature-map grid pairs before training begins...")
    small_feature_maps = get_feature_maps_for_first_n(
        feature_extractor, train_dataset, device, num_samples=16
    )
    vis_out_dir = f"./mixup_pairs_features_np{num_parties}"
    save_featuremap_grid_and_mixup_pairs(
        original_feature_maps=small_feature_maps,
        radius=radius,
        output_dir=vis_out_dir,
        num_pairs=5,
    )

    # 4) Federated split
    seed_for_split = DEFAULT_SEED if bench_mode else int(time.time()) % (2**31 - 1)
    party_subsets = split_dataset_equally(train_dataset, num_parties=num_parties, seed=seed_for_split)

    # We'll use party 0 for the baseline (no mixup)
    party0_loader = DataLoader(
        party_subsets[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # All parties for the mixup-union experiment
    party_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        for subset in party_subsets
    ]

    # Global test loader
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

    # -------------------- Baseline: train on ONE shard (party 0), NO mixup -------------------- #
    print("\n==================== Baseline: single-party (no mixup) ====================")
    classifier_baseline = DenseClassifier(embedding_dim, num_classes).to(device)
    optimizer_baseline = optim.SGD(
        classifier_baseline.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler_baseline = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_baseline, T_max=num_epochs, eta_min=1e-5
    )

    best_acc_baseline = train_classifier_standard(
        device=device,
        feature_extractor=feature_extractor,
        classifier=classifier_baseline,
        optimizer=optimizer_baseline,
        scheduler=scheduler_baseline,
        trainloader=party0_loader,
        testloader=test_loader,
        num_epochs=num_epochs,
    )

    baseline_ckpt = checkpoint_path(
        prefix="single_party_no_mixup",
        model="resnet18",
        dataset_name="cifar10",
        num_epochs=num_epochs,
        num_parties=num_parties,
    )
    torch.save(classifier_baseline.state_dict(), baseline_ckpt)
    print(f"[Baseline] Best test accuracy (party 0, no mixup): {best_acc_baseline:.2f}%")
    print(f"[Baseline] Saved checkpoint: {baseline_ckpt}")

    # -------------------- Mixup-Union: train on UNION of parties (feature-space) -------------------- #
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
        testloader=test_loader,
        num_epochs=num_epochs,
        radius=radius,
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
    print(f"Baseline (party 0, no mixup) best test acc: {best_acc_baseline:.2f}%")
    print(f"Mixup-Union (all parties) best test acc:   {best_acc_union:.2f}%")
    print("============================================================")

    workflow_duration = time.time() - workflow_start_time
    print(f"Experiment (num_parties={num_parties}) execution time: {workflow_duration:.2f} s")

    # Optional final eval (uses already-trained models in memory)
    if finish_with_model_eval:
        print("\nRe-evaluating both models once more (no checkpoint reload)...")
        acc_base_final = evaluate_classifier(classifier_baseline, feature_extractor, device, test_loader)
        acc_union_final = evaluate_classifier(classifier_union, feature_extractor, device, test_loader)
        print(f"Baseline final test accuracy:   {acc_base_final:.2f}%")
        print(f"Mixup-Union final test accuracy:{acc_union_final:.2f}%")

    print(f"============ Experiment END (num_parties={num_parties}) ============\n")


# --------------------------- CLI --------------------------- #

if __name__ == "__main__":
    batch_size = 128
    device = get_optimal_device()

    parser = argparse.ArgumentParser(
        prog="Augmented Mixup (Single-Party Baseline vs Mixup-Union)",
        description="Runs three experiments over num_parties in {10, 20, 30}.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Mixing radius r in feature space.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
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
        help="Learning rate for the classifier.",
    )

    args = parser.parse_args()

    if args.bench:
        set_deterministic_behavior(DEFAULT_SEED)

    # Create a single unified log file and redirect all stdout for all experiments
    unified_log = generate_unified_log_filename(
        tag="federated_multi",
        model="resnet18",
        dataset_name="cifar10",
        num_epochs=args.epochs,
    )
    tee = Tee(unified_log)
    print(f"Saving ALL experiments to a single log file: {unified_log}")
    print(f"Run started on: {datetime.datetime.now().ctime()}")

    try:
        experiments = [10, 20, 30]
        print(f"Planned experiments over num_parties: {experiments}")

        for np_val in experiments:
            augment_workflow(
                device=device,
                num_epochs=args.epochs,
                batch_size=batch_size,
                radius=args.radius,
                lr=args.lr,
                bench_mode=args.bench,
                num_parties=np_val,
                finish_with_model_eval=True,
                tee=tee,
            )

        print("\n==================== ALL EXPERIMENTS COMPLETED ====================")
        print(f"Experiments finished on: {datetime.datetime.now().ctime()}")
        print("===================================================================\n")
    finally:
        tee.close()
