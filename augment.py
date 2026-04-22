import os
import time
import datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import (
    DEFAULT_SEED, SUPPORTED_DATASETS, SUPPORTED_MODELS, RADIUS_MAPPING,
    get_dataset, build_resnet_feature_extractor, get_optimal_device,
    set_deterministic_behavior, generate_log_and_ckpt_files,
    Tee, DenseClassifier, mixup_batch_in_feature_space,
    get_feature_maps_for_first_n, save_featuremap_grid_and_mixup_pairs,
    cross_entropy_with_soft_targets,
)


# --------------------------- Training / evaluation --------------------------- #


def train_and_eval(
    device,
    feature_extractor,
    classifier,
    optimizer,
    scheduler,
    trainloader,
    testloader,
    num_epochs,
    checkpoint_file,
    radius,
    num_classes,
    resume_checkpoint: str | None = None,
) -> None:
    classifier.to(device)
    feature_extractor.to(device)
    feature_extractor.eval()

    start_epoch = 0
    best_acc = 0.0

    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        classifier.load_state_dict(ckpt['classifier_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        print(f"Resumed from checkpoint: {resume_checkpoint}")
        print(f"  Starting at epoch {start_epoch + 1}/{num_epochs} | best_acc so far: {best_acc:.2f}%")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        classifier.train()
        train_loss = 0.0
        for imgs, labels in tqdm(trainloader, dynamic_ncols=True):
            optimizer.zero_grad()

            mix_feats, mix_soft_labels = mixup_batch_in_feature_space(
                feature_extractor,
                imgs,
                labels,
                device,
                radius,
                num_classes,
            )

            outputs = classifier(mix_feats)
            loss = cross_entropy_with_soft_targets(outputs, mix_soft_labels)
            train_loss += loss.item() * mix_feats.size(0)
            loss.backward()
            optimizer.step()

        train_loss /= len(trainloader.dataset)
        scheduler.step()

        # -------------------- Evaluation (hard labels, on-the-fly features) -------------------- #
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                fmap = feature_extractor(imgs)
                feats = fmap.view(fmap.size(0), -1)
                outputs = classifier(feats)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = correct / total * 100.0

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, checkpoint_file)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

    print(f"Best test accuracy: {best_acc:.2f}%")


def test_model(classifier, feature_extractor, device, test_loader):
    """
    Final evaluation with best saved classifier, computing features on demand.
    """
    classifier.eval()
    classifier.to(device)
    feature_extractor.eval()

    eval_start = time.time()
    with torch.no_grad():
        total = 0
        correct = 0
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            fmap = feature_extractor(imgs)
            feats = fmap.view(fmap.size(0), -1)
            y = classifier(feats)
            preds = torch.argmax(y, dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.shape[0]

    acc = correct / total * 100.0
    eval_finish = time.time() - eval_start
    print(
        f"Final test accuracy with best model: {acc:.2f}% [{eval_finish:.2f} s]")


# --------------------------- Main workflow --------------------------- #


def augment_workflow(
    feature_extractor_type: str,
    dataset_type: str,
    classifier_type: str,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    radius: float,
    lr: float,
    bench_mode: bool,
    finish_with_model_eval: bool = True,
    resume_checkpoint: str | None = None,
) -> None:
    log_id, log_filename, checkpoint_file = generate_log_and_ckpt_files(
        feature_extractor_type, dataset_type, num_epochs
    )

    # If resuming, save back to the same checkpoint file
    if resume_checkpoint is not None:
        checkpoint_file = resume_checkpoint

    # 1) Datasets
    train_dataset, test_dataset, num_classes = get_dataset(dataset_type)

    # 2) Feature extractor (ResNet backbone cut after layer4)
    feature_extractor = build_resnet_feature_extractor(
        feature_extractor_type, device
    )
    feature_layers = list(feature_extractor.children())

    # ALWAYS log to file using Tee
    tee = Tee(log_filename)
    print(f"Saving logs to: {log_filename}")

    print(f"------------ Augmented Mixup (Feature Space) ------------")
    print(f"Started on: {datetime.datetime.now().ctime()} (Log ID: #{log_id})")
    print(f"Torch seed: {torch.random.initial_seed()}")
    print(f"Deterministic behavior: {bench_mode}")
    print(f"Using device: {device}")
    print(f"Model: {feature_extractor_type} [{len(feature_layers)} layers]")
    print(f"Dataset: {dataset_type} | Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"radius (r): {radius}")
    print(f"--------------------------------------------------------------------------")

    workflow_start_time = time.time()

    # 3) Visualize a few pairs of original/mixup feature-map grids BEFORE training
    print("Saving original/mixup feature-map grid pairs before training begins...")
    small_feature_maps = get_feature_maps_for_first_n(
        feature_extractor, train_dataset, device, num_samples=16
    )
    save_featuremap_grid_and_mixup_pairs(
        original_feature_maps=small_feature_maps,
        radius=radius,
        output_dir="./mixup_pairs_features",
        num_pairs=5,
    )

    # 4) Standard image loaders; features are computed on-the-fly inside training/eval
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False)

    # 5) Classifier on top of feature vectors
    if classifier_type == "dense":
        # Infer embedding dimension from a single sample
        with torch.no_grad():
            sample_img, _ = train_dataset[0]
            sample_img = sample_img.unsqueeze(0).to(device)
            fmap = feature_extractor(sample_img)
            embedding_dim = fmap.view(1, -1).size(1)
        classifier = DenseClassifier(embedding_dim, num_classes).to(device)
    else:
        raise NotImplementedError("Only a DenseClassifier is supported...")

    optimizer = optim.SGD(
        classifier.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5)

    print(f"--------------- Classifier size ---------------")
    classifier.get_classifier_size()
    print(f"-----------------------------------------------")
    print(
        f"Training classifier on feature-space mixup ({feature_extractor_type}, {dataset_type}) "
        f"on {device} for {num_epochs} epochs"
    )

    # 6) Train & evaluate (features and mixup on demand)
    train_and_eval(
        device=device,
        feature_extractor=feature_extractor,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=train_loader,
        testloader=test_loader,
        num_epochs=num_epochs,
        checkpoint_file=checkpoint_file,
        radius=radius,
        num_classes=num_classes,
        resume_checkpoint=resume_checkpoint,
    )

    workflow_duration = time.time() - workflow_start_time
    print(f"Total execution time: {workflow_duration:.2f} s")
    print(f"-----------------------------------------------")

    # Load best classifier and test (again, features on demand)
    if finish_with_model_eval:
        ckpt = torch.load(checkpoint_file, map_location=device)
        if isinstance(ckpt, dict) and 'classifier_state_dict' in ckpt:
            classifier.load_state_dict(ckpt['classifier_state_dict'])
        else:
            classifier.load_state_dict(ckpt)  # backward compat with old checkpoints
        test_model(classifier, feature_extractor, device, test_loader)

    # ALWAYS close Tee at the end
    tee.close()


# --------------------------- CLI --------------------------- #


if __name__ == "__main__":
    batch_size = 128
    device = get_optimal_device()

    parser = argparse.ArgumentParser(
        prog="Augmented Mixup", description="Augmented Mixup (Feature Space, memory-efficient)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Mixing radius r in feature space.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnet18",
        help="The ResNet architecture to use for feature extraction (resnet18, resnet34, resnet50).",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset to use: mnist, cifar10, cifar100, tiny-imagenet.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=200,
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    args = parser.parse_args()

    args.dataset = str(args.dataset).lower()
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset <{args.dataset}> not supported. Currently supported datasets: {SUPPORTED_DATASETS}"
        )
    args.model = str(args.model).lower()
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model <{args.model}> not supported. Currently supported models: {SUPPORTED_MODELS}")

    if args.bench:
        set_deterministic_behavior(DEFAULT_SEED)

    if args.radius is None:
        args.radius = RADIUS_MAPPING[args.dataset]
        print(
            f'Setting the radius automatically -> r({args.dataset}) = {args.radius}')

    # python3 augment.py --model effnet_v2 --epochs 1 --dataset cifar10 --bench
    augment_workflow(
        feature_extractor_type=args.model,
        dataset_type=args.dataset,
        classifier_type="dense",
        device=device,
        num_epochs=args.epochs,
        batch_size=batch_size,
        radius=args.radius,
        lr=args.lr,
        bench_mode=args.bench,
        resume_checkpoint=args.checkpoint,
    )
