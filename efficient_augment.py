import argparse

# local imports
from augment import augment_workflow
from utils import get_optimal_device, set_deterministic_behavior, RADIUS_MAPPING, SUPPORTED_DATASETS


def efficient_net_workflow(dataset_type: str, seed: int, num_epochs: int, learning_rate: float, radius: float, use_efficientnet_v2: bool = True, quick_run: bool = False, resume_checkpoint: str | None = None):
    device = get_optimal_device()
    set_deterministic_behavior(seed)

    batch_size = 128
    if quick_run:
        print(f"Fast run enabled")
        num_epochs = 1
        batch_size = 128

    feature_extractor_type = "effnet_v2" if use_efficientnet_v2 else "effnet_v1"
    augment_workflow(
        feature_extractor_type=feature_extractor_type,
        dataset_type=dataset_type,
        classifier_type="dense",
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        radius=radius,
        lr=learning_rate,
        bench_mode=True,
        finish_with_model_eval=True,
        resume_checkpoint=resume_checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description="Train ImageNet on EfficientNet-B0")
    parser.add_argument("--epochs", type=int, default=10,
                        help="The number of training episodes")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="The learning rate during training")
    parser.add_argument("--radius", type=float, default=None,
                        help="The learning rate during training")
    parser.add_argument("--v2", action="store_true",
                        help="Use EfficientNet V2")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick smoke test")
    parser.add_argument("--seed", type=int, default=1137,
                        help="Run a quick smoke test")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset used for training")
    parser.add_argument("--shards-per-chunk", type=int, default=1,
                        help="Number of parquet shards to load at once for "
                        "ImageNet (default: 1, ~130 MB per shard). "
                        "Ignored for non-ImageNet datasets.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint file to resume training from.")
    args = parser.parse_args()

    args.dataset = str(args.dataset).lower()
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset <{args.dataset}> not supported. Currently supported datasets: {SUPPORTED_DATASETS}"
        )
    if args.radius is None:
        args.radius = RADIUS_MAPPING[args.dataset]
        print(
            f'Setting the radius automatically -> r({args.dataset}) = {args.radius}')

    # python3 efficient_augment.py --dataset cifar10
    efficient_net_workflow(
        dataset_type=args.dataset,
        seed=args.seed,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        radius=args.radius,
        use_efficientnet_v2=args.v2,
        quick_run=args.quick,
        resume_checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
