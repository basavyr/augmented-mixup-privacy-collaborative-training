from __future__ import annotations

import argparse
import glob
import io
import os
import random
import time
from typing import Callable, Iterator, Literal, Optional, Tuple

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Standard ImageNet normalization constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_train_transform(image_size: int = 224) -> transforms.Compose:
    """Standard ImageNet training augmentation."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def default_val_transform(
    image_size: int = 224, resize_size: int = 256
) -> transforms.Compose:
    """Standard ImageNet validation transform (resize + center-crop)."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_shards(
    data_dir: str, split: str
) -> list[str]:
    """Find and validate parquet shard files for the given split."""
    pattern = os.path.join(data_dir, f"{split}-*.parquet")
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(
            f"No parquet shards found for split='{split}' in {data_dir!r}.\n"
            f"Expected files matching: {pattern}\n"
            f"Make sure you cloned the HuggingFace repo and the 'data/' "
            f"directory contains files like '{split}-00000-of-NNNNN.parquet'."
        )
    return shard_paths


def _read_shard(path: str) -> list[Tuple[bytes, int]]:
    """
    Read a single parquet shard and return a list of (jpeg_bytes, label).

    The Arrow table is released immediately after extraction.
    """
    table = pq.read_table(path, columns=["image", "label"])
    image_col = table.column("image")
    label_col = table.column("label")

    samples: list[Tuple[bytes, int]] = []
    for i in range(table.num_rows):
        image_struct = image_col[i].as_py()
        if isinstance(image_struct, dict):
            jpeg_bytes = image_struct["bytes"]
        else:
            jpeg_bytes = image_struct
        label = label_col[i].as_py()
        samples.append((jpeg_bytes, label))

    del table, image_col, label_col
    return samples


def _decode_sample(
    jpeg_bytes: bytes, label: int, transform: Callable
) -> Tuple[torch.Tensor, int]:
    """Decode JPEG bytes to a transformed tensor and return (tensor, label)."""
    image = Image.open(io.BytesIO(jpeg_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = transform(image)
    return image, label


def _resolve_transform(
    transform: Optional[Callable], split: str, image_size: int
) -> Callable:
    """Pick the right transform: user-provided, or default for the split."""
    if transform is not None:
        return transform
    if split == "train":
        return default_train_transform(image_size)
    return default_val_transform(image_size)


# ---------------------------------------------------------------------------
# Streaming dataset (IterableDataset) -- for the training split
# ---------------------------------------------------------------------------
class ImageNetStreamingDataset(IterableDataset):
    """
    Streams ImageNet images from parquet shards in configurable chunks.

    At each epoch the shard order is randomized.  Shards are loaded in
    chunks of ``shards_per_chunk`` and their samples are pooled and
    shuffled together before yielding.  This keeps RAM usage bounded
    while providing good cross-shard mixing within each chunk.

    ``__len__`` is implemented (returns total images across all shards) so
    that callers like ``train_loss /= len(trainloader.dataset)`` work
    correctly.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the parquet files.
    split : str
        Which split to load (``"train"`` or ``"validation"``).
    transform : callable, optional
        Torchvision transform applied to each PIL image.  If *None* the
        standard ImageNet transform for the split is used.
    image_size : int
        Target spatial resolution (default 224).
    shards_per_chunk : int
        Number of shards to load into memory at once (default 1).
        Each shard is ~130 MB of JPEG bytes, so memory usage is roughly
        ``shards_per_chunk * 130 MB``.  Examples:

        - ``1``   -> ~130 MB  (minimum memory, intra-shard shuffle only)
        - ``10``  -> ~1.3 GB  (good cross-shard mixing)
        - ``50``  -> ~6.5 GB  (very good mixing)
        - ``294`` -> ~38 GB   (full dataset in memory, global shuffle)
    """

    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "validation"] = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        shards_per_chunk: int = 1,
    ) -> None:
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.split = split
        self.image_size = image_size
        self.transform = _resolve_transform(transform, split, image_size)
        self.shards_per_chunk = max(1, shards_per_chunk)

        # Discover shards and read metadata (row counts only -- no pixel data).
        self._shard_paths = _discover_shards(self.data_dir, split)

        self._shard_row_counts: list[int] = []
        self._total = 0
        for path in self._shard_paths:
            pf = pq.ParquetFile(path)
            n = pf.metadata.num_rows
            self._shard_row_counts.append(n)
            self._total += n

        mem_estimate_mb = self.shards_per_chunk * 130
        print(
            f"[ImageNetStreamingDataset] split={split}  "
            f"shards={len(self._shard_paths)}  "
            f"images={self._total:,}  "
            f"shards_per_chunk={self.shards_per_chunk}  "
            f"(~{mem_estimate_mb} MB)"
        )

    def __len__(self) -> int:
        return self._total

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        """
        Iterate over all shards once (= one epoch).

        Shard order is shuffled, then shards are loaded in chunks of
        ``shards_per_chunk``.  All samples within a chunk are pooled and
        shuffled together before yielding.
        """
        shard_order = list(range(len(self._shard_paths)))
        random.shuffle(shard_order)

        # Process shards in chunks.
        for chunk_start in range(0, len(shard_order), self.shards_per_chunk):
            chunk_indices = shard_order[
                chunk_start : chunk_start + self.shards_per_chunk
            ]

            # Pool all samples from the shards in this chunk.
            chunk_samples: list[Tuple[bytes, int]] = []
            for shard_idx in chunk_indices:
                chunk_samples.extend(_read_shard(self._shard_paths[shard_idx]))

            random.shuffle(chunk_samples)

            for jpeg_bytes, label in chunk_samples:
                yield _decode_sample(jpeg_bytes, label, self.transform)

            del chunk_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Random access for a single sample.

        This is *not* used during training iteration (``__iter__`` is used
        instead).  It exists so that callers that probe the dataset for a
        single sample (e.g. to determine the embedding dimension) can do
        so without loading the full dataset.
        """
        if index < 0 or index >= self._total:
            raise IndexError(
                f"index {index} out of range for dataset of size {self._total}"
            )

        # Walk cumulative counts to find the right shard.
        cumulative = 0
        for shard_idx, count in enumerate(self._shard_row_counts):
            if index < cumulative + count:
                row_in_shard = index - cumulative
                break
            cumulative += count
        else:
            raise IndexError(f"index {index} could not be mapped to a shard")

        # Read just that shard and extract the needed row.
        samples = _read_shard(self._shard_paths[shard_idx])
        jpeg_bytes, label = samples[row_in_shard]
        del samples
        return _decode_sample(jpeg_bytes, label, self.transform)


# ---------------------------------------------------------------------------
# Preloaded dataset (map-style) -- for the validation split
# ---------------------------------------------------------------------------
class ImageNetPreloadedDataset(Dataset):
    """
    Preloads all images (as raw JPEG bytes) into memory at init time.

    After loading, ``__getitem__`` is an O(1) list lookup followed by JPEG
    decoding and transform application.  Ideal for the validation split
    (14 shards, ~50K images, ~1.5 GB of JPEG bytes).

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the parquet files.
    split : str
        Which split to load (``"train"`` or ``"validation"``).
    transform : callable, optional
        Torchvision transform applied to each PIL image.
    image_size : int
        Target spatial resolution (default 224).
    """

    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "validation"] = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.split = split
        self.image_size = image_size
        self.transform = _resolve_transform(transform, split, image_size)

        shard_paths = _discover_shards(self.data_dir, split)

        print(
            f"[ImageNetPreloadedDataset] Loading {split} split from "
            f"{len(shard_paths)} shards..."
        )

        self._samples: list[Tuple[bytes, int]] = []
        for path in tqdm(shard_paths, desc=f"Loading {split} shards",
                         unit="shard", dynamic_ncols=True):
            self._samples.extend(_read_shard(path))

        print(
            f"[ImageNetPreloadedDataset] split={split}  "
            f"shards={len(shard_paths)}  "
            f"images={len(self._samples):,}  "
            f"loaded into memory"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index < 0 or index >= len(self._samples):
            raise IndexError(
                f"index {index} out of range for dataset of size "
                f"{len(self._samples)}"
            )
        jpeg_bytes, label = self._samples[index]
        return _decode_sample(jpeg_bytes, label, self.transform)


# ---------------------------------------------------------------------------
# Convenience: build train & val DataLoaders in one call
# ---------------------------------------------------------------------------
def create_imagenet_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    image_size: int = 224,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation ``DataLoader`` objects.

    The train loader uses ``ImageNetStreamingDataset`` (shard-streaming,
    bounded memory).  The val loader uses ``ImageNetPreloadedDataset``
    (preloaded, fast random access).

    Parameters
    ----------
    data_dir : str
        Path to the directory with parquet shard files.
    batch_size : int
        Mini-batch size (default 128).
    image_size : int
        Crop / resize target (default 224).
    pin_memory : bool
        Whether to use pinned (page-locked) memory for faster GPU transfer.
    train_transform : callable, optional
        Custom transform for the training split.
    val_transform : callable, optional
        Custom transform for the validation split.

    Returns
    -------
    train_loader, val_loader : tuple[DataLoader, DataLoader]
    """
    train_ds = ImageNetStreamingDataset(
        data_dir=data_dir,
        split="train",
        transform=train_transform,
        image_size=image_size,
    )
    val_ds = ImageNetPreloadedDataset(
        data_dir=data_dir,
        split="validation",
        transform=val_transform,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,       # streaming dataset handles shuffling internally
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI: quick sanity check
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check the ImageNet parquet dataset loader."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing parquet shards "
        "(e.g. ./imagenet-1k/data).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Which split to test (default: validation).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the test DataLoader (default: 32).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="How many batches to iterate for the test (default: 5).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size (default: 224).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ImageNet Parquet Dataset - Sanity Check")
    print("=" * 60)

    # Use streaming for train, preloaded for validation.
    if args.split == "train":
        ds = ImageNetStreamingDataset(
            data_dir=args.data_dir,
            split=args.split,
            image_size=args.image_size,
        )
        use_shuffle = False  # streaming handles it
    else:
        ds = ImageNetPreloadedDataset(
            data_dir=args.data_dir,
            split=args.split,
            image_size=args.image_size,
        )
        use_shuffle = False

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=use_shuffle,
        num_workers=0,
        pin_memory=False,
    )

    print(
        f"\nIterating {args.num_batches} batches "
        f"(batch_size={args.batch_size})...\n"
    )

    t0 = time.perf_counter()
    for i, (images, labels) in enumerate(loader):
        if i >= args.num_batches:
            break
        elapsed = time.perf_counter() - t0
        print(
            f"  batch {i + 1:3d}  |  images: {list(images.shape)}  "
            f"dtype={images.dtype}  |  labels: {list(labels.shape)}  "
            f"min={labels.min().item()} max={labels.max().item()}  |  "
            f"time: {elapsed:.2f}s"
        )
        t0 = time.perf_counter()

    # Quick single-sample check.
    print("\nSingle-sample check (index 0):")
    img, lbl = ds[0]
    print(f"  image shape: {list(img.shape)}  dtype: {img.dtype}")
    print(f"  pixel range:  [{img.min().item():.3f}, {img.max().item():.3f}]")
    print(f"  label: {lbl}")
    print("\nDone.")


if __name__ == "__main__":
    main()
