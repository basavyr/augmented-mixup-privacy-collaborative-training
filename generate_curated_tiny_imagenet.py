import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

# Mapping of WordNet IDs to CIFAR-10 semantic classes
CIFAR10_TO_WORDNET = {
    "dog":        ["n02124075"],                           # Domestic dog
    "cat":        ["n02123394", "n02123045"],              # Cat, Tabby cat
    # Various birds
    "bird":       ["n01983481", "n01855672", "n01945685", "n02206856"],
    "frog":       ["n01641577"],                           # Frog
    "horse":      ["n01950731"],                           # Horse
    "deer":       ["n02129165"],                           # Antelope
    # Pickup truck, Truck
    "truck":      ["n04099969", "n03930313"],
    # Minibus, Jinrikisha (car-like)
    "automobile": ["n02814533", "n02814860"],
    "ship":       ["n04507155"],                           # Ship
    # "airplane": NOT IN TINY-IMAGENET (would be n02690373, n04270147 but not in the 200 classes)
}


def get_tiny_imagenet_wnids(tiny_imagenet_root: str) -> Set[str]:
    """Load the list of WordNet IDs in Tiny-ImageNet."""
    wnids_file = os.path.join(tiny_imagenet_root, "wnids.txt")
    with open(wnids_file) as f:
        return set(line.strip() for line in f)


def verify_cifar10_mapping(tiny_imagenet_root: str) -> Dict[str, List[str]]:
    """Verify which CIFAR-10 classes and their WordNet equivalents exist in Tiny-ImageNet."""
    tiny_wnids = get_tiny_imagenet_wnids(tiny_imagenet_root)

    verified_mapping = {}
    for cifar_class, wnids in CIFAR10_TO_WORDNET.items():
        found_wnids = [w for w in wnids if w in tiny_wnids]
        if found_wnids:
            verified_mapping[cifar_class] = found_wnids

    print("CIFAR-10 Classes and Their Tiny-ImageNet Equivalents:")
    print("=" * 70)
    for cifar_class in sorted(CIFAR10_TO_WORDNET.keys()):
        found = verified_mapping.get(cifar_class, [])
        if found:
            print(f"  ✓ {cifar_class:12} -> {found}")
        else:
            print(f"  ✗ {cifar_class:12} -> NOT FOUND IN TINY-IMAGENET")
    print("=" * 70)
    print(f"Total: {len(verified_mapping)}/10 CIFAR-10 classes available\n")

    return verified_mapping


def create_curated_subset(tiny_imagenet_root: str, output_root: str,
                          verified_mapping: Dict[str, List[str]]) -> None:
    """
    Create a curated subset of Tiny-ImageNet by symlinking only CIFAR-10-matching classes.

    Structure:
      output_root/
        train/
          dog/ -> symlink to tiny-imagenet-200/train/n02124075/
          cat/ -> symlink to tiny-imagenet-200/train/n02123394/
          ...
        val/
          (same structure)
    """

    print(f"Creating curated subset at: {output_root}\n")

    for split in ["train", "val"]:
        split_dir = os.path.join(output_root, split)
        os.makedirs(split_dir, exist_ok=True)

        for cifar_class, wnids in verified_mapping.items():
            # Create a class directory for this CIFAR-10 class
            class_dir = os.path.join(split_dir, cifar_class)
            os.makedirs(class_dir, exist_ok=True)

            # For each WordNet ID in this class, symlink its images
            for wnid in wnids:
                src_dir = os.path.join(
                    tiny_imagenet_root, split, wnid, "images")

                if not os.path.exists(src_dir):
                    print(f"  Warning: {src_dir} does not exist, skipping")
                    continue

                # Symlink all images from this WordNet class to the CIFAR-10 class directory
                for img_file in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, img_file)
                    dst_path = os.path.join(class_dir, f"{wnid}_{img_file}")

                    # Create symlink if it doesn't exist
                    if not os.path.exists(dst_path):
                        try:
                            os.symlink(src_path, dst_path)
                        except FileExistsError:
                            pass

                print(f"  Linked {split}/{cifar_class}/ <- {wnid}")

    print("\n✓ Curated subset created successfully!\n")


def count_images(curated_root: str) -> None:
    """Count and print statistics of the curated subset."""
    print("Image Count in Curated Subset:")
    print("=" * 70)

    total_train = 0
    total_val = 0

    for split in ["train", "val"]:
        split_dir = os.path.join(curated_root, split)
        print(f"\n{split.upper()}:")

        for cifar_class in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, cifar_class)
            if os.path.isdir(class_dir):
                count = len(os.listdir(class_dir))
                print(f"  {cifar_class:12}: {count:5} images")
                if split == "train":
                    total_train += count
                else:
                    total_val += count

    print("=" * 70)
    print(f"TOTAL TRAIN: {total_train} images")
    print(f"TOTAL VAL:   {total_val} images\n")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a curated Tiny-ImageNet subset matching CIFAR-10 classes."
    )
    parser.add_argument(
        "--tiny-imagenet-root", type=str, required=True,
        help="Path to the Tiny-ImageNet-200 root directory."
    )
    parser.add_argument(
        "--output-root", type=str, required=True,
        help="Path where the curated subset will be saved."
    )
    cli_args = parser.parse_args()

    tiny_imagenet_root = cli_args.tiny_imagenet_root
    output_root = cli_args.output_root

    print("=" * 70)
    print("CURATED TINY-IMAGENET SUBSET FOR CIFAR-10 MATCHING")
    print("=" * 70 + "\n")

    # Verify mapping
    verified_mapping = verify_cifar10_mapping(tiny_imagenet_root)

    # Create subset
    if verified_mapping:
        create_curated_subset(tiny_imagenet_root,
                              output_root, verified_mapping)

        # Count images
        count_images(output_root)

        print(f"Use this dataset in non_linear_attack.py with:")
        print(f"  --public-dataset tiny-imagenet-cifar10-matched")
        print(
            f"  OR add support for custom dataset paths via --public-dataset {output_root}")
    else:
        print("Error: No CIFAR-10 classes found in Tiny-ImageNet!")
        sys.exit(1)
