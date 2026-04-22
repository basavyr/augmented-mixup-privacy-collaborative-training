# Augmented Mixup Procedure for Privacy-Preserving Collaborative Training

**Paper:** [OpenReview](https://openreview.net/forum?id=1SrZyNgmpY)
**Venue:** Transactions on Machine Learning Research (TMLR)
**Authors:** Mihail-Iulian Plesa, Fabrice Clerot, Simona Elena David, Robert Poenaru

---

## Overview

This repository contains the official code for *Augmented Mixup Procedure for Privacy-Preserving Collaborative Training*. We propose a feature-space mixup method based on InstaHide-style mixing with dataset-calibrated noise, enabling privacy-preserving collaborative training across multiple parties without sharing raw data.

The codebase covers:

- **Feature-space mixup training** with frozen pretrained backbones (ResNet-18/34/50, EfficientNet-B0/V2-S)
- **Federated and collaborative training** with FedProx, Dirichlet non-IID splits, and mixup-union
- **Privacy attack evaluation** via linear reconstruction (known mixing graph + TV/L2 priors) and non-linear U-Net attacks (zero-shot transfer)
- **Outlier analysis** using LPIPS-based perceptual filtering
- **Radius estimation** for dataset-calibrated noise across privacy parameter tau

---

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)

---

## Dataset Preparation

**Automatically downloaded:** MNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet are downloaded to `./data` on first use.

**Manual setup required for large datasets:**

```bash
export CIFAR_5M_FULL_PATH="/path/to/cifar5m"   # directory containing part0.npz ... part5.npz
export IMAGENET_FULL_PATH="/path/to/imagenet"   # directory with parquet shards
```

**Curated Tiny-ImageNet subset** (for non-linear attack experiments):

```bash
python generate_curated_tiny_imagenet.py --tiny-imagenet-root /path/to/tiny-imagenet-200 \
    --output-root /path/to/tiny-imagenet-cifar10-matched
```

---

## Compute Requirements

Most experiments were run on a single NVIDIA GPU with 16–24 GB VRAM. Training times vary:

| Experiment | Approximate Time |
|---|---|
| `augment.py` (ResNet-18, CIFAR-10, 200 epochs) | ~30 min |
| `collaborative_training.py` (10 parties, 5 epochs) | ~15 min |
| `linear_attack.py` (single dataset, 6 tau values) | ~10 min |
| `non_linear_attack.py` (U-Net, 30 epochs) | ~45 min |

CPU execution is supported but significantly slower.

---

## Scripts

### utils.py

Shared utilities used by all other scripts: device selection, deterministic seeding, dataset/transform loading, normalization helpers, noise generation, classifier, feature-space mixup, and visualization.

### augment.py

Single-machine feature-space mixup training with a frozen ResNet backbone and a dense classifier. Supports resuming from a checkpoint via `--checkpoint`.

```bash
python augment.py --model resnet18 --dataset cifar10 --epochs 200 --bench
python augment.py --model resnet50 --dataset tiny-imagenet --epochs 100 --radius 631.28 --lr 0.001

# Resume from a saved checkpoint
python augment.py --model resnet18 --dataset cifar10 --epochs 200 --bench \
    --checkpoint checkpoints/best_instahide_classifier_resnet18_cifar10_200epochs.pth
```

### efficient_augment.py

Same workflow as `augment.py` but using EfficientNet (B0 or V2-S) as the backbone. Also supports `--checkpoint` for resume.

```bash
python efficient_augment.py --dataset cifar10 --epochs 10 --v2
python efficient_augment.py --dataset cifar100 --epochs 20 --lr 0.01 --quick

# Resume from a saved checkpoint
python efficient_augment.py --dataset cifar5m --epochs 200 --v2 \
    --checkpoint checkpoints/best_instahide_classifier_effnet_v2_cifar5m_200epochs.pth
```

### federated_augment.py

Federated learning experiment: compares a single-party baseline (no mixup) vs mixup-union across all parties (equal splits).

```bash
python federated_augment.py --epochs 5 --bench --radius 1.0
```

### collaborative_training.py

FedProx vs mixup-union with Dirichlet non-IID splits and per-party radius from tau. Supports size-skew experiments (one data-poor party via `--rho`) and minimum party size guarantees.

```bash
python collaborative_training.py --num-parties 10 --tau 1e-6 --epochs 5 --bench
python collaborative_training.py --num-parties 20 --tau 1e-3 --dirichlet-alpha 0.1 --mu 0.01

# Size-skew: party 0 gets rho=0.1 of the data (10x smaller than others)
python collaborative_training.py --num-parties 5 --tau 1e-6 --epochs 10 --bench \
    --rho 0.1 --poor-party 0 --min-party-size 50
```

### radius_approx.py

Estimates feature-space displacement (radius) induced by dataset-calibrated noise levels across datasets and tau values.

```bash
python radius_approx.py --backbone resnet18 --datasets cifar10 cifar100
python radius_approx.py --backbone resnet50 --datasets tiny-imagenet --taus 1e-2 1e-4 1e-6
```

### linear_attack.py

Linear reconstruction attack on mixup (known mixing graph, TV+L2 priors). Reports SNR, SSIM, LPIPS.

```bash
python linear_attack.py --datasets cifar10 --taus 1e-1 1e-3 1e-6 --sub_size 256
python linear_attack.py --datasets mnist cifar10 cifar100 --attack_steps 300 --seed 42
```

### non_linear_attack.py

Non-linear (U-Net) reconstruction attack. Trains on a public dataset, evaluates zero-shot on CIFAR-10.

```bash
python non_linear_attack.py --taus "1e-1, 1e-3, 1e-6" --epochs 30 --seed 1137
python non_linear_attack.py --curated --taus "1e-2, 1e-4" --epochs 50
```

### eval_metrics.py

Evaluate saved U-Net checkpoints (SNR, SSIM, LPIPS) without retraining.

```bash
python eval_metrics.py --checkpoint unet_attack_figs/seed_1137/unet_tau0.01_seed1137.pt --seed 1137 --alpha 0.7 --tau 0.01
python eval_metrics.py --checkpoint-dir unet_attack_figs/seed_1137 --seed 1137 --alpha 0.7
```

### outliers.py

LPIPS-based outlier detection: measure how many mixup images are perceptually too similar to originals.

```bash
# Analysis mode: sweep tau at fixed LPIPS threshold
python outliers.py --data cifar10 --subset_size 512 --analysis
python outliers.py --data cifar100 --subset_size 1024 --analysis --analysis_lpips_th 0.5

# Single-run mode: fixed tau and LPIPS threshold
python outliers.py --data cifar10 --tau 1e-6 --lpips_th 0.6 --subset_size 512

# With inspect PNGs (saves sample pairs per threshold)
python outliers.py --data cifar10 --tau 1e-6 --lpips_th 0.6 --subset_size 512 --inspect
```

### generate_curated_tiny_imagenet.py

Creates a curated Tiny-ImageNet subset matching CIFAR-10 semantic classes.

```bash
python generate_curated_tiny_imagenet.py --tiny-imagenet-root /path/to/tiny-imagenet-200 \
    --output-root /path/to/tiny-imagenet-cifar10-matched
```

### imagenet_dataset.py

Parquet-backed ImageNet-1K dataset loaders (streaming and preloaded). Used internally by other scripts; not run directly.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{plesa2026augmented,
  title   = {Augmented Mixup Procedure for Privacy-Preserving Collaborative Training},
  author  = {Plesa, Mihail-Iulian and Clerot, Fabrice and David, Simona Elena and Poenaru, Robert},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2026},
  url     = {https://openreview.net/forum?id=1SrZyNgmpY}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
