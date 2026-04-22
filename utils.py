import torchvision
from torchvision import transforms
from torch import nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import os
import sys
import math
import random
import numpy as np
import time
from typing import List, Tuple

# local imports
from imagenet_dataset import ImageNetStreamingDataset, ImageNetPreloadedDataset


# ---------------------------
# Basic config / utils
# ---------------------------
DEFAULT_SEED = 1137
LOG_DIR = "./logs"
CHECKPOINTS_DIR = "./checkpoints"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_DATASETS = ["mnist", "cifar5m", "cifar10",
                      "cifar100", "tiny-imagenet", "imagenet"]
SUPPORTED_MODELS = ["resnet18", "resnet34",
                    "resnet50", "effnet_v1", "effnet_v2"]

# has to be manually configured because of very large size
IMAGENET_FULL_PATH = os.getenv("IMAGENET_FULL_PATH", None)
CIFAR_5M_FULL_PATH = os.getenv("CIFAR_5M_FULL_PATH", None)
RADIUS_MAPPING = {
    "mnist": 567.726562,
    "cifar5m": 573.834595,
    "cifar10": 569.761841,
    "cifar100": 614.544922,
    "tiny-imagenet": 631.283386,
    "imagenet": 635.055420
}


def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_deterministic_behavior(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


class Tee:
    """
    Tee that duplicates all stdout writes to both terminal and a log file,
    flushing on every write so the log file is updated in (near) real time.
    """

    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
        sys.stdout = self  # redirect global stdout to self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        # Flush immediately so that the log file is updated in real time
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


def generate_log_and_ckpt_files(model, dataset_name, num_epochs):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    log_id = int(time.time())
    log_filename = f"{LOG_DIR}/instahide_{model}_{dataset_name}_epochs{num_epochs}-ID-{log_id}.log"
    checkpoint_file = (
        f"{CHECKPOINTS_DIR}/best_instahide_classifier_{model}_{dataset_name}_{num_epochs}epochs.pth"
    )

    return log_id, log_filename, checkpoint_file


# ---------------------------
# Model builders
# ---------------------------
def build_resnet_feature_extractor(model_name: str, device: torch.types.Device,
                                   cut_layers: int = 2):
    """
    Build a pretrained backbone and remove the last ``cut_layers`` layers.

    Default ``cut_layers=2`` removes [avgpool, fc] giving spatial feature maps
    after the last conv block.  Use ``cut_layers=3`` to also remove the final
    residual block (e.g. layer4 in ResNet).
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = torchvision.models.ResNet34_Weights.DEFAULT
        model = torchvision.models.resnet34(weights=weights)
    elif model_name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
    elif model_name == "effnet_v1":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights)
    elif model_name == "effnet_v2":
        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_s(weights=weights)
    else:
        raise ValueError(
            f"Model <{model_name}> is not in the supported list: {SUPPORTED_MODELS}")

    modules = list(model.children())[:-cut_layers]
    feature_extractor = nn.Sequential(*modules).to(device)
    feature_extractor.eval()
    return feature_extractor


# ---------------------------
# Datasets & transforms
# ---------------------------
def get_transforms(dataset_type: str):
    """
    Transforms compatible with ImageNet-pretrained ResNet backbones.
    All datasets are normalized with ImageNet statistics since we use
    pretrained ImageNet models as frozen feature extractors.
    """
    dataset_type = dataset_type.lower()
    if dataset_type == "mnist":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif dataset_type == "cifar5m":
        return transforms.Compose([
            transforms.ToPILImage(),  # Converts NumPy (HWC) to PIL
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif dataset_type in ["cifar10", "cifar100", "tiny-imagenet"]:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif dataset_type == "imagenet":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type} (Supported datasets: {SUPPORTED_DATASETS})")


def get_dataset(dataset_type: str, root_dir: str = "./data",
                shards_per_chunk: int = 1):
    dataset_type = dataset_type.lower()
    tfm = get_transforms(dataset_type)

    if dataset_type == "mnist":
        train = torchvision.datasets.MNIST(
            root=root_dir, train=True, download=True, transform=tfm)
        test = torchvision.datasets.MNIST(
            root=root_dir, train=False, download=True, transform=tfm)
        num_classes = 10
    elif dataset_type == "cifar5m":
        assert CIFAR_5M_FULL_PATH is not None, "The environments variable <CIFAR_5M_FULL_PATH> is not set"
        train = CIFAR5m(CIFAR_5M_FULL_PATH, transform=tfm, train=True)
        test = CIFAR5m(CIFAR_5M_FULL_PATH, transform=tfm, train=False)
        num_classes = 10
    elif dataset_type == "cifar10":
        train = torchvision.datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=True, transform=tfm)
        num_classes = 10
    elif dataset_type == "cifar100":
        train = torchvision.datasets.CIFAR100(
            root=root_dir, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR100(
            root=root_dir, train=False, download=True, transform=tfm)
        num_classes = 100
    elif dataset_type == "tiny-imagenet":
        train_dir = os.path.join(root_dir, "tiny-imagenet-200", "train")
        val_dir = os.path.join(root_dir, "tiny-imagenet-200", "val")
        test_dir = os.path.join(root_dir, "tiny-imagenet-200", "test")
        eval_dir = val_dir if os.path.isdir(val_dir) else test_dir
        train = torchvision.datasets.ImageFolder(root=train_dir, transform=tfm)
        test = torchvision.datasets.ImageFolder(root=eval_dir, transform=tfm)
        num_classes = 200
    elif dataset_type == "imagenet":
        assert IMAGENET_FULL_PATH is not None, "The environment variable <IMAGENET_FULL_PATH> is not set"
        train = ImageNetStreamingDataset(
            IMAGENET_FULL_PATH, split="train", transform=tfm,
            shards_per_chunk=shards_per_chunk)
        test = ImageNetPreloadedDataset(
            IMAGENET_FULL_PATH, split="validation", transform=tfm)
        num_classes = 1000
    else:
        raise ValueError(
            f"Dataset <{dataset_type}> not supported. Currently supported datasets: {SUPPORTED_DATASETS}")
    return train, test, num_classes


class CIFAR5m(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose | None, train: bool):
        import numpy as np
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.npz_files = [
            f'{root_dir}/part0.npz',
            f'{root_dir}/part1.npz',
            f'{root_dir}/part2.npz',
            f'{root_dir}/part3.npz',
            f'{root_dir}/part4.npz',
            f'{root_dir}/part5.npz']
        self.x = []
        self.y_true = []

        # Train: part0.npz, part1.npz, ... (first K files)
        # Test:  part5.npz (last file)
        files_to_load = self.npz_files[:4] if self.train else self.npz_files[5:]

        for npz_file in files_to_load:
            npz_data = np.load(npz_file)
            self.x.append(npz_data['X'])
            self.y_true.extend(npz_data['Y'])
        self.x = np.vstack(self.x)  # already (N, 32, 32, 3) — HWC

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y_true = self.y_true[idx]

        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x).float()
        y_true_torch = torch.tensor(y_true, dtype=torch.long)
        return x, y_true_torch

    def __len__(self):
        return len(self.x)


# ---------------------------
# Normalization helpers
# ---------------------------
def clamp_imagenet_normalized(x: torch.Tensor) -> torch.Tensor:
    """Clamp by denormalizing to [0,1], clipping, then renormalizing back."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, -1, 1, 1)
    x01 = (x * std + mean).clamp(0.0, 1.0)
    return (x01 - mean) / std


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalized [B,3,H,W] -> [0,1] range."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def denorm_to_lpips(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalized [B,3,H,W] -> [-1,1] range for LPIPS."""
    return denorm_to_01(x) * 2.0 - 1.0


def chw_to_numpy_img01(x_chw: torch.Tensor) -> np.ndarray:
    """x_chw: [3,H,W] normalized; return HxWx3 in [0,1] numpy."""
    mean = torch.tensor(IMAGENET_MEAN, device=x_chw.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x_chw.device).view(3, 1, 1)
    x01 = (x_chw * std + mean).clamp(0, 1)
    return x01.permute(1, 2, 0).detach().cpu().numpy()


# ---------------------------
# Subset & dataset statistics
# ---------------------------
def make_subdataset(dataset, max_images: int, seed: int = 0):
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_images, n), replace=False)
    return Subset(dataset, sorted(idx.tolist()))


@torch.no_grad()
def estimate_average_distance(dataset, device: torch.types.Device,
                              max_images: int = 2048,
                              batch_size: int = 64) -> float:
    """Average L2 distance between random pairs of images (after transforms)."""
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=2, drop_last=False)
    xs, total = [], 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        xs.append(imgs)
        total += imgs.size(0)
        if total >= max_images:
            break
    X = torch.cat(xs, dim=0)  # [N,C,H,W]
    N = X.size(0)
    perm = torch.randperm(N, device=X.device)
    perm = torch.roll(perm, shifts=1)
    dists = torch.norm((X - X[perm]).view(N, -1), p=2, dim=1)
    return float(dists.mean().item())


@torch.no_grad()
def estimate_per_pixel_variance_global_mean(
    dataset, device: torch.types.Device,
    max_images: int = 2048,
    batch_size: int = 64
) -> Tuple[float, int]:
    """
    Estimate global scalar variance v_hat and dimensionality d (C*H*W).
    v_hat is computed over all pixels and channels using a single global mean.
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=2, drop_last=False)
    xs, total = [], 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        xs.append(imgs)
        total += imgs.size(0)
        if total >= max_images:
            break
    X = torch.cat(xs, dim=0)   # [N,C,H,W]
    N, C, H, W = X.shape
    d = C * H * W
    Xv = X.view(N, -1)
    mu = Xv.mean()                                # GLOBAL mean
    v_hat = ((Xv - mu)**2).mean().item()
    return v_hat, d


def compute_c_factor(r: float, v_hat: float, d: int) -> float:
    return (r * r) / (2.0 * d * v_hat + 1e-20)


def mf_from_tau(alpha: float, tau: float, c: float) -> float:
    """mf >= sqrt( (alpha^2/(tau(1-alpha)^2) - 1) / (2c) )"""
    a2 = alpha * alpha
    inner = max(a2 / (max(tau, 1e-20) * (1 - alpha)**2) - 1.0, 0.0)
    return float(np.sqrt(inner / (2.0 * max(c, 1e-20))))


# ---------------------------
# Noise helpers
# ---------------------------
def sample_ball_noise(shape, radius, device=None):
    """
    Sample noise uniformly from a ball of radius ``radius`` in R^shape[1].
    """
    noise = torch.randn(*shape, device=device)
    norm = noise.norm(dim=1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return noise / norm * radius


def add_noise_with_l2_norm_batch(x: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Add independent Gaussian noise per sample, scaled to L2 norm exactly target_norm."""
    B = x.size(0)
    noise = torch.randn_like(x)
    norms = torch.norm(noise.view(B, -1), p=2, dim=1)
    scales = (target_norm / (norms + 1e-12)).view(B, 1, 1, 1)
    return x + noise * scales


# ---------------------------
# Classifier
# ---------------------------
class DenseClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(DenseClassifier, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Handle batch_size == 1 in training mode to avoid BatchNorm1d error
        if self.training and x.size(0) == 1:
            was_training = self.layers.training
            self.layers.eval()          # run BN in eval mode for this forward
            out = self.layers(x)
            if was_training:
                self.layers.train()     # restore previous state
            return out
        return self.layers(x)

    def get_classifier_size(self):
        for k, v in list(self.layers.named_children()):
            print(f"Layer {int(k) + 1}: {v}")


# ---------------------------
# Feature-space mixup
# ---------------------------
def mixup_batch_in_feature_space(
    feature_extractor,
    imgs,
    labels,
    device,
    radius,
    num_classes,
):
    """
    Perform feature-space mixup for a batch, without ever storing global features.
    """
    imgs = imgs.to(device)
    labels = labels.to(device)

    feature_extractor.eval()
    with torch.no_grad():
        fmap = feature_extractor(imgs)           # (B, C, H, W)
        feats = fmap.view(fmap.size(0), -1)      # (B, D)

    B, D = feats.shape

    # Edge-case: single sample in batch -- fall back to no mixup
    if B == 1:
        lam = torch.tensor([0.5], device=device)
        noise = sample_ball_noise((1, D), radius, device=device)
        mix_feats = lam.view(1, 1) * feats + \
            (1.0 - lam).view(1, 1) * (feats + noise)
        soft_labels = torch.zeros(1, num_classes, device=device)
        soft_labels[0, labels[0]] = 1.0
        return mix_feats, soft_labels

    indices = torch.arange(B, device=device)
    partner = torch.roll(indices, shifts=1)  # cyclic shift pairing

    lam = 0.3 + 0.4 * torch.rand(B, device=device)  # (B,)
    noise = sample_ball_noise((B, D), radius, device=device)

    perturbed = feats[partner] + noise
    lam_view = lam.view(B, 1)
    mix_feats = lam_view * feats + (1.0 - lam_view) * perturbed

    soft_labels = torch.zeros(B, num_classes, device=device)
    soft_labels[indices, labels] += lam
    soft_labels[indices, labels[partner]] += (1.0 - lam)

    return mix_feats, soft_labels


def cross_entropy_with_soft_targets(logits: torch.Tensor,
                                    soft_targets: torch.Tensor) -> torch.Tensor:
    return -(soft_targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def evaluate_classifier(classifier, feature_extractor, device, testloader) -> float:
    classifier.eval()
    feature_extractor.eval()
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
    return 100.0 * correct / max(1, total)


# ---------------------------
# Federated helpers
# ---------------------------
def split_dataset_equally(dataset, num_parties: int,
                          seed: int = 0) -> List[Subset]:
    """
    Randomly split a dataset into ``num_parties`` equally-sized
    (as close as possible) disjoint subsets.
    """
    n = len(dataset)
    lengths = [n // num_parties] * num_parties
    for i in range(n % num_parties):
        lengths[i] += 1
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(n, generator=g).tolist()
    subsets = []
    start = 0
    for length in lengths:
        end = start + length
        subsets.append(Subset(dataset, indices[start:end]))
        start = end
    return subsets


# ---------------------------
# Feature-map visualization
# ---------------------------
def get_feature_maps_for_first_n(
    feature_extractor,
    dataset,
    device,
    num_samples=16,
):
    """
    Return feature maps for the first ``num_samples`` samples in the dataset.
    Used only for visualization, so we keep it small.
    """
    num_samples = min(num_samples, len(dataset))
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)

    feature_extractor.eval()
    with torch.no_grad():
        fmap = feature_extractor(imgs)  # (num_samples, C, H, W)

    return fmap.cpu()


def feature_map_to_squared_grid(feat_map: torch.Tensor) -> np.ndarray:
    """
    Arrange channels of a feature map (C, H, W) into a tight square grid image.
    """
    assert feat_map.dim() == 3, "feat_map must have shape (C, H, W)"
    C, H, W = feat_map.shape
    grid_size = int(math.ceil(math.sqrt(C)))
    grid = torch.zeros(grid_size * H, grid_size * W, dtype=torch.float32)
    for idx in range(C):
        ch = feat_map[idx]
        ch = ch - ch.min()
        denom = ch.max()
        if denom > 0:
            ch = ch / denom
        row = idx // grid_size
        col = idx % grid_size
        grid[row * H:(row + 1) * H, col * W:(col + 1) * W] = ch
    return grid.cpu().numpy()


def save_featuremap_grid_and_mixup_pairs(
    original_feature_maps: torch.Tensor,
    radius: float,
    output_dir: str = "./mixup_pairs_features",
    num_pairs: int = 5,
):
    """
    Save num_pairs images (original vs a synthetic mixup feature map) for visualization only.
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    n, C, H, W = original_feature_maps.shape
    num_pairs = min(num_pairs, n)
    flat_feats = original_feature_maps.view(n, -1)  # (n, D)

    for i in range(num_pairs):
        idx = np.random.randint(0, n)
        j = (idx + np.random.randint(1, n)) % n  # ensure j != idx

        orig_map = original_feature_maps[idx]  # (C, H, W)
        orig_grid = feature_map_to_squared_grid(orig_map)

        anchor = flat_feats[idx]
        other = flat_feats[j]
        noise = sample_ball_noise(
            (1, other.numel()), radius, device=other.device).view_as(other)
        lam = 0.3 + 0.4 * torch.rand(1, device=other.device)  # [0.3, 0.7]
        mix_flat = lam * anchor + (1.0 - lam) * (other + noise)
        mix_map = mix_flat.view(C, H, W)
        mix_grid = feature_map_to_squared_grid(mix_map)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(orig_grid, cmap="viridis", aspect="equal")
        axes[0].set_title(f"Original feat (idx={idx})")
        axes[0].axis("off")
        axes[1].imshow(mix_grid, cmap="viridis", aspect="equal")
        axes[1].set_title(f"Mixup feat (idx={idx}, j={j})")
        axes[1].axis("off")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95,
                            bottom=0.05, wspace=0.05, hspace=0.0)

        save_path = os.path.join(output_dir, f"featuremap_grid_pair_{i+1}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved feature-map grid mixup pair {i+1} to: {save_path}")
