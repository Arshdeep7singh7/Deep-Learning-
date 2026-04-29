from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .augmentations import augmix_image, corrupt_pil_image

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10C_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


class ConsistencyCIFAR10(datasets.CIFAR10):
    """Return clean and corrupted versions of each training image."""

    def __init__(self, *args, corruption_severity: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.clean_transform = train_transform()
        self.corruption_severity = corruption_severity

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        corrupted = corrupt_pil_image(img, severity=self.corruption_severity)
        return self.clean_transform(img), self.clean_transform(corrupted), target


class AugMixCIFAR10(datasets.CIFAR10):
    """Return clean plus two AugMix views for JSD consistency training."""

    def __init__(self, *args, severity: int = 3, width: int = 3, depth: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_transform = train_transform()
        self.severity = severity
        self.width = width
        self.depth = depth

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        aug1 = augmix_image(img, severity=self.severity, width=self.width, depth=self.depth)
        aug2 = augmix_image(img, severity=self.severity, width=self.width, depth=self.depth)
        return self.base_transform(img), self.base_transform(aug1), self.base_transform(aug2), target


class CIFAR10C(Dataset):
    """CIFAR-10-C corruption dataset.

    Expected layout:
        data/CIFAR-10-C/gaussian_noise.npy
        data/CIFAR-10-C/labels.npy
    Each corruption file stores five 10k-image severity slices.
    """

    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int = 1,
        transform: transforms.Compose | None = None,
    ):
        if corruption not in CIFAR10C_CORRUPTIONS:
            raise ValueError(f"Unknown corruption '{corruption}'.")
        if severity < 1 or severity > 5:
            raise ValueError("CIFAR-10-C severity must be in [1, 5].")

        root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform or test_transform()
        corruption_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"
        if not corruption_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                "CIFAR-10-C files were not found. Download and extract CIFAR-10-C so that "
                f"{corruption_path} and {labels_path} exist."
            )
        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = np.load(corruption_path, mmap_mode="r")[start:end]
        labels = np.load(labels_path)
        self.targets = labels[:10000] if len(labels) >= 10000 else labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        img = Image.fromarray(np.asarray(self.data[index]).astype("uint8"))
        target = int(self.targets[index])
        return self.transform(img), target


def make_cifar10_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
    mode: Literal["baseline", "consistency", "augmix"] = "baseline",
    corruption_severity: int = 2,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    if mode == "baseline":
        train_set = datasets.CIFAR10(data_dir, train=True, download=download, transform=train_transform())
    elif mode == "consistency":
        train_set = ConsistencyCIFAR10(data_dir, train=True, download=download, corruption_severity=corruption_severity)
    elif mode == "augmix":
        train_set = AugMixCIFAR10(data_dir, train=True, download=download)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    test_set = datasets.CIFAR10(data_dir, train=False, download=download, transform=test_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def make_cifar10c_loader(
    cifar10c_dir: str | Path,
    corruption: str,
    severity: int,
    batch_size: int = 128,
    num_workers: int = 2,
) -> DataLoader:
    dataset = CIFAR10C(cifar10c_dir, corruption=corruption, severity=severity, transform=test_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
