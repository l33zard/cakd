"""CIFAR-100 data loading utilities.

The training split is created deterministically from the original CIFAR-100
training set.  Training and validation use separate dataset instances so that
validation never receives the random training augmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PathLike: TypeAlias = str | Path

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)
_TRAIN_SIZE = 45_000
_VALIDATION_SIZE = 5_000


def _build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Create the training and evaluation transforms used for CIFAR-100."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    return train_transform, evaluation_transform


def _split_training_indices(split_seed: int) -> tuple[list[int], list[int]]:
    """Return deterministic train and validation indices for CIFAR-100."""
    generator = torch.Generator()
    generator.manual_seed(split_seed)
    permutation = torch.randperm(50_000, generator=generator).tolist()
    validation_indices = permutation[:_VALIDATION_SIZE]
    train_indices = permutation[_VALIDATION_SIZE:]
    return train_indices, validation_indices


def get_cifar100_loaders(
    data_dir: PathLike,
    batch_size: int,
    num_workers: int,
    split_seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test CIFAR-100 dataloaders.

    The original 50,000-image training split is deterministically divided into
    45,000 training images and 5,000 validation images using ``split_seed``.
    The test loader contains the original 10,000-image test split.

    Args:
        data_dir: Directory where torchvision stores CIFAR-100 data.
        batch_size: Number of images per loader batch.
        num_workers: Number of subprocesses used by each DataLoader.
        split_seed: Seed controlling the train/validation index permutation.

    Returns:
        A ``(train_loader, val_loader, test_loader)`` tuple.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    train_transform, evaluation_transform = _build_transforms()
    train_indices, validation_indices = _split_training_indices(split_seed)

    data_path = Path(data_dir)
    train_dataset = datasets.CIFAR100(
        root=data_path,
        train=True,
        transform=train_transform,
        download=True,
    )
    validation_dataset = datasets.CIFAR100(
        root=data_path,
        train=True,
        transform=evaluation_transform,
        download=False,
    )
    test_dataset = datasets.CIFAR100(
        root=data_path,
        train=False,
        transform=evaluation_transform,
        download=True,
    )

    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    assert len(train_subset) == _TRAIN_SIZE
    assert len(validation_subset) == _VALIDATION_SIZE
    assert len(test_dataset) == 10_000

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader, test_loader


def sanity_check(
    data_dir: PathLike = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    split_seed: int = 42,
) -> None:
    """Print basic sizes and one training batch for a pipeline smoke test."""
    train_loader, validation_loader, test_loader = get_cifar100_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        split_seed=split_seed,
    )
    images, labels = next(iter(train_loader))

    print(f"train size: {len(train_loader.dataset)}")
    print(f"validation size: {len(validation_loader.dataset)}")
    print(f"test size: {len(test_loader.dataset)}")
    print(f"batch image shape: {tuple(images.shape)}")
    print(f"label shape: {tuple(labels.shape)}")
    print(f"min label: {labels.min().item()}")
    print(f"max label: {labels.max().item()}")


if __name__ == "__main__":
    sanity_check()
