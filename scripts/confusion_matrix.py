"""Analyze aligned-KD classification errors on the official CIFAR-100 test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_test import _load_model
from src.data import _build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/kd_student_seed42_aligned_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def _save_heatmap(normalized, counts, pairs, names, path: Path) -> None:
    values = np.array([normalized[true, predicted] for true, predicted in pairs])
    fig, ax = plt.subplots(figsize=(11, 10), layout="constrained")
    heatmap = ax.imshow(values[:, None], cmap="YlOrRd", vmin=0,
                        vmax=max(float(values.max()), 0.01), aspect="auto")
    ax.set_yticks(range(len(pairs)),
                  [f"{names[true]} → {names[predicted]}" for true, predicted in pairs])
    ax.set_xticks([0], ["Predictions for the indicated true → predicted pair"])
    ax.set_ylabel("True class → predicted class (highest error first)")
    ax.set_title("CIFAR-100 aligned KD: top 20 off-diagonal confusions")
    for row, ((true, predicted), value) in enumerate(zip(pairs, values)):
        ax.text(0, row, f"{value:.0%} ({counts[true, predicted]} images)",
                ha="center", va="center",
                color="white" if value > 0.6 * values.max() else "black")
    fig.colorbar(heatmap, ax=ax, label="Fraction of true-class test images")
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    # The official test set is fixed; no train/validation split is applied.
    torch.manual_seed(args.split_seed)
    device = torch.device(args.device)
    model = _load_model(args.checkpoint, "aligned").to(device).eval()
    _, evaluation_transform = _build_transforms()
    dataset = CIFAR100(args.data_dir, train=False, download=False,
                       transform=evaluation_transform)
    names = dataset.classes
    if len(names) != 100 or len(dataset) != 10_000:
        raise RuntimeError("Expected the 100-class, 10000-image CIFAR-100 test set")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    counts = torch.zeros(100, 100, dtype=torch.int64)
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        counts += torch.bincount(labels * 100 + predictions, minlength=10_000).reshape(100, 100)
    counts = counts.numpy()
    if counts.sum() != 10_000 or np.any(counts.sum(axis=1) != 100):
        raise RuntimeError("Expected 100 test observations per class")
    normalized = counts / counts.sum(axis=1, keepdims=True)
    # Stable index tie-breaking makes equally frequent errors reproducible.
    pairs = sorted(((true, predicted) for true in range(100)
                    for predicted in range(100) if true != predicted),
                   key=lambda pair: (-normalized[pair], pair[0], pair[1]))
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "confusion_matrix.npy", normalized)
    _save_heatmap(normalized, counts, pairs[:20], names, output_dir / "confusion_matrix.png")

    print("Top 10 confused pairs (true -> predicted):")
    for true, predicted in pairs[:10]:
        print(f"{names[true]} -> {names[predicted]}: "
              f"{counts[true, predicted]}/100 ({normalized[true, predicted]:.2%})")
    print("10 worst classes (per-class accuracy, ascending):")
    for index in np.argsort(normalized.diagonal(), kind="stable")[:10]:
        print(f"{names[index]}: {normalized[index, index]:.2%} "
              f"({counts[index, index]}/100)")
    print(f"Overall test top-1: {100.0 * counts.trace() / counts.sum():.2f}%")


if __name__ == "__main__":
    main()
