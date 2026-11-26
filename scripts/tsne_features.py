"""Compare teacher and KD student features on the same seeded CIFAR-100 test images."""

from __future__ import annotations

import argparse
import inspect
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_test import _load_model
from src.data import _build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=Path("checkpoints/teacher_best.pt"))
    parser.add_argument("--student-before", type=Path,
                        default=Path("checkpoints/kd_student_seed42_best.pt"))
    parser.add_argument("--student-after", type=Path,
                        default=Path("checkpoints/kd_student_seed42_aligned_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def _superclass_metadata(dataset: CIFAR100):
    """Read the official fine-to-coarse mapping and names, without a lookup table."""
    directory = Path(dataset.root) / dataset.base_folder
    with (directory / "test").open("rb") as file:
        test = pickle.load(file, encoding="latin1")
    with (directory / "meta").open("rb") as file:
        metadata = pickle.load(file, encoding="latin1")
    if list(metadata["fine_label_names"]) != dataset.classes:
        raise RuntimeError("Dataset classes do not match the official metadata")
    fine = np.asarray(test["fine_labels"], dtype=np.int64)
    coarse = np.asarray(test["coarse_labels"], dtype=np.int64)
    if not np.array_equal(fine, dataset.targets):
        raise RuntimeError("Official superclass labels do not match the dataset order")
    mapping = np.full(len(dataset.classes), -1, dtype=np.int64)
    for fine_label in range(len(dataset.classes)):
        groups = np.unique(coarse[fine == fine_label])
        if len(groups) != 1:
            raise RuntimeError("Each fine class must belong to exactly one superclass")
        mapping[fine_label] = groups[0]
    names = metadata["coarse_label_names"]
    if len(names) != 20 or not np.all(np.bincount(mapping, minlength=20) == 5):
        raise RuntimeError("Expected 20 official superclasses with five fine classes each")
    return mapping, names


@torch.no_grad()
def _extract_features(model, loader, device, *, teacher: bool) -> np.ndarray:
    batches = []

    def hook(_module, _inputs, output):
        features = output[0] if isinstance(output, (tuple, list)) else output
        if teacher:
            if tuple(features.shape[1:]) != (1024, 8, 8):
                raise RuntimeError("Expected teacher layer3 shape (B, 1024, 8, 8)")
            pooled = features.mean(dim=(2, 3))
        else:
            if tuple(features.shape[1:]) != (65, 192):
                raise RuntimeError("Expected student blocks[7] shape (B, 65, 192)")
            pooled = features[:, 1:, :].mean(dim=1)
        batches.append(pooled.cpu())

    model.to(device).eval().requires_grad_(False)
    layer = model.layer3 if teacher else model.blocks[7]
    handle = layer.register_forward_hook(hook)
    try:
        for images, _labels in loader:
            model(images.to(device))
    finally:
        handle.remove()
    return torch.cat(batches).numpy()


def _embed(features: np.ndarray) -> np.ndarray:
    # sklearn renamed n_iter to max_iter; preserve the requested 1000 iterations.
    iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    return TSNE(perplexity=30, random_state=42,
                **{iteration_key: 1000}).fit_transform(features)


def _save_figure(embeddings, labels, superclass_names, path: Path) -> None:
    colors = plt.get_cmap("tab20")(np.arange(20))
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    titles = ("Teacher: ResNet-50 layer3", "Student before: KD block 8",
              "Student after: aligned KD block 8")
    for ax, coords, title in zip(axes, embeddings, titles):
        ax.scatter(coords[:, 0], coords[:, 1], c=colors[labels], s=9,
                   alpha=0.7, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[index],
                      markersize=6, label=name.replace("_", " "))
               for index, name in enumerate(superclass_names)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("CIFAR-100 mean-pooled features: independent t-SNE embeddings\n"
                 "Same seeded test images; colors show official superclasses. "
                 "Panel positions and orientations are not aligned.", fontsize=12)
    fig.tight_layout(rect=(0, 0.17, 1, 0.90))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 30 < args.n_samples <= 10_000:
        raise ValueError("--n-samples must be between 31 and 10000 (perplexity is 30)")
    torch.manual_seed(args.split_seed)
    _, evaluation_transform = _build_transforms()
    dataset = CIFAR100(args.data_dir, train=False, download=False,
                       transform=evaluation_transform)
    mapping, superclass_names = _superclass_metadata(dataset)
    generator = torch.Generator().manual_seed(args.split_seed)
    indices = torch.randperm(len(dataset), generator=generator)[:args.n_samples].numpy()
    fine_labels = np.asarray(dataset.targets)[indices]
    superclass_labels = mapping[fine_labels]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=64,
                        shuffle=False, num_workers=0)
    device = torch.device(args.device)
    embeddings = []
    for name, checkpoint, arch in (
        ("teacher", args.teacher, "teacher"),
        ("student_before", args.student_before, "student"),
        ("student_after", args.student_after, "aligned"),
    ):
        print(f"Extracting {name} features from {checkpoint.name}", flush=True)
        model = _load_model(checkpoint, arch)
        features = _extract_features(model, loader, device, teacher=(arch == "teacher"))
        del model
        print(f"Running t-SNE for {name}: {features.shape}", flush=True)
        embeddings.append(_embed(features))
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "tsne_coords.npz", teacher=embeddings[0],
             student_before=embeddings[1], student_after=embeddings[2],
             fine_labels=fine_labels, superclass_labels=superclass_labels,
             sample_indices=indices, class_names=np.asarray(dataset.classes),
             superclass_names=np.asarray(superclass_names), fine_to_coarse=mapping)
    _save_figure(embeddings, superclass_labels, superclass_names, output_dir / "tsne.png")
    print("Saved results/tsne.png and results/tsne_coords.npz", flush=True)


if __name__ == "__main__":
    main()
