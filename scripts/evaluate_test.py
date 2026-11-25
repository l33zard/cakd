"""Evaluate a teacher, student, or aligned-KD checkpoint on CIFAR-100 test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alignment import TeacherStudentAlignment
from src.data import get_cifar100_loaders
from src.models import create_student, create_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arch", choices=("auto", "teacher", "student", "aligned"), default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _detect_arch(state: dict[str, torch.Tensor]) -> str:
    if any(key.startswith(("teacher.", "student.", "projection.")) for key in state):
        return "aligned"
    if any(key.startswith(("cls_token", "blocks.", "head.")) for key in state):
        return "student"
    return "teacher"


def _load_model(path: Path, arch: str) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu")["model_state_dict"]
    if arch == "auto":
        arch = _detect_arch(state)
    if arch == "aligned":
        alignment = TeacherStudentAlignment(create_teacher(), create_student())
        try:
            alignment.load_state_dict(state)
            # Classification uses the trained student, without feature hooks.
            return alignment.student
        finally:
            alignment.remove_hooks()
    model = create_student() if arch == "student" else create_teacher()
    model.load_state_dict(state)
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = _load_model(args.checkpoint, args.arch).to(device).eval()
    train_loader, validation_loader, test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=64,
        num_workers=0,
        split_seed=args.split_seed,
    )
    del train_loader, validation_loader
    # Only the official torchvision CIFAR100(train=False) dataset is evaluated.
    # split_seed affects the unused train/validation split, never this test set.
    top1_correct = top5_correct = n_samples = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).topk(5, dim=1).indices
        matches = predictions.eq(labels[:, None])
        top1_correct += matches[:, 0].sum().item()
        top5_correct += matches.any(dim=1).sum().item()
        n_samples += labels.size(0)
    if n_samples != 10_000:
        raise RuntimeError(f"Expected 10000 test images, evaluated {n_samples}")
    result = {
        "test_top1": 100.0 * top1_correct / n_samples,
        "test_top5": 100.0 * top5_correct / n_samples,
        "n_samples": n_samples,
        "checkpoint": args.checkpoint.name,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"checkpoint={args.checkpoint.name} "
        f"test_top1={result['test_top1']:.2f} test_top5={result['test_top5']:.2f}"
    )


if __name__ == "__main__":
    main()
