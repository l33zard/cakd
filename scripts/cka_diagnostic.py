"""Compare raw/projected, pooled/batch CKA on one fixed validation subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.alignment import TeacherStudentAlignment
from src.cka import linear_cka
from src.data import get_cifar100_loaders
from src.models import create_student, create_teacher
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--aligned-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("model_state_dict", checkpoint)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.n_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--n-samples and --batch-size must be positive")
    set_seed(args.split_seed)
    device = torch.device(args.device)

    teacher = create_teacher()
    teacher.load_state_dict(_load_state(args.teacher_checkpoint))
    alignment = TeacherStudentAlignment(teacher, create_student())
    try:
        # Strict loading restores teacher.*, student.*, and projection.* together.
        # The frozen teacher stored in the aligned checkpoint is authoritative.
        alignment.load_state_dict(_load_state(args.aligned_checkpoint))
        alignment.to(device).eval().requires_grad_(False)

        train_loader, validation_loader, test_loader = get_cifar100_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=0,
            split_seed=args.split_seed,
        )
        del train_loader, test_loader
        if args.n_samples > len(validation_loader.dataset):
            raise ValueError("--n-samples exceeds the validation set size")
        # The validation split is seeded and ordered, with no random augmentation.
        loader = DataLoader(
            Subset(validation_loader.dataset, range(args.n_samples)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        teacher_batches, student_batches = [], []
        raw_scores, projected_scores = [], []
        for images, _labels in loader:
            images = images.to(device)
            teacher_tokens = alignment.forward_teacher(images)
            student_tokens = alignment.forward_student(images)
            batch_size = images.shape[0]
            if tuple(alignment._teacher_feat.shape) != (batch_size, 1024, 8, 8):
                raise RuntimeError("Expected teacher layer3 shape (B, 1024, 8, 8)")
            if tuple(alignment._student_feat.shape) != (batch_size, 65, 192):
                raise RuntimeError("Expected student blocks[7] shape (B, 65, 192)")
            raw_scores.append(linear_cka(teacher_tokens, student_tokens).item())
            projected_scores.append(
                linear_cka(alignment.projection(teacher_tokens), student_tokens).item()
            )
            teacher_batches.append(teacher_tokens.cpu())
            student_batches.append(student_tokens.cpu())

        # Pool the actual observations, then center once inside linear_cka.
        teacher_tokens = torch.cat(teacher_batches).to(device)
        student_tokens = torch.cat(student_batches).to(device)
        projected_teacher = alignment.projection(teacher_tokens)
        raw_pooled = linear_cka(teacher_tokens, student_tokens).item()
        projected_pooled = linear_cka(projected_teacher, student_tokens).item()
        # Rank is measured on P(F_T) itself, without additional centering.
        singular_values = torch.linalg.svdvals(projected_teacher)
        rank = (singular_values > 1e-4 * singular_values.max()).sum().item()
        raw = torch.tensor(raw_scores, dtype=torch.float64)
        projected = torch.tensor(projected_scores, dtype=torch.float64)

        print(f"Raw pooled CKA:           {raw_pooled:.4f}")
        print(f"Projected pooled CKA:     {projected_pooled:.4f}")
        # Population standard deviation over the evaluated batches (also valid for one).
        print(f"Raw batch CKA (mean):     {raw.mean():.4f} (+/- {raw.std(correction=0):.4f})")
        print(
            f"Projected batch CKA (mean): {projected.mean():.4f} "
            f"(+/- {projected.std(correction=0):.4f})"
        )
        print(f"Effective rank of P(F_T): {rank} (singular values > 1e-4 * max_sv)")
    finally:
        alignment.remove_hooks()


if __name__ == "__main__":
    main()
