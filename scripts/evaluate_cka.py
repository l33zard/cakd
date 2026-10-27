"""Evaluate teacher/student feature similarity with linear centered CKA."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import torch

from src.cka import linear_cka, run_sanity_checks
from src.data import get_cifar100_loaders
from src.models import create_student, create_teacher
from src.utils import get_device, set_seed


DEFAULT_DATA_DIR = Path("data")
DEFAULT_TEACHER_CHECKPOINT = Path("checkpoints/teacher_best.pt")
DEFAULT_STUDENT_CHECKPOINT = Path("checkpoints/student_seed42_best.pt")
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0
DEFAULT_SPLIT_SEED = 42
DEFAULT_SEED = 42
TEACHER_FEATURES = 1024
STUDENT_FEATURES = 192
PATCH_TOKENS = 64
STUDENT_BLOCK_INDEX = 7  # block 8 in one-based experiment/report notation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"CIFAR-100 directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=DEFAULT_TEACHER_CHECKPOINT,
        help=f"Teacher checkpoint (default: {DEFAULT_TEACHER_CHECKPOINT})",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=DEFAULT_STUDENT_CHECKPOINT,
        help=f"Student checkpoint (default: {DEFAULT_STUDENT_CHECKPOINT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Validation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"DataLoader worker processes (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"Fixed train/validation split seed (default: {DEFAULT_SPLIT_SEED})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic setup (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Evaluation device (default: auto)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional deterministic prefix of validation batches to evaluate",
    )
    return parser.parse_args()


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return get_device()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def _load_model_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])


def _feature_hook(storage: list[torch.Tensor]):
    def hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        storage.append(output.detach())

    return hook


def _collect_validation_features(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    validation_loader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_features: list[torch.Tensor] = []
    student_features: list[torch.Tensor] = []
    teacher_hook = teacher.layer3.register_forward_hook(_feature_hook(teacher_features))
    student_hook = student.blocks[STUDENT_BLOCK_INDEX].register_forward_hook(
        _feature_hook(student_features)
    )
    loader = validation_loader if max_batches is None else islice(validation_loader, max_batches)

    try:
        with torch.no_grad():
            for images, _labels in loader:
                images = images.to(device)
                teacher(images)
                student(images)
                if len(teacher_features) != len(student_features):
                    raise RuntimeError("Teacher and student hooks produced different batch counts")
    finally:
        teacher_hook.remove()
        student_hook.remove()

    if not teacher_features:
        raise RuntimeError("No validation batches were processed")

    teacher_matrices = []
    student_matrices = []
    for teacher_batch, student_batch in zip(teacher_features, student_features):
        if tuple(teacher_batch.shape[1:]) != (TEACHER_FEATURES, 8, 8):
            raise RuntimeError(
                f"Expected teacher layer3 shape (B, 1024, 8, 8), got {tuple(teacher_batch.shape)}"
            )
        if tuple(student_batch.shape[1:]) != (PATCH_TOKENS + 1, STUDENT_FEATURES):
            raise RuntimeError(
                f"Expected student block 8 shape (B, 65, 192), got {tuple(student_batch.shape)}"
            )

        teacher_matrices.append(
            teacher_batch.permute(0, 2, 3, 1).reshape(-1, TEACHER_FEATURES).cpu()
        )
        student_matrices.append(student_batch[:, 1:, :].reshape(-1, STUDENT_FEATURES).cpu())

    return torch.cat(teacher_matrices, dim=0), torch.cat(student_matrices, dim=0)


def main() -> None:
    args = parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when supplied")

    set_seed(args.seed)
    device = _select_device(args.device)
    run_sanity_checks()

    train_loader, validation_loader, _test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
    )
    del train_loader

    teacher = create_teacher().to(device)
    student = create_student().to(device)
    _load_model_checkpoint(teacher, args.teacher_checkpoint, device)
    _load_model_checkpoint(student, args.student_checkpoint, device)
    teacher.eval()
    student.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    teacher_matrix, student_matrix = _collect_validation_features(
        teacher,
        student,
        validation_loader,
        device,
        args.max_batches,
    )
    score = linear_cka(teacher_matrix, student_matrix)

    print(f"device: {device}")
    print(f"validation observations: {teacher_matrix.shape[0]}")
    print(f"teacher matrix shape: {tuple(teacher_matrix.shape)}")
    print(f"student matrix shape: {tuple(student_matrix.shape)}")
    print(f"linear centered CKA: {score.item():.6f}")


if __name__ == "__main__":
    main()
