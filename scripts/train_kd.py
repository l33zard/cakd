"""Train a DeiT-Tiny student with vanilla logit knowledge distillation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from src.data import get_cifar100_loaders
from src.engine import EpochMetrics, _correct_count, evaluate, save_checkpoint_if_best
from src.losses import vanilla_kd_loss
from src.models import create_student, create_teacher
from src.utils import append_csv_row, get_device, set_seed


# These defaults intentionally match scripts/train_student.py.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_ALPHA = 0.5
DEFAULT_TAU = 4.0
DEFAULT_SEED = 42
DEFAULT_SPLIT_SEED = 42
DEFAULT_SMOKE_BATCHES = 1

SMOKE_CHECKPOINT_PATH = Path("checkpoints/smoke_kd_student.pt")
SMOKE_METRICS_PATH = Path("results/smoke_kd_student_metrics.csv")
SMOKE_SUMMARY_PATH = Path("results/smoke_kd_student_summary.json")

METRICS_FIELDNAMES = [
    "epoch",
    "learning_rate",
    "train_loss",
    "train_top1",
    "train_top5",
    "val_loss",
    "val_top1",
    "val_top5",
    "checkpoint_saved",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        required=True,
        help="Frozen ResNet-50 teacher checkpoint",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="KD checkpoint path (default: checkpoints/kd_student_seed<seed>_best.pt)",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="KD epoch CSV path (default: results/kd_student_seed<seed>_metrics.csv)",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="KD summary JSON path (default: results/kd_student_seed<seed>_summary.json)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one batch for training, validation, and test",
    )
    parser.add_argument("--smoke-batches", type=int, default=DEFAULT_SMOKE_BATCHES)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Destroy existing KD outputs before training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume KD training from the student checkpoint if it exists",
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


def _limit_loader(loader, max_batches: int):
    if max_batches <= 0:
        raise ValueError("smoke-batches must be positive")
    return islice(loader, max_batches)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Match the supervised DeiT baseline's warmup-plus-cosine schedule."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        progress = min(1.0, (epoch - warmup_epochs) / cosine_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _prepare_metrics_csv(path: Path, *, resume: bool) -> None:
    if resume and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        csv.DictWriter(csv_file, fieldnames=METRICS_FIELDNAMES).writeheader()


def _prepare_output_files(
    paths: list[Path], *, overwrite: bool, resume: bool
) -> None:
    if resume:
        return
    conflicts = [path for path in paths if path.exists()]
    if conflicts and not overwrite:
        conflict_text = "\n".join(f"  - {path}" for path in conflicts)
        raise SystemExit(
            "Refusing to overwrite existing KD outputs:\n"
            f"{conflict_text}\n"
            "Rerun with --overwrite to destroy these prior results."
        )
    if overwrite:
        for path in conflicts:
            path.unlink()


def _move_batch(batch, device: torch.device):
    images, targets = batch
    non_blocking = device.type == "cuda"
    return (
        images.to(device, non_blocking=non_blocking),
        targets.to(device, non_blocking=non_blocking),
    )


def _train_one_kd_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    alpha: float,
    tau: float,
    show_progress: bool,
) -> EpochMetrics:
    student.train()
    teacher.eval()
    iterator = tqdm(loader, desc="train-kd", leave=False) if show_progress else loader

    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total_examples = 0

    for batch in iterator:
        images, targets = _move_batch(batch, device)
        batch_size = targets.size(0)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        loss = vanilla_kd_loss(
            student_logits,
            teacher_logits,
            targets,
            alpha=alpha,
            tau=tau,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        top1_correct += _correct_count(student_logits.detach(), targets, k=1)
        top5_correct += _correct_count(student_logits.detach(), targets, k=5)
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("cannot calculate metrics for an empty data loader")
    return EpochMetrics(
        loss=total_loss / total_examples,
        top1=100.0 * top1_correct / total_examples,
        top5=100.0 * top5_correct / total_examples,
    )


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.warmup_epochs <= 0:
        raise ValueError("warmup-epochs must be positive")
    if args.warmup_epochs > args.epochs:
        raise ValueError("warmup-epochs cannot exceed epochs")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers cannot be negative")
    if args.smoke_batches <= 0:
        raise ValueError("smoke-batches must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if args.tau <= 0.0:
        raise ValueError("tau must be positive")
    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")

    checkpoint_path = args.checkpoint_path or Path(
        f"checkpoints/kd_student_seed{args.seed}_best.pt"
    )
    metrics_path = args.metrics_path or Path(
        f"results/kd_student_seed{args.seed}_metrics.csv"
    )
    summary_path = args.summary_path or Path(
        f"results/kd_student_seed{args.seed}_summary.json"
    )
    if args.smoke_test:
        checkpoint_path = SMOKE_CHECKPOINT_PATH
        metrics_path = SMOKE_METRICS_PATH
        summary_path = SMOKE_SUMMARY_PATH
        args.epochs = 1
        args.warmup_epochs = 1

    _prepare_output_files(
        [checkpoint_path, metrics_path, summary_path],
        overwrite=args.overwrite,
        resume=args.resume,
    )
    _prepare_metrics_csv(metrics_path, resume=args.resume)

    started_at = time.perf_counter()
    set_seed(args.seed)
    device = _select_device(args.device)
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"alpha: {args.alpha}")
    print(f"tau: {args.tau}")
    print(f"teacher checkpoint: {args.teacher_checkpoint}")

    train_loader, validation_loader, test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
    )

    teacher = create_teacher().to(device)
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    if "model_state_dict" not in teacher_checkpoint:
        raise KeyError("teacher checkpoint must contain model_state_dict")
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = create_student().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_top1 = checkpoint.get("best_val_top1", 0.0)
        scheduler.last_epoch = start_epoch - 1
        # The first resumed loop epoch is zero-based start_epoch - 1.
        resumed_multiplier = scheduler.lr_lambdas[0](start_epoch - 1)
        resumed_lrs = [
            base_lr * resumed_multiplier for base_lr in scheduler.base_lrs
        ]
        for parameter_group, learning_rate in zip(optimizer.param_groups, resumed_lrs):
            parameter_group["lr"] = learning_rate
        scheduler._last_lr = resumed_lrs
        scheduler._step_count = 0
    else:
        if args.resume:
            print(
                f"Warning: checkpoint not found at {checkpoint_path}; "
                "starting KD training from epoch 1."
            )
        start_epoch = 1
        best_val_top1 = 0.0

    # Epoch indices are zero-based only inside the loop; logs and checkpoints
    # use one-based epoch numbers, matching the supervised student script.
    end_epoch = start_epoch if args.smoke_test else args.epochs
    for epoch in range(start_epoch - 1, end_epoch):
        epoch_multiplier = scheduler.lr_lambdas[0](epoch)
        epoch_lrs = [
            base_lr * epoch_multiplier for base_lr in scheduler.base_lrs
        ]
        for parameter_group, learning_rate in zip(
            optimizer.param_groups, epoch_lrs
        ):
            parameter_group["lr"] = learning_rate

        train_epoch_loader = train_loader
        validation_epoch_loader = validation_loader
        if args.smoke_test:
            train_epoch_loader = _limit_loader(train_loader, args.smoke_batches)
            validation_epoch_loader = _limit_loader(validation_loader, args.smoke_batches)

        train_metrics = _train_one_kd_epoch(
            student,
            teacher,
            train_epoch_loader,
            optimizer,
            device,
            alpha=args.alpha,
            tau=args.tau,
            show_progress=not args.smoke_test,
        )
        validation_metrics = evaluate(
            model=student,
            loader=validation_epoch_loader,
            criterion=criterion,
            device=device,
            show_progress=not args.smoke_test,
        )
        best_val_top1, checkpoint_saved = save_checkpoint_if_best(
            path=checkpoint_path,
            model=student,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=validation_metrics,
            best_val_top1=best_val_top1,
        )
        append_csv_row(
            metrics_path,
            {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics.loss,
                "train_top1": train_metrics.top1,
                "train_top5": train_metrics.top5,
                "val_loss": validation_metrics.loss,
                "val_top1": validation_metrics.top1,
                "val_top5": validation_metrics.top5,
                "checkpoint_saved": checkpoint_saved,
            },
            fieldnames=METRICS_FIELDNAMES,
        )
        print(
            f"epoch {epoch + 1}/{end_epoch} | "
            f"train loss {train_metrics.loss:.4f} | "
            f"train top1 {train_metrics.top1:.2f}% | "
            f"val loss {validation_metrics.loss:.4f} | "
            f"val top1 {validation_metrics.top1:.2f}% | "
            f"val top5 {validation_metrics.top5:.2f}%"
        )
        scheduler.step()

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(
            f"No checkpoint found at {checkpoint_path}; "
            "evaluating the current (final-epoch) model on the test set."
        )
    test_loader_for_run = test_loader
    if args.smoke_test:
        test_loader_for_run = _limit_loader(test_loader, args.smoke_batches)
    test_metrics = evaluate(
        model=student,
        loader=test_loader_for_run,
        criterion=criterion,
        device=device,
        show_progress=not args.smoke_test,
    )
    summary = {
        "seed": args.seed,
        "best_validation_top1": best_val_top1,
        "test_top1": test_metrics.top1,
        "test_top5": test_metrics.top5,
        "runtime_seconds": time.perf_counter() - started_at,
    }
    _write_summary(summary_path, summary)
    print(
        f"test loss {test_metrics.loss:.4f} | "
        f"test top1 {test_metrics.top1:.2f}% | "
        f"test top5 {test_metrics.top5:.2f}%"
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
