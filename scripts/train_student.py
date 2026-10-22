"""Train the DeiT-Tiny student with supervised CIFAR-100 learning."""

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

from src.data import get_cifar100_loaders
from src.engine import EpochMetrics, evaluate, save_checkpoint_if_best, train_one_epoch
from src.models import create_student
from src.utils import append_csv_row, get_device, set_seed


# Defaults are declared together so the experiment settings are easy to audit.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_SEED = 42
DEFAULT_SPLIT_SEED = 42
DEFAULT_SMOKE_BATCHES = 1
SMOKE_CHECKPOINT_PATH = Path("checkpoints/smoke_student.pt")
SMOKE_METRICS_PATH = Path("results/smoke_student_metrics.csv")
SMOKE_SUMMARY_PATH = Path("results/smoke_student_summary.json")

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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"CIFAR-100 directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Best-checkpoint path (default: checkpoints/student_seed<seed>_best.pt)",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Epoch CSV path (default: results/student_seed<seed>_metrics.csv)",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Final summary JSON path (default: results/student_seed<seed>_summary.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"DataLoader worker processes (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"AdamW peak learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"AdamW weight decay (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help=f"Linear warmup duration (default: {DEFAULT_WARMUP_EPOCHS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Model, shuffling, and training randomness seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"Fixed CIFAR-100 train/validation split seed (default: {DEFAULT_SPLIT_SEED})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device (default: auto)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one batch for training, validation, and test",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help=(
            "Destroy existing checkpoint, metrics CSV, and summary JSON for "
            "this seed before training"
        ),
    )
    parser.add_argument(
        "--smoke-batches",
        type=int,
        default=DEFAULT_SMOKE_BATCHES,
        help=f"Batches per split in smoke mode (default: {DEFAULT_SMOKE_BATCHES})",
    )
    return parser.parse_args()


def _select_device(device_name: str) -> torch.device:
    """Resolve an automatic or explicitly requested device."""
    if device_name == "auto":
        return get_device()

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def _limit_loader(loader, max_batches: int):
    """Limit an iterable loader for the smoke test."""
    if max_batches <= 0:
        raise ValueError("smoke-batches must be positive")
    return islice(loader, max_batches)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a linear-warmup then cosine-decay epoch scheduler."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs

        cosine_epochs = max(1, total_epochs - warmup_epochs)
        progress = min(1.0, (epoch - warmup_epochs) / cosine_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _metrics_row(
    epoch: int,
    learning_rate: float,
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    checkpoint_saved: bool,
) -> dict[str, object]:
    """Build one CSV row for an epoch."""
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train_metrics.loss,
        "train_top1": train_metrics.top1,
        "train_top5": train_metrics.top5,
        "val_loss": validation_metrics.loss,
        "val_top1": validation_metrics.top1,
        "val_top5": validation_metrics.top5,
        "checkpoint_saved": checkpoint_saved,
    }


def _prepare_metrics_csv(path: Path) -> None:
    """Create a fresh metrics CSV with the stable experiment header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        csv.DictWriter(csv_file, fieldnames=METRICS_FIELDNAMES).writeheader()


def _prepare_output_files(paths: list[Path], overwrite: bool) -> None:
    """Refuse stale experiment outputs unless the user explicitly opts in."""
    conflicts = [path for path in paths if path.exists()]
    if conflicts and not overwrite:
        conflict_text = "\n".join(f"  - {path}" for path in conflicts)
        raise SystemExit(
            "Refusing to overwrite existing student experiment outputs:\n"
            f"{conflict_text}\n"
            "Rerun with --overwrite to destroy these prior results."
        )

    if overwrite:
        for path in conflicts:
            path.unlink()


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    """Write the final run summary as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.warmup_epochs <= 0:
        raise ValueError("warmup-epochs must be positive")
    if args.warmup_epochs > args.epochs:
        raise ValueError("warmup-epochs cannot exceed epochs")
    if args.smoke_batches <= 0:
        raise ValueError("smoke-batches must be positive")

    checkpoint_path = args.checkpoint_path or Path(
        f"checkpoints/student_seed{args.seed}_best.pt"
    )
    metrics_path = args.metrics_path or Path(
        f"results/student_seed{args.seed}_metrics.csv"
    )
    summary_path = args.summary_path or Path(
        f"results/student_seed{args.seed}_summary.json"
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
    )

    started_at = time.perf_counter()
    set_seed(args.seed)
    device = _select_device(args.device)
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"smoke test: {args.smoke_test}")

    _prepare_metrics_csv(metrics_path)
    train_loader, validation_loader, test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
    )

    model = create_student().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    best_val_top1 = float("-inf")
    epochs_to_run = 1 if args.smoke_test else args.epochs
    for epoch in range(epochs_to_run):
        train_epoch_loader = train_loader
        validation_epoch_loader = validation_loader
        if args.smoke_test:
            train_epoch_loader = _limit_loader(train_loader, args.smoke_batches)
            validation_epoch_loader = _limit_loader(validation_loader, args.smoke_batches)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_epoch_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            show_progress=not args.smoke_test,
        )
        validation_metrics = evaluate(
            model=model,
            loader=validation_epoch_loader,
            criterion=criterion,
            device=device,
            show_progress=not args.smoke_test,
        )
        best_val_top1, checkpoint_saved = save_checkpoint_if_best(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=validation_metrics,
            best_val_top1=best_val_top1,
        )

        append_csv_row(
            metrics_path,
            _metrics_row(
                epoch=epoch + 1,
                learning_rate=optimizer.param_groups[0]["lr"],
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                checkpoint_saved=checkpoint_saved,
            ),
            fieldnames=METRICS_FIELDNAMES,
        )
        print(
            f"epoch {epoch + 1}/{epochs_to_run} | "
            f"train loss {train_metrics.loss:.4f} | "
            f"train top1 {train_metrics.top1:.2f}% | "
            f"val loss {validation_metrics.loss:.4f} | "
            f"val top1 {validation_metrics.top1:.2f}% | "
            f"val top5 {validation_metrics.top5:.2f}%"
        )
        scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader_for_run = test_loader
    if args.smoke_test:
        test_loader_for_run = _limit_loader(test_loader, args.smoke_batches)
    test_metrics = evaluate(
        model=model,
        loader=test_loader_for_run,
        criterion=criterion,
        device=device,
        show_progress=not args.smoke_test,
    )
    runtime_seconds = time.perf_counter() - started_at
    summary = {
        "seed": args.seed,
        "best_validation_top1": best_val_top1,
        "test_top1": test_metrics.top1,
        "test_top5": test_metrics.top5,
        "runtime_seconds": runtime_seconds,
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
