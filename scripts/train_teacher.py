"""Train the CIFAR-adapted ResNet-50 teacher with supervised learning."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import torch
from torch import nn

from src.data import get_cifar100_loaders
from src.engine import EpochMetrics, evaluate, save_checkpoint_if_best, train_one_epoch
from src.models import create_teacher
from src.utils import append_csv_row, get_device, set_seed


# Defaults are intentionally declared in one place so experiment settings are
# visible in the script and can be overridden from the command line.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/teacher_best.pt")
DEFAULT_METRICS_PATH = Path("results/teacher_metrics.csv")
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_MOMENTUM = 0.9
DEFAULT_SEED = 42
DEFAULT_SPLIT_SEED = 42
DEFAULT_SMOKE_BATCHES = 1
SMOKE_CHECKPOINT_PATH = Path("checkpoints/smoke_teacher.pt")
SMOKE_METRICS_PATH = Path("results/smoke_teacher_metrics.csv")


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
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Best-checkpoint path (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"CSV epoch-metrics path (default: {DEFAULT_METRICS_PATH})",
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
        help=f"Initial SGD learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"SGD weight decay (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=DEFAULT_MOMENTUM,
        help=f"SGD momentum (default: {DEFAULT_MOMENTUM})",
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
        help="Run one batch for training, validation, and test instead of a full job",
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
    """Limit an iterable loader for the smoke test without changing the loader."""
    if max_batches <= 0:
        raise ValueError("smoke-batches must be positive")
    return islice(loader, max_batches)


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


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.smoke_batches <= 0:
        raise ValueError("smoke-batches must be positive")

    if args.smoke_test:
        args.checkpoint_path = SMOKE_CHECKPOINT_PATH
        args.metrics_path = SMOKE_METRICS_PATH

    set_seed(args.seed)
    device = _select_device(args.device)
    print(f"device: {device}")
    print(f"smoke test: {args.smoke_test}")

    train_loader, validation_loader, test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
    )

    model = create_teacher().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    epochs_to_run = 1 if args.smoke_test else args.epochs
    best_val_top1 = float("-inf")

    for epoch in range(1, epochs_to_run + 1):
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
            path=args.checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=validation_metrics,
            best_val_top1=best_val_top1,
        )

        learning_rate = optimizer.param_groups[0]["lr"]
        append_csv_row(
            args.metrics_path,
            _metrics_row(
                epoch=epoch,
                learning_rate=learning_rate,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                checkpoint_saved=checkpoint_saved,
            ),
        )
        print(
            f"epoch {epoch}/{epochs_to_run} | "
            f"train loss {train_metrics.loss:.4f} | "
            f"train top1 {train_metrics.top1:.2f}% | "
            f"val loss {validation_metrics.loss:.4f} | "
            f"val top1 {validation_metrics.top1:.2f}% | "
            f"val top5 {validation_metrics.top5:.2f}%"
        )
        scheduler.step()

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
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
    print(
        f"test loss {test_metrics.loss:.4f} | "
        f"test top1 {test_metrics.top1:.2f}% | "
        f"test top5 {test_metrics.top5:.2f}%"
    )


if __name__ == "__main__":
    main()
