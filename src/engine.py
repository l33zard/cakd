"""Basic supervised training and evaluation utilities.

This module deliberately contains no knowledge-distillation logic.  Models,
optimizers, criteria, loaders, and devices are supplied by the caller so the
functions do not rely on hidden global state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(frozen=True)
class EpochMetrics:
    """Dataset-level metrics returned by training and evaluation."""

    loss: float
    top1: float
    top5: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a plain dictionary for logging or checkpoints."""
        return asdict(self)


def _correct_count(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    """Count examples whose target is among the top ``k`` predictions."""
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("logits must be 2D and targets must be 1D")
    if logits.size(0) != targets.size(0):
        raise ValueError("logits and targets must contain the same batch size")
    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, logits.size(1))
    predictions = logits.topk(k, dim=1).indices
    correct = predictions.eq(targets.unsqueeze(1)).any(dim=1)
    return int(correct.sum().item())


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Calculate top-k accuracy for one batch as a percentage."""
    return 100.0 * _correct_count(logits, targets, k) / targets.size(0)


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate top-1 accuracy for one batch as a percentage."""
    return top_k_accuracy(logits, targets, k=1)


def top5_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate top-5 accuracy for one batch as a percentage."""
    return top_k_accuracy(logits, targets, k=5)


def _prepare_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move an image/label batch to the requested device."""
    images, targets = batch
    non_blocking = device.type == "cuda"
    return (
        images.to(device, non_blocking=non_blocking),
        targets.to(device, non_blocking=non_blocking),
    )


def _metrics_from_totals(
    total_loss: float,
    top1_correct: int,
    top5_correct: int,
    total_examples: int,
) -> EpochMetrics:
    """Convert accumulated example-level totals into epoch metrics."""
    if total_examples == 0:
        raise ValueError("cannot calculate metrics for an empty data loader")
    return EpochMetrics(
        loss=total_loss / total_examples,
        top1=100.0 * top1_correct / total_examples,
        top5=100.0 * top5_correct / total_examples,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    show_progress: bool = False,
) -> EpochMetrics:
    """Train ``model`` for one supervised epoch and return its metrics."""
    model.train()
    device = torch.device(device)
    iterator = tqdm(loader, desc="train", leave=False) if show_progress else loader

    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total_examples = 0

    for batch in iterator:
        images, targets = _prepare_batch(batch, device)
        batch_size = targets.size(0)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        top1_correct += _correct_count(logits.detach(), targets, k=1)
        top5_correct += _correct_count(logits.detach(), targets, k=5)
        total_examples += batch_size

    return _metrics_from_totals(
        total_loss,
        top1_correct,
        top5_correct,
        total_examples,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    show_progress: bool = False,
) -> EpochMetrics:
    """Evaluate ``model`` without tracking gradients and return its metrics."""
    model.eval()
    device = torch.device(device)
    iterator = tqdm(loader, desc="eval", leave=False) if show_progress else loader

    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total_examples = 0

    for batch in iterator:
        images, targets = _prepare_batch(batch, device)
        batch_size = targets.size(0)
        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * batch_size
        top1_correct += _correct_count(logits, targets, k=1)
        top5_correct += _correct_count(logits, targets, k=5)
        total_examples += batch_size

    return _metrics_from_totals(
        total_loss,
        top1_correct,
        top5_correct,
        total_examples,
    )


def save_checkpoint_if_best(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None,
    epoch: int,
    metrics: EpochMetrics | Mapping[str, float],
    best_val_top1: float,
) -> tuple[float, bool]:
    """Save a checkpoint only when validation top-1 accuracy improves.

    Returns:
        A tuple containing the updated best top-1 value and whether a file was
        saved during this call.
    """
    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        existing_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        existing_metrics = existing_checkpoint.get("metrics", {})
        existing_best = float(
            existing_checkpoint.get(
                "best_val_top1",
                existing_metrics.get("top1", float("-inf")),
            )
        )
        best_val_top1 = max(best_val_top1, existing_best)

    current_metrics = metrics.as_dict() if isinstance(metrics, EpochMetrics) else dict(metrics)
    current_top1 = float(current_metrics["top1"])
    if current_top1 <= best_val_top1:
        return best_val_top1, False

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": current_metrics,
        "best_val_top1": current_top1,
    }
    torch.save(checkpoint, checkpoint_path)
    return current_top1, True
