"""Small shared utilities for reproducible experiments.

This module intentionally contains no model or training logic.  It provides
only the pieces that are useful across the project's future scripts:
reproducibility, device selection, directory creation, and CSV logging.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed: The integer seed to use for the experiment.
        deterministic: When true, request deterministic cuDNN behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return the best available PyTorch device.

    CUDA is preferred when available, followed by Apple's MPS backend and
    finally the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def append_csv_row(
    path: str | Path,
    metrics: Mapping[str, object],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    """Append one metric row to a CSV file, creating its header if needed.

    ``fieldnames`` can be supplied to enforce a stable column order.  If it is
    omitted, the keys from ``metrics`` are used in their insertion order.
    """
    csv_path = Path(path)
    ensure_dir(csv_path.parent)

    columns = fieldnames or list(metrics.keys())
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow({column: metrics.get(column, "") for column in columns})
