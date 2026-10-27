"""Summarize per-seed experiment JSON files for the midterm report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from collections import Counter


EXPERIMENT_PATTERNS = {
    "scratch": "student_seed*_summary.json",
    "vanilla_kd": "kd_student_seed*_summary.json",
}
EXPECTED_SEEDS = {42, 123, 999}
HYPERPARAMETER_FIELDS = ("learning_rate", "batch_size", "epochs", "alpha", "tau")


def _read_summaries(
    results_dir: Path, pattern: str
) -> list[tuple[Path, dict[str, object]]]:
    """Read all matching JSON summaries, sorted by filename."""
    summaries = []
    for path in sorted(results_dir.glob(pattern)):
        with path.open("r", encoding="utf-8") as json_file:
            summary = json.load(json_file)
        if not isinstance(summary, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        summaries.append((path, summary))
    return summaries


def _number(summary: dict[str, object], key: str, path_label: str) -> float:
    """Read a numeric summary field with a useful error message."""
    value = summary.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path_label} is missing numeric field '{key}'")
    return float(value)


def _format_stat(values: list[float], *, sample_std: bool = False) -> str:
    """Format a mean or sample standard deviation, or n/a if unavailable."""
    if not values:
        return "n/a"
    if sample_std:
        # Sample standard deviation is undefined for a single observation.
        if len(values) < 2:
            return "n/a"
        return f"{stdev(values):.4f}"
    return f"{mean(values):.4f}"


def _summarize(summaries: list[dict[str, object]]) -> dict[str, str]:
    """Calculate report statistics from summary JSON values."""
    test_top1 = [_number(item, "test_top1", "summary") for item in summaries]
    test_top5 = [_number(item, "test_top5", "summary") for item in summaries]
    runtime = [
        float(item["runtime_seconds"])
        for item in summaries
        if isinstance(item.get("runtime_seconds"), (int, float))
        and not isinstance(item.get("runtime_seconds"), bool)
        and math.isfinite(float(item["runtime_seconds"]))
    ]
    return {
        "test_top1_mean": _format_stat(test_top1),
        "test_top1_std": _format_stat(test_top1, sample_std=True),
        "test_top5_mean": _format_stat(test_top5),
        "test_top5_std": _format_stat(test_top5, sample_std=True),
        "runtime_mean": _format_stat(runtime),
        "runs": str(len(summaries)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize scratch and vanilla-KD result JSON files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing per-seed summary JSON files (default: results)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Results directory: {args.results_dir}")
    print("Standard deviation: sample standard deviation (n-1).")
    print()
    print(
        f"{'Experiment':<14} {'Runs':>4} {'Top-1 mean':>12} "
        f"{'Top-1 std':>11} {'Top-5 mean':>12} {'Top-5 std':>11} "
        f"{'Runtime mean (s)':>17}"
    )
    print("-" * 96)

    for experiment, pattern in EXPERIMENT_PATTERNS.items():
        records = _read_summaries(args.results_dir, pattern)
        summaries = [summary for _, summary in records]
        seeds = [summary.get("seed") for summary in summaries]
        seed_counts = Counter(seed for seed in seeds if isinstance(seed, int))

        print(f"\n{experiment} files included:")
        if records:
            for path, summary in records:
                print(f"  - {path} (seed={summary.get('seed', 'missing')})")
        else:
            print("  (none)")
        print(f"{experiment} seeds included: {seeds or 'none'}")

        missing_seeds = sorted(EXPECTED_SEEDS - set(seed_counts))
        if missing_seeds:
            print(
                f"WARNING: {experiment} is missing expected final seed(s): "
                f"{missing_seeds}"
            )
        duplicate_seeds = sorted(seed for seed, count in seed_counts.items() if count > 1)
        if duplicate_seeds:
            print(
                f"WARNING: {experiment} contains duplicate seed(s): "
                f"{duplicate_seeds}; all matching files are being included."
            )

        for field in HYPERPARAMETER_FIELDS:
            values = [summary.get(field) for summary in summaries]
            if not summaries or not all(value is not None for value in values):
                print(
                    f"{experiment} {field}: consistency cannot be verified "
                    "because it is not stored in all summaries."
                )
                continue
            if len(set(map(str, values))) > 1:
                print(
                    f"WARNING: {experiment} has inconsistent {field} values: "
                    f"{values}"
                )
            else:
                print(f"{experiment} {field}: consistent ({values[0]})")

        statistics = _summarize(summaries)
        print(
            f"{experiment:<14} {statistics['runs']:>4} "
            f"{statistics['test_top1_mean']:>12} "
            f"{statistics['test_top1_std']:>11} "
            f"{statistics['test_top5_mean']:>12} "
            f"{statistics['test_top5_std']:>11} "
            f"{statistics['runtime_mean']:>17}"
        )


if __name__ == "__main__":
    main()
