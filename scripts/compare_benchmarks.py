#!/usr/bin/env python3
"""Compare benchmark results from two different folders."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ChangeEntry:
    """Tracks changes between two benchmark runs."""

    config: tuple[str, str, str, bool, bool]
    before: "BenchmarkRow"
    after: "BenchmarkRow"
    samples_change: float
    samples_percent: float
    iterations_change: float
    iterations_percent: float
    speedup_change: float
    speedup_percent: float


@dataclass
class BenchmarkRow:
    """Represents a single benchmark result."""

    model_size: str
    model_parameters: int
    data_config: str
    max_digits: int
    max_operands: int
    training_mode: str
    use_gumbel: bool
    fp16: bool
    optimal_batch_size: int
    optimal_iterations_per_second: float
    optimal_samples_per_second: float
    baseline_batch_size: int
    baseline_samples_per_second: float
    speedup: float
    avg_sequence_length: float

    @property
    def key(self) -> tuple[str, str, str, bool, bool]:
        """Return a unique key for this configuration."""
        return (
            self.model_size,
            self.data_config,
            self.training_mode,
            self.use_gumbel,
            self.fp16,
        )

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "BenchmarkRow":
        """Create a BenchmarkRow from a CSV row."""
        return cls(
            model_size=row["model_size"],
            model_parameters=int(row["model_parameters"]),
            data_config=row["data_config"],
            max_digits=int(row["max_digits"]),
            max_operands=int(row["max_operands"]),
            training_mode=row["training_mode"],
            use_gumbel=row["use_gumbel"] == "True",
            fp16=row["fp16"] == "True",
            optimal_batch_size=int(row["optimal_batch_size"]),
            optimal_iterations_per_second=float(row["optimal_iterations_per_second"]),
            optimal_samples_per_second=float(row["optimal_samples_per_second"]),
            baseline_batch_size=int(row["baseline_batch_size"]),
            baseline_samples_per_second=float(row["baseline_samples_per_second"]),
            speedup=float(row["speedup"]),
            avg_sequence_length=float(row["avg_sequence_length"]),
        )


def load_benchmarks(
    file_path: Path,
) -> dict[tuple[str, str, str, bool, bool], BenchmarkRow]:
    """Load benchmark results from a CSV file."""
    results = {}
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            benchmark = BenchmarkRow.from_csv_row(row)
            results[benchmark.key] = benchmark
    return results


def format_change(old_value: float, new_value: float) -> str:
    """Format the change between two values."""
    absolute_change = new_value - old_value
    if old_value != 0:
        percent_change = ((new_value - old_value) / old_value) * 100
        return f"{absolute_change:+.2f} ({percent_change:+.1f}%)"
    else:
        return f"{absolute_change:+.2f} (N/A%)"


def matches_filters(
    row: BenchmarkRow,
    model_filter: Optional[str],
    dataset_filter: Optional[str],
    mode_filter: Optional[str],
) -> bool:
    """Check if a row matches the given filters."""
    if model_filter and row.model_size != model_filter:
        return False
    if dataset_filter and row.data_config != dataset_filter:
        return False
    if mode_filter and row.training_mode != mode_filter:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results from two different folders"
    )
    parser.add_argument(
        "before_path", type=Path, help="Path to the 'before' benchmark results CSV file"
    )
    parser.add_argument(
        "after_path", type=Path, help="Path to the 'after' benchmark results CSV file"
    )
    parser.add_argument(
        "--model",
        choices=["xsmall", "small", "medium", "large"],
        help="Filter by model size",
    )
    parser.add_argument(
        "--dataset", help="Filter by dataset (e.g., 1d_2op, 2d_4op, 5d_5op)"
    )
    parser.add_argument(
        "--mode", choices=["normal", "gumbel"], help="Filter by training mode"
    )
    parser.add_argument(
        "--metric",
        choices=["samples_per_second", "iterations_per_second", "speedup", "all"],
        default="all",
        help="Which metric(s) to compare (default: all)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["improvement", "regression", "absolute"],
        default="improvement",
        help="Sort results by improvement, regression, or absolute change",
    )

    args = parser.parse_args()

    # Load benchmark results
    before_results = load_benchmarks(args.before_path)
    after_results = load_benchmarks(args.after_path)

    # Find common configurations
    common_keys = set(before_results.keys()) & set(after_results.keys())

    if not common_keys:
        print("No common configurations found between the two benchmark files.")
        return

    # Calculate changes
    changes: list[ChangeEntry] = []
    for key in common_keys:
        before = before_results[key]
        after = after_results[key]

        # Apply filters
        if not matches_filters(before, args.model, args.dataset, args.mode):
            continue

        # Calculate performance changes
        samples_change = (
            after.optimal_samples_per_second - before.optimal_samples_per_second
        )
        samples_percent = (
            (samples_change / before.optimal_samples_per_second) * 100
            if before.optimal_samples_per_second != 0
            else 0
        )

        iterations_change = (
            after.optimal_iterations_per_second - before.optimal_iterations_per_second
        )
        iterations_percent = (
            (iterations_change / before.optimal_iterations_per_second) * 100
            if before.optimal_iterations_per_second != 0
            else 0
        )

        speedup_change = after.speedup - before.speedup
        speedup_percent = (
            (speedup_change / before.speedup) * 100 if before.speedup != 0 else 0
        )

        changes.append(
            ChangeEntry(
                config=key,
                before=before,
                after=after,
                samples_change=samples_change,
                samples_percent=samples_percent,
                iterations_change=iterations_change,
                iterations_percent=iterations_percent,
                speedup_change=speedup_change,
                speedup_percent=speedup_percent,
            )
        )

    if not changes:
        print("No configurations match the specified filters.")
        return

    # Sort results
    if args.sort_by == "improvement":
        changes.sort(key=lambda x: x.samples_percent, reverse=True)
    elif args.sort_by == "regression":
        changes.sort(key=lambda x: x.samples_percent)
    else:  # absolute
        changes.sort(key=lambda x: abs(x.samples_percent), reverse=True)

    # Print header
    print(f"\nComparing: {args.before_path} → {args.after_path}")
    print("=" * 120)

    # Print results
    for change in changes:
        config = change.config
        before = change.before
        after = change.after

        print(
            f"\n{config[0]} | {config[1]} | {config[2]} | Gumbel: {config[3]} | FP16: {config[4]}"
        )
        print("-" * 120)

        if args.metric in ["samples_per_second", "all"]:
            print(
                f"  Samples/sec:     {before.optimal_samples_per_second:8.2f} → {after.optimal_samples_per_second:8.2f}  "
                f"Change: {format_change(before.optimal_samples_per_second, after.optimal_samples_per_second)}"
            )

        if args.metric in ["iterations_per_second", "all"]:
            print(
                f"  Iterations/sec:  {before.optimal_iterations_per_second:8.2f} → {after.optimal_iterations_per_second:8.2f}  "
                f"Change: {format_change(before.optimal_iterations_per_second, after.optimal_iterations_per_second)}"
            )

        if args.metric in ["speedup", "all"]:
            print(
                f"  Speedup:         {before.speedup:8.2f}x → {after.speedup:8.2f}x  "
                f"Change: {format_change(before.speedup, after.speedup)}"
            )

        if args.metric == "all":
            print(
                f"  Optimal batch:   {before.optimal_batch_size:8d} → {after.optimal_batch_size:8d}"
            )

    # Print summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("-" * 120)

    avg_samples_percent = sum(c.samples_percent for c in changes) / len(changes)
    avg_iterations_percent = sum(c.iterations_percent for c in changes) / len(changes)

    improvements = sum(1 for c in changes if c.samples_percent > 0)
    regressions = sum(1 for c in changes if c.samples_percent < 0)

    print(f"Total configurations compared: {len(changes)}")
    print(
        f"Improvements: {improvements} | Regressions: {regressions} | No change: {len(changes) - improvements - regressions}"
    )
    print(f"Average samples/sec change: {avg_samples_percent:+.1f}%")
    print(f"Average iterations/sec change: {avg_iterations_percent:+.1f}%")

    # Find best and worst changes
    if changes:
        best = max(changes, key=lambda x: x.samples_percent)
        worst = min(changes, key=lambda x: x.samples_percent)

        print(
            f"\nBiggest improvement: {best.config[0]} {best.config[1]} {best.config[2]} "
            f"({best.samples_percent:+.1f}% samples/sec)"
        )
        print(
            f"Biggest regression: {worst.config[0]} {worst.config[1]} {worst.config[2]} "
            f"({worst.samples_percent:+.1f}% samples/sec)"
        )


if __name__ == "__main__":
    main()
