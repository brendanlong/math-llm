#!/usr/bin/env python3
"""Benchmark data loader performance with different configurations."""

import argparse
import sys
import time
from pathlib import Path
from typing import TypedDict

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_dataloader


class BenchmarkResult(TypedDict):
    """Result from data loader benchmark."""

    batch_size: int
    total_samples: int
    total_batches: int
    elapsed_time: float
    samples_per_second: float
    batches_per_second: float
    time_per_batch: float


def benchmark_dataloader(
    data_path: Path,
    batch_size: int,
    max_batches: int = 100,
    warmup_batches: int = 10,
) -> BenchmarkResult:
    """Benchmark data loader performance.

    Args:
        data_path: Path to data file
        batch_size: Batch size to test
        max_batches: Maximum number of batches to process
        warmup_batches: Number of warmup batches (excluded from timing)

    Returns:
        Dictionary with benchmark results
    """

    dataloader = create_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        shuffle=True,
    )

    # Warmup phase
    print(f"Warming up with {warmup_batches} batches...")
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break

    # Benchmark phase
    print(f"Benchmarking {max_batches} batches...")
    start_time = time.time()
    total_samples = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        total_samples += batch["input_ids"].size(0)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate metrics
    samples_per_second = total_samples / elapsed_time
    batches_per_second = max_batches / elapsed_time
    time_per_batch = elapsed_time / max_batches

    return {
        "batch_size": batch_size,
        "total_samples": total_samples,
        "total_batches": max_batches,
        "elapsed_time": elapsed_time,
        "samples_per_second": samples_per_second,
        "batches_per_second": batches_per_second,
        "time_per_batch": time_per_batch,
    }


def run_benchmark_suite(data_path: Path) -> None:
    """Run a comprehensive benchmark suite."""
    print(f"Benchmarking data loader performance on {data_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    # Test configurations
    batch_sizes = [16, 32, 64, 128]

    results: list[BenchmarkResult] = []

    for batch_size in batch_sizes:
        print(f"Testing batch_size={batch_size}")

        try:
            result = benchmark_dataloader(
                data_path=data_path,
                batch_size=batch_size,
            )
            results.append(result)

            print(f"  Samples/sec: {result['samples_per_second']:.1f}")
            print(f"  Batches/sec: {result['batches_per_second']:.1f}")
            print(f"  Time/batch: {result['time_per_batch']:.3f}s")
            print()

        except Exception as e:
            print(f"  Error: {e}")
            print()

    # Find best configuration
    if results:
        best_result = max(results, key=lambda x: x["samples_per_second"])
        print("Best configuration:")
        print(f"  Batch size: {best_result['batch_size']}")
        print(f"  Samples/sec: {best_result['samples_per_second']:.1f}")
        print(f"  Time/batch: {best_result['time_per_batch']:.3f}s")


def main() -> None:
    """Main benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark data loader performance")

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/train.jsonl"),
        help="Path to data file for benchmarking",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Single batch size to test (runs full suite if not specified)",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Maximum number of batches to process per test",
    )

    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=10,
        help="Number of warmup batches (excluded from timing)",
    )

    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"Error: Data file {args.data_path} does not exist")
        print("Run 'python scripts/generate_data.py' to create training data")
        return

    if args.batch_size is not None:
        # Single configuration test
        print("Running single configuration benchmark...")
        result = benchmark_dataloader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            warmup_batches=args.warmup_batches,
        )

        print(f"Results for batch_size={args.batch_size}:")
        print(f"  Total samples: {result['total_samples']}")
        print(f"  Elapsed time: {result['elapsed_time']:.2f}s")
        print(f"  Samples/sec: {result['samples_per_second']:.1f}")
        print(f"  Batches/sec: {result['batches_per_second']:.1f}")
        print(f"  Time/batch: {result['time_per_batch']:.3f}s")
    else:
        # Full benchmark suite
        run_benchmark_suite(args.data_path)


if __name__ == "__main__":
    main()
