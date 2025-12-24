#!/usr/bin/env python3
"""Benchmark script for measuring training throughput across model sizes and data complexity."""

import argparse
import csv
import json
import logging
import os
import platform
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import colorlog
import torch
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data import ArithmeticDataset
from src.generation import generate_addition_examples
from src.model import Model, ModelSizeStr, create_model
from src.tokenizer import tokenizer
from src.training import (
    compute_metrics,
    data_collator,
    setup_training_optimizations,
)


@dataclass
class DataConfig:
    """Configuration for benchmark data generation."""

    max_digits: int
    max_operands: int
    name: str


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    model_size: str
    model_parameters: int
    data_config: str
    max_digits: int
    max_operands: int
    fp16: bool
    optimal_batch_size: int | None
    optimal_iterations_per_second: float
    optimal_samples_per_second: float
    baseline_batch_size: int
    baseline_samples_per_second: float | None
    speedup: float | None
    avg_sequence_length: float


def setup_logging() -> None:
    """Setup colored logging configuration."""
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def generate_benchmark_data(
    max_digits: int, max_operands: int, num_examples: int = 1000
) -> list[str]:
    """Generate benchmark data for specific complexity."""
    return generate_addition_examples(
        num_examples=num_examples,
        max_digits=max_digits,
        max_operands=max_operands,
        seed=42,
        include_chain_of_thought=True,
    )


def create_benchmark_dataset(examples: list[str], max_length: int) -> ArithmeticDataset:
    """Create dataset from examples."""
    # Create dataset dict format expected by ArithmeticDataset
    dataset_dict = {
        "examples": examples,
        "metadata": {
            "longest_example_length": max_length,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dataset_dict, f)
        temp_path = f.name

    try:
        dataset = ArithmeticDataset(
            data_path=temp_path,
            max_length=max_length,
        )
        return dataset
    finally:
        os.unlink(temp_path)


def benchmark_training_step(
    model: Model,
    dataloader: DataLoader[Any],
    device: torch.device,
    num_steps: int = 20,  # Reduced for speed
    fp16: bool = False,
) -> tuple[float, float]:
    """Benchmark training throughput.

    Returns:
        Tuple of (iterations_per_second, samples_per_second)
    """
    # Create training args optimized for benchmarking
    training_args = TrainingArguments(
        output_dir="/tmp/benchmark",
        overwrite_output_dir=True,
        # Training hyperparameters
        num_train_epochs=1,
        per_device_train_batch_size=32,  # Will be ignored, using dataloader batch size
        learning_rate=1e-3,  # Same as train.py default
        weight_decay=0.01,
        warmup_steps=500,
        # Optimization
        fp16=fp16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        # Disable evaluation and minimize logging for speed
        eval_strategy="no",
        logging_steps=10000,  # Large number to minimize logging
        save_steps=10000,  # Large number to avoid saving
        save_total_limit=1,
        # Disable W&B and other integrations
        report_to="none",
        # Other settings
        seed=42,
        torch_compile=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataloader.dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    model.train()

    # Create optimizer manually since trainer setup is complex for benchmarking
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,  # Same as train.py default
        weight_decay=0.01,
    )

    # Warmup steps using trainer's compute_loss method
    warmup_steps_count = 3  # Reduced for speed
    for i, batch in enumerate(dataloader):
        if i >= warmup_steps_count:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=fp16):
            loss = trainer.compute_loss(model, batch)

        # Handle case where compute_loss returns a tensor or tuple
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        optimizer.step()

    # Actual benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    total_samples = 0

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        batch_size = batch["input_ids"].shape[0]
        total_samples += batch_size

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=fp16):
            loss = trainer.compute_loss(model, batch)

        # Handle case where compute_loss returns a tensor or tuple
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_time = end_time - start_time
    iterations_per_second = num_steps / total_time
    samples_per_second = total_samples / total_time

    return iterations_per_second, samples_per_second


def run_benchmark() -> list[BenchmarkResult]:
    """Run comprehensive benchmark across all configurations."""
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup training optimizations exactly like train.py
    setup_training_optimizations()

    # Simplified configuration for initial benchmark
    model_sizes: list[ModelSizeStr] = [
        "xsmall",
        "small",
    ]  # Only run on small models for speed
    data_configs = [
        DataConfig(max_digits=1, max_operands=2, name="1d_2op"),
        DataConfig(max_digits=2, max_operands=4, name="2d_4op"),
        DataConfig(max_digits=5, max_operands=5, name="5d_5op"),
    ]

    # Batch sizes to try in descending order
    batch_sizes_to_try = [128, 64, 32, 16, 8, 4, 2, 1]
    num_benchmark_steps = 100  # Fast benchmark

    results = []

    for data_config in data_configs:
        logger.info(f"Data config: {data_config.name}")

        # Generate benchmark data once per data config
        examples = generate_benchmark_data(
            max_digits=data_config.max_digits,
            max_operands=data_config.max_operands,
            num_examples=500,  # Small dataset for speed
        )
        max_length = max(len(example) for example in examples)

        dataset = create_benchmark_dataset(examples, max_length)

        for model_size in model_sizes:
            logger.info(f"  Benchmarking {model_size} model")
            model = create_model(model_size)
            model.to(device)
            param_count = model.count_parameters()

            # Find optimal batch size by starting large and working down
            optimal_batch_size = None
            optimal_result = None

            # First, always test batch_size=1 for baseline (no memory check)
            batch_size = 1
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )

            try:
                iterations_per_sec, samples_per_sec = benchmark_training_step(
                    model=model,
                    dataloader=dataloader,
                    device=device,
                    num_steps=num_benchmark_steps,
                    fp16=False,
                )

                baseline_result = {
                    "batch_size": batch_size,
                    "iterations_per_second": iterations_per_sec,
                    "samples_per_second": samples_per_sec,
                }
                logger.info(
                    f"    Batch size 1: {iterations_per_sec:.1f} it/s, {samples_per_sec:.1f} samples/s"
                )
            except Exception as e:
                logger.error(f"    Failed at batch size 1: {e}")
                baseline_result = None

            # Now find the optimal batch size
            for batch_size in batch_sizes_to_try:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                )

                try:
                    iterations_per_sec, samples_per_sec = benchmark_training_step(
                        model=model,
                        dataloader=dataloader,
                        device=device,
                        num_steps=num_benchmark_steps,
                        fp16=False,
                    )

                    logger.info(
                        f"    Batch size {batch_size}: {iterations_per_sec:.1f} it/s, {samples_per_sec:.1f} samples/s"
                    )

                    # This is our optimal batch size
                    optimal_batch_size = batch_size
                    optimal_result = {
                        "batch_size": batch_size,
                        "iterations_per_second": iterations_per_sec,
                        "samples_per_second": samples_per_sec,
                    }
                    break  # Found the largest working batch size

                except Exception as e:
                    if "out of memory" in str(e).lower():
                        logger.info(f"    Batch size {batch_size}: OOM, trying smaller")
                    else:
                        logger.error(f"    Batch size {batch_size} failed: {e}")
                    continue

            # Record the best result
            if optimal_result:
                result = BenchmarkResult(
                    model_size=model_size,
                    model_parameters=param_count,
                    data_config=data_config.name,
                    max_digits=data_config.max_digits,
                    max_operands=data_config.max_operands,
                    fp16=False,
                    optimal_batch_size=optimal_batch_size,
                    optimal_iterations_per_second=round(
                        optimal_result["iterations_per_second"], 2
                    ),
                    optimal_samples_per_second=round(
                        optimal_result["samples_per_second"], 2
                    ),
                    baseline_batch_size=1,
                    baseline_samples_per_second=round(
                        baseline_result["samples_per_second"], 2
                    )
                    if baseline_result
                    else None,
                    speedup=round(
                        optimal_result["samples_per_second"]
                        / baseline_result["samples_per_second"],
                        2,
                    )
                    if baseline_result
                    else None,
                    avg_sequence_length=round(
                        sum(len(tokenizer.encode(ex)) for ex in examples[:50]) / 50,
                        1,
                    ),
                )
                results.append(result)

                logger.info(
                    f"    ✓ Optimal: batch_size={optimal_batch_size}, {optimal_result['samples_per_second']:.1f} samples/s"
                )
                if (
                    baseline_result
                    and optimal_batch_size is not None
                    and optimal_batch_size > 1
                ):
                    speedup = (
                        optimal_result["samples_per_second"]
                        / baseline_result["samples_per_second"]
                    )
                    logger.info(f"    → Speedup vs batch_size=1: {speedup:.2f}x")

    return results


@dataclass
class SystemInfo:
    """System information for benchmark reports."""

    hostname: str
    platform: str
    cpu: str
    memory_gb: float
    gpu_count: int
    gpu_name: str
    gpu_memory_gb: float
    cuda_version: str
    pytorch_version: str
    python_version: str


def get_system_info() -> SystemInfo:
    """Get system information including hostname, CPU, GPU, and memory."""
    hostname = socket.gethostname()
    platform_info = platform.platform()
    cpu = platform.processor() or "Unknown"
    python_version = platform.python_version()
    pytorch_version = torch.__version__

    # Get CPU info from /proc/cpuinfo if available
    try:
        with open("/proc/cpuinfo", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("model name"):
                    cpu = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, IndexError):
        pass

    # Get memory info
    memory_gb = 0.0
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("MemTotal"):
                    mem_kb = int(line.split()[1])
                    memory_gb = round(mem_kb / 1024 / 1024, 1)
                    break
    except (FileNotFoundError, ValueError, IndexError):
        memory_gb = 0.0

    # Get GPU info
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
        cuda_version = torch.version.cuda or "N/A"  # type: ignore[attr-defined]
    else:
        gpu_count = 0
        gpu_name = "None"
        gpu_memory_gb = 0.0
        cuda_version = "N/A"

    return SystemInfo(
        hostname=hostname,
        platform=platform_info,
        cpu=cpu,
        memory_gb=memory_gb,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        cuda_version=cuda_version,
        pytorch_version=pytorch_version,
        python_version=python_version,
    )


def save_results(results: list[BenchmarkResult], output_dir: str) -> None:
    """Save benchmark results to JSON, CSV, and Markdown files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = Path(output_dir) / "benchmark_results.json"
    with open(json_path, "w") as f:
        # Convert dataclasses to dicts for JSON serialization
        results_dict = [asdict(result) for result in results]
        json.dump(results_dict, f, indent=2)

    # Save CSV
    csv_path = Path(output_dir) / "benchmark_results.csv"
    if results:
        fieldnames = asdict(results[0]).keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([asdict(result) for result in results])

    # Save Markdown report
    md_path = Path(output_dir) / "benchmark_report.md"
    write_markdown_report(results, md_path)

    print(f"Results saved to {json_path}, {csv_path}, and {md_path}")


def write_markdown_report(results: list[BenchmarkResult], output_path: Path) -> None:
    """Write a comprehensive markdown report of benchmark results."""
    system_info = get_system_info()

    with open(output_path, "w") as f:
        # Header
        f.write("# Training Throughput Benchmark Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # System Information
        f.write("## System Information\n\n")
        f.write(f"- **Hostname**: {system_info.hostname}\n")
        f.write(f"- **Platform**: {system_info.platform}\n")
        f.write(f"- **CPU**: {system_info.cpu}\n")
        f.write(f"- **Memory**: {system_info.memory_gb} GB\n")
        f.write(f"- **GPU**: {system_info.gpu_name}")
        if system_info.gpu_count > 0:
            f.write(f" ({system_info.gpu_memory_gb} GB VRAM)\n")
        else:
            f.write("\n")
        f.write(f"- **CUDA Version**: {system_info.cuda_version}\n")
        f.write(f"- **PyTorch Version**: {system_info.pytorch_version}\n")
        f.write(f"- **Python Version**: {system_info.python_version}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        if results:
            best_overall = max(results, key=lambda x: x.optimal_samples_per_second)
            f.write(
                f"- **Best Performance**: {best_overall.optimal_samples_per_second:.1f} samples/s\n"
            )
            f.write(
                f"- **Best Configuration**: {best_overall.model_size} model, {best_overall.data_config} dataset\n"
            )
            f.write(f"- **Optimal Batch Size**: {best_overall.optimal_batch_size}\n")
            if best_overall.speedup:
                f.write(f"- **Speedup vs Batch=1**: {best_overall.speedup:.2f}x\n")
        f.write(f"- **Total Configurations Tested**: {len(results)}\n\n")

        # Detailed Results by Dataset
        data_configs = sorted(set(result.data_config for result in results))

        for data_config in data_configs:
            dataset_results = [r for r in results if r.data_config == data_config]
            if not dataset_results:
                continue

            f.write(f"## {data_config.upper().replace('_', ' ')} Dataset Results\n\n")

            # Dataset info
            sample_result = dataset_results[0]
            f.write(f"- **Max Digits**: {sample_result.max_digits}\n")
            f.write(f"- **Max Operands**: {sample_result.max_operands}\n")
            f.write(
                f"- **Average Sequence Length**: {sample_result.avg_sequence_length:.1f} tokens\n\n"
            )

            # Results table
            f.write(
                "| Model | Parameters | Optimal Batch | Samples/s | Baseline (b=1) | Speedup |\n"
            )
            f.write(
                "|-------|------------|---------------|-----------|----------------|----------|\n"
            )

            model_sizes = ["xsmall", "small", "medium", "large"]
            for model_size in model_sizes:
                model_results = [
                    r for r in dataset_results if r.model_size == model_size
                ]
                for result in model_results:
                    speedup_str = f"{result.speedup:.2f}x" if result.speedup else "N/A"
                    baseline_str = (
                        f"{result.baseline_samples_per_second:.1f}"
                        if result.baseline_samples_per_second
                        else "N/A"
                    )

                    f.write(
                        f"| {result.model_size} | {result.model_parameters:,} | {result.optimal_batch_size} | {result.optimal_samples_per_second:.1f} | {baseline_str} | {speedup_str} |\n"
                    )

            f.write("\n")

        # Performance Analysis
        f.write("## Performance Analysis\n\n")

        # Best performers by model size
        f.write("### Top Performers by Model Size\n\n")
        model_sizes = ["xsmall", "small", "medium", "large"]
        for model_size in model_sizes:
            model_results = [r for r in results if r.model_size == model_size]
            if model_results:
                best = max(model_results, key=lambda x: x.optimal_samples_per_second)
                f.write(
                    f"**{model_size.upper()}** ({best.model_parameters:,} parameters):\n"
                )
                f.write(
                    f"- Best performance: {best.optimal_samples_per_second:.1f} samples/s\n"
                )
                f.write(
                    f"- Configuration: {best.data_config} dataset, batch size {best.optimal_batch_size}\n"
                )
                if best.speedup:
                    f.write(f"- Speedup vs batch=1: {best.speedup:.2f}x\n")
                f.write("\n")

        # Batch size analysis
        f.write("### Batch Size Impact\n\n")
        batch_sizes = sorted(
            set(
                result.optimal_batch_size
                for result in results
                if result.optimal_batch_size
            )
        )
        for batch_size in batch_sizes:
            batch_results = [r for r in results if r.optimal_batch_size == batch_size]
            if batch_results:
                avg_perf = sum(
                    r.optimal_samples_per_second for r in batch_results
                ) / len(batch_results)
                f.write(
                    f"- **Batch size {batch_size}**: {avg_perf:.1f} samples/s average ({len(batch_results)} configs)\n"
                )

        f.write("\n## Raw Data\n\n")
        f.write(
            "Complete benchmark results are available in the accompanying JSON and CSV files.\n"
        )


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Group by data config for cleaner display
    data_configs = sorted(set(result.data_config for result in results))
    model_sizes = ["xsmall", "small", "medium", "large"]

    for data_config in data_configs:
        print(f"\n{data_config.upper()} Dataset")
        print("-" * 80)
        print(
            f"{'Model':<8} {'Optimal':<8} {'Samples/s':<12} {'Baseline':<12} {'Speedup':<8} {'Seq Len':<8}"
        )
        print(
            f"{'     ':<8} {'Batch':<8} {'        ':<12} {'(b=1)':<12} {'       ':<8} {'       ':<8}"
        )
        print("-" * 80)

        for model_size in model_sizes:
            # Get results for this model/data combo
            model_results = [
                r
                for r in results
                if r.model_size == model_size and r.data_config == data_config
            ]

            for result in model_results:
                speedup_str = f"{result.speedup:.2f}x" if result.speedup else "N/A"
                baseline_str = (
                    f"{result.baseline_samples_per_second:.1f}"
                    if result.baseline_samples_per_second
                    else "N/A"
                )

                print(
                    f"{model_size:<8} {result.optimal_batch_size:<8} "
                    f"{result.optimal_samples_per_second:<12.1f} {baseline_str:<12} "
                    f"{speedup_str:<8} {result.avg_sequence_length:<8.1f}"
                )

    print("\n" + "=" * 80)

    # Add overall best performers summary
    print("TOP PERFORMERS BY MODEL SIZE")
    print("=" * 80)

    for model_size in model_sizes:
        model_results = [r for r in results if r.model_size == model_size]
        if model_results:
            # Find best result by samples/second
            best = max(model_results, key=lambda x: x.optimal_samples_per_second)
            print(f"\n{model_size.upper()} ({best.model_parameters:,} params):")
            print(
                f"  Best config: {best.data_config} / batch={best.optimal_batch_size}"
            )
            print(f"  Performance: {best.optimal_samples_per_second:.1f} samples/s")
            if best.speedup:
                print(f"  Speedup vs batch=1: {best.speedup:.2f}x")

    print("\n" + "=" * 80)


def main() -> None:
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark model training throughput")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting benchmark")
    logger.info("This will test all model sizes with various data complexities")

    # Enable optimizations exactly like train.py
    setup_training_optimizations()

    # Run benchmark
    results = run_benchmark()

    # Save and display results
    save_results(results, args.output_dir)
    print_summary(results)

    logger.info("Benchmark completed!")


if __name__ == "__main__":
    main()
