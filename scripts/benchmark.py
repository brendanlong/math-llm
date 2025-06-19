#!/usr/bin/env python3
"""Benchmark script for measuring training throughput across model sizes and data complexity."""

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import colorlog
import torch
from torch.utils.data import DataLoader
from transformers.training_args import TrainingArguments

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data import ArithmeticDataset
from src.generation import generate_addition_examples
from src.model import ArithmeticModel, ModelSizeStr, create_model
from src.tokenizer import tokenizer
from src.training import (
    GumbelTrainer,
    compute_metrics,
    data_collator,
    setup_training_optimizations,
)


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
    model: ArithmeticModel,
    dataloader: DataLoader[Any],
    device: torch.device,
    use_gumbel: bool = False,
    gumbel_temperature: float = 1.0,
    num_steps: int = 20,  # Reduced for speed
    fp16: bool = False,
) -> tuple[float, float]:
    """Benchmark training throughput using GumbelTrainer like train.py.

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
        torch_compile=not use_gumbel,  # Disable compile for Gumbel mode
    )

    trainer = GumbelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataloader.dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        use_gumbel=use_gumbel,
        gumbel_temperature=gumbel_temperature,
        mask_reasoning=False,
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


def run_benchmark() -> list[dict[str, Any]]:
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
    ]  # Only run on small models for speed , "medium", "large"]
    data_configs = [
        {"max_digits": 1, "max_operands": 2, "name": "1d_2op"},
        {"max_digits": 2, "max_operands": 4, "name": "2d_4op"},
        {"max_digits": 5, "max_operands": 5, "name": "5d_5op"},
    ]
    training_modes = [
        {"use_gumbel": False, "name": "normal", "fp16": False},
        {"use_gumbel": True, "name": "gumbel", "fp16": False},
    ]

    # Batch sizes to try in descending order
    batch_sizes_to_try = [64, 32, 16, 8, 4, 2, 1]
    num_benchmark_steps = 20  # Fast benchmark

    results = []

    for data_config in data_configs:
        logger.info(f"Data config: {data_config['name']}")

        # Generate benchmark data once per data config
        examples = generate_benchmark_data(
            max_digits=int(data_config["max_digits"]),
            max_operands=int(data_config["max_operands"]),
            num_examples=500,  # Small dataset for speed
        )
        max_length = max(len(example) for example in examples)

        dataset = create_benchmark_dataset(examples, max_length)

        for model_size in model_sizes:
            logger.info(f"  Benchmarking {model_size} model")
            model = create_model(model_size)
            model.to(device)
            param_count = model.count_parameters()

            for training_mode in training_modes:
                logger.info(f"    Mode: {training_mode['name']}")

                # Find optimal batch size by starting large and working down
                optimal_batch_size = None
                optimal_result = None

                # First, always test batch_size=1 for baseline
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
                        use_gumbel=bool(training_mode["use_gumbel"]),
                        num_steps=num_benchmark_steps,
                        fp16=bool(training_mode["fp16"]),
                    )

                    baseline_result = {
                        "batch_size": batch_size,
                        "iterations_per_second": iterations_per_sec,
                        "samples_per_second": samples_per_sec,
                    }
                    logger.info(
                        f"      Batch size 1: {iterations_per_sec:.1f} it/s, {samples_per_sec:.1f} samples/s"
                    )
                except Exception as e:
                    logger.error(f"      Failed at batch size 1: {e}")
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
                            use_gumbel=bool(training_mode["use_gumbel"]),
                            num_steps=num_benchmark_steps,
                            fp16=bool(training_mode["fp16"]),
                        )

                        logger.info(
                            f"      Batch size {batch_size}: {iterations_per_sec:.1f} it/s, {samples_per_sec:.1f} samples/s"
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
                            logger.info(
                                f"      Batch size {batch_size}: OOM, trying smaller"
                            )
                        else:
                            logger.error(f"      Batch size {batch_size} failed: {e}")
                        continue

                # Record the best result
                if optimal_result:
                    result = {
                        "model_size": model_size,
                        "model_parameters": param_count,
                        "data_config": data_config["name"],
                        "max_digits": data_config["max_digits"],
                        "max_operands": data_config["max_operands"],
                        "training_mode": training_mode["name"],
                        "use_gumbel": training_mode["use_gumbel"],
                        "fp16": training_mode["fp16"],
                        "optimal_batch_size": optimal_batch_size,
                        "optimal_iterations_per_second": round(
                            optimal_result["iterations_per_second"], 2
                        ),
                        "optimal_samples_per_second": round(
                            optimal_result["samples_per_second"], 2
                        ),
                        "baseline_batch_size": 1,
                        "baseline_samples_per_second": round(
                            baseline_result["samples_per_second"], 2
                        )
                        if baseline_result
                        else None,
                        "speedup": round(
                            optimal_result["samples_per_second"]
                            / baseline_result["samples_per_second"],
                            2,
                        )
                        if baseline_result
                        else None,
                        "avg_sequence_length": round(
                            sum(len(tokenizer.encode(ex)) for ex in examples[:50]) / 50,
                            1,
                        ),
                    }
                    results.append(result)

                    logger.info(
                        f"      ✓ Optimal: batch_size={optimal_batch_size}, {optimal_result['samples_per_second']:.1f} samples/s"
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
                        logger.info(f"      → Speedup vs batch_size=1: {speedup:.2f}x")

    return results


def save_results(results: list[dict[str, Any]], output_dir: str) -> None:
    """Save benchmark results to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = Path(output_dir) / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    csv_path = Path(output_dir) / "benchmark_results.csv"
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"Results saved to {json_path} and {csv_path}")


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Group by data config for cleaner display
    data_configs = sorted(set(result["data_config"] for result in results))
    model_sizes = ["xsmall", "small", "medium", "large"]

    for data_config in data_configs:
        print(f"\n{data_config.upper()} Dataset")
        print("-" * 80)
        print(
            f"{'Model':<8} {'Mode':<10} {'Optimal':<8} {'Samples/s':<12} {'Baseline':<12} {'Speedup':<8} {'Seq Len':<8}"
        )
        print(
            f"{'     ':<8} {'    ':<10} {'Batch':<8} {'        ':<12} {'(b=1)':<12} {'       ':<8} {'       ':<8}"
        )
        print("-" * 80)

        for model_size in model_sizes:
            # Get results for this model/data combo
            model_results = [
                r
                for r in results
                if r["model_size"] == model_size and r["data_config"] == data_config
            ]

            for result in sorted(model_results, key=lambda x: x["training_mode"]):
                speedup_str = (
                    f"{result['speedup']:.2f}x" if result.get("speedup") else "N/A"
                )
                baseline_str = (
                    f"{result['baseline_samples_per_second']:.1f}"
                    if result.get("baseline_samples_per_second")
                    else "N/A"
                )

                print(
                    f"{model_size:<8} {result['training_mode']:<10} {result['optimal_batch_size']:<8} "
                    f"{result['optimal_samples_per_second']:<12.1f} {baseline_str:<12} "
                    f"{speedup_str:<8} {result['avg_sequence_length']:<8.1f}"
                )

    print("\n" + "=" * 80)

    # Add overall best performers summary
    print("TOP PERFORMERS BY MODEL SIZE")
    print("=" * 80)

    for model_size in model_sizes:
        model_results = [r for r in results if r["model_size"] == model_size]
        if model_results:
            # Find best result by samples/second
            best = max(model_results, key=lambda x: x["optimal_samples_per_second"])
            print(f"\n{model_size.upper()} ({best['model_parameters']:,} params):")
            print(
                f"  Best config: {best['data_config']} / {best['training_mode']} / batch={best['optimal_batch_size']}"
            )
            print(f"  Performance: {best['optimal_samples_per_second']:.1f} samples/s")
            if best.get("speedup"):
                print(f"  Speedup vs batch=1: {best['speedup']:.2f}x")

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
