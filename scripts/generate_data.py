#!/usr/bin/env python3
"""Generate arithmetic training data for the math LLM with chain-of-thought reasoning.

This script generates addition problems of varying complexity in the format:
- Simple: "3+5=8<end>"
- With reasoning: "3+5+2=<think>3+5+2=8+2=01</think>10<end>"

Supports single-digit through multi-digit addition (up to 10 digits) with
a distribution that ensures simpler examples remain well-represented.

The data is saved as JSON files with train/validation/test splits (80/10/10).
"""

import argparse
import json
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.generation import generate_addition_examples_parallel, split_data
from src.types import DatasetDict, DatasetMetadata


def save_dataset(
    examples: list[str],
    output_path: Path,
    split_name: str,
    metadata: DatasetMetadata | None = None,
) -> None:
    """Save dataset to JSON file.

    Args:
        examples: List of examples to save
        output_path: Directory to save the file
        split_name: Name of the split (train/val/test)
    """
    output_path.mkdir(parents=True, exist_ok=True)

    dataset: DatasetDict = {"examples": examples}
    if metadata:
        dataset["metadata"] = metadata

    # Save to JSON file
    output_file = output_path / f"{split_name}.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset['examples'])} examples to {output_file}")


def main():
    """Generate and save arithmetic training data."""
    parser = argparse.ArgumentParser(description="Generate arithmetic training data")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10000,
        help="Number of examples to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--max-digits",
        type=int,
        default=2,
        help="Maximum number of digits per operand (default: %(default)s)",
    )
    parser.add_argument(
        "--max-operands",
        type=int,
        default=3,
        help="Maximum number of operands (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data",
        help="Output directory for datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: %(default)s)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--no-include-cot",
        action="store_true",
        help="Disable including reasoning in <think> tags",
    )
    parser.add_argument(
        "--fixed-length-cot",
        action="store_true",
        help="Enable fixed-length chain-of-thought padding with <noop> tokens",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes for parallel generation (default: %(default)s)",
    )

    args = parser.parse_args()

    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("train_ratio + val_ratio must be less than 1.0")

    print(f"Generating {args.num_examples} addition examples...")
    print(f"Maximum digits per operand: {args.max_digits}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of workers: {args.num_workers}")
    print(
        f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {1 - args.train_ratio - args.val_ratio}"
    )

    # Generate examples
    examples = generate_addition_examples_parallel(
        num_examples=args.num_examples,
        max_digits=args.max_digits,
        seed=args.seed,
        fixed_length_cot=args.fixed_length_cot,
        include_chain_of_thought=not args.no_include_cot,
        max_operands=args.max_operands,
        num_workers=args.num_workers,
    )
    print(f"Generated {len(examples)} examples")

    # Find longest example
    longest_example = max(examples, key=len)
    longest_length = len(longest_example)
    print(f"Longest example length: {longest_length} characters")
    print(f"Longest example: {longest_example}")

    # Show some sample examples
    print("\nSample examples:")
    for i, example in enumerate(examples[:5]):
        print(f"  {i + 1}: {example}")

    # Split data
    train_examples, val_examples, test_examples = split_data(
        examples, args.train_ratio, args.val_ratio, args.seed
    )

    print("\nData splits:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")

    # Create metadata
    metadata: DatasetMetadata = {
        "max_digits": args.max_digits,
        "max_operands": args.max_operands,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "include_chain_of_thought": not args.no_include_cot,
        "fixed_length_cot": args.fixed_length_cot,
        "num_examples": args.num_examples,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "generation_timestamp": datetime.now().isoformat(),
        "generation_version": "1.0.0",
        "longest_example_length": longest_length,
        "longest_example": longest_example,
    }

    # Save datasets
    print(f"\nSaving datasets to {args.output_dir}/...")
    save_dataset(train_examples, args.output_dir, "train", metadata)
    save_dataset(val_examples, args.output_dir, "val", metadata)
    save_dataset(test_examples, args.output_dir, "test", metadata)

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
