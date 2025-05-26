#!/usr/bin/env python3
"""Generate arithmetic training data for the math LLM.

This script generates single-digit addition problems in the format:
"operand1+operand2=result<end>"

The data is saved as JSON files with train/validation/test splits (80/10/10).
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import TypedDict

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.tokenizer import ArithmeticTokenizer


class ExampleDict(TypedDict):
    """Type definition for a single training example."""

    text: str
    tokens: list[int]
    length: int


class MetadataDict(TypedDict):
    """Type definition for dataset metadata."""

    split: str
    num_examples: int
    vocab_size: int
    format: str


class DatasetDict(TypedDict):
    """Type definition for the complete dataset."""

    examples: list[ExampleDict]
    metadata: MetadataDict


def generate_single_digit_addition(num_examples: int, seed: int = 42) -> list[str]:
    """Generate single-digit addition examples.

    Args:
        num_examples: Number of examples to generate
        seed: Random seed for reproducibility

    Returns:
        List of arithmetic expressions in format "a+b=c<end>"
    """
    random.seed(seed)
    examples = []

    # Generate all possible single-digit addition combinations
    all_combinations = []
    for a in range(10):
        for b in range(10):
            result = a + b
            example = f"{a}+{b}={result}<end>"
            all_combinations.append(example)

    # If we need fewer examples than all combinations, sample randomly
    if num_examples <= len(all_combinations):
        examples = random.sample(all_combinations, num_examples)
    else:
        # If we need more examples, sample with replacement
        examples = random.choices(all_combinations, k=num_examples)

    return examples


def split_data(
    examples: list[str], train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[list[str], list[str], list[str]]:
    """Split data into train/validation/test sets.

    Args:
        examples: List of examples to split
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set (test gets remainder)

    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    total = len(examples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # Shuffle examples before splitting
    shuffled = examples.copy()
    random.shuffle(shuffled)

    train_examples = shuffled[:train_size]
    val_examples = shuffled[train_size : train_size + val_size]
    test_examples = shuffled[train_size + val_size :]

    return train_examples, val_examples, test_examples


def save_dataset(examples: list[str], output_path: Path, split_name: str) -> None:
    """Save dataset to JSON file.

    Args:
        examples: List of examples to save
        output_path: Directory to save the file
        split_name: Name of the split (train/val/test)
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dataset with both raw text and tokenized versions
    tokenizer = ArithmeticTokenizer()
    dataset: DatasetDict = {
        "examples": [],
        "metadata": {
            "split": split_name,
            "num_examples": len(examples),
            "vocab_size": tokenizer.vocab_size,
            "format": "operand1+operand2=result<end>",
        },
    }

    for text in examples:
        try:
            tokens = tokenizer.encode(text)
            dataset["examples"].append(
                {"text": text, "tokens": tokens, "length": len(tokens)}
            )
        except ValueError as e:
            print(f"Warning: Skipping invalid example '{text}': {e}")

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
        help="Number of examples to generate (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data",
        help="Output directory for datasets (default: data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )

    args = parser.parse_args()

    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("train_ratio + val_ratio must be less than 1.0")

    print(f"Generating {args.num_examples} single-digit addition examples...")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {1 - args.train_ratio - args.val_ratio}"
    )

    # Generate examples
    examples = generate_single_digit_addition(args.num_examples, args.seed)
    print(f"Generated {len(examples)} examples")

    # Show some sample examples
    print("\nSample examples:")
    for i, example in enumerate(examples[:5]):
        print(f"  {i + 1}: {example}")

    # Split data
    train_examples, val_examples, test_examples = split_data(
        examples, args.train_ratio, args.val_ratio
    )

    print("\nData splits:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")

    # Save datasets
    print(f"\nSaving datasets to {args.output_dir}/...")
    save_dataset(train_examples, args.output_dir, "train")
    save_dataset(val_examples, args.output_dir, "val")
    save_dataset(test_examples, args.output_dir, "test")

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
