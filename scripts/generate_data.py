#!/usr/bin/env python3
"""Generate arithmetic training data for the math LLM.

This script generates addition problems of varying complexity in the format:
"operand1+operand2=result<end>"

Supports single-digit through multi-digit addition (up to 10 digits) with
a distribution that ensures simpler examples remain well-represented.

The data is saved as JSON files with train/validation/test splits (80/10/10).
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.tokenizer import ArithmeticTokenizer
from src.types import DatasetDict


def generate_addition_examples(
    num_examples: int, max_digits: int = 8, seed: int = 42
) -> list[str]:
    """Generate addition examples with uniform distribution across digit lengths.

    Args:
        num_examples: Number of examples to generate
        max_digits: Maximum number of digits per operand (1-8)
        seed: Random seed for reproducibility

    Returns:
        List of arithmetic expressions in format "a+b=c<end>"
    """
    random.seed(seed)
    examples = []

    for _ in range(num_examples):
        # Generate each operand independently with random digit count
        num_digits_a = random.randint(1, max_digits)
        num_digits_b = random.randint(1, max_digits)

        # Generate operands (0 to 10^digits - 1)
        a = random.randint(10 ** (num_digits_a - 1), 10**num_digits_a - 1)
        b = random.randint(10 ** (num_digits_b - 1), 10**num_digits_b - 1)

        result = a + b
        example = f"{a}+{b}={result}<end>"
        examples.append(example)

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
        "--max-digits",
        type=int,
        default=8,
        help="Maximum number of digits per operand (default: 8)",
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

    print(f"Generating {args.num_examples} addition examples...")
    print(f"Maximum digits per operand: {args.max_digits}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {1 - args.train_ratio - args.val_ratio}"
    )

    # Generate examples
    examples = generate_addition_examples(args.num_examples, args.max_digits, args.seed)
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
