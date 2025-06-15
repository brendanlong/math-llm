#!/usr/bin/env python3
"""Count duplicate examples in training data."""

import json
from collections import Counter
from pathlib import Path


def count_duplicates(data_path: Path) -> None:
    """Count and display duplicate examples in the dataset."""
    print(f"Loading {data_path}...")
    with open(data_path, "r") as f:
        data = json.load(f)

    # Extract the examples list
    examples = data["examples"]

    # Count occurrences
    counter = Counter(examples)

    # Find duplicates (count > 1)
    duplicates: dict[str, int] = {
        text: count for text, count in counter.items() if count > 1
    }

    print(f"\nTotal examples: {len(examples):,}")
    print(f"Unique examples: {len(counter):,}")
    print(f"Number of duplicate examples: {len(examples) - len(counter):,}")
    print(f"Number of unique texts that have duplicates: {len(duplicates):,}")

    if duplicates:
        print("\nTop 20 most duplicated examples:")
        for text, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[
            :20
        ]:
            print(f"  '{text}' appears {count} times")


if __name__ == "__main__":
    data_path = Path("data/train.json")
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        exit(1)

    count_duplicates(data_path)
