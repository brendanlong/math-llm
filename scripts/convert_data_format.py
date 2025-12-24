#!/usr/bin/env python3
"""Convert between JSON and JSONL data formats.

JSON format: {"examples": ["example1", "example2", ...], "metadata": {...}}
JSONL format: one example per line (plain text, no JSON encoding)
"""

import argparse
import json
import sys
from pathlib import Path


def json_to_jsonl(json_path: Path, jsonl_path: Path) -> int:
    """Convert JSON format to JSONL format.

    Args:
        json_path: Path to input JSON file
        jsonl_path: Path to output JSONL file

    Returns:
        Number of examples converted
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    examples = data["examples"]

    with open(jsonl_path, "w") as f:
        for example in examples:
            f.write(example + "\n")

    # Also save metadata if present
    if "metadata" in data:
        metadata_path = jsonl_path.parent / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(data["metadata"], f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    return len(examples)


def jsonl_to_json(jsonl_path: Path, json_path: Path) -> int:
    """Convert JSONL format to JSON format.

    Args:
        jsonl_path: Path to input JSONL file
        json_path: Path to output JSON file

    Returns:
        Number of examples converted
    """
    examples: list[str] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(line)

    data: dict[str, object] = {"examples": examples}

    # Try to load metadata if it exists
    metadata_path = jsonl_path.parent / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            data["metadata"] = json.load(f)
        print(f"Included metadata from {metadata_path}")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert between JSON and JSONL data formats"
    )
    parser.add_argument("input", type=Path, help="Input file path")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output file path (default: same name with different extension)",
    )
    parser.add_argument(
        "--to-jsonl",
        action="store_true",
        help="Convert JSON to JSONL (auto-detected from extension if not specified)",
    )
    parser.add_argument(
        "--to-json",
        action="store_true",
        help="Convert JSONL to JSON (auto-detected from extension if not specified)",
    )

    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Auto-detect direction from input extension
    if not args.to_jsonl and not args.to_json:
        if input_path.suffix == ".json":
            args.to_jsonl = True
        elif input_path.suffix == ".jsonl":
            args.to_json = True
        else:
            print("Error: Cannot auto-detect format. Use --to-jsonl or --to-json")
            sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.to_jsonl:
        output_path = input_path.with_suffix(".jsonl")
    else:
        output_path = input_path.with_suffix(".json")

    # Convert
    if args.to_jsonl:
        count = json_to_jsonl(input_path, output_path)
        print(f"Converted {count} examples from {input_path} to {output_path}")
    else:
        count = jsonl_to_json(input_path, output_path)
        print(f"Converted {count} examples from {input_path} to {output_path}")


if __name__ == "__main__":
    main()
