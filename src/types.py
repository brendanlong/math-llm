"""Shared type definitions for the math LLM project."""

from typing import NotRequired, TypedDict


class DatasetMetadata(TypedDict):
    """Metadata for dataset generation."""

    max_digits: int
    max_operands: int
    seed: int
    num_workers: int
    include_chain_of_thought: bool
    fixed_length_cot: bool
    num_examples: int
    split_ratios: dict[str, float]
    generation_timestamp: str
    generation_version: str
    longest_example_length: int
    longest_example: str


class DatasetDict(TypedDict):
    """Type definition for the complete dataset."""

    examples: list[str]
    metadata: NotRequired[DatasetMetadata]
