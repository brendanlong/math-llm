"""Shared type definitions for the math LLM project."""

from typing import TypedDict


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
