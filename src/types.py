"""Shared type definitions for the math LLM project."""

from typing import TypedDict


class DatasetDict(TypedDict):
    """Type definition for the complete dataset."""

    examples: list[str]
