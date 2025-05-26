"""Snapshot tests for data generation to ensure consistent output."""

from typing import Any

from src.generation import (
    generate_chain_of_thought,
    generate_recursive_chain_of_thought,
)


def test_chain_of_thought_snapshots(snapshot: Any) -> None:
    """Test chain-of-thought generation with snapshots."""
    # Single digit (should be empty)
    assert generate_chain_of_thought(3, 5) == snapshot

    # Two digit without carry
    assert generate_chain_of_thought(12, 34) == snapshot

    # Two digit with carry
    assert generate_chain_of_thought(28, 17) == snapshot

    # Multi-digit with multiple carries
    assert generate_chain_of_thought(658, 189) == snapshot


def test_recursive_chain_of_thought_snapshots(snapshot: Any) -> None:
    """Test recursive chain-of-thought generation with snapshots."""
    # Two operands (should use regular chain-of-thought)
    assert generate_recursive_chain_of_thought([28, 17]) == snapshot

    # Three operands simple
    assert generate_recursive_chain_of_thought([3, 5, 2]) == snapshot

    # Three operands with carries
    assert generate_recursive_chain_of_thought([28, 17, 94]) == snapshot

    # Three operands complex
    assert generate_recursive_chain_of_thought([658, 189, 234]) == snapshot
