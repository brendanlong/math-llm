"""Snapshot tests for data generation to ensure consistent output."""

from typing import Any

from src.generation import generate_addition_examples


def test_chain_of_thought_snapshots(snapshot: Any) -> None:
    """Test test generation with chain of thought."""
    # Two operands
    assert (
        generate_addition_examples(
            num_examples=2,
            max_digits=1,
            seed=12,
            max_operands=2,
            include_chain_of_thought=True,
        )
        == snapshot
    )

    # Three operands simple
    assert (
        generate_addition_examples(
            num_examples=2,
            max_digits=1,
            seed=13,
            max_operands=3,
            include_chain_of_thought=True,
        )
        == snapshot
    )

    # Three operands with more digits
    assert (
        generate_addition_examples(
            num_examples=2,
            max_digits=3,
            seed=12,
            max_operands=3,
            include_chain_of_thought=True,
        )
        == snapshot
    )

    # Large number of operands and digits
    assert (
        generate_addition_examples(
            num_examples=2,
            max_digits=5,
            seed=12,
            max_operands=5,
            include_chain_of_thought=True,
        )
        == snapshot
    )


def test_no_chain_of_thought_snapshots(snapshot: Any) -> None:
    """Test test generation with no chain of thought."""
    # Two operands
    assert (
        generate_addition_examples(
            num_examples=2,
            max_digits=1,
            seed=12,
            max_operands=2,
            include_chain_of_thought=False,
        )
        == snapshot
    )

    # Three operands simple
    assert (
        generate_addition_examples(
            num_examples=2,
            max_digits=1,
            seed=13,
            max_operands=3,
            include_chain_of_thought=False,
        )
        == snapshot
    )

    # Three operands with more digits
    assert (
        generate_addition_examples(
            num_examples=2,
            max_digits=3,
            seed=12,
            max_operands=3,
            include_chain_of_thought=False,
        )
        == snapshot
    )

    # Large number of operands and digits
    assert (
        generate_addition_examples(
            num_examples=2,
            max_digits=5,
            seed=12,
            max_operands=5,
            include_chain_of_thought=False,
        )
        == snapshot
    )


def test_zero_pad_reversed_format_snapshots(snapshot: Any) -> None:
    """Test generation with zero padding in reversed format."""
    # Small operands
    assert (
        generate_addition_examples(
            num_examples=3,
            max_digits=2,
            seed=42,
            max_operands=2,
            reversed_format=True,
            zero_pad=True,
        )
        == snapshot
    )

    # Larger operands
    assert (
        generate_addition_examples(
            num_examples=3,
            max_digits=3,
            seed=42,
            max_operands=3,
            reversed_format=True,
            zero_pad=True,
        )
        == snapshot
    )


def test_zero_pad_normal_format_snapshots(snapshot: Any) -> None:
    """Test generation with zero padding in normal format (no CoT)."""
    # Small operands
    assert (
        generate_addition_examples(
            num_examples=3,
            max_digits=2,
            seed=42,
            max_operands=2,
            include_chain_of_thought=False,
            zero_pad=True,
        )
        == snapshot
    )

    # Larger operands
    assert (
        generate_addition_examples(
            num_examples=3,
            max_digits=3,
            seed=42,
            max_operands=3,
            include_chain_of_thought=False,
            zero_pad=True,
        )
        == snapshot
    )


def test_zero_pad_with_cot_snapshots(snapshot: Any) -> None:
    """Test generation with zero padding and chain-of-thought."""
    assert (
        generate_addition_examples(
            num_examples=3,
            max_digits=2,
            seed=42,
            max_operands=2,
            include_chain_of_thought=True,
            zero_pad=True,
        )
        == snapshot
    )
