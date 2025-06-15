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
