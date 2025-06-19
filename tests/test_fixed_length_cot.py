"""Tests for fixed-length chain-of-thought padding functionality."""

from src.generation import (
    generate_addition_examples,
)
from src.tokenizer import tokenizer


def test_generate_addition_examples_fixed_length_cot():
    """Test data generation with fixed-length CoT enabled."""
    examples = generate_addition_examples(
        num_examples=10,
        max_digits=2,
        seed=42,
        max_operands=3,
        fixed_length_cot=True,
    )

    # Check that examples with reasoning contain <noop> tokens
    cot_examples = [ex for ex in examples if "<think>" in ex]
    if cot_examples:  # Only test if we have CoT examples
        for example in cot_examples:
            assert "<noop>" in example, f"Example missing <noop>: {example}"

            # Verify tokenizer can handle it
            tokens = tokenizer.encode(example)
            assert tokenizer.vocab["<noop>"] in tokens


def test_noop_token_in_vocabulary():
    """Test that <noop> token is properly added to vocabulary."""

    # Check token is in vocabulary
    assert "<noop>" in tokenizer.vocab
    assert tokenizer.vocab["<noop>"] == 15

    # Test encoding/decoding
    text = "3+5=<noop><end>"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_fixed_length_preserves_correctness():
    """Test that fixed-length padding doesn't break arithmetic correctness."""
    examples = generate_addition_examples(
        num_examples=5, max_digits=2, seed=123, fixed_length_cot=True
    )

    for example in examples:
        # Extract the final answer and operands
        if "=" in example and "<end>" in example:
            # Find the first = which separates operands from reasoning/answer
            first_eq = example.find("=")
            operand_part = example[:first_eq]
            rest = example[first_eq + 1 :].replace("<end>", "")

            # For complex nested reasoning, extract just the final number
            # by finding all closing tags and taking what comes after the last one
            final_answer = rest

            # Remove all reasoning content by finding the rightmost position after all closing tags
            closing_tags = ["</think>"]
            last_tag_end = -1

            for tag in closing_tags:
                pos = rest.rfind(tag)
                if pos >= 0:
                    last_tag_end = max(last_tag_end, pos + len(tag))

            if last_tag_end >= 0:
                final_answer = rest[last_tag_end:]

            # Clean up any remaining tags or noop tokens
            final_answer = final_answer.replace("<noop>", "").strip()

            # Parse operands and expected result - only if operand_part looks valid
            if "+" in operand_part and not any(
                tag in operand_part for tag in ["<think>", "<noop>"]
            ):
                try:
                    operands = [int(x.strip()) for x in operand_part.split("+")]
                    expected = sum(operands)

                    # Skip if final_answer is not a pure number (contains remaining markup)
                    if final_answer.isdigit():
                        actual = int(final_answer)
                        assert actual == expected, (
                            f"Wrong answer in: {example}\nExpected: {expected}, Got: {actual}, Final answer extracted: '{final_answer}'"
                        )
                except ValueError:
                    # Skip malformed examples in test
                    continue
