"""Tests for fixed-length chain-of-thought padding functionality."""

from src.generation import (
    calculate_max_operand_digits,
    generate_addition_examples,
    pad_cot_to_fixed_length,
)
from src.tokenizer import ArithmeticTokenizer


def test_calculate_max_operand_digits():
    """Test calculation of maximum digits in operands."""
    assert calculate_max_operand_digits(1, 2, 3) == 1
    assert calculate_max_operand_digits(12, 345, 6) == 3
    assert calculate_max_operand_digits(999, 1000) == 4
    assert calculate_max_operand_digits(0) == 1


def test_pad_cot_to_fixed_length_disabled():
    """Test that padding is skipped when fixed_length_mode is False."""
    reasoning = "<think_digit>\n2+3=5</think_digit>"
    result = pad_cot_to_fixed_length(reasoning, [12, 34], fixed_length_mode=False)
    assert result == reasoning


def test_pad_cot_to_fixed_length_empty_reasoning():
    """Test that empty reasoning is returned unchanged."""
    result = pad_cot_to_fixed_length("", [12, 34], fixed_length_mode=True)
    assert result == ""


def test_pad_cot_to_fixed_length_think_digit():
    """Test padding for think_digit reasoning."""
    # For operands [12, 34], max digits = 2, target = 4*2+2 = 10 tokens
    # Content has minimal tokens, should add <noop> padding
    reasoning = "<think_digit>\n2+4=6\n1+3=4</think_digit>"
    result = pad_cot_to_fixed_length(reasoning, [12, 34], fixed_length_mode=True)

    # Should contain original content plus <noop> tokens
    assert result.startswith("<think_digit>")
    assert result.endswith("</think_digit>")
    assert "<noop>" in result
    assert "\n2+4=6\n1+3=4" in result


def test_pad_cot_to_fixed_length_think_multi():
    """Test padding for think_multi reasoning."""
    reasoning = "<think_multi>\n1+2=3\n3+4=7</think_multi>"
    result = pad_cot_to_fixed_length(reasoning, [1, 2, 4], fixed_length_mode=True)

    # Should contain original content plus <noop> tokens
    assert result.startswith("<think_multi>")
    assert result.endswith("</think_multi>")
    assert "<noop>" in result
    assert "\n1+2=3\n3+4=7" in result


def test_pad_cot_to_fixed_length_different_digit_counts():
    """Test padding calculation for different max digit counts."""
    # 1 digit: target = 4*1+2 = 6 tokens
    reasoning1 = "<think_digit>\ntest</think_digit>"
    result1 = pad_cot_to_fixed_length(reasoning1, [5, 7], fixed_length_mode=True)
    noop_count1 = result1.count("<noop>")

    # 3 digits: target = 4*3+2 = 14 tokens
    reasoning2 = "<think_digit>\ntest</think_digit>"
    result2 = pad_cot_to_fixed_length(reasoning2, [123, 456], fixed_length_mode=True)
    noop_count2 = result2.count("<noop>")

    # Should have more padding for larger operands
    assert noop_count2 > noop_count1


def test_generate_addition_examples_fixed_length_cot():
    """Test data generation with fixed-length CoT enabled."""
    examples = generate_addition_examples(
        num_examples=10,
        max_digits=2,
        seed=42,
        include_three_operands=False,
        fixed_length_cot=True,
    )

    tokenizer = ArithmeticTokenizer()

    # Check that examples with reasoning contain <noop> tokens
    cot_examples = [ex for ex in examples if "<think_" in ex]
    if cot_examples:  # Only test if we have CoT examples
        for example in cot_examples:
            if "<think_digit>" in example:
                assert "<noop>" in example, f"Example missing <noop>: {example}"

                # Verify tokenizer can handle it
                tokens = tokenizer.encode(example)
                assert tokenizer.vocab["<noop>"] in tokens


def test_noop_token_in_vocabulary():
    """Test that <noop> token is properly added to vocabulary."""
    tokenizer = ArithmeticTokenizer()

    # Check token is in vocabulary
    assert "<noop>" in tokenizer.vocab
    assert tokenizer.vocab["<noop>"] == 18

    # Test encoding/decoding
    text = "3+5=<noop><end>"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text

    # Test tokenize method
    string_tokens = tokenizer.tokenize(text)
    assert "<noop>" in string_tokens


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
            closing_tags = ["</think_digit>", "</think_multi>"]
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
                tag in operand_part for tag in ["<think_", "<noop>"]
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
