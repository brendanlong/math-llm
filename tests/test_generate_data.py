"""Tests for data generation utilities."""

from src.generation import (
    generate_addition_examples,
    generate_chain_of_thought,
    split_data,
)
from src.tokenizer import ArithmeticTokenizer


class TestChainOfThought:
    """Tests for chain-of-thought generation."""

    def test_single_digit_no_reasoning(self):
        """Test that single-digit addition doesn't generate reasoning."""
        reasoning = generate_chain_of_thought(3, 5)
        assert reasoning == ""

        reasoning = generate_chain_of_thought(9, 9)
        assert reasoning == ""

    def test_multi_digit_has_reasoning(self):
        """Test that multi-digit addition generates reasoning."""
        reasoning = generate_chain_of_thought(12, 34)
        assert reasoning != ""
        assert "<think>" in reasoning
        assert "</think>" in reasoning

    def test_reasoning_format(self):
        """Test the format of generated reasoning."""
        reasoning = generate_chain_of_thought(25, 17)

        # Should be exact format
        expected = "<think>\n5+7=12\n2+1+1=4</think>"
        assert reasoning == expected

    def test_carry_reasoning(self):
        """Test reasoning with carry operations."""
        reasoning = generate_chain_of_thought(8, 9)

        # 8+9 should be single digit, no reasoning
        assert reasoning == ""

        # Multi-digit with carry
        reasoning = generate_chain_of_thought(18, 9)
        expected = "<think>\n8+9=17\n1+0+1=2</think>"
        assert reasoning == expected

    def test_complex_carry(self):
        """Test complex multi-digit carry operations."""
        reasoning = generate_chain_of_thought(658, 189)

        # Should be exact format with explicit carries
        expected = "<think>\n8+9=17\n5+8+1=14\n6+1+1=8</think>"
        assert reasoning == expected

    def test_large_numbers(self):
        """Test reasoning with larger numbers."""
        reasoning = generate_chain_of_thought(999, 999)

        # Should be exact format with carries
        expected = "<think>\n9+9=18\n9+9+1=19\n9+9+1=19</think>"
        assert reasoning == expected


class TestDataGeneration:
    """Tests for data generation functions."""

    def test_generate_basic_examples(self):
        """Test basic example generation."""
        examples = generate_addition_examples(num_examples=5, max_digits=1, seed=42)

        assert len(examples) == 5

        # All should be simple additions without reasoning
        for example in examples:
            assert example.endswith("<end>")
            assert "+" in example
            assert "=" in example
            # Single-digit should not have reasoning
            assert "<think>" not in example

    def test_generate_multi_digit_examples(self):
        """Test multi-digit example generation."""
        examples = generate_addition_examples(num_examples=10, max_digits=2, seed=42)

        assert len(examples) == 10

        # Some should have reasoning (multi-digit cases)
        has_reasoning = any("<think>" in example for example in examples)
        assert has_reasoning

    def test_example_validity(self):
        """Test that generated examples are mathematically correct."""
        examples = generate_addition_examples(num_examples=10, max_digits=3, seed=42)
        tokenizer = ArithmeticTokenizer()

        for example in examples:
            # Should be tokenizable
            tokens = tokenizer.encode(example)
            decoded = tokenizer.decode(tokens)
            assert decoded == example

            # Extract the arithmetic expression
            if "<think>" in example:
                # Remove reasoning part for validation
                problem = example.split("=<think>")[0] + "="
                answer_part = example.split("</think>")[1].replace("<end>", "")
            else:
                problem = example.replace("<end>", "")
                answer_part = problem.split("=")[1]
                problem = problem.split("=")[0] + "="

            # Parse operands and result
            operands = problem.replace("=", "").split("+")
            a, b = int(operands[0]), int(operands[1])
            result = int(answer_part)

            # Check arithmetic
            assert a + b == result

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        examples1 = generate_addition_examples(num_examples=5, max_digits=2, seed=123)
        examples2 = generate_addition_examples(num_examples=5, max_digits=2, seed=123)

        assert examples1 == examples2

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        examples1 = generate_addition_examples(num_examples=10, max_digits=2, seed=123)
        examples2 = generate_addition_examples(num_examples=10, max_digits=2, seed=456)

        # Should be different (very unlikely to be identical by chance)
        assert examples1 != examples2

    def test_max_digits_parameter(self):
        """Test that max_digits parameter is respected."""
        examples = generate_addition_examples(num_examples=20, max_digits=1, seed=42)

        for example in examples:
            # Extract operands
            problem = example.split("=")[0]
            operands = problem.split("+")
            a, b = int(operands[0]), int(operands[1])

            # Should be single digit
            assert 1 <= a <= 9
            assert 1 <= b <= 9

    def test_reasoning_consistency(self):
        """Test that reasoning leads to correct answer."""
        examples = generate_addition_examples(num_examples=10, max_digits=3, seed=42)

        for example in examples:
            if "<think>" not in example:
                continue

            # Parse the example
            parts = example.split("=")
            operands = parts[0].split("+")
            a, b = int(operands[0]), int(operands[1])

            # Extract answer after reasoning
            answer_part = example.split("</think>")[1].replace("<end>", "")
            result = int(answer_part)

            # Verify arithmetic
            assert a + b == result


class TestDataSplitting:
    """Tests for data splitting functionality."""

    def test_split_ratios(self):
        """Test that data is split according to specified ratios."""
        examples = [f"example_{i}" for i in range(100)]

        train, val, test = split_data(examples, train_ratio=0.8, val_ratio=0.1)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_no_duplicates(self):
        """Test that splits contain no duplicate examples."""
        examples = [f"example_{i}" for i in range(50)]

        train, val, test = split_data(examples, train_ratio=0.6, val_ratio=0.2)

        all_split_examples = train + val + test
        assert len(set(all_split_examples)) == len(all_split_examples)  # No duplicates
        assert len(all_split_examples) == len(examples)  # All examples included

    def test_split_different_ratios(self):
        """Test splitting with different ratios."""
        examples = [f"example_{i}" for i in range(200)]

        train, val, test = split_data(examples, train_ratio=0.7, val_ratio=0.15)

        assert len(train) == 140  # 70% of 200
        assert len(val) == 30  # 15% of 200
        assert len(test) == 30  # Remaining 15%

    def test_split_reproducibility(self):
        """Test that splits are reproducible with same random state."""
        examples = [f"example_{i}" for i in range(30)]

        # Reset random state before each split
        import random

        random.seed(42)
        train1, val1, test1 = split_data(examples)

        random.seed(42)
        train2, val2, test2 = split_data(examples)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2
