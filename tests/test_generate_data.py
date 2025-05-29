"""Tests for data generation utilities."""

from src.generation import (
    generate_addition_examples,
    split_data,
)
from src.tokenizer import ArithmeticTokenizer


class TestDataGeneration:
    """Tests for data generation functions."""

    def test_generate_basic_examples(self):
        """Test basic example generation."""
        examples = generate_addition_examples(
            num_examples=5, max_digits=1, seed=42, max_operands=2
        )

        assert len(examples) == 5

        # All should be simple additions without reasoning (when forced to 2-operand only)
        for example in examples:
            assert example.endswith("<end>")
            assert "+" in example
            assert "=" in example
            # Count plus signs to ensure it's 2-operand
            assert example.count("+") == 1

    def test_generate_multi_digit_examples(self):
        """Test multi-digit example generation."""
        examples = generate_addition_examples(num_examples=10, max_digits=2, seed=42)

        assert len(examples) == 10

        # Some should have reasoning (multi-digit cases)
        has_reasoning = any(
            "<think_digit>" in example or "<think_multi>" in example
            for example in examples
        )
        assert has_reasoning

    def test_example_validity(self):
        """Test that generated examples are mathematically correct."""
        examples = generate_addition_examples(
            num_examples=10, max_digits=2, seed=42, max_operands=2
        )
        tokenizer = ArithmeticTokenizer()

        for example in examples:
            # Should be tokenizable
            tokens = tokenizer.encode(example)
            decoded = tokenizer.decode(tokens)
            assert decoded == example

            # Extract the arithmetic expression
            if "<think_digit>" in example:
                # Remove reasoning part for validation
                problem = example.split("=<think_digit>")[0] + "="
                answer_part = example.split("</think_digit>")[-1].replace("<end>", "")
            elif "<think_multi>" in example:
                # Remove reasoning part for validation
                problem = example.split("=<think_multi>")[0] + "="
                answer_part = example.split("</think_multi>")[-1].replace("<end>", "")
            else:
                problem = example.replace("<end>", "")
                answer_part = problem.split("=")[1]
                problem = problem.split("=")[0] + "="

            # Parse operands and result - handle 2-operand only
            operands = problem.replace("=", "").split("+")
            if len(operands) == 2:
                a, b = int(operands[0]), int(operands[1])
                expected_result = a + b
            else:
                continue  # Skip 3-operand for this test

            result = int(answer_part)

            # Check arithmetic
            assert expected_result == result

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
            assert 0 <= a <= 9
            assert 0 <= b <= 9

    def test_reasoning_consistency(self):
        """Test that reasoning leads to correct answer."""
        examples = generate_addition_examples(
            num_examples=10, max_digits=2, seed=42, max_operands=2
        )

        for example in examples:
            if "<think_digit>" not in example and "<think_multi>" not in example:
                continue

            # Parse the example - handle 2-operand only
            parts = example.split("=")
            operands = parts[0].split("+")
            if len(operands) != 2:
                continue  # Skip 3-operand examples

            a, b = int(operands[0]), int(operands[1])

            # Extract answer after reasoning
            if "</think_digit>" in example:
                answer_part = example.split("</think_digit>")[-1].replace("<end>", "")
            else:
                answer_part = example.split("</think_multi>")[-1].replace("<end>", "")
            result = int(answer_part)

            # Verify arithmetic
            assert a + b == result

    def test_three_operand_examples(self):
        """Test generation of 3-operand examples."""
        examples = generate_addition_examples(
            num_examples=20, max_digits=1, seed=42, max_operands=3
        )

        # Should have both 2-operand and 3-operand examples
        two_operand_count = 0
        three_operand_count = 0

        for example in examples:
            # Extract just the problem part (before the = and any reasoning)
            problem = example.split("=")[0]
            plus_count = problem.count("+")

            if plus_count == 1:
                two_operand_count += 1
            elif plus_count == 2:
                three_operand_count += 1

        assert two_operand_count > 0
        assert three_operand_count > 0
        assert two_operand_count + three_operand_count == len(examples)

    def test_three_operand_validity(self):
        """Test that 3-operand examples are mathematically correct."""
        examples = generate_addition_examples(
            num_examples=20, max_digits=1, seed=42, max_operands=3
        )

        for example in examples:
            # Extract just the problem part to count operands
            problem = example.split("=")[0]
            if problem.count("+") != 2:  # Skip 2-operand examples
                continue

            # Extract the arithmetic expression - get the problem part before any thinking
            problem_part = example.split("=")[0]

            # Extract the final answer after all thinking
            # Find the last closing thinking tag and get everything after it
            last_close_multi = example.rfind("</think_multi>")
            last_close_digit = example.rfind("</think_digit>")

            if last_close_multi > last_close_digit:
                answer_part = example[
                    last_close_multi + len("</think_multi>") :
                ].replace("<end>", "")
            elif last_close_digit >= 0:
                answer_part = example[
                    last_close_digit + len("</think_digit>") :
                ].replace("<end>", "")
            else:
                answer_part = example.split("=")[1].replace("<end>", "")

            # Parse operands and result
            operands = problem_part.split("+")
            a, b, c = int(operands[0]), int(operands[1]), int(operands[2])
            result = int(answer_part)

            # Check arithmetic
            assert a + b + c == result


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
