"""Tests for data generation utilities."""

from src.generation import (
    generate_addition_examples,
    generate_addition_examples_parallel,
    split_data,
)
from src.tokenizer import ArithmeticTokenizer


class TestDataGeneration:
    """Tests for data generation functions."""

    def test_example_validity(self, tokenizer: ArithmeticTokenizer):
        """Test that generated examples are mathematically correct."""
        examples = generate_addition_examples(
            num_examples=10, max_digits=2, seed=42, max_operands=2
        )

        for example in examples:
            # Should be tokenizable
            tokens = tokenizer.encode(example)
            decoded = tokenizer.decode(tokens)
            assert decoded == example

            # Extract the arithmetic expression
            if "<think>" in example:
                # Remove reasoning part for validation
                problem = example.split("=<think>")[0] + "="
                answer_part = example.split("</think>")[-1].replace("<end>", "")
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
        for max_digits in range(1, 9):
            examples = generate_addition_examples(
                num_examples=20, max_digits=max_digits, seed=42, max_operands=5
            )

            all_operands = []
            for example in examples:
                # Extract operands
                problem = example.split("=")[0]
                operands = list(map(int, problem.split("+")))
                # All of the operands should be less than 10^max_digits, i.e. 2-digit numbers should be < 100
                out_of_range = [
                    operand
                    for operand in operands
                    if not (0 <= operand < 10**max_digits)
                ]
                assert not out_of_range, (
                    f"All operands for max_digits={max_digits} should be between 0 and {10**max_digits} but got {out_of_range}"
                )

                all_operands.extend(operands)
            # Some of the operands should be greater than 10^(max_digits-1), i.e. 2-digit numbers should be > 9
            full_digit = [
                operand for operand in all_operands if operand > 10 ** (max_digits - 1)
            ]
            assert full_digit, (
                f"At least one operand for max_digits={max_digits} should be greater than {10 ** (max_digits - 1)} but got {all_operands}"
            )

    def test_max_operands_parameter(self):
        """Test that max_operands parameter is respected."""
        for max_operands in range(2, 9):
            examples = generate_addition_examples(
                num_examples=20, max_digits=5, seed=42, max_operands=max_operands
            )

            all_operands: list[list[int]] = []
            for example in examples:
                # Extract operands
                problem = example.split("=")[0]
                operands = list(map(int, problem.split("+")))
                assert 2 <= len(operands) <= max_operands

                all_operands.append(operands)
            # Some of the examples should have max_operands
            assert [example for example in all_operands if len(example) == max_operands]

    def test_no_duplicates(self):
        # Note that we could get duplicates with enough examples since there's only so many 1-digit operands, but
        # not in 100 examples
        examples = generate_addition_examples(
            num_examples=100, max_digits=3, seed=42, max_operands=5
        )
        assert len(set(examples)) == len(examples)


class TestGenerationWorkers:
    def test_no_duplicates_parallel(self):
        # Note that we could get duplicates with enough examples since there's only so many 1-digit operands, but
        # not in 100 examples
        examples = generate_addition_examples_parallel(
            num_examples=100, max_digits=3, seed=42, max_operands=5, num_workers=10
        )
        assert len(set(examples)) == len(examples)


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

        train1, val1, test1 = split_data(examples, seed=42)
        train2, val2, test2 = split_data(examples, seed=42)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_split_different_seed(self):
        """Test that splits with a different random state are different."""
        examples = [f"example_{i}" for i in range(30)]

        train1, val1, test1 = split_data(examples, seed=1)
        train2, val2, test2 = split_data(examples, seed=2)

        assert train1 != train2
        assert val1 != val2
        assert test1 != test2
