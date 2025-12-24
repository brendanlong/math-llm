"""Tests for data generation utilities."""

from src.generation import (
    calculate_max_result_digits,
    format_number,
    generate_addition_examples,
    generate_addition_examples_parallel,
    split_data,
)
from src.tokenizer import tokenizer


class TestDataGeneration:
    """Tests for data generation functions."""

    def test_example_validity(
        self,
    ):
        """Test that generated examples are mathematically correct."""
        examples = generate_addition_examples(
            num_examples=10, max_digits=2, seed=42, max_operands=2
        )

        for example in examples:
            # Should be tokenizable
            tokens = tokenizer.encode(example)
            decoded = tokenizer.decode(tokens)
            assert decoded == example
            print(example)
            example = example.replace("<begin>", "").replace("<end>", "")
            problem, answer_part = example.split("=", 1)

            # Parse operands and result
            operands = problem.split("+")
            expected_result = sum(map(int, operands))

            final_answer = (
                answer_part.split("</think>", 1)[1]
                if "<think>" in answer_part
                else answer_part
            )
            result = int(final_answer)

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
                problem = example.replace("<begin>", "").split("=")[0]
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
                problem = example.replace("<begin>", "").split("=")[0]
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


class TestZeroPadding:
    """Tests for zero-padding functionality."""

    def test_calculate_max_result_digits(self):
        """Test calculation of maximum result digits."""
        # 2 operands with 1 digit max: max sum is 9+9=18, so 2 digits
        assert calculate_max_result_digits(1, 2) == 2

        # 3 operands with 1 digit max: max sum is 9+9+9=27, so 2 digits
        assert calculate_max_result_digits(1, 3) == 2

        # 2 operands with 2 digits max: max sum is 99+99=198, so 3 digits
        assert calculate_max_result_digits(2, 2) == 3

        # 3 operands with 3 digits max: max sum is 999*3=2997, so 4 digits
        assert calculate_max_result_digits(3, 3) == 4

        # 5 operands with 5 digits max: max sum is 99999*5=499995, so 6 digits
        assert calculate_max_result_digits(5, 5) == 6

    def test_format_number_no_padding(self):
        """Test format_number with no padding (width=0)."""
        assert format_number(123, 0, reversed_format=False) == "123"
        assert format_number(123, 0, reversed_format=True) == "321"
        assert format_number(5, 0, reversed_format=False) == "5"
        assert format_number(5, 0, reversed_format=True) == "5"

    def test_format_number_normal_padding(self):
        """Test format_number with left zero-padding (normal format)."""
        assert format_number(5, 3, reversed_format=False) == "005"
        assert format_number(12, 3, reversed_format=False) == "012"
        assert format_number(123, 3, reversed_format=False) == "123"
        assert format_number(1234, 3, reversed_format=False) == "1234"  # No truncation

    def test_format_number_reversed_padding(self):
        """Test format_number with right zero-padding (reversed format)."""
        # 5 reversed is "5", padded to 3 on right is "500"
        assert format_number(5, 3, reversed_format=True) == "500"
        # 12 reversed is "21", padded to 3 on right is "210"
        assert format_number(12, 3, reversed_format=True) == "210"
        # 123 reversed is "321", already 3 chars
        assert format_number(123, 3, reversed_format=True) == "321"
        # 1234 reversed is "4321", 4 chars (no truncation)
        assert format_number(1234, 3, reversed_format=True) == "4321"

    def test_zero_pad_reversed_format_operands(self):
        """Test that reversed format with zero_pad pads operands correctly."""
        examples = generate_addition_examples(
            num_examples=20,
            max_digits=3,
            max_operands=2,
            seed=42,
            reversed_format=True,
            zero_pad=True,
        )

        for example in examples:
            # Extract operands from "<begin>XXX+YYY=ZZZ<end>"
            content = example.replace("<begin>", "").replace("<end>", "")
            problem, _ = content.split("=")
            operands = problem.split("+")

            # All operands should be exactly 3 characters (max_digits)
            for op in operands:
                assert len(op) == 3, f"Operand {op} should be 3 chars in {example}"

    def test_zero_pad_reversed_format_result(self):
        """Test that reversed format with zero_pad pads results correctly."""
        examples = generate_addition_examples(
            num_examples=20,
            max_digits=3,
            max_operands=3,
            seed=42,
            reversed_format=True,
            zero_pad=True,
        )

        # For max_digits=3, max_operands=3, max result is 999*3=2997, so 4 digits
        expected_result_width = 4

        for example in examples:
            content = example.replace("<begin>", "").replace("<end>", "")
            _, result = content.split("=")

            assert len(result) == expected_result_width, (
                f"Result {result} should be {expected_result_width} chars in {example}"
            )

    def test_zero_pad_normal_format_operands(self):
        """Test that normal format with zero_pad pads operands correctly."""
        examples = generate_addition_examples(
            num_examples=20,
            max_digits=3,
            max_operands=2,
            seed=42,
            reversed_format=False,
            zero_pad=True,
            include_chain_of_thought=False,
        )

        for example in examples:
            content = example.replace("<begin>", "").replace("<end>", "")
            problem, _ = content.split("=")
            operands = problem.split("+")

            # All operands should be exactly 3 characters (max_digits)
            for op in operands:
                assert len(op) == 3, f"Operand {op} should be 3 chars in {example}"
                # Zeros should be on the left (leading zeros)
                assert op == op.zfill(3), f"Operand {op} should have leading zeros"

    def test_zero_pad_normal_format_result(self):
        """Test that normal format with zero_pad pads results correctly."""
        examples = generate_addition_examples(
            num_examples=20,
            max_digits=3,
            max_operands=3,
            seed=42,
            reversed_format=False,
            zero_pad=True,
            include_chain_of_thought=False,
        )

        expected_result_width = 4  # max_digits=3, max_operands=3 -> 4 digits

        for example in examples:
            content = example.replace("<begin>", "").replace("<end>", "")
            _, result = content.split("=")

            assert len(result) == expected_result_width, (
                f"Result {result} should be {expected_result_width} chars in {example}"
            )
            # Zeros should be on the left
            assert result == result.zfill(expected_result_width)

    def test_zero_pad_arithmetic_correctness(self):
        """Test that zero-padded examples are still arithmetically correct."""
        for reversed_format in [True, False]:
            examples = generate_addition_examples(
                num_examples=20,
                max_digits=3,
                max_operands=3,
                seed=42,
                reversed_format=reversed_format,
                zero_pad=True,
                include_chain_of_thought=False,
            )

            for example in examples:
                content = example.replace("<begin>", "").replace("<end>", "")
                problem, result_str = content.split("=")
                operand_strs = problem.split("+")

                if reversed_format:
                    # Reverse back to get original values
                    operands = [int(op[::-1]) for op in operand_strs]
                    result = int(result_str[::-1])
                else:
                    operands = [int(op) for op in operand_strs]
                    result = int(result_str)

                expected = sum(operands)
                assert result == expected, (
                    f"Expected {expected}, got {result} in {example}"
                )

    def test_zero_pad_with_cot(self):
        """Test that zero-padding works with chain-of-thought (operands/result only)."""
        examples = generate_addition_examples(
            num_examples=10,
            max_digits=2,
            max_operands=2,
            seed=42,
            reversed_format=False,
            zero_pad=True,
            include_chain_of_thought=True,
        )

        for example in examples:
            content = example.replace("<begin>", "").replace("<end>", "")
            problem, answer_part = content.split("=", 1)
            operand_strs = problem.split("+")

            # Operands should be zero-padded to 2 digits
            for op in operand_strs:
                assert len(op) == 2, f"Operand {op} should be 2 chars in {example}"

            # Final answer after </think> should be zero-padded
            final_answer = answer_part.split("</think>")[1]
            # max result for 2 operands with 2 digits: 99+99=198 -> 3 digits
            assert len(final_answer) == 3, (
                f"Result {final_answer} should be 3 chars in {example}"
            )
