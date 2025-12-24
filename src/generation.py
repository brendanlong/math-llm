"""Data generation utilities for arithmetic expressions with chain-of-thought reasoning."""

import multiprocessing
import random

from .tokenizer import tokenizer


def reverse_operand(operand: int) -> str:
    """Reverse the digits of an operand.

    Args:
        operand: Operand to reverse

    Returns:
        Reversed operand (as a string)
    """
    str_operand = str(operand)
    return str_operand[::-1]


def calculate_max_result_digits(max_digits: int, max_operands: int) -> int:
    """Calculate the maximum number of digits the result can have.

    Args:
        max_digits: Maximum digits per operand
        max_operands: Maximum number of operands

    Returns:
        Maximum number of digits in any possible result
    """
    max_sum = max_operands * (10**max_digits - 1)
    return len(str(max_sum))


def format_number(n: int, width: int, reversed_format: bool = False) -> str:
    """Format a number with optional zero-padding and reversal.

    Args:
        n: The number to format
        width: Target width (0 for no padding)
        reversed_format: If True, reverse digits and pad on right; else pad on left

    Returns:
        Formatted string representation
    """
    s = str(n)
    if width == 0:
        if reversed_format:
            return s[::-1]
        return s

    if reversed_format:
        # Reverse first, then pad on right
        return s[::-1].ljust(width, "0")
    else:
        # Pad on left
        return s.zfill(width)


def generate_chain_of_thought(operands: list[int]) -> str:
    """Generate recursive left-to-right chain-of-thought for multiple operands.

    Args:
        operands: List of operands to add (e.g., [3, 5, 2])

    Returns:
        Chain-of-thought reasoning string showing recursive addition with nested thinking tags
    """
    # Show recursive left-to-right addition
    reasoning: list[str] = []

    # Add each group of two operands recursively
    remaining_operands = operands.copy()
    while len(remaining_operands) > 1:
        # Show the addition step
        # Reverse the operands to make this easier for a left-to-right LLM
        reasoning.append("+".join(map(reverse_operand, remaining_operands)))
        reasoning.append("=")

        # Pop the first two operands
        operand_a = remaining_operands.pop(0)
        operand_b = remaining_operands.pop(0)

        result = operand_a + operand_b
        remaining_operands.insert(0, result)
    reasoning.append(reverse_operand(remaining_operands[0]))

    return "".join(reasoning)


def generate_addition_examples(
    num_examples: int,
    max_digits: int = 3,
    seed: int = 42,
    max_operands: int = 3,
    fixed_length_cot: bool = False,
    include_chain_of_thought: bool = True,
    reversed_format: bool = False,
    zero_pad: bool = False,
) -> list[str]:
    """Generate addition examples with chain-of-thought for multi-digit problems.

    Args:
        num_examples: Number of examples to generate
        max_digits: Maximum number of digits per operand (1-8)
        seed: Random seed for reproducibility
        max_operands: The maximum number of operands to add
        fixed_length_cot: Whether to pad CoT to fixed length with <noop> tokens.
        include_chain_of_thought: Whether to include chain-of-thought reasoning.
        reversed_format: Whether to reverse digit order in operands and result (no CoT).
        zero_pad: Whether to zero-pad all numbers to fixed width. Operands are padded
            to max_digits, results are padded to the maximum possible result width.
            In reversed_format, zeros appear on the right; otherwise on the left.

    Returns:
        List of arithmetic expressions in format "a+b+c=<think>...</think>d<end>" with recursive reasoning,
        or "a+b+c=d<end>" with reversed digits if reversed_format is True.
    """
    assert max_digits >= 1
    assert max_operands >= 2

    # Calculate padding widths if zero_pad is enabled
    operand_width = max_digits if zero_pad else 0
    result_width = (
        calculate_max_result_digits(max_digits, max_operands) if zero_pad else 0
    )

    cot_length = (
        len(
            tokenizer.encode(
                generate_chain_of_thought([10**max_digits - 1] * max_operands)
            )
        )
        if fixed_length_cot
        else 0
    )

    r = random.Random(seed)
    examples = []

    for _ in range(num_examples):
        # First select number of operands
        num_operands = r.randint(2, max_operands)

        # Generate operands with uniform distribution of digit counts
        operands = []
        for _ in range(num_operands):
            # Select number of digits uniformly from 1 to max_digits
            num_digits = r.randint(1, max_digits)

            if num_digits == 1:
                # For 1 digit: 0-9
                operand = r.randint(0, 9)
            else:
                # For n digits: 10^(n-1) to 10^n - 1
                min_value = 10 ** (num_digits - 1)
                max_value = 10**num_digits - 1
                operand = r.randint(min_value, max_value)

            operands.append(operand)
        result = sum(operands)

        # Generate example based on format
        if reversed_format:
            # Reversed format: no CoT, digits reversed in operands and result
            # With zero_pad, zeros appear on the right (after reversal)
            formatted_operands = [
                format_number(op, operand_width, reversed_format=True)
                for op in operands
            ]
            formatted_result = format_number(result, result_width, reversed_format=True)
            example = f"<begin>{'+'.join(formatted_operands)}={formatted_result}<end>"
        else:
            # Standard format with optional CoT
            if include_chain_of_thought:
                reasoning = generate_chain_of_thought(operands)

                # Apply fixed-length padding if enabled
                if fixed_length_cot:
                    reasoning_length = len(tokenizer.encode(reasoning))
                    pad_length = cot_length - reasoning_length
                    if pad_length < 0:
                        # Error out if our 4x assumption is wrong
                        raise ValueError(
                            f"Generated CoT is longer than fixed-length CoT: {reasoning_length} > {cot_length}: {'+'.join(map(str, operands))}={reasoning}"
                        )
                    elif pad_length > 0:
                        reasoning += "<noop>" * pad_length
                reasoning = f"<think>{reasoning}</think>"
            else:
                reasoning = ""

            # Format operands and result with optional zero-padding
            formatted_operands = [
                format_number(op, operand_width, reversed_format=False)
                for op in operands
            ]
            formatted_result = format_number(
                result, result_width, reversed_format=False
            )
            example = f"<begin>{'+'.join(formatted_operands)}={reasoning}{formatted_result}<end>"
        examples.append(example)

    return examples


def generate_addition_examples_parallel(
    num_examples: int,
    max_digits: int = 3,
    seed: int = 42,
    max_operands: int = 3,
    fixed_length_cot: bool = False,
    include_chain_of_thought: bool = True,
    reversed_format: bool = False,
    zero_pad: bool = False,
    num_workers: int | None = None,
) -> list[str]:
    """Generate addition examples using multiple processes.

    Args:
        num_examples: Total number of examples to generate
        max_digits: Maximum number of digits per operand
        seed: Base random seed
        max_operands: Maximum number of operands
        fixed_length_cot: Whether to use fixed-length chain-of-thought
        include_chain_of_thought: Whether to include chain-of-thought reasoning
        reversed_format: Whether to reverse digit order (no CoT)
        zero_pad: Whether to zero-pad all numbers to fixed width
        num_workers: Number of worker processes (default: CPU count)

    Returns:
        List of generated examples
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # If we have only one worker, use serial generation
    if num_workers == 1:
        return generate_addition_examples(
            num_examples=num_examples,
            max_digits=max_digits,
            seed=seed,
            max_operands=max_operands,
            fixed_length_cot=fixed_length_cot,
            include_chain_of_thought=include_chain_of_thought,
            reversed_format=reversed_format,
            zero_pad=zero_pad,
        )

    # Calculate examples per worker
    examples_per_worker = num_examples // num_workers
    remaining_examples = num_examples % num_workers

    # Create work chunks with different seeds for each worker
    work_chunks = []
    for i in range(num_workers):
        chunk_size = examples_per_worker + (1 if i < remaining_examples else 0)
        if chunk_size > 0:
            work_chunks.append(
                (
                    chunk_size,
                    max_digits,
                    seed + i,  # Different seed for each worker
                    max_operands,
                    fixed_length_cot,
                    include_chain_of_thought,
                    reversed_format,
                    zero_pad,
                )
            )

    # Generate examples in parallel
    with multiprocessing.Pool(processes=len(work_chunks)) as pool:
        results = pool.starmap(generate_addition_examples, work_chunks)

    # Flatten results
    examples = []
    for chunk_examples in results:
        examples.extend(chunk_examples)

    return examples


def split_data(
    examples: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split data into train/validation/test sets.

    Args:
        examples: List of examples to split
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set (test gets remainder)

    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    total = len(examples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # Shuffle examples before splitting
    shuffled = examples.copy()
    r = random.Random(seed)
    r.shuffle(shuffled)

    train_examples = shuffled[:train_size]
    val_examples = shuffled[train_size : train_size + val_size]
    test_examples = shuffled[train_size + val_size :]

    return train_examples, val_examples, test_examples
