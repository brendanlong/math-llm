"""Data generation utilities for arithmetic expressions with chain-of-thought reasoning."""

import multiprocessing
import random

from .tokenizer import ArithmeticTokenizer


def reverse_operand(operand: int) -> str:
    """Reverse the digits of an operand.

    Args:
        operand: Operand to reverse

    Returns:
        Reversed operand (as a string)
    """
    str_operand = str(operand)
    return str_operand[::-1]


def generate_chain_of_thought(operands: list[int]) -> str:
    """Generate recursive left-to-right chain-of-thought for multiple operands.

    Args:
        operands: List of operands to add (e.g., [3, 5, 2])

    Returns:
        Chain-of-thought reasoning string showing recursive addition with nested thinking tags
    """
    # Show recursive left-to-right addition
    reasoning = ["<think>"]

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

    reasoning.append("</think>")
    return "".join(reasoning)


def calculate_max_operand_digits(operands: list[int]) -> int:
    return max(len(str(operand)) for operand in operands)


def generate_addition_examples(
    num_examples: int,
    max_digits: int = 3,
    seed: int = 42,
    max_operands: int = 3,
    fixed_length_cot: bool = False,
    include_chain_of_thought: bool = True,
) -> list[str]:
    """Generate addition examples with chain-of-thought for multi-digit problems.

    Args:
        num_examples: Number of examples to generate
        max_digits: Maximum number of digits per operand (1-8)
        seed: Random seed for reproducibility
        max_operands: The maximum number of operands to add
        fixed_length_cot: Whether to pad CoT to fixed length with <noop> tokens

    Returns:
        List of arithmetic expressions in format "a+b+c=<think>...</think>d<end>" with recursive reasoning
        If fixed_length_cot is True, pads reasoning to 10 * max(digits in operands) * max(number of operands)
    """
    assert max_digits >= 1
    assert max_operands >= 2

    random.seed(seed)
    examples = []
    tokenizer = ArithmeticTokenizer()
    cot_length = 20 * max_digits * max_operands

    for _ in range(num_examples):
        # First select number of operands
        num_operands = random.randint(2, max_operands)

        # Generate operands with uniform distribution of digit counts
        operands = []
        for _ in range(num_operands):
            # Select number of digits uniformly from 1 to max_digits
            num_digits = random.randint(1, max_digits)

            if num_digits == 1:
                # For 1 digit: 0-9
                operand = random.randint(0, 9)
            else:
                # For n digits: 10^(n-1) to 10^n - 1
                min_value = 10 ** (num_digits - 1)
                max_value = 10**num_digits - 1
                operand = random.randint(min_value, max_value)

            operands.append(operand)
        result = sum(operands)

        # Generate chain-of-thought reasoning
        reasoning = (
            generate_chain_of_thought(operands) if include_chain_of_thought else ""
        )

        # Apply fixed-length padding if enabled
        if fixed_length_cot:
            max_digits = calculate_max_operand_digits(operands)
            reasoning_length = len(tokenizer.encode(reasoning))
            pad_length = cot_length - reasoning_length
            if pad_length < 0:
                # Error out if our 4x assumption is wrong
                raise ValueError(
                    f"Generated CoT is longer than fixed-length CoT: {reasoning_length} > {cot_length}: {'+'.join(map(str, operands))}={reasoning}"
                )
            elif pad_length > 0:
                reasoning += "<noop>" * pad_length

        example = f"{'+'.join(map(str, operands))}={reasoning}{result}<end>"
        examples.append(example)

    return examples


def _generate_examples_chunk(
    num_examples: int,
    max_digits: int,
    seed: int,
    max_operands: int,
    fixed_length_cot: bool,
    include_chain_of_thought: bool,
) -> list[str]:
    """Generate a chunk of examples in a worker process.

    Args:
        num_examples: Number of examples to generate in this chunk
        max_digits: Maximum number of digits per operand
        seed: Random seed for this worker
        max_operands: Maximum number of operands
        fixed_length_cot: Whether to use fixed-length chain-of-thought

    Returns:
        List of generated examples
    """
    return generate_addition_examples(
        num_examples=num_examples,
        max_digits=max_digits,
        seed=seed,
        max_operands=max_operands,
        fixed_length_cot=fixed_length_cot,
        include_chain_of_thought=include_chain_of_thought,
    )


def generate_addition_examples_parallel(
    num_examples: int,
    max_digits: int = 3,
    seed: int = 42,
    max_operands: int = 3,
    fixed_length_cot: bool = False,
    include_chain_of_thought: bool = True,
    num_workers: int | None = None,
) -> list[str]:
    """Generate addition examples using multiple processes.

    Args:
        num_examples: Total number of examples to generate
        max_digits: Maximum number of digits per operand
        seed: Base random seed
        max_operands: Maximum number of operands
        fixed_length_cot: Whether to use fixed-length chain-of-thought
        num_workers: Number of worker processes (default: CPU count)

    Returns:
        List of generated examples
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # If we have few examples or only one worker, use serial generation
    if num_examples < num_workers * 10 or num_workers == 1:
        return generate_addition_examples(
            num_examples=num_examples,
            max_digits=max_digits,
            seed=seed,
            max_operands=max_operands,
            fixed_length_cot=fixed_length_cot,
            include_chain_of_thought=include_chain_of_thought,
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
                )
            )

    # Generate examples in parallel
    with multiprocessing.Pool(processes=len(work_chunks)) as pool:
        results = pool.starmap(_generate_examples_chunk, work_chunks)

    # Flatten results
    examples = []
    for chunk_examples in results:
        examples.extend(chunk_examples)

    return examples


def split_data(
    examples: list[str], train_ratio: float = 0.8, val_ratio: float = 0.1
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
    random.shuffle(shuffled)

    train_examples = shuffled[:train_size]
    val_examples = shuffled[train_size : train_size + val_size]
    test_examples = shuffled[train_size + val_size :]

    return train_examples, val_examples, test_examples
