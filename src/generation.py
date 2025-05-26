"""Data generation utilities for arithmetic expressions with chain-of-thought reasoning."""

import random


def generate_chain_of_thought(a: int, b: int) -> str:
    """Generate chain-of-thought reasoning for addition.

    Args:
        a: First operand
        b: Second operand

    Returns:
        Chain-of-thought reasoning string
    """
    str_a = str(a)
    str_b = str(b)

    # For single-digit addition, no reasoning needed
    if len(str_a) == 1 and len(str_b) == 1:
        return ""

    # Pad numbers to same length for column addition
    max_len = max(len(str_a), len(str_b))
    str_a = str_a.zfill(max_len)
    str_b = str_b.zfill(max_len)

    reasoning = ["<think>"]
    carry = 0

    # Work from right to left
    for i in range(max_len - 1, -1, -1):
        digit_a = int(str_a[i])
        digit_b = int(str_b[i])

        # Add digits plus any carry
        sum_digits = digit_a + digit_b + carry

        # Show the addition step with explicit carry when present
        if carry > 0:
            reasoning.append(f"\n{digit_a}+{digit_b}+{carry}={sum_digits}")
        else:
            reasoning.append(f"\n{digit_a}+{digit_b}={sum_digits}")

        # Update carry for next iteration
        carry = sum_digits // 10 if sum_digits >= 10 else 0

    reasoning.append("</think>")
    return "".join(reasoning)


def generate_addition_examples(
    num_examples: int, max_digits: int = 8, seed: int = 42
) -> list[str]:
    """Generate addition examples with chain-of-thought for multi-digit problems.

    Args:
        num_examples: Number of examples to generate
        max_digits: Maximum number of digits per operand (1-8)
        seed: Random seed for reproducibility

    Returns:
        List of arithmetic expressions in format "a+b=<think>...</think>c<end>"
        Chain-of-thought reasoning explicitly shows carries (e.g., "5+8+1=14")
    """
    random.seed(seed)
    examples = []

    for _ in range(num_examples):
        # Generate each operand independently with random digit count
        num_digits_a = random.randint(1, max_digits)
        num_digits_b = random.randint(1, max_digits)

        # Generate operands (0 to 10^digits - 1)
        a = random.randint(10 ** (num_digits_a - 1), 10**num_digits_a - 1)
        b = random.randint(10 ** (num_digits_b - 1), 10**num_digits_b - 1)

        result = a + b

        # Generate chain-of-thought reasoning
        reasoning = generate_chain_of_thought(a, b)

        if reasoning:
            example = f"{a}+{b}={reasoning}{result}<end>"
        else:
            example = f"{a}+{b}={result}<end>"

        examples.append(example)

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
