"""Data generation utilities for arithmetic expressions with chain-of-thought reasoning."""

import random

from .tokenizer import ArithmeticTokenizer


def generate_two_operand_chain_of_thought(a: int, b: int) -> str:
    """Generate chain-of-thought reasoning for addition.

    Args:
        a: First operand
        b: Second operand

    Returns:
        Chain-of-thought reasoning string with multi-digit thinking tokens
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

    reasoning = ["<think_digit>"]
    carry = 0

    # Work from right to left
    for i in range(max_len - 1, -1, -1):
        digit_a = int(str_a[i])
        digit_b = int(str_b[i])

        # Add digits plus any carry
        sum_digits = digit_a + digit_b + carry

        # Show the addition step with recursive thinking when there's a carry (3 numbers)
        if carry > 0:
            reasoning.append(f"\n{digit_a}+{digit_b}+{carry}=")
            # Use recursive thinking for three-number addition
            recursive_reasoning = generate_chain_of_thought([digit_a, digit_b, carry])
            reasoning.append(recursive_reasoning)
            reasoning.append(str(sum_digits))
        else:
            reasoning.append(f"\n{digit_a}+{digit_b}={sum_digits}")

        # Update carry for next iteration
        carry = sum_digits // 10 if sum_digits >= 10 else 0

    reasoning.append("</think_digit>")
    return "".join(reasoning)


def generate_chain_of_thought(operands: list[int]) -> str:
    """Generate recursive left-to-right chain-of-thought for multiple operands.

    Args:
        operands: List of operands to add (e.g., [3, 5, 2])

    Returns:
        Chain-of-thought reasoning string showing recursive addition with nested thinking tags
    """
    if len(operands) < 3:
        # For 2 operands, use original chain-of-thought
        return generate_two_operand_chain_of_thought(operands[0], operands[1])

    # For 3+ operands, show recursive left-to-right addition
    reasoning = ["<think_multi>"]

    # Start with first operand
    current_sum = operands[0]

    for i in range(1, len(operands)):
        next_operand = operands[i]

        # Show the addition step
        reasoning.append(f"\n{current_sum}+{next_operand}=")

        # Get the reasoning for this step (with nested thinking tags if multi-digit)
        step_reasoning = generate_two_operand_chain_of_thought(
            current_sum, next_operand
        )
        if step_reasoning:
            # Keep the nested thinking tags for recursive reasoning
            reasoning.append(step_reasoning)

        # Calculate and show the result
        current_sum = current_sum + next_operand
        reasoning.append(str(current_sum))

    reasoning.append("</think_multi>")
    return "".join(reasoning)


def calculate_max_operand_digits(operands: list[int]) -> int:
    return max(len(str(operand)) for operand in operands)


def generate_addition_examples(
    num_examples: int,
    max_digits: int = 3,
    seed: int = 42,
    max_operands: int = 3,
    fixed_length_cot: bool = False,
) -> list[str]:
    """Generate addition examples with chain-of-thought for multi-digit problems.

    Args:
        num_examples: Number of examples to generate
        max_digits: Maximum number of digits per operand (1-8)
        seed: Random seed for reproducibility
        max_operands: The maximum number of operands to add
        fixed_length_cot: Whether to pad CoT to fixed length with <noop> tokens

    Returns:
        List of arithmetic expressions in format "a+b=<think_digit>...</think_digit>c<end>"
        or "a+b+c=<think_multi>...</think_multi>d<end>" with recursive reasoning
        If fixed_length_cot is True, pads reasoning to 10 * max(digits in operands) * max(number of operands)
    """
    assert max_digits >= 1
    assert max_operands >= 2

    random.seed(seed)
    examples = []
    max_value = 10**max_digits - 1
    tokenizer = ArithmeticTokenizer()
    cot_length = 20 * max_digits * max_operands

    for _ in range(num_examples):
        operands = [
            random.randint(0, max_value) for _ in range(random.randint(2, max_operands))
        ]
        result = sum(operands)

        # Generate chain-of-thought reasoning
        reasoning = generate_chain_of_thought(operands)

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
