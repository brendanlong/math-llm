"""Data generation utilities for arithmetic expressions with chain-of-thought reasoning."""

import random


def calculate_max_operand_digits(*operands: int) -> int:
    """Calculate maximum number of digits in any operand.

    Args:
        *operands: Variable number of operands

    Returns:
        Maximum number of digits across all operands
    """
    return max(len(str(operand)) for operand in operands)


def pad_cot_to_fixed_length(
    reasoning: str, operands: list[int], fixed_length_mode: bool = False
) -> str:
    """Pad chain-of-thought reasoning to fixed length with <noop> tokens.

    Args:
        reasoning: Original reasoning string (e.g., "<think_digit>...</think_digit>")
        operands: List of operands to calculate padding from
        fixed_length_mode: Whether to apply fixed-length padding

    Returns:
        Padded reasoning string with <noop> tokens if fixed_length_mode is True
    """
    if not fixed_length_mode or not reasoning:
        return reasoning

    # Calculate target length: 4 * max(digits in operands) + 2
    max_digits = calculate_max_operand_digits(*operands)
    target_tokens = 4 * max_digits + 2

    # Count existing reasoning tokens (everything between outer think tags)
    # For nested cases, we only pad the outermost level
    if reasoning.startswith("<think_digit>") and reasoning.endswith("</think_digit>"):
        # Extract content between outer tags
        inner_content = reasoning[13:-14]  # Remove <think_digit> and </think_digit>
        existing_tokens = len(inner_content.split()) if inner_content.strip() else 0

        # Add padding before closing tag
        padding_needed = max(
            0, target_tokens - existing_tokens - 2
        )  # -2 for open/close tags
        padding = "<noop>" * padding_needed

        return f"<think_digit>{inner_content}{padding}</think_digit>"

    elif reasoning.startswith("<think_multi>") and reasoning.endswith("</think_multi>"):
        # Extract content between outer tags
        inner_content = reasoning[13:-14]  # Remove <think_multi> and </think_multi>
        existing_tokens = len(inner_content.split()) if inner_content.strip() else 0

        # Add padding before closing tag
        padding_needed = max(
            0, target_tokens - existing_tokens - 2
        )  # -2 for open/close tags
        padding = "<noop>" * padding_needed

        return f"<think_multi>{inner_content}{padding}</think_multi>"

    return reasoning


def generate_chain_of_thought(a: int, b: int) -> str:
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
            # Use recursive thinking for three-number addition
            recursive_reasoning = generate_recursive_chain_of_thought(
                [digit_a, digit_b, carry]
            )
            reasoning.append(recursive_reasoning)
            reasoning.append(str(sum_digits))
        else:
            reasoning.append(f"{sum_digits}")

        # Update carry for next iteration
        carry = sum_digits // 10 if sum_digits >= 10 else 0

    reasoning.append("</think_digit>")
    return "".join(reasoning)


def generate_recursive_chain_of_thought(operands: list[int]) -> str:
    """Generate recursive left-to-right chain-of-thought for multiple operands.

    Args:
        operands: List of operands to add (e.g., [3, 5, 2])

    Returns:
        Chain-of-thought reasoning string showing recursive addition with nested thinking tags
    """
    if len(operands) < 3:
        # For 2 operands, use original chain-of-thought
        return generate_chain_of_thought(operands[0], operands[1])

    # For 3+ operands, show recursive left-to-right addition
    reasoning = ["<think_multi>"]

    # Start with first operand
    current_sum = operands[0]

    for i in range(1, len(operands)):
        next_operand = operands[i]

        # Get the reasoning for this step (with nested thinking tags if multi-digit)
        step_reasoning = generate_chain_of_thought(current_sum, next_operand)
        if step_reasoning:
            # Keep the nested thinking tags for recursive reasoning
            reasoning.append(step_reasoning)

        # Calculate and show the result
        current_sum = current_sum + next_operand
        reasoning.append(str(current_sum))

    reasoning.append("</think_multi>")
    return "".join(reasoning)


def generate_addition_examples(
    num_examples: int,
    max_digits: int = 8,
    seed: int = 42,
    include_three_operands: bool = True,
    fixed_length_cot: bool = False,
) -> list[str]:
    """Generate addition examples with chain-of-thought for multi-digit problems.

    Args:
        num_examples: Number of examples to generate
        max_digits: Maximum number of digits per operand (1-8)
        seed: Random seed for reproducibility
        include_three_operands: Whether to include 3-operand examples
        fixed_length_cot: Whether to pad CoT to fixed length with <noop> tokens

    Returns:
        List of arithmetic expressions in format "a+b=<think_digit>...</think_digit>c<end>"
        or "a+b+c=<think_multi>...</think_multi>d<end>" with recursive reasoning
        If fixed_length_cot is True, pads reasoning to 4 * max(digits in operands) + 2 tokens
    """
    random.seed(seed)
    examples = []
    max_value = 10**max_digits - 1

    # Determine the split between 2-operand and 3-operand examples
    if include_three_operands:
        two_operand_count = int(num_examples * 0.7)  # 70% two operands
        three_operand_count = num_examples - two_operand_count  # 30% three operands
    else:
        two_operand_count = num_examples
        three_operand_count = 0

    # Generate 2-operand examples
    for _ in range(two_operand_count):
        a = random.randint(0, max_value)
        b = random.randint(0, max_value)
        result = a + b

        # Generate chain-of-thought reasoning
        reasoning = generate_chain_of_thought(a, b)

        # Apply fixed-length padding if enabled
        reasoning = pad_cot_to_fixed_length(reasoning, [a, b], fixed_length_cot)

        if reasoning:
            example = f"{a}+{b}={reasoning}{result}<end>"
        else:
            example = f"{a}+{b}={result}<end>"

        examples.append(example)

    # Generate 3-operand examples
    for _ in range(three_operand_count):
        a = random.randint(0, max_value)
        b = random.randint(0, max_value)
        c = random.randint(0, max_value)
        result = a + b + c

        # Generate recursive chain-of-thought reasoning
        reasoning = generate_recursive_chain_of_thought([a, b, c])

        # Apply fixed-length padding if enabled
        reasoning = pad_cot_to_fixed_length(reasoning, [a, b, c], fixed_length_cot)

        if reasoning:
            example = f"{a}+{b}+{c}={reasoning}{result}<end>"
        else:
            example = f"{a}+{b}+{c}={result}<end>"

        examples.append(example)

    # Shuffle to randomize order
    random.shuffle(examples)
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
