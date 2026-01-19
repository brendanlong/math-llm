#!/usr/bin/env python3
"""Interactive inference script for arithmetic transformer model.

This script loads a trained model checkpoint and provides an interactive
interface where users can input arithmetic expressions and see the model's
completions.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from src.config import find_checkpoint_in_output_dir, find_config_in_checkpoint
from src.model import Model
from src.tokenizer import END_TOKEN_ID, tokenizer
from src.utils import get_device, load_model, setup_logging

BEGIN_TOKEN = "<begin>"


def ensure_begin_token(text: str) -> str:
    """Prepend <begin> token to text if not already present.

    Args:
        text: Input text

    Returns:
        Text with <begin> token prepended if not already present
    """
    if not text.startswith(BEGIN_TOKEN):
        return BEGIN_TOKEN + text
    return text


def greedy_generate_with_probs(
    model: Model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
) -> tuple[torch.Tensor, list[float]]:
    """Generate tokens using greedy decoding (argmax) and return probabilities.

    Args:
        model: The model to use for generation
        input_ids: Initial input tokens of shape (1, seq_len)
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        Tuple of:
            - Generated tokens of shape (1, seq_len + num_generated)
            - List of probabilities for each generated token
    """
    model.eval()
    probabilities: list[float] = []

    for _ in range(max_new_tokens):
        # Get predictions for current sequence
        with torch.no_grad():
            outputs = model.forward(input_ids)
            # Extract logits (forward returns dict when labels provided, tensor otherwise)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            # Ensure logits is a tensor (not a tuple from attentions)
            assert isinstance(logits, torch.Tensor)

            # Get logits for last token
            logits = logits[:, -1, :]

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Use argmax for greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Store the probability of the selected token
            selected_token_idx = int(next_token[0].item())
            token_prob = probs[0, selected_token_idx].item()
            probabilities.append(token_prob)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if we hit the end token
            if next_token.item() == END_TOKEN_ID:
                break

    return input_ids, probabilities


def greedy_generate(
    model: Model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
) -> torch.Tensor:
    """Generate tokens using greedy decoding (argmax).

    Args:
        model: The model to use for generation
        input_ids: Initial input tokens of shape (1, seq_len)
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        Generated tokens of shape (1, seq_len + num_generated)
    """
    generated_ids, _ = greedy_generate_with_probs(model, input_ids, max_new_tokens)
    return generated_ids


def get_probability_background(prob: float) -> str:
    """Get ANSI background color based on probability.

    Args:
        prob: Probability value between 0 and 1

    Returns:
        ANSI escape code for background color
    """
    if prob >= 1.0:
        return "\033[48;5;16m"  # Black (100%)
    elif prob >= 0.9:
        return "\033[48;5;234m"  # Very dark grey (90%+)
    elif prob >= 0.8:
        return "\033[48;5;236m"  # Dark grey (80%+)
    elif prob >= 0.7:
        return "\033[48;5;238m"  # Medium-dark grey (70%+)
    elif prob >= 0.6:
        return "\033[48;5;240m"  # Medium grey (60%+)
    elif prob >= 0.5:
        return "\033[48;5;242m"  # Lighter grey (50%+)
    elif prob >= 0.4:
        return "\033[48;5;244m"  # Light grey (40%+)
    elif prob >= 0.3:
        return "\033[48;5;246m"  # Very light grey (30%+)
    elif prob >= 0.2:
        return "\033[48;5;248m"  # Almost white grey (20%+)
    elif prob >= 0.1:
        return "\033[48;5;250m"  # Near white (10%+)
    else:
        return "\033[48;5;252m"  # Very near white (<10%)


class ThinkingNode:
    """Represents a node in the thinking tree structure."""

    def __init__(
        self,
        tag_type: Literal["think", "text"],
        content: str,
        children: Optional[list["ThinkingNode"]] = None,
    ):
        self.tag_type = tag_type
        self.content = content
        self.children = children or []

    def __repr__(self) -> str:
        return f"ThinkingNode({self.tag_type}, content='{self.content[:20]}...', children={len(self.children)})"


def parse_thinking_tags(text: str) -> list[ThinkingNode]:
    """Parse thinking tags using a streaming-like approach with tag open/close events.

    Args:
        text: Text containing potentially nested thinking tags

    Returns:
        List of ThinkingNode objects representing the parsed structure
    """
    nodes: list[ThinkingNode] = []
    stack: list[tuple[str, int, int]] = []  # [(tag_type, start_pos, content_start)]
    i = 0

    while i < len(text):
        # Look for any tag (opening or closing)
        tag_match = re.search(r"<(/?)(think)>", text[i:])

        if not tag_match:
            # No more tags, rest is plain text
            if i < len(text):
                content = text[i:].strip()
                if content:
                    if stack:
                        # We're inside a tag, this will be handled when we close
                        pass
                    else:
                        nodes.append(ThinkingNode("text", content))
            break

        tag_pos = i + tag_match.start()
        is_closing = tag_match.group(1) == "/"
        tag_type = tag_match.group(2)
        tag_text = tag_match.group(0)

        # Add any text before this tag
        if tag_pos > i:
            content = text[i:tag_pos].strip()
            if content:
                if stack:
                    # We're inside a tag, this will be handled when we close
                    pass
                else:
                    nodes.append(ThinkingNode("text", content))

        if is_closing:
            # Closing tag event
            if stack and stack[-1][0] == tag_type:
                # Matching closing tag found
                _parent_tag_type, start_pos, content_start = stack.pop()
                thinking_content = text[content_start:tag_pos]

                # Recursively parse the content inside this tag
                children = parse_thinking_tags(thinking_content)
                # _parent_tag_type is always "think" since that's the only tag we match
                node = ThinkingNode("think", thinking_content, children)

                if stack:
                    # We're still inside another tag, this will be handled later
                    pass
                else:
                    nodes.append(node)
            else:
                # Unmatched closing tag - treat as text
                if not stack:
                    nodes.append(ThinkingNode("text", tag_text))

        else:
            # Opening tag event
            content_start = tag_pos + len(tag_text)
            stack.append((tag_type, tag_pos, content_start))

        i = tag_pos + len(tag_text)

    # Handle any unclosed tags - treat as text
    while stack:
        tag_type, start_pos, content_start = stack.pop()
        remaining_content = text[start_pos:].strip()
        if remaining_content:
            nodes.append(ThinkingNode("text", remaining_content))

    return nodes


def get_top_k_predictions(
    model: Model,
    input_ids: torch.Tensor,
    k: int = 5,
) -> list[tuple[str, float, int]]:
    """Get top-k predictions for next token with probabilities.

    Args:
        model: The model to use for predictions
        input_ids: Current sequence of token IDs
        k: Number of top predictions to return

    Returns:
        List of (token_string, probability, token_id) tuples
    """
    model.eval()

    with torch.no_grad():
        # Get logits for the sequence
        outputs = model.forward(input_ids)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        # Ensure logits is a tensor (not a tuple from attentions)
        assert isinstance(logits, torch.Tensor)

        # Get logits for last position - always use temperature=1.0 for probability display
        logits = logits[:, -1, :]

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Get top-k
        top_probs, top_indices = torch.topk(probs[0], min(k, probs.size(-1)))

        # Convert to list of (token, probability, token_id) tuples
        predictions: list[tuple[str, float, int]] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token_str = tokenizer.decode([idx])
            predictions.append((token_str, prob, idx))

    return predictions


def display_thinking_tree(nodes: list[ThinkingNode], indent: int = 0) -> None:
    """Display thinking tree structure with proper indentation.

    Args:
        nodes: List of ThinkingNode objects to display
        indent: Current indentation level
    """
    indent_str = "  " * indent

    for node in nodes:
        if node.tag_type == "text":
            # Display text content, splitting on newlines for better formatting
            lines = node.content.replace("\\n", "\n").split("\n")
            for line in lines:
                if line.strip():
                    print(f"{indent_str}{line.strip()}")
        else:
            # Display thinking tag
            tag_symbol = "\U0001f9ee"  # abacus emoji
            print(f"{indent_str}{tag_symbol} Thinking:")

            # Recursively display children with increased indent
            if node.children:
                display_thinking_tree(node.children, indent + 1)
            else:
                # If no children, display the raw content
                lines = node.content.replace("\\n", "\n").split("\n")
                for line in lines:
                    if line.strip():
                        print(f"{indent_str}  {line.strip()}")


def interactive_session_with_probabilities(
    model: Model,
    device: torch.device,
    max_new_tokens: int = 512,
) -> None:
    """Run interactive inference session with probability display and manual input.

    Args:
        model: Loaded model
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
    """
    model.eval()

    print("\n\U0001f9ee Math LLM Interactive Inference (Probability Mode)")
    print("=" * 50)
    print("This mode shows top-5 next token predictions with probabilities.")
    print("You can:")
    print("  - Press ENTER to accept the top prediction")
    print("  - Type your own token(s) to continue (e.g., '<think>')")
    print("  - Type 'done' to finish the current expression")
    print("  - Type 'quit' or 'exit' to stop")
    print("=" * 50)

    while True:
        try:
            # Get initial input
            user_input = input("\n\u27a4 Enter initial expression: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! \U0001f44b")
                break

            # Skip empty input
            if not user_input:
                continue

            # Start with user input, ensuring <begin> token is present
            current_text = ensure_begin_token(user_input)
            token_count = 0

            # Interactive generation loop
            while token_count < max_new_tokens:
                # Encode current sequence
                try:
                    input_ids = tokenizer.encode(current_text)
                    input_tensor = torch.tensor(
                        input_ids, dtype=torch.long, device=device
                    ).unsqueeze(0)
                except ValueError as e:
                    print(f"\u26a0\ufe0f  Error encoding input: {e}")
                    break

                # Get top-k predictions with temperature=1.0 for probability display
                predictions = get_top_k_predictions(model, input_tensor, k=5)

                # Display current state
                print(f"\n\U0001f4dd Current: {current_text}")
                print("\n\U0001f3af Top 5 predictions:")
                for i, (token, prob, token_id) in enumerate(predictions):
                    # Format token display
                    display_token = token
                    if token == "\n":
                        display_token = "\\n"
                    print(f"  {i + 1}. '{display_token}' ({prob:.2%}) [id={token_id}]")

                # Get user choice
                choice = input(
                    "\n\u279c Press ENTER for top choice, or type token(s): "
                ).strip()

                if choice.lower() == "done":
                    print(f"\n\u2705 Final: {current_text}")
                    break
                elif choice.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye! \U0001f44b")
                    return
                elif choice == "":
                    # Accept top prediction
                    top_token = predictions[0][0]
                    current_text += top_token
                    token_count += 1
                    print(f"   \u2192 Added: '{top_token}'")

                    # Check if we hit end token
                    if predictions[0][2] == END_TOKEN_ID:
                        print(f"\n\u2705 Complete: {current_text}")
                        break
                else:
                    # User provided custom token(s)
                    try:
                        # Validate that we can encode the new text
                        tokenizer.encode(current_text + choice)
                        current_text += choice
                        # Count how many tokens were added
                        new_ids = tokenizer.encode(choice)
                        token_count += len(new_ids)
                        print(f"   \u2192 Added: '{choice}' ({len(new_ids)} tokens)")
                    except ValueError as e:
                        print(f"\u26a0\ufe0f  Error: Invalid token sequence - {e}")
                        continue

            if token_count >= max_new_tokens:
                print(f"\n\u26a0\ufe0f  Hit token limit ({max_new_tokens} tokens)")
                print(f"\u2705 Final: {current_text}")

        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! \U0001f44b")
            break
        except Exception as e:
            print(f"\u26a0\ufe0f  Unexpected error: {e}")


def interactive_session(
    model: Model,
    device: torch.device,
    max_new_tokens: int = 512,
) -> None:
    """Run interactive inference session with chain-of-thought display.

    Args:
        model: Loaded model
        device: Device to run on
        max_new_tokens: Maximum tokens to generate (default: 512)
    """
    model.eval()

    print("\n\U0001f9ee Math LLM Interactive Inference")
    print("=" * 40)
    print("Enter arithmetic expressions for the model to complete.")
    print("Examples:")
    print("  '3+5=' \u2192 model completes with '8<end>'")
    print("  '12+34=' \u2192 model shows reasoning and completes with '46<end>'")
    print("  '7+' \u2192 model completes with operand and result")
    print("\nToken backgrounds show model confidence:")
    print("  Black = 100% | Dark grey = 90%+ | Light grey = <50%")
    print("\nType 'quit' or 'exit' to stop.")
    print("=" * 40)

    while True:
        try:
            # Get user input
            user_input = input("\n\u27a4 Enter expression: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! \U0001f44b")
                break

            # Skip empty input
            if not user_input:
                continue

            # Ensure <begin> token is present
            input_text = ensure_begin_token(user_input)

            # Encode input
            try:
                input_ids = tokenizer.encode(input_text)
                input_tensor = torch.tensor(
                    input_ids, dtype=torch.long, device=device
                ).unsqueeze(0)
            except ValueError as e:
                print(f"\u26a0\ufe0f  Error encoding input: {e}")
                continue

            # Generate completion
            print(f"\U0001f4ad Generating completion for: {input_text}")

            with torch.no_grad():
                initial_length = input_tensor.size(1)
                # Use greedy decoding (argmax) with probabilities
                generated_ids, probabilities = greedy_generate_with_probs(
                    model,
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                )

            # Check if we hit the token limit
            final_length = generated_ids.size(1)
            tokens_generated = final_length - initial_length
            last_token = generated_ids[0, -1].item()

            if tokens_generated >= max_new_tokens and last_token != END_TOKEN_ID:
                print(
                    f"\u26a0\ufe0f  Warning: Hit token limit ({max_new_tokens} tokens) - generation may be incomplete"
                )

            # Decode result
            try:
                generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())

                # Build output with colored backgrounds for generated tokens
                print("\u2728 Model output: ", end="")

                # Print input without background
                print(input_text, end="")

                # Get the generated tokens (after the input)
                generated_token_ids = generated_ids[0, initial_length:].cpu().tolist()

                # Print each generated token with background based on probability
                for token_id, prob in zip(generated_token_ids, probabilities):
                    token_str = tokenizer.decode([token_id])
                    bg_color = get_probability_background(prob)
                    reset_color = "\033[0m"

                    # Handle special display for newline
                    if token_str == "\n":
                        print(f"{bg_color}\\n{reset_color}", end="")
                    else:
                        print(f"{bg_color}{token_str}{reset_color}", end="")

                print()  # New line after output

                # Parse reasoning from completion
                thinking_nodes = parse_thinking_tags(generated_text)
                # Check if we have any thinking tags (not just text)
                has_thinking_tags = any(
                    node.tag_type != "text" for node in thinking_nodes
                )

                if has_thinking_tags:
                    # Display thinking tree structure
                    print("\U0001f914 Chain of thought:")
                    display_thinking_tree(thinking_nodes, indent=1)

                print(
                    f"\u2705 Answer: {''.join(n.content for n in thinking_nodes if n.tag_type == 'text').removesuffix('<end>')}"
                )

            except Exception as e:
                print(f"\u26a0\ufe0f  Error decoding output: {e}")

        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! \U0001f44b")
            break
        except Exception as e:
            print(f"\u26a0\ufe0f  Unexpected error: {e}")


def main() -> None:
    """Main interactive function."""
    parser = argparse.ArgumentParser(
        description="Interactive inference with arithmetic transformer model"
    )

    # Model arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory containing model checkpoint and config",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to model configuration YAML file (auto-detected from output dir if not specified)",
    )

    # Generation arguments
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )

    # System arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=["normal", "probability"],
        help="Interactive mode: 'normal' for standard generation, 'probability' for step-by-step with probabilities",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Determine device
    device = get_device(args.device)
    logging.info(f"Using device: {device}")

    # Check output directory exists
    if not args.output_dir.exists():
        logging.error(f"Output directory not found: {args.output_dir}")
        sys.exit(1)

    # Find checkpoint file in output directory
    checkpoint_path = find_checkpoint_in_output_dir(args.output_dir)
    if checkpoint_path is None:
        logging.error(
            f"No checkpoint file found in {args.output_dir}. "
            "Expected model.safetensors or pytorch_model.bin."
        )
        sys.exit(1)
    logging.info(f"Found checkpoint: {checkpoint_path}")

    # Find or use provided config
    if args.config is not None:
        config_path = args.config
    else:
        config_path = find_config_in_checkpoint(args.output_dir)
        if config_path is None:
            logging.error(
                "No model_config.yaml found in output directory. "
                "Please specify --config path."
            )
            sys.exit(1)
        logging.info(f"Auto-detected config: {config_path}")

    # Load model
    logging.info(f"Loading model from {checkpoint_path} with config {config_path}")
    try:
        model = load_model(checkpoint_path, config_path)
        model.to(device)
        logging.info(
            f"Model loaded successfully ({model.count_parameters():,} parameters)"
        )
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Start interactive session based on mode
    if args.mode == "probability":
        interactive_session_with_probabilities(
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        interactive_session(
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
