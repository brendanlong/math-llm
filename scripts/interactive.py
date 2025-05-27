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
from typing import List, Optional, Tuple

import colorlog
import torch

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.model import (
    ArithmeticModel,
    create_large_model,
    create_medium_model,
    create_small_model,
)
from src.tokenizer import ArithmeticTokenizer


def setup_logging() -> None:
    """Setup colored logging configuration."""
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def load_model(checkpoint_path: Path, model_size: str) -> ArithmeticModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_size: Model size ("small", "medium", or "large")

    Returns:
        Loaded model
    """
    # Create model architecture
    if model_size == "small":
        model = create_small_model()
    elif model_size == "medium":
        model = create_medium_model()
    elif model_size == "large":
        model = create_large_model()
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    # Load checkpoint - handle different formats
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
        model.load_state_dict(state_dict)
    else:
        # Load PyTorch format
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume checkpoint is the state dict directly
            model.load_state_dict(checkpoint)

    return model


class ThinkingNode:
    """Represents a node in the thinking tree structure."""

    def __init__(
        self,
        tag_type: str,
        content: str,
        children: Optional[List["ThinkingNode"]] = None,
    ):
        self.tag_type = tag_type  # 'think_digit', 'think_multi', or 'text'
        self.content = content
        self.children = children or []

    def __repr__(self) -> str:
        return f"ThinkingNode({self.tag_type}, content='{self.content[:20]}...', children={len(self.children)})"


def parse_thinking_recursively(text: str) -> List[ThinkingNode]:
    """Parse thinking tags recursively to build a tree structure.

    Args:
        text: Text containing potentially nested thinking tags

    Returns:
        List of ThinkingNode objects representing the parsed structure
    """
    nodes = []
    i = 0

    while i < len(text):
        # Look for opening thinking tags
        think_match = re.search(r"<(think_digit|think_multi)>", text[i:])

        if not think_match:
            # No more thinking tags, rest is plain text
            if i < len(text):
                content = text[i:].strip()
                if content:
                    nodes.append(ThinkingNode("text", content))
            break

        # Add any text before the thinking tag
        start_pos = i + think_match.start()
        if start_pos > i:
            content = text[i:start_pos].strip()
            if content:
                nodes.append(ThinkingNode("text", content))

        # Find the matching closing tag
        tag_type = think_match.group(1)
        opening_tag = f"<{tag_type}>"
        closing_tag = f"</{tag_type}>"

        # Start after the opening tag
        content_start = start_pos + len(opening_tag)

        # Find matching closing tag, handling nesting
        depth = 1
        pos = content_start
        closing_tag_found = False

        while pos < len(text) and depth > 0:
            # Look for next opening or closing tag of same type
            next_open = text.find(opening_tag, pos)
            next_close = text.find(closing_tag, pos)

            if next_close == -1:
                # No closing tag found
                break

            if next_open != -1 and next_open < next_close:
                # Found nested opening tag
                depth += 1
                pos = next_open + len(opening_tag)
            else:
                # Found closing tag
                depth -= 1
                if depth == 0:
                    # This is our matching closing tag
                    thinking_content = text[content_start:next_close]

                    # Recursively parse the content inside thinking tags
                    children = parse_thinking_recursively(thinking_content)
                    nodes.append(ThinkingNode(tag_type, thinking_content, children))

                    i = next_close + len(closing_tag)
                    closing_tag_found = True
                    break
                else:
                    pos = next_close + len(closing_tag)

        if not closing_tag_found:
            # Malformed - no matching closing tag found
            # Treat everything from the opening tag onward as normal text
            remaining_content = text[start_pos:].strip()
            if remaining_content:
                nodes.append(ThinkingNode("text", remaining_content))
            break

    return nodes


def display_thinking_tree(nodes: List[ThinkingNode], indent: int = 0) -> None:
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
            tag_symbol = "🧮" if node.tag_type == "think_digit" else "🎯"
            print(
                f"{indent_str}{tag_symbol} {node.tag_type.replace('_', ' ').title()}:"
            )

            # Recursively display children with increased indent
            if node.children:
                display_thinking_tree(node.children, indent + 1)
            else:
                # If no children, display the raw content
                lines = node.content.replace("\\n", "\n").split("\n")
                for line in lines:
                    if line.strip():
                        print(f"{indent_str}  {line.strip()}")


def parse_reasoning_output(text: str) -> Tuple[str, List[ThinkingNode], str]:
    """Parse model output to separate reasoning from answer.

    Handles both <think_digit> and <think_multi> reasoning sections recursively.

    Args:
        text: Full model output text

    Returns:
        Tuple of (prefix, thinking_nodes, answer)
    """
    # Find the first thinking tag
    first_think = re.search(r"<(think_digit|think_multi)>", text)

    if not first_think:
        # No thinking tags found
        return text, [], ""

    # Split into prefix, thinking section, and answer
    prefix = text[: first_think.start()]

    # Parse thinking section recursively
    thinking_section = text[first_think.start() :]

    # Find where thinking ends (after the last closing tag)
    last_close_digit = thinking_section.rfind("</think_digit>")
    last_close_multi = thinking_section.rfind("</think_multi>")

    if last_close_digit == -1 and last_close_multi == -1:
        # No closing tags found
        thinking_nodes = parse_thinking_recursively(thinking_section)
        return prefix, thinking_nodes, ""

    # Find the actual end of thinking section
    if last_close_digit > last_close_multi:
        thinking_end = last_close_digit + len("</think_digit>")
    else:
        thinking_end = last_close_multi + len("</think_multi>")

    thinking_content = thinking_section[:thinking_end]
    answer = thinking_section[thinking_end:]

    thinking_nodes = parse_thinking_recursively(thinking_content)

    return prefix, thinking_nodes, answer


def interactive_session(
    model: ArithmeticModel,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.01,
) -> None:
    """Run interactive inference session with chain-of-thought display.

    Args:
        model: Loaded model
        tokenizer: Tokenizer instance
        device: Device to run on
        max_new_tokens: Maximum tokens to generate (default: 512)
        temperature: Sampling temperature
    """
    model.eval()

    print("\n🧮 Math LLM Interactive Inference")
    print("=" * 40)
    print("Enter arithmetic expressions for the model to complete.")
    print("Examples:")
    print("  '3+5=' → model completes with '8<end>'")
    print("  '12+34=' → model shows reasoning and completes with '46<end>'")
    print("  '7+' → model completes with operand and result")
    print("\nType 'quit' or 'exit' to stop.")
    print("=" * 40)

    while True:
        try:
            # Get user input
            user_input = input("\n➤ Enter expression: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! 👋")
                break

            # Skip empty input
            if not user_input:
                continue

            # Validate input contains only valid characters
            valid_chars = set("0123456789+=")
            if not all(c in valid_chars for c in user_input):
                print("⚠️  Error: Input can only contain digits, '+', and '='")
                continue

            # Encode input
            try:
                input_ids = tokenizer.encode(user_input)
                input_tensor = torch.tensor(
                    input_ids, dtype=torch.long, device=device
                ).unsqueeze(0)
            except ValueError as e:
                print(f"⚠️  Error encoding input: {e}")
                continue

            # Generate completion
            print(f"💭 Generating completion for: '{user_input}'")

            with torch.no_grad():
                initial_length = input_tensor.size(1)
                generated_ids = model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    end_token_id=tokenizer.end_token_id,
                )

            # Check if we hit the token limit
            final_length = generated_ids.size(1)
            tokens_generated = final_length - initial_length
            last_token = generated_ids[0, -1].item()

            if (
                tokens_generated >= max_new_tokens
                and last_token != tokenizer.end_token_id
            ):
                print(
                    f"⚠️  Warning: Hit token limit ({max_new_tokens} tokens) - generation may be incomplete"
                )

            # Decode result
            try:
                generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
                print(f"✨ Model output: '{generated_text}'")

                # Extract just the completion part
                if generated_text.startswith(user_input):
                    completion = generated_text[len(user_input) :]

                    # Parse reasoning from completion
                    prefix, thinking_nodes, answer = parse_reasoning_output(completion)

                    if thinking_nodes:
                        # Display thinking tree structure
                        print("🤔 Chain of thought:")
                        display_thinking_tree(thinking_nodes, indent=1)

                        if answer.strip():
                            print(f"✨ Final answer: {prefix}{answer}")
                        print(
                            f"📝 Complete: '{user_input}' → '{user_input}{completion}'"
                        )
                    else:
                        print(
                            f"✨ Model completion: '{user_input}' → '{user_input}{completion}'"
                        )
                else:
                    print(f"✨ Model output: '{generated_text}'")

                # Show token breakdown if helpful (only for very short sequences)
                if len(generated_ids[0]) <= 10:  # Only for very short sequences
                    tokens = tokenizer.tokenize(generated_text)
                    print(f"🔍 Tokens: {' | '.join(tokens)}")

            except Exception as e:
                print(f"⚠️  Error decoding output: {e}")

        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")


def main() -> None:
    """Main interactive function."""
    parser = argparse.ArgumentParser(
        description="Interactive inference with arithmetic transformer model"
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Model size configuration",
    )

    # Generation arguments
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (lower = more deterministic)",
    )

    # System arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logging.info(f"Using device: {device}")

    # Check checkpoint exists
    if not args.checkpoint.exists():
        logging.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Initialize tokenizer
    tokenizer = ArithmeticTokenizer()

    # Load model
    logging.info(f"Loading {args.model_size} model from {args.checkpoint}")
    try:
        model = load_model(args.checkpoint, args.model_size)
        model.to(device)
        logging.info(
            f"Model loaded successfully ({model.count_parameters():,} parameters)"
        )
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Start interactive session
    interactive_session(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
