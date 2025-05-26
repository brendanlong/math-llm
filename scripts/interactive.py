#!/usr/bin/env python3
"""Interactive inference script for arithmetic transformer model.

This script loads a trained model checkpoint and provides an interactive
interface where users can input arithmetic expressions and see the model's
completions.
"""

import argparse
import logging
import sys
from pathlib import Path

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


def interactive_session(
    model: ArithmeticModel,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    max_new_tokens: int = 20,
    temperature: float = 0.1,
) -> None:
    """Run interactive inference session.

    Args:
        model: Loaded model
        tokenizer: Tokenizer instance
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    model.eval()

    print("\n🧮 Math LLM Interactive Inference")
    print("=" * 40)
    print("Enter arithmetic expressions for the model to complete.")
    print("Examples:")
    print("  '3+5=' → model completes with '8<end>'")
    print("  '12+34=' → model completes with '46<end>'")
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
                generated_ids = model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    end_token_id=tokenizer.end_token_id,
                )

            # Decode result
            try:
                generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())

                # Extract just the completion part
                if generated_text.startswith(user_input):
                    completion = generated_text[len(user_input) :]
                    print(
                        f"✨ Model completion: '{user_input}' → '{user_input}{completion}'"
                    )
                else:
                    print(f"✨ Model output: '{generated_text}'")

                # Show token breakdown if helpful
                if len(generated_ids[0]) <= 15:  # Only for short sequences
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
        default=20,
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
