#!/usr/bin/env python3
"""Evaluation script for arithmetic transformer model.

This script evaluates a trained transformer model on arithmetic expressions,
computing exact match accuracy and token-level accuracy.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sized, cast

import colorlog
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data import create_dataloader
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
            "%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s %(message)s",
            datefmt="%H:%M:%S",
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
        # Load safetensors format
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


def compute_exact_match_accuracy(
    model: ArithmeticModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    max_new_tokens: int = 20,
) -> float:
    """Compute exact match accuracy by generating complete sequences.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation data
        tokenizer: Tokenizer instance
        device: Device to run evaluation on
        max_new_tokens: Maximum tokens to generate

    Returns:
        Exact match accuracy (0.0 to 1.0)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing exact match accuracy"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids.size(0)

            for i in range(batch_size):
                # Get the target sequence (remove padding)
                target_ids = labels[i]
                target_ids = target_ids[target_ids != -100]  # Remove masked tokens
                target_text = tokenizer.decode(target_ids.cpu().tolist())

                # Find the "=" in the input to get the prompt
                input_text = tokenizer.decode(input_ids[i].cpu().tolist())
                if "=" in input_text:
                    prompt = input_text.split("=")[0] + "="
                    prompt_ids = torch.tensor(
                        tokenizer.encode(prompt), dtype=torch.long, device=device
                    ).unsqueeze(0)

                    # Generate completion
                    generated_ids = model.generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.1,  # Low temperature for deterministic output
                        end_token_id=tokenizer.end_token_id,
                    )

                    # Extract only the generated part
                    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())

                    # Check if generated text matches target
                    if generated_text.strip() == target_text.strip():
                        correct += 1

                total += 1

    return correct / total if total > 0 else 0.0


def compute_token_accuracy(
    model: ArithmeticModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    """Compute token-level accuracy on teacher-forced predictions.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Token-level accuracy (0.0 to 1.0)
    """
    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing token accuracy"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Get model predictions
            outputs = model(input_ids, labels=labels)
            logits = outputs["logits"]

            # Get predicted tokens
            predictions = torch.argmax(logits, dim=-1)

            # Shift predictions and labels for next-token prediction
            shift_predictions = predictions[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten and mask padding tokens
            shift_predictions = shift_predictions.view(-1)
            shift_labels = shift_labels.view(-1)

            mask = shift_labels != -100
            correct = (shift_predictions == shift_labels) & mask

            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    return total_correct / total_tokens if total_tokens > 0 else 0.0


def evaluate_on_dataset(
    model: ArithmeticModel,
    data_path: Path,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
) -> dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        data_path: Path to dataset JSON file
        tokenizer: Tokenizer instance
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length

    Returns:
        Dictionary with evaluation metrics
    """
    # Create dataloader
    dataloader = create_dataloader(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
        num_workers=0,  # Use single process for deterministic results
    )

    # Compute metrics
    logging.info(f"Evaluating on {len(cast(Sized, dataloader.dataset))} examples")

    token_accuracy = compute_token_accuracy(model, dataloader, device)
    exact_match_accuracy = compute_exact_match_accuracy(
        model, dataloader, tokenizer, device
    )

    return {
        "token_accuracy": token_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "num_examples": len(cast(Sized, dataloader.dataset)),
    }


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate arithmetic transformer model"
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Model size configuration",
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to evaluation data JSON file (if not provided, uses test.json from data dir)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train/val/test JSON files",
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to save evaluation results JSON",
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

    # Initialize tokenizer
    tokenizer = ArithmeticTokenizer()

    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.model_size)
    model.to(device)

    # Determine data path
    if args.data_path:
        data_path: Path = args.data_path
    else:
        data_path = args.data_dir / "test.json"

    # Evaluate model
    logging.info(f"Evaluating on {data_path}")
    results = evaluate_on_dataset(
        model=model,
        data_path=data_path,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Print results
    logging.info("Evaluation Results:")
    logging.info(f"  Token Accuracy: {results['token_accuracy']:.4f}")
    logging.info(f"  Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
    logging.info(f"  Number of Examples: {results['num_examples']}")

    # Save results if requested
    if args.output_file:
        with args.output_file.open("w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
