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

from src.config import find_config_in_checkpoint, load_config
from src.data import create_dataloader
from src.model import Model, create_model_from_config
from src.tokenizer import VOCAB, tokenizer


def remove_thinking_sections(text: str) -> str:
    """Remove thinking sections from text.

    Handles <think>...</think> sections.
    Uses first thinking tag and last corresponding closing tag to handle nested tokens.

    Args:
        text: Input text that may contain thinking sections

    Returns:
        Text with thinking sections removed
    """
    # Remove thinking sections
    while True:
        first_think = text.find("<think>")
        if first_think == -1:
            break
        last_think = text.rfind("</think>")
        if last_think == -1 or last_think < first_think:
            break
        text = text[:first_think] + text[last_think + 8 :]

    return text


def create_thinking_mask(
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Create a mask that excludes tokens inside thinking sections.

    Handles <think>...</think> sections.
    Uses first thinking tag and last corresponding closing tag to handle nested tokens.

    Args:
        token_ids: Tensor of token IDs (any shape)
        tokenizer: Tokenizer instance to get special token IDs

    Returns:
        Boolean tensor with same shape, True for tokens to include in accuracy
    """
    # Get token IDs for thinking
    think_id = tokenizer.vocab["<think>"]
    end_think_id = tokenizer.vocab["</think>"]

    mask = torch.ones_like(token_ids, dtype=torch.bool)

    # Handle batched input
    if token_ids.dim() == 2:
        for batch_idx in range(token_ids.size(0)):
            seq = token_ids[batch_idx]

            # Find all thinking sections and mask them
            for pos in range(len(seq)):
                # Check for thinking start
                if seq[pos] == think_id:
                    # Find matching close tag
                    for end_pos in range(pos + 1, len(seq)):
                        if seq[end_pos] == end_think_id:
                            mask[batch_idx, pos : end_pos + 1] = False
                            break
    else:
        # Handle 1D tensor
        for pos in range(len(token_ids)):
            # Check for thinking start
            if token_ids[pos] == think_id:
                # Find matching close tag
                for end_pos in range(pos + 1, len(token_ids)):
                    if token_ids[end_pos] == end_think_id:
                        mask[pos : end_pos + 1] = False
                        break

    return mask


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


def load_model(
    checkpoint_path: Path,
    config_path: Path,
) -> Model:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model configuration YAML file

    Returns:
        Loaded model
    """
    config = load_config(config_path)
    model = create_model_from_config(config)

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
    model: Model,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    max_new_tokens: int = 512,
) -> float:
    """Compute exact match accuracy by generating complete sequences.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation data
        tokenizer: Tokenizer instance
        device: Device to run evaluation on
        max_new_tokens: Maximum tokens to generate (default: 512)

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
                        end_token_id=VOCAB["<end>"],
                    )

                    # Extract only the generated part
                    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())

                    # Remove thinking sections before comparison
                    generated_clean = remove_thinking_sections(generated_text.strip())
                    target_clean = remove_thinking_sections(target_text.strip())

                    # Check if generated text matches target
                    if generated_clean == target_clean:
                        correct += 1

                total += 1

    return correct / total if total > 0 else 0.0


def compute_token_accuracy(
    model: Model,
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

            # Remove thinking sections from both predictions and labels for proper alignment
            batch_size, _ = shift_predictions.shape

            for i in range(batch_size):
                pred_seq = shift_predictions[i]
                label_seq = shift_labels[i]

                # Create thinking mask for this sequence
                thinking_mask = create_thinking_mask(label_seq)

                # Only include non-padding, non-thinking tokens
                valid_mask = (label_seq != -100) & thinking_mask

                if valid_mask.sum() > 0:
                    # Extract tokens outside thinking sections
                    pred_filtered = pred_seq[valid_mask]
                    label_filtered = label_seq[valid_mask]

                    # Compare aligned tokens
                    correct = pred_filtered == label_filtered
                    total_correct += correct.sum().item()
                    total_tokens += len(correct)

    return total_correct / total_tokens if total_tokens > 0 else 0.0


def evaluate_on_dataset(
    model: Model,
    data_path: Path,
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
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
        num_workers=0,  # Use single process for deterministic results
    )

    # Compute metrics
    logging.info(f"Evaluating on {len(cast(Sized, dataloader.dataset))} examples")

    token_accuracy = compute_token_accuracy(model, dataloader, device)
    exact_match_accuracy = compute_exact_match_accuracy(model, dataloader, device)

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
        "--config",
        type=Path,
        default=None,
        help="Path to model configuration YAML file (auto-detected from checkpoint dir if not specified)",
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

    # Find or use provided config
    if args.config is not None:
        config_path = args.config
    else:
        config_path = find_config_in_checkpoint(args.checkpoint)
        if config_path is None:
            logging.error(
                "No model_config.yaml found in checkpoint directory. "
                "Please specify --config path."
            )
            sys.exit(1)
        logging.info(f"Auto-detected config: {config_path}")

    # Load model
    logging.info(f"Loading model from {args.checkpoint} with config {config_path}")
    model = load_model(args.checkpoint, config_path)
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
