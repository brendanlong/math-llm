#!/usr/bin/env python3
"""Training script for arithmetic transformer model.

This script trains a small transformer model on arithmetic expressions using
HuggingFace Transformers with W&B logging and automatic checkpointing.
"""

import os

# Disable tokenizers parallelism to avoid fork warning with DataLoader workers.
# Our simple 17-token vocabulary doesn't benefit from parallel tokenization.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Sized, cast

import colorlog
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, schedule
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed
from transformers.training_args import TrainingArguments

import wandb

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config, save_config
from src.data import ArithmeticDataset, load_splits
from src.model import create_model_from_config
from src.tokenizer import VOCAB_SIZE
from src.training import (
    compute_metrics,
    data_collator,
    setup_training_optimizations,
)


def setup_logging() -> None:
    """Setup colored logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Setup colored console handler
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

    # Setup file handler (no colors for file output)
    file_handler = logging.FileHandler("logs/training.log")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train arithmetic transformer model")

    # Model arguments
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to model configuration YAML file (e.g., config/standard-small.yaml)",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing train/val/test JSON files",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: longest example length from training data)",
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=1000,
        help="Evaluation every N steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Log every N steps",
    )

    # System arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint in output directory",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run one epoch with torch.profiler and save results",
    )

    args = parser.parse_args()

    # Setup
    setup_logging()
    set_random_seeds(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("Starting training script")
    logger.info(f"Arguments: {args}")

    # Initialize tokenizer
    logger.info("Initializing tokenizer")

    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    train_loader, val_loader, test_loader = load_splits(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    # Get the actual max_length from the dataset (in case it was set from metadata)
    actual_max_length = cast(ArithmeticDataset, train_loader.dataset).max_length
    if args.max_length is None:
        args.max_length = actual_max_length
        logger.info(f"Using max_length={actual_max_length} from dataset metadata")
    else:
        assert actual_max_length == args.max_length

    # Load model configuration
    logger.info(f"Loading model configuration from {args.config}")
    model_config = load_config(args.config)

    # Get config name for logging
    config_name = (
        args.config.stem
    )  # e.g., "standard-small" from "config/standard-small.yaml"

    # Initialize W&B
    if not args.no_wandb:
        wandb_name = (
            f"arithmetic-{config_name}-{args.batch_size}batch-{args.learning_rate}lr"
        )
        wandb.init(
            project="math-llm",
            config={
                "config_file": str(args.config),
                "architecture": model_config.architecture,
                "d_model": model_config.d_model,
                "n_layers": model_config.n_layers,
                "n_heads": model_config.n_heads,
                "d_ff": model_config.d_ff,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_length": args.max_length,
                "seed": args.seed,
            },
            name=wandb_name,
        )

    logger.info(f"Train samples: {len(cast(Sized, train_loader.dataset))}")
    logger.info(f"Validation samples: {len(cast(Sized, val_loader.dataset))}")
    logger.info(f"Test samples: {len(cast(Sized, test_loader.dataset))}")

    # Create model
    arch_desc = (
        "Universal Transformer"
        if model_config.architecture == "universal"
        else "standard transformer"
    )
    logger.info(f"Creating {config_name} {arch_desc}")
    model = create_model_from_config(model_config)
    logger.info(f"Model parameters: {model.count_parameters():,}")
    if model_config.architecture == "universal":
        logger.info(
            f"Universal Transformer: {model.n_layers} layers × {model.n_loops} loops = {model.sequential_depth} sequential depth"
        )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # Enable TensorFloat32 for better performance on modern GPUs
    setup_training_optimizations()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=not args.resume,
        # Training hyperparameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        # Optimization
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        # Evaluation and logging
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_accumulation_steps=32,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only 3 most recent checkpoints
        # W&B integration
        report_to="wandb" if not args.no_wandb else "none",
        run_name=f"arithmetic-{config_name}",
        # Other settings
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_token_accuracy",
        greater_is_better=True,
        torch_compile=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Save model configuration to output directory
    model_config_path = Path(args.output_dir) / "model_config.yaml"
    save_config(model_config, model_config_path)
    logger.info(f"Saved model configuration to {model_config_path}")

    # Save training configuration
    training_config = {
        "config_file": str(args.config),
        "architecture": model_config.architecture,
        "model_parameters": model.count_parameters(),
        "vocab_size": VOCAB_SIZE,
        "max_length": args.max_length,
        "training_args": training_args.to_dict(),
    }
    if model_config.architecture == "universal":
        training_config["n_layers"] = model.n_layers
        training_config["n_loops"] = model.n_loops
        training_config["sequential_depth"] = model.sequential_depth

    training_config_path = Path(args.output_dir) / "training_config.json"
    with open(training_config_path, "w") as f:
        json.dump(training_config, f, indent=2)

    logger.info(f"Saved training configuration to {training_config_path}")

    # Profile training if requested
    if args.profile:
        logger.info("Running profiling for one epoch")
        profile_dir = Path(args.output_dir) / "profiles"
        profile_dir.mkdir(exist_ok=True)

        # Setup profiler with simpler schedule for better TensorBoard compatibility
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        # Run one epoch with profiling
        model.train()
        profiler.start()

        # Store losses without forcing sync until the end
        losses = []

        step = 0
        for batch in train_loader:
            if step >= 12:  # Profile 12 steps to match our schedule (2+2+6+2)
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs["loss"]

            loss.backward()

            # Store loss tensor without calling .item() (avoids sync)
            losses.append(loss.detach())

            profiler.step()
            step += 1

            if step % 2 == 0:
                logger.info(f"Profiling step {step}/12")

        # Log losses after profiling is complete (single sync)
        logger.info("Loss values:")
        for i, loss_tensor in enumerate(losses):
            logger.info(f"  Step {i + 1}: {loss_tensor.item():.4f}")

        profiler.stop()

        # Print profiler summary to console
        logger.info("Profiler Summary:")
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        logger.info(f"Profiling completed. Results saved to {profile_dir}")
        logger.info("TensorBoard traces available in the profiles directory")
        logger.info(
            "View detailed traces with: tensorboard --logdir=checkpoints/profiles"
        )
        logger.info(
            "Or install tensorboard-plugin-profile: pip install tensorboard-plugin-profile"
        )
        return

    # Start training
    logger.info("Starting training")
    trainer.train(args.resume)

    # Save final model
    logger.info("Saving final model")
    trainer.save_model(args.output_dir)

    # Final evaluation on test set
    logger.info("Running final evaluation on test set")
    test_results = trainer.evaluate(eval_dataset=test_loader.dataset)
    logger.info(f"Test results: {test_results}")

    # Save test results
    results_path = Path(args.output_dir) / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Saved test results to {results_path}")

    # Finish W&B run
    if not args.no_wandb:
        wandb.finish()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
