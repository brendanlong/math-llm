#!/usr/bin/env python3
"""Training script for arithmetic transformer model.

This script trains a small transformer model on arithmetic expressions using
HuggingFace Transformers with W&B logging and automatic checkpointing.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional, Sized, cast

import colorlog
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, schedule
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalPrediction, set_seed
from transformers.training_args import TrainingArguments

import wandb

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data import ArithmeticDataset, load_splits
from src.model import ArithmeticModel, create_model
from src.tokenizer import VOCAB, ArithmeticTokenizer


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


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute completion-only evaluation metrics to match training objective.

    Args:
        eval_pred: Predictions and labels from trainer

    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred

    # Get predicted tokens (argmax)
    predictions = np.argmax(predictions, axis=-1)

    # Shift labels left by 1 to align with predictions
    # Model predicts next token, so labels should be shifted left
    labels_shifted = np.full_like(labels, -100)  # Initialize with ignore index
    # Shift labels left by 1
    labels_shifted[:, :-1] = labels[:, 1:]  # type:ignore

    # Use original predictions and shifted labels
    labels = labels_shifted

    # Flatten and generate basic mask
    predictions_flat = predictions.reshape(-1)
    labels_flat = labels.reshape(-1)
    mask = labels_flat != -100

    # Mask everything after first <end> token (token_id=12)
    end_token_id = VOCAB["<end>"]
    batch_size, seq_len = labels.shape

    # Find first <end> token in each sequence using numpy
    end_mask = labels == end_token_id

    # Get indices of first <end> per sequence (or seq_len if no <end>)
    # Use argmax to find first True value in each row
    # If no <end> token exists, argmax returns 0, so we need to check
    first_end_indices = np.where(end_mask.any(axis=1), end_mask.argmax(axis=1), seq_len)

    # Create sequence position indices
    positions = np.arange(seq_len)[None, :].repeat(batch_size, axis=0)

    # Mask positions after first <end> token
    after_end_mask = positions > first_end_indices[:, None]

    # Flatten the after_end_mask and combine with valid_mask
    after_end_mask_flat = after_end_mask.reshape(-1)
    valid_mask = mask & ~after_end_mask_flat

    predictions_masked = predictions_flat[valid_mask]
    labels_masked = labels_flat[valid_mask]

    # Compute accuracy only on completion tokens (answer portion)
    if len(predictions_masked) > 0:
        completion_accuracy = np.mean(predictions_masked == labels_masked)
    else:
        completion_accuracy = 0.0

    return {
        "token_accuracy": float(completion_accuracy),
    }


# Custom data collator since we're not using HuggingFace tokenizer
def data_collator(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Simple data collator for our custom tokenizer."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


class GumbelTrainer(Trainer):
    """Custom trainer that supports Gumbel-Softmax generation."""

    def __init__(
        self,
        use_gumbel: bool = False,
        gumbel_temperature: float = 1.0,
        tokenizer: Optional[ArithmeticTokenizer] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert self.model is not None, "Model must be provided to GumbelTrainer"
        self.use_gumbel = use_gumbel
        self.gumbel_temperature = gumbel_temperature
        self.tokenizer = tokenizer

    def compute_loss(
        self,
        model: ArithmeticModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Override compute_loss to pass Gumbel-Softmax parameters."""
        # Only use Gumbel during training, not evaluation
        if self.use_gumbel and model.training:
            inputs["use_gumbel"] = torch.tensor(True)
            inputs["gumbel_temperature"] = torch.tensor(self.gumbel_temperature)

        if return_outputs:
            outputs = model(**inputs)
            loss = outputs["loss"]
            return loss, outputs
        else:
            return model(**inputs)["loss"]

    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Custom evaluation that includes generation-based metrics for Gumbel training."""
        # First run standard evaluation
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # If using Gumbel training, add generation-based evaluation
        if self.use_gumbel and self.tokenizer is not None:
            gen_results = self._evaluate_generation(eval_dataset or self.eval_dataset)
            # Add generation metrics with prefix
            for key, value in gen_results.items():
                results[f"{metric_key_prefix}_gen_{key}"] = value

        return results

    def _evaluate_generation(
        self, eval_dataset: Any, num_samples: int = 100
    ) -> dict[str, float]:
        """Evaluate model using actual generation on a subset of examples."""
        import random

        assert self.model is not None
        self.model.eval()
        correct = 0
        total = 0

        # Sample random examples from eval dataset
        dataset_size = len(eval_dataset)
        sample_indices = random.sample(
            range(dataset_size), min(num_samples, dataset_size)
        )

        with torch.no_grad():
            for idx in sample_indices:
                example = eval_dataset[idx]
                input_ids = example["input_ids"].unsqueeze(0).to(self.model.device)
                labels = example["labels"].unsqueeze(0).to(self.model.device)

                # Find the prompt part (before the answer)
                equals_token = VOCAB["="]
                equals_pos = (input_ids == equals_token).nonzero(as_tuple=True)
                if len(equals_pos[1]) > 0:
                    prompt_end = equals_pos[1][0].item() + 1
                    prompt = input_ids[:, :prompt_end]

                    # Generate completion
                    model = cast(ArithmeticModel, self.model)
                    generated = model.generate(
                        prompt,
                        max_new_tokens=10,
                        temperature=0.1,  # Low temperature for deterministic generation
                        end_token_id=VOCAB["<end>"],
                    )

                    # Extract the answer part and compare with expected
                    expected_answer = labels[0, prompt_end:].cpu()
                    generated_answer = generated[0, prompt_end:].cpu()

                    # Compare up to the length of expected answer
                    min_len = min(len(expected_answer), len(generated_answer))
                    if min_len > 0:
                        # Check if answers match (ignoring padding/end tokens after first <end>)
                        exp_seq = expected_answer[:min_len]
                        gen_seq = generated_answer[:min_len]

                        # Find first <end> token in each
                        end_token = VOCAB["<end>"]
                        exp_end = (exp_seq == end_token).nonzero(as_tuple=True)
                        gen_end = (gen_seq == end_token).nonzero(as_tuple=True)

                        exp_len = (
                            exp_end[0][0].item() + 1
                            if len(exp_end[0]) > 0
                            else len(exp_seq)
                        )
                        gen_len = (
                            gen_end[0][0].item() + 1
                            if len(gen_end[0]) > 0
                            else len(gen_seq)
                        )

                        # Compare sequences up to first <end> token
                        match_len = min(exp_len, gen_len)
                        if torch.equal(exp_seq[:match_len], gen_seq[:match_len]):
                            correct += 1

                total += 1

        generation_accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": generation_accuracy, "samples": total}


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train arithmetic transformer model")

    # Model arguments
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["xsmall", "small", "medium", "large"],
        help="Model size configuration",
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
        help="Maximum sequence length (default: longest_example_length from training data metadata + 10)",
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
    parser.add_argument(
        "--use-gumbel",
        action="store_true",
        help="Use Gumbel-Softmax for differentiable sequence generation instead of teacher forcing",
    )
    parser.add_argument(
        "--gumbel-temperature",
        type=float,
        default=1.0,
        help="Temperature for Gumbel-Softmax (lower = more discrete)",
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
    tokenizer = ArithmeticTokenizer()

    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    train_loader, val_loader, test_loader = load_splits(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
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

    # Initialize W&B
    if not args.no_wandb:
        wandb_name = f"arithmetic-{args.model_size}-{args.batch_size}batch-{args.learning_rate}lr"
        if args.use_gumbel:
            wandb_name += f"-gumbel{args.gumbel_temperature}"

        wandb.init(
            project="math-llm",
            config={
                "model_size": args.model_size,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_length": args.max_length,
                "seed": args.seed,
                "use_gumbel": args.use_gumbel,
                "gumbel_temperature": args.gumbel_temperature,
            },
            name=wandb_name,
        )

    logger.info(f"Train samples: {len(cast(Sized, train_loader.dataset))}")
    logger.info(f"Validation samples: {len(cast(Sized, val_loader.dataset))}")
    logger.info(f"Test samples: {len(cast(Sized, test_loader.dataset))}")

    # Create model
    logger.info(f"Creating {args.model_size} model")
    model = create_model(args.model_size)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # Enable TensorFloat32 for better performance on modern GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        logger.info("Enabled TensorFloat32 for faster matrix multiplication")

    # Log Gumbel-Softmax settings
    if args.use_gumbel:
        logger.info(
            f"Using Gumbel-Softmax generation with temperature {args.gumbel_temperature}"
        )
        logger.info(
            "Disabling torch.compile for Gumbel mode due to autoregressive generation"
        )

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
        run_name=f"arithmetic-{args.model_size}",
        # Other settings
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model=(
            "eval_gen_accuracy" if args.use_gumbel else "eval_token_accuracy"
        ),
        greater_is_better=True,
        torch_compile=not args.use_gumbel,  # Disable compile for Gumbel mode due to .item() calls
    )

    # Create trainer
    trainer = GumbelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        use_gumbel=args.use_gumbel,
        gumbel_temperature=args.gumbel_temperature,
        tokenizer=tokenizer,
    )

    # Save training configuration
    config = {
        "model_size": args.model_size,
        "model_parameters": model.count_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "max_length": args.max_length,
        "use_gumbel": args.use_gumbel,
        "gumbel_temperature": args.gumbel_temperature,
        "training_args": training_args.to_dict(),
    }

    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved training configuration to {config_path}")

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
