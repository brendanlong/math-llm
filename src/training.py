"""Training utilities for arithmetic transformer model."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from .activation_stats import ActivationStatsCollector, format_stats_summary


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute evaluation metrics.

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
    labels_shifted[:, :-1] = labels[:, 1:]  # type:ignore

    # Flatten and mask out ignored positions
    predictions_flat = predictions.reshape(-1)
    labels_flat = labels_shifted.reshape(-1)
    valid_mask = labels_flat != -100

    predictions_masked = predictions_flat[valid_mask]
    labels_masked = labels_flat[valid_mask]

    # Compute accuracy only on valid tokens
    assert len(predictions_masked) > 0, "No valid tokens found for evaluation"
    accuracy = np.mean(predictions_masked == labels_masked)

    return {
        "token_accuracy": float(accuracy),
    }


def data_collator(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Simple data collator for our custom tokenizer."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def setup_training_optimizations() -> None:
    """Setup training optimizations like TensorFloat32."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        logger = logging.getLogger(__name__)
        logger.info("Enabled TensorFloat32 for faster matrix multiplication")


class ActivationStatsCallback(TrainerCallback):
    """HuggingFace Trainer callback for computing activation statistics.

    Computes activation statistics (kurtosis, outliers, attention entropy) during
    evaluation and logs them to W&B. Saves final stats to output directory at
    end of training.
    """

    def __init__(
        self,
        eval_dataloader: DataLoader[dict[str, torch.Tensor]],
        use_softmax1: bool = False,
        compute_every_n_evals: int = 1,
        max_batches: Optional[int] = 10,
    ):
        """Initialize the activation stats callback.

        Args:
            eval_dataloader: DataLoader for computing stats (should be eval set)
            use_softmax1: Whether the model uses softmax1 attention
            compute_every_n_evals: Compute stats every N evaluations
            max_batches: Max batches to process per evaluation (None for all)
        """
        self.eval_dataloader = eval_dataloader
        self.use_softmax1 = use_softmax1
        self.compute_every_n_evals = compute_every_n_evals
        self.max_batches = max_batches
        self.eval_count = 0
        self.logger = logging.getLogger(__name__)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        **kwargs: object,
    ) -> None:
        """Compute activation stats after evaluation."""
        del args, control, kwargs  # Unused

        if model is None:
            return

        self.eval_count += 1
        if self.eval_count % self.compute_every_n_evals != 0:
            return

        self.logger.info("Computing activation statistics...")
        collector = ActivationStatsCollector(model, use_softmax1=self.use_softmax1)

        model.eval()
        device = next(model.parameters()).device

        with collector:
            with torch.no_grad():
                for batch_idx, batch in enumerate(
                    tqdm(self.eval_dataloader, desc="Computing activation stats")
                ):
                    if self.max_batches is not None and batch_idx >= self.max_batches:
                        break

                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    # Forward pass with attention outputs
                    outputs = model(
                        input_ids,
                        labels=labels,
                        output_attentions=True,
                        output_attention_scores=True,
                    )

                    # Process attention outputs
                    attention_weights = outputs.get("attentions")
                    attention_scores = outputs.get("attention_scores")
                    collector.process_attention_outputs(
                        attention_weights, attention_scores
                    )

        stats = collector.compute_statistics()
        stats_dict = stats.to_dict()

        # Log aggregate stats to W&B via trainer's logging
        try:
            import wandb

            if wandb.run is not None:
                agg = stats_dict["aggregate"]
                wandb.log(
                    {
                        "activation_stats/hidden_kurtosis_mean": agg[
                            "hidden_kurtosis_mean"
                        ],
                        "activation_stats/hidden_max_abs": agg["hidden_max_abs"],
                        "activation_stats/hidden_outlier_fraction_mean": agg[
                            "hidden_outlier_fraction_mean"
                        ],
                        "activation_stats/attention_entropy_mean": agg[
                            "attention_entropy_mean"
                        ],
                        "activation_stats/attention_sparsity_mean": agg[
                            "attention_sparsity_mean"
                        ],
                        "activation_stats/attention_abstention_mean": agg[
                            "attention_abstention_mean"
                        ],
                    },
                    step=state.global_step,
                )
        except ImportError:
            pass

        self.logger.info(f"\n{format_stats_summary(stats)}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        **kwargs: object,
    ) -> None:
        """Save final activation stats at end of training."""
        del state, control, kwargs  # Unused

        if model is None:
            return

        if args.output_dir is None:
            self.logger.warning(
                "No output directory specified, skipping activation stats save"
            )
            return

        self.logger.info("Computing final activation statistics...")
        collector = ActivationStatsCollector(model, use_softmax1=self.use_softmax1)

        model.eval()
        device = next(model.parameters()).device

        with collector:
            with torch.no_grad():
                for batch_idx, batch in enumerate(
                    tqdm(self.eval_dataloader, desc="Final activation stats")
                ):
                    if self.max_batches is not None and batch_idx >= self.max_batches:
                        break

                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(
                        input_ids,
                        labels=labels,
                        output_attentions=True,
                        output_attention_scores=True,
                    )

                    attention_weights = outputs.get("attentions")
                    attention_scores = outputs.get("attention_scores")
                    collector.process_attention_outputs(
                        attention_weights, attention_scores
                    )

        stats = collector.compute_statistics()

        # Save to output directory
        output_path = Path(args.output_dir) / "activation_stats.json"
        stats.save(output_path)
        self.logger.info(f"Activation stats saved to {output_path}")
        self.logger.info(f"\n{format_stats_summary(stats)}")
