"""Training utilities for arithmetic transformer model."""

import logging

import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction


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
