"""Training utilities for arithmetic transformer model."""

import logging
import random
from typing import Any, Optional, cast

import numpy as np
import torch
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalPrediction

from .model import ArithmeticModel
from .tokenizer import VOCAB


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute evaluation metrics to match training objective.

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

    # Flatten and mask out ignored positions
    predictions_flat = predictions.reshape(-1)
    labels_flat = labels.reshape(-1)
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


class GumbelTrainer(Trainer):
    """Custom trainer that supports Gumbel-Softmax generation."""

    def __init__(
        self,
        use_gumbel: bool = False,
        gumbel_temperature: float = 1.0,
        mask_reasoning: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert self.model is not None, "Model must be provided to GumbelTrainer"
        self.use_gumbel = use_gumbel
        self.gumbel_temperature = gumbel_temperature
        self.mask_reasoning = mask_reasoning

    def compute_loss(
        self,
        model: ArithmeticModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,  # noqa: ARG002
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Override compute_loss to pass Gumbel-Softmax parameters."""
        # Only use Gumbel during training, not evaluation
        if self.use_gumbel and model.training:
            inputs["use_gumbel"] = torch.tensor(True)
            inputs["gumbel_temperature"] = torch.tensor(self.gumbel_temperature)

        # Pass mask_reasoning parameter
        inputs["mask_reasoning"] = torch.tensor(self.mask_reasoning)

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
        if self.use_gumbel and self.processing_class is not None:
            gen_results = self._evaluate_generation(eval_dataset or self.eval_dataset)
            # Add generation metrics with prefix, using same name as token accuracy
            results[f"{metric_key_prefix}_token_accuracy"] = gen_results["accuracy"]

        return results

    def _evaluate_generation(
        self, eval_dataset: Any, num_samples: int = 100
    ) -> dict[str, float]:
        """Evaluate model using actual generation on a subset of examples."""

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


def setup_training_optimizations() -> None:
    """Setup training optimizations like TensorFloat32."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        logger = logging.getLogger(__name__)
        logger.info("Enabled TensorFloat32 for faster matrix multiplication")
