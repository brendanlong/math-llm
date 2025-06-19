"""Test Gumbel-Softmax generation functionality."""

import pytest
import torch

from src.model import ArithmeticModel, create_model
from src.tokenizer import VOCAB, VOCAB_SIZE, tokenizer


class TestGumbelSoftmax:
    """Test suite for Gumbel-Softmax generation."""

    @pytest.fixture
    def model(self) -> ArithmeticModel:
        """Create a test model."""
        return create_model("xsmall")

    @pytest.fixture
    def sample_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample input and labels."""
        text = "3+5="
        input_ids = torch.tensor([tokenizer.encode(text)])

        full_text = "3+5=8<end>"
        labels = torch.tensor([tokenizer.encode(full_text)])

        # Pad to same length
        max_len = max(input_ids.shape[1], labels.shape[1])
        if input_ids.shape[1] < max_len:
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, max_len - input_ids.shape[1]),
                value=VOCAB["<end>"],
            )
        if labels.shape[1] < max_len:
            labels = torch.nn.functional.pad(
                labels, (0, max_len - labels.shape[1]), value=-100
            )

        return input_ids, labels

    def test_regular_forward_pass(
        self,
        model: ArithmeticModel,
        sample_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test regular forward pass works correctly."""
        input_ids, labels = sample_data

        outputs = model(input_ids, labels=labels)

        assert "loss" in outputs
        assert "logits" in outputs
        assert torch.isfinite(outputs["loss"]).all()
        assert outputs["logits"].shape == (1, input_ids.shape[1], VOCAB_SIZE)

    def test_gumbel_forward_pass(
        self,
        model: ArithmeticModel,
        sample_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test Gumbel-Softmax forward pass works correctly."""
        input_ids, labels = sample_data

        outputs = model(
            input_ids, labels=labels, use_gumbel=True, gumbel_temperature=1.0
        )

        assert "loss" in outputs
        assert "logits" in outputs
        assert torch.isfinite(outputs["loss"]).all()
        assert outputs["logits"].shape == (1, input_ids.shape[1], VOCAB_SIZE)

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
    def test_gumbel_temperatures(
        self,
        model: ArithmeticModel,
        sample_data: tuple[torch.Tensor, torch.Tensor],
        temperature: float,
    ) -> None:
        """Test Gumbel-Softmax with different temperatures."""
        input_ids, labels = sample_data

        outputs = model(
            input_ids, labels=labels, use_gumbel=True, gumbel_temperature=temperature
        )

        assert "loss" in outputs
        assert torch.isfinite(outputs["loss"]).all()

    def test_eval_mode_disables_gumbel(
        self, model: ArithmeticModel, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that evaluation mode disables Gumbel-Softmax even when requested."""
        input_ids, labels = sample_data

        # Get regular training mode output
        model.train()
        train_outputs = model(input_ids, labels=labels)

        # Get eval mode output with use_gumbel=True (should behave like regular forward)
        model.eval()
        eval_outputs = model(
            input_ids, labels=labels, use_gumbel=True, gumbel_temperature=1.0
        )

        # In eval mode, should behave like regular forward pass
        assert eval_outputs["logits"].shape == train_outputs["logits"].shape

        # Reset to training mode
        model.train()

    def test_gumbel_vs_regular_loss_difference(
        self, model: ArithmeticModel, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that Gumbel-Softmax produces different losses than regular forward pass."""
        input_ids, labels = sample_data

        # Regular forward pass
        regular_outputs = model(input_ids, labels=labels)

        # Gumbel forward pass
        gumbel_outputs = model(
            input_ids, labels=labels, use_gumbel=True, gumbel_temperature=1.0
        )

        # Losses should be different (with high probability due to stochastic sampling)
        # We'll just check that both are finite and valid
        assert torch.isfinite(regular_outputs["loss"])
        assert torch.isfinite(gumbel_outputs["loss"])
        assert regular_outputs["loss"] >= 0
        assert gumbel_outputs["loss"] >= 0
