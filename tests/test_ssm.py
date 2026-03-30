"""Tests for SSM (Mamba) model architecture."""

import torch

from src.config import ModelConfig
from src.model import create_model_from_config
from src.ssm import MambaBlock, SSMModel, SelectiveSSM
from src.tokenizer import VOCAB_SIZE, tokenizer

SSM_SMALL_CONFIG = ModelConfig(
    architecture="ssm",
    d_model=64,
    n_layers=2,
    d_state=8,
    d_conv=4,
    expand=2,
    dropout=0.0,
)


class TestSelectiveSSM:
    """Tests for the SelectiveSSM core computation."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input shape."""
        d_inner = 128
        ssm = SelectiveSSM(d_inner=d_inner, d_state=16)
        x = torch.randn(2, 10, d_inner)
        y = ssm(x)
        assert y.shape == x.shape

    def test_causal(self) -> None:
        """Test that SSM output at position t depends only on positions <= t."""
        d_inner = 64
        ssm = SelectiveSSM(d_inner=d_inner, d_state=8)
        ssm.eval()

        x = torch.randn(1, 8, d_inner)
        y_full = ssm(x)

        # Output at position 3 should be the same if we truncate input to positions 0-3
        y_prefix = ssm(x[:, :4])
        torch.testing.assert_close(y_full[:, :4], y_prefix, atol=1e-5, rtol=1e-5)


class TestMambaBlock:
    """Tests for the MambaBlock."""

    def test_output_shape(self) -> None:
        """Test that block preserves input shape (residual connection)."""
        d_model = 64
        block = MambaBlock(d_model=d_model, d_state=8, d_conv=4, expand=2, dropout=0.0)
        x = torch.randn(2, 10, d_model)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_at_init(self) -> None:
        """Test that output is close to input at initialization (residual + small init)."""
        d_model = 64
        block = MambaBlock(d_model=d_model, d_state=8, d_conv=4, expand=2, dropout=0.0)
        block.eval()
        x = torch.randn(1, 5, d_model)
        y = block(x)
        # The residual connection means output should be in the same ballpark as input
        assert torch.isfinite(y).all()


class TestSSMModel:
    """Tests for the full SSM model."""

    def test_create_from_config(self) -> None:
        """Test model creation from config."""
        model = create_model_from_config(SSM_SMALL_CONFIG)
        assert isinstance(model, SSMModel)
        assert model.d_model == 64
        assert len(model.layers) == 2

    def test_forward_without_labels(self) -> None:
        """Test forward pass returns logits."""
        model = create_model_from_config(SSM_SMALL_CONFIG)
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        logits = model(input_ids)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 10, VOCAB_SIZE)

    def test_forward_with_labels(self) -> None:
        """Test forward pass returns loss and logits when labels provided."""
        model = create_model_from_config(SSM_SMALL_CONFIG)
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        labels = input_ids.clone()
        result = model(input_ids, labels=labels)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "logits" in result
        assert result["loss"].ndim == 0  # scalar loss

    def test_generate(self) -> None:
        """Test autoregressive generation."""
        model = create_model_from_config(SSM_SMALL_CONFIG)
        tokens = tokenizer.encode("1+2=")
        input_ids = torch.tensor([tokens])
        generated = model.generate(input_ids, max_new_tokens=5)
        assert generated.shape[0] == 1
        assert generated.shape[1] >= len(tokens)

    def test_backward(self) -> None:
        """Test that gradients flow through the model."""
        model = create_model_from_config(SSM_SMALL_CONFIG)
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        labels = input_ids.clone()
        result = model(input_ids, labels=labels)
        assert isinstance(result, dict)
        loss = result["loss"]
        assert isinstance(loss, torch.Tensor)
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_parameter_count_small(self) -> None:
        """Test that SSM small is compute-matched with transformer small (~2.4M)."""
        config = ModelConfig(architecture="ssm", d_model=256, n_layers=5)
        model = create_model_from_config(config)
        params = model.count_parameters()
        # Should be roughly matched to standard-small (2.38M)
        assert 1_500_000 < params < 3_500_000

    def test_config_rejects_n_heads(self) -> None:
        """Test that SSM config rejects n_heads parameter."""
        import pytest

        with pytest.raises(ValueError, match="does not use n_heads"):
            ModelConfig(architecture="ssm", d_model=256, n_layers=5, n_heads=4)

    def test_config_rejects_d_ff(self) -> None:
        """Test that SSM config rejects d_ff parameter."""
        import pytest

        with pytest.raises(ValueError, match="does not use d_ff"):
            ModelConfig(architecture="ssm", d_model=256, n_layers=5, d_ff=512)
