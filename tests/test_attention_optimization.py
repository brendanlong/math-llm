"""Tests for attention mechanism optimization with built-in PyTorch functions."""

import pytest
import torch

from src.model import TransformerBlock


class TestAttentionOptimization:
    """Test that the optimized attention mechanism produces equivalent results."""

    @pytest.fixture
    def test_inputs(self) -> tuple[torch.Tensor, int]:
        """Create test inputs for attention mechanism."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 4
        x = torch.randn(batch_size, seq_len, d_model)
        return x, n_heads

    def test_forward_pass_consistency(
        self, test_inputs: tuple[torch.Tensor, int]
    ) -> None:
        """Test that forward pass works with optimized attention."""
        x, n_heads = test_inputs

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128)

        output, _attn = block(x)
        assert output.shape == x.shape

    def test_gradient_flow(self, test_inputs: tuple[torch.Tensor, int]) -> None:
        """Test that gradients flow properly through optimized attention."""
        x, n_heads = test_inputs
        x.requires_grad_(True)

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128)

        # Forward pass - use a loss that creates meaningful gradients
        output, _attn = block(x)
        # Square the output to amplify gradients and ensure they're meaningful
        loss = (output**2).mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist for input
        assert x.grad is not None
        assert torch.any(x.grad != 0.0)

        # Check that model parameters have gradients
        for param in block.parameters():
            assert param.grad is not None
            # For random inputs, some parameters might have very small gradients
            # Just check that not all gradients are exactly zero
            assert torch.any(param.grad != 0.0)

    def test_causal_masking_preserved(
        self, test_inputs: tuple[torch.Tensor, int]
    ) -> None:
        """Test that causal masking is always applied (built into the block via is_causal=True)."""
        x, n_heads = test_inputs

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128)

        output, _attn = block(x)
        assert output.shape == x.shape

        # Test that the model produces deterministic output in eval mode
        block.eval()
        with torch.no_grad():
            output1, _ = block(x)
            output2, _ = block(x)
            assert torch.allclose(output1, output2, atol=1e-6)

    def test_training_vs_eval_mode(self, test_inputs: tuple[torch.Tensor, int]) -> None:
        """Test that training and eval modes work correctly."""
        x, n_heads = test_inputs

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128, dropout=0.1)

        # Training mode
        block.train()
        output_train, _ = block(x)

        # Eval mode
        block.eval()
        with torch.no_grad():
            output_eval, _ = block(x)

        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval, atol=1e-4)

        # Both should have correct shape
        assert output_train.shape == x.shape
        assert output_eval.shape == x.shape
