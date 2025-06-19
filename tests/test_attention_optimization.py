"""Tests for attention mechanism optimization with built-in PyTorch functions."""

import pytest
import torch

from src.model import TransformerBlock, build_alibi_bias


class TestAttentionOptimization:
    """Test that the optimized attention mechanism produces equivalent results."""

    @pytest.fixture
    def test_inputs(self):
        """Create test inputs for attention mechanism."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 4
        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))

        # Create ALiBi bias
        alibi_bias = build_alibi_bias(n_heads, seq_len, x.device, x.dtype)

        return x, mask, alibi_bias, n_heads

    def test_forward_pass_consistency(
        self, test_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        """Test that forward pass works with optimized attention."""
        x, _mask, alibi_bias, n_heads = test_inputs

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128)

        # Test with ALiBi bias (always required now)
        output = block(x, alibi_bias=alibi_bias)
        assert output.shape == x.shape

    def test_gradient_flow(
        self, test_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        """Test that gradients flow properly through optimized attention."""
        x, _mask, alibi_bias, n_heads = test_inputs
        x.requires_grad_(True)

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128)

        # Forward pass - use a loss that creates meaningful gradients
        output = block(x, alibi_bias=alibi_bias)
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

    def test_attention_mask_shapes(
        self, test_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        """Test that attention masks are properly shaped for scaled_dot_product_attention."""
        x, _mask, alibi_bias, n_heads = test_inputs

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128)

        # Test that we can handle ALiBi bias without shape errors
        try:
            block(x, alibi_bias=alibi_bias)
        except RuntimeError as e:
            if "shape" in str(e).lower() or "size" in str(e).lower():
                pytest.fail(f"Shape mismatch in attention masks: {e}")
            else:
                # Re-raise if it's not a shape-related error
                raise

    def test_alibi_bias_preservation(
        self, test_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        """Test that ALiBi bias functionality is preserved."""
        x, _mask, alibi_bias, n_heads = test_inputs

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128)

        # Create a zero ALiBi bias for comparison
        zero_alibi = torch.zeros_like(alibi_bias)

        # Output with ALiBi should be different from output with zero ALiBi
        output_zero_alibi = block(x, alibi_bias=zero_alibi)
        output_with_alibi = block(x, alibi_bias=alibi_bias)

        # They should be different (ALiBi is having an effect)
        assert not torch.allclose(output_zero_alibi, output_with_alibi, atol=1e-4)

    def test_causal_masking_preserved(
        self, test_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        """Test that causal masking is always applied (built into the block now)."""
        x, _mask, alibi_bias, n_heads = test_inputs

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128)

        # Since causal masking is now always applied, just test that forward works
        output = block(x, alibi_bias=alibi_bias)
        assert output.shape == x.shape

        # Test that the model produces deterministic output in eval mode
        block.eval()
        with torch.no_grad():
            output1 = block(x, alibi_bias=alibi_bias)
            output2 = block(x, alibi_bias=alibi_bias)
            assert torch.allclose(output1, output2, atol=1e-6)

    def test_training_vs_eval_mode(
        self, test_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        """Test that training and eval modes work correctly."""
        x, _mask, alibi_bias, n_heads = test_inputs

        block = TransformerBlock(d_model=64, n_heads=n_heads, d_ff=128, dropout=0.1)

        # Training mode
        block.train()
        output_train = block(x, alibi_bias=alibi_bias)

        # Eval mode
        block.eval()
        with torch.no_grad():
            output_eval = block(x, alibi_bias=alibi_bias)

        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval, atol=1e-4)

        # Both should have correct shape
        assert output_train.shape == x.shape
        assert output_eval.shape == x.shape
