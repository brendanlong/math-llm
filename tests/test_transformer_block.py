"""Unit tests for TransformerBlock."""

import torch

from src.model import TransformerBlock


class TestTransformerBlock:
    """Test transformer block module."""

    def test_init(self):
        """Test transformer block initialization."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)
        assert block.d_model == 256
        assert block.n_heads == 4
        assert block.head_dim == 64

    def test_forward_shape(self):
        """Test transformer block forward pass shapes."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)
        x = torch.randn(2, 10, 256)  # (batch_size, seq_len, d_model)

        output = block(x)
        assert output.shape == x.shape

    def test_forward_causal(self):
        """Test transformer block with causal masking (built-in via is_causal=True)."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)
        x = torch.randn(2, 10, 256)

        output = block(x)
        assert output.shape == x.shape
