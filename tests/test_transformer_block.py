"""Unit tests for TransformerBlock."""

import torch

from src.model import TransformerBlock, build_alibi_bias


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

    def test_forward_with_mask_and_alibi(self):
        """Test transformer block with attention mask and ALiBi bias."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)
        x = torch.randn(2, 10, 256)
        mask = torch.triu(torch.ones(10, 10), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))

        alibi_bias = build_alibi_bias(4, 10, x.device, x.dtype)

        output = block(x, mask=mask, alibi_bias=alibi_bias)
        assert output.shape == x.shape
