"""Tests for ALiBi (Attention with Linear Biases) implementation."""

import pytest
import torch

from src.model import build_alibi_bias


class TestAliBiBias:
    """Test ALiBi bias computation."""

    def test_build_alibi_bias_shape(self):
        """Test that ALiBi bias has correct shape."""
        n_heads = 4
        seq_len = 10
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        assert alibi_bias.shape == (n_heads, seq_len, seq_len)

    def test_build_alibi_bias_diagonal_zero(self):
        """Test that diagonal elements are zero (no self-bias)."""
        n_heads = 4
        seq_len = 10
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        for h in range(n_heads):
            diagonal = torch.diagonal(alibi_bias[h])
            assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)

    def test_build_alibi_bias_non_positive(self):
        """Test that all bias values are non-positive (penalize distances)."""
        n_heads = 4
        seq_len = 10
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        for h in range(n_heads):
            assert (alibi_bias[h] <= 0).all()

    def test_build_alibi_bias_distance_penalty(self):
        """Test that distant positions have more negative bias."""
        n_heads = 4
        seq_len = 10
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        for h in range(n_heads):
            center = seq_len // 2
            # Check that farther positions have more negative bias
            if center + 2 < seq_len:
                bias_1 = alibi_bias[h, center, center + 1]  # 1 step away
                bias_2 = alibi_bias[h, center, center + 2]  # 2 steps away
                assert bias_2 <= bias_1, (
                    f"Head {h}: bias_2 ({bias_2}) should be <= bias_1 ({bias_1})"
                )

    def test_build_alibi_bias_head_slopes(self):
        """Test that different heads have different slopes."""
        n_heads = 4
        seq_len = 10
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        # Check that different heads have different bias magnitudes at the same position
        center = seq_len // 2
        if center + 1 < seq_len:
            biases = [alibi_bias[h, center, center + 1] for h in range(n_heads)]

            # All biases should be different (different slopes per head)
            for i in range(n_heads):
                for j in range(i + 1, n_heads):
                    assert not torch.allclose(biases[i], biases[j])

    def test_build_alibi_bias_symmetric_distance(self):
        """Test that bias depends only on distance, not direction."""
        n_heads = 4
        seq_len = 10
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        for h in range(n_heads):
            center = seq_len // 2
            if center > 0 and center + 1 < seq_len:
                # Distance 1 in both directions should have same magnitude
                bias_left = alibi_bias[h, center, center - 1]
                bias_right = alibi_bias[h, center, center + 1]
                assert torch.allclose(bias_left, bias_right, atol=1e-6)

    @pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
    def test_build_alibi_bias_different_head_counts(self, n_heads: int):
        """Test ALiBi bias with different numbers of heads."""
        seq_len = 8
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        assert alibi_bias.shape == (n_heads, seq_len, seq_len)
        assert (alibi_bias <= 0).all()

    @pytest.mark.parametrize("seq_len", [1, 5, 16, 32])
    def test_build_alibi_bias_different_sequence_lengths(self, seq_len: int):
        """Test ALiBi bias with different sequence lengths."""
        n_heads = 4
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        assert alibi_bias.shape == (n_heads, seq_len, seq_len)
        assert (alibi_bias <= 0).all()

    def test_build_alibi_bias_dtype_device(self):
        """Test that ALiBi bias respects device and dtype parameters."""
        n_heads = 2
        seq_len = 5
        device = torch.device("cpu")
        dtype = torch.float16

        alibi_bias = build_alibi_bias(n_heads, seq_len, device, dtype)

        assert alibi_bias.device == device
        assert alibi_bias.dtype == dtype

    def test_build_alibi_bias_slope_calculation(self):
        """Test that slopes follow the expected geometric progression."""
        n_heads = 4
        seq_len = 5
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        # Extract slopes by looking at unit distance bias
        slopes = []
        for h in range(n_heads):
            # Get bias at distance 1 (which equals -slope * 1)
            slope = -alibi_bias[
                h, 0, 1
            ].item()  # Negative because we negate in the function
            slopes.append(slope)

        # Check that slopes follow geometric progression 2^(-8/n), 2^(-16/n), etc.
        expected_slopes = [2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)]

        for actual, expected in zip(slopes, expected_slopes):
            assert abs(actual - expected) < 1e-6, (
                f"Slope mismatch: {actual} vs {expected}"
            )

    def test_build_alibi_bias_single_position(self):
        """Test ALiBi bias with sequence length 1."""
        n_heads = 2
        seq_len = 1
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        assert alibi_bias.shape == (n_heads, 1, 1)
        # Only element should be zero (self-attention)
        assert torch.allclose(alibi_bias, torch.zeros_like(alibi_bias))

    def test_build_alibi_bias_large_sequence(self):
        """Test ALiBi bias with a larger sequence length."""
        n_heads = 8
        seq_len = 64
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        assert alibi_bias.shape == (n_heads, seq_len, seq_len)
        assert (alibi_bias <= 0).all()

        # Test that bias magnitude increases with distance
        for h in range(n_heads):
            center = seq_len // 2
            distances = [1, 5, 10, 20]
            prev_bias = 0

            for dist in distances:
                if center + dist < seq_len:
                    current_bias = alibi_bias[h, center, center + dist]
                    assert current_bias <= prev_bias, (
                        "Bias should decrease with distance"
                    )
                    prev_bias = current_bias


if __name__ == "__main__":
    pytest.main([__file__])
