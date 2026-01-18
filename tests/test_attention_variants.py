"""Tests for attention variants: PoPE and softmax1."""

import torch

from src.config import ModelConfig
from src.model import (
    ArithmeticModel,
    PoPE,
    TransformerBlock,
    UniversalTransformerModel,
    create_model_from_config,
    softmax1,
)
from src.tokenizer import VOCAB_SIZE


class TestSoftmax1:
    """Tests for the softmax1 function."""

    def test_output_shape(self) -> None:
        """Softmax1 should preserve input shape."""
        x = torch.randn(2, 4, 8)
        result = softmax1(x, dim=-1)
        assert result.shape == x.shape

    def test_sum_less_than_one(self) -> None:
        """Softmax1 outputs should sum to less than 1."""
        x = torch.randn(2, 4, 8)
        result = softmax1(x, dim=-1)
        sums = result.sum(dim=-1)
        assert (sums < 1.0).all()
        assert (sums > 0.0).all()

    def test_preserves_relative_ratios(self) -> None:
        """Softmax1 should preserve relative ratios like standard softmax."""
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = softmax1(x, dim=-1)
        ratios = result[0, 1:] / result[0, :-1]
        expected_ratios = torch.exp(torch.tensor([1.0, 1.0]))
        assert torch.allclose(ratios, expected_ratios, rtol=1e-5)

    def test_escape_hatch_behavior(self) -> None:
        """When all inputs are equal, the +1 reduces each output.

        With standard softmax on [-100, -100, -100], each gets 1/3.
        With softmax1, the +1 in denominator makes each get exp(0)/(1+3*exp(0)) = 0.25.
        This demonstrates the "escape hatch" - the sum is less than 1.
        """
        x = torch.tensor([[-100.0, -100.0, -100.0]])
        result = softmax1(x, dim=-1)
        # Each element should be approximately 0.25 (1/4)
        assert torch.allclose(result, torch.tensor([[0.25, 0.25, 0.25]]), atol=1e-6)
        # Sum should be 0.75, less than 1
        assert torch.isclose(result.sum(), torch.tensor(0.75), atol=1e-6)

    def test_gradient_flow(self) -> None:
        """Softmax1 should allow gradient flow."""
        x = torch.randn(2, 4, requires_grad=True)
        result = softmax1(x, dim=-1)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestPoPE:
    """Tests for Polar Positional Encoding (PoPE)."""

    def test_init(self) -> None:
        """PoPE should initialize correctly."""
        pope = PoPE(d_model=64, n_heads=4)
        assert pope.d_model == 64
        assert pope.n_heads == 4
        assert pope.head_dim == 16
        assert pope.phase_bias.shape == (4, 16)

    def test_forward_output_shape(self) -> None:
        """PoPE forward should output doubled head dimension."""
        pope = PoPE(d_model=64, n_heads=4)
        batch_size, n_heads, seq_len, head_dim = 2, 4, 8, 16

        q = torch.randn(batch_size, n_heads, seq_len, head_dim)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        positions = torch.arange(seq_len)

        q_pope, k_pope = pope(q, k, positions)

        assert q_pope.shape == (batch_size, n_heads, seq_len, head_dim * 2)
        assert k_pope.shape == (batch_size, n_heads, seq_len, head_dim * 2)

    def test_forward_values_finite(self) -> None:
        """PoPE outputs should be finite."""
        pope = PoPE(d_model=64, n_heads=4)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        positions = torch.arange(8)

        q_pope, k_pope = pope(q, k, positions)

        assert torch.isfinite(q_pope).all()
        assert torch.isfinite(k_pope).all()

    def test_learnable_phase_bias(self) -> None:
        """Phase bias should be learnable and affect output."""
        pope = PoPE(d_model=64, n_heads=4)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        positions = torch.arange(8)

        q1, k1 = pope(q, k, positions)

        # Modify phase bias
        pope.phase_bias.data.fill_(0.5)
        q2, k2 = pope(q, k, positions)

        assert not torch.allclose(q1, q2)
        assert not torch.allclose(k1, k2)


class TestTransformerBlockVariants:
    """Tests for TransformerBlock with attention variants."""

    def test_standard_softmax(self) -> None:
        """Block with standard softmax should work."""
        block = TransformerBlock(
            d_model=64, n_heads=4, d_ff=128, softmax_variant="standard"
        )
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_softmax1_variant(self) -> None:
        """Block with softmax1 should work."""
        block = TransformerBlock(
            d_model=64, n_heads=4, d_ff=128, softmax_variant="softmax1"
        )
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_pope_variant(self) -> None:
        """Block with PoPE positional encoding should work."""
        block = TransformerBlock(
            d_model=64, n_heads=4, d_ff=128, positional_encoding="pope"
        )
        x = torch.randn(2, 8, 64)
        positions = torch.arange(8)
        out = block(x, positions)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_pope_with_softmax1(self) -> None:
        """Block with both PoPE and softmax1 should work."""
        block = TransformerBlock(
            d_model=64,
            n_heads=4,
            d_ff=128,
            positional_encoding="pope",
            softmax_variant="softmax1",
        )
        x = torch.randn(2, 8, 64)
        positions = torch.arange(8)
        out = block(x, positions)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestModelVariants:
    """Tests for full model with attention variants."""

    def test_arithmetic_model_pope(self) -> None:
        """ArithmeticModel with PoPE should work."""
        model = ArithmeticModel(
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            positional_encoding="pope",
        )
        input_ids = torch.randint(0, 10, (2, 8))
        out = model(input_ids)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8, VOCAB_SIZE)

    def test_arithmetic_model_softmax1(self) -> None:
        """ArithmeticModel with softmax1 should work."""
        model = ArithmeticModel(
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            softmax_variant="softmax1",
        )
        input_ids = torch.randint(0, 10, (2, 8))
        out = model(input_ids)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8, VOCAB_SIZE)

    def test_universal_model_pope(self) -> None:
        """UniversalTransformerModel with PoPE should work."""
        model = UniversalTransformerModel(
            d_model=64,
            n_layers=1,
            n_loops=2,
            n_heads=4,
            d_ff=128,
            positional_encoding="pope",
        )
        input_ids = torch.randint(0, 10, (2, 8))
        out = model(input_ids)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8, VOCAB_SIZE)

    def test_universal_model_softmax1(self) -> None:
        """UniversalTransformerModel with softmax1 should work."""
        model = UniversalTransformerModel(
            d_model=64,
            n_layers=1,
            n_loops=2,
            n_heads=4,
            d_ff=128,
            softmax_variant="softmax1",
        )
        input_ids = torch.randint(0, 10, (2, 8))
        out = model(input_ids)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8, VOCAB_SIZE)


class TestConfigVariants:
    """Tests for creating models from config with variants."""

    def test_create_pope_model_from_config(self) -> None:
        """Should create model with PoPE from config."""
        config = ModelConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            positional_encoding="pope",
        )
        model = create_model_from_config(config)
        assert isinstance(model, ArithmeticModel)
        assert model.positional_encoding == "pope"

    def test_create_softmax1_model_from_config(self) -> None:
        """Should create model with softmax1 from config."""
        config = ModelConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            softmax_variant="softmax1",
        )
        model = create_model_from_config(config)
        assert isinstance(model, ArithmeticModel)
        assert model.softmax_variant == "softmax1"

    def test_feedback_rejects_pope(self) -> None:
        """Feedback architecture should reject PoPE."""
        config = ModelConfig(
            architecture="feedback",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            positional_encoding="pope",
        )
        try:
            create_model_from_config(config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "PoPE is not supported" in str(e)

    def test_feedback_accepts_softmax1(self) -> None:
        """Feedback architecture should accept softmax1."""
        config = ModelConfig(
            architecture="feedback",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            softmax_variant="softmax1",
        )
        model = create_model_from_config(config)
        assert model.softmax_variant == "softmax1"
