"""Unit tests for Universal Transformer model."""

import pytest
import torch

from src.config import ModelConfig
from src.model import (
    UniversalTransformerModel,
    create_model_from_config,
)
from src.tokenizer import VOCAB, VOCAB_SIZE, tokenizer

# Test configs matching the original factory functions
UT_SMALL_CONFIG = ModelConfig(
    architecture="universal",
    d_model=256,
    n_layers=2,
    n_loops=4,
    n_heads=4,
    d_ff=512,
    dropout=0.1,
    use_loop_embeddings=True,
)

UT_MEDIUM_CONFIG = ModelConfig(
    architecture="universal",
    d_model=512,
    n_layers=2,
    n_loops=4,
    n_heads=8,
    d_ff=1024,
    dropout=0.1,
    use_loop_embeddings=True,
)

UT_LARGE_CONFIG = ModelConfig(
    architecture="universal",
    d_model=512,
    n_layers=2,
    n_loops=6,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    use_loop_embeddings=True,
)

STANDARD_SMALL_CONFIG = ModelConfig(
    architecture="standard",
    d_model=256,
    n_layers=4,
    n_heads=4,
    d_ff=512,
    dropout=0.1,
)


class TestUniversalTransformerConfigs:
    """Test Universal Transformer creation from configs."""

    def test_create_ut_small_model(self) -> None:
        """Test small UT model creation."""
        model = create_model_from_config(UT_SMALL_CONFIG)

        assert isinstance(model, UniversalTransformerModel)
        assert model.d_model == 256
        assert model.n_layers == 2
        assert model.n_loops == 4
        assert model.sequential_depth == 8
        assert model.use_loop_embeddings is True

        # Should have fewer parameters than standard model with same depth
        param_count = model.count_parameters()
        assert 100_000 < param_count < 2_000_000

    def test_create_ut_medium_model(self) -> None:
        """Test medium UT model creation."""
        model = create_model_from_config(UT_MEDIUM_CONFIG)

        assert isinstance(model, UniversalTransformerModel)
        assert model.d_model == 512
        assert model.n_layers == 2
        assert model.n_loops == 4
        assert model.sequential_depth == 8

        param_count = model.count_parameters()
        assert 500_000 < param_count < 5_000_000

    def test_create_ut_large_model(self) -> None:
        """Test large UT model creation."""
        model = create_model_from_config(UT_LARGE_CONFIG)

        assert isinstance(model, UniversalTransformerModel)
        assert model.d_model == 512
        assert model.n_layers == 2
        assert model.n_loops == 6
        assert model.sequential_depth == 12

        param_count = model.count_parameters()
        assert 500_000 < param_count < 10_000_000

    def test_all_ut_models_same_vocab_size(self) -> None:
        """Test that all UT model configs use correct vocab size."""
        models = [
            create_model_from_config(UT_SMALL_CONFIG),
            create_model_from_config(UT_MEDIUM_CONFIG),
            create_model_from_config(UT_LARGE_CONFIG),
        ]

        for model in models:
            assert model.token_embedding.num_embeddings == VOCAB_SIZE
            assert model.lm_head.out_features == VOCAB_SIZE


class TestCreateModelWithArchitecture:
    """Test create_model_from_config function with different architectures."""

    def test_create_standard_model(self) -> None:
        """Test creating standard model via create_model_from_config."""
        model = create_model_from_config(STANDARD_SMALL_CONFIG)
        assert model.architecture == "standard"
        assert not isinstance(model, UniversalTransformerModel)

    def test_create_universal_model(self) -> None:
        """Test creating universal model via create_model_from_config."""
        model = create_model_from_config(UT_SMALL_CONFIG)
        assert model.architecture == "universal"
        assert isinstance(model, UniversalTransformerModel)

    def test_universal_requires_n_loops(self) -> None:
        """Test that universal architecture requires n_loops."""
        with pytest.raises(ValueError, match="requires n_loops"):
            ModelConfig(
                architecture="universal",
                d_model=256,
                n_layers=2,
                n_heads=4,
                d_ff=512,
            )


class TestUniversalTransformerForward:
    """Test Universal Transformer forward pass."""

    def test_forward_pass_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        model = create_model_from_config(UT_SMALL_CONFIG)
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        output = model(input_ids)

        assert output.shape == (batch_size, seq_len, VOCAB_SIZE)

    def test_forward_with_labels(self) -> None:
        """Test forward pass with labels returns loss and logits."""
        model = create_model_from_config(UT_SMALL_CONFIG)
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        labels = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

        output = model(input_ids, labels=labels)

        assert isinstance(output, dict)
        assert "loss" in output
        assert "logits" in output
        assert output["logits"].shape == (batch_size, seq_len, VOCAB_SIZE)
        assert output["loss"].dim() == 0  # Scalar loss

    def test_forward_without_loop_embeddings(self) -> None:
        """Test forward pass works without loop embeddings."""
        model = UniversalTransformerModel(
            d_model=64,
            n_layers=2,
            n_loops=3,
            n_heads=4,
            d_ff=128,
            use_loop_embeddings=False,
        )

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 5))
        output = model(input_ids)

        assert output.shape == (1, 5, VOCAB_SIZE)


class TestUniversalTransformerWeightSharing:
    """Test weight sharing property of Universal Transformer."""

    def test_layers_are_reused_across_loops(self) -> None:
        """Verify that the same layer weights are used across loops."""
        model = UniversalTransformerModel(
            d_model=64,
            n_layers=2,
            n_loops=4,
            n_heads=4,
            d_ff=128,
        )

        # Only 2 unique layers should exist
        assert len(model.layers) == 2

        # But effective depth is 8
        assert model.sequential_depth == 8

    def test_fewer_params_than_equivalent_standard(self) -> None:
        """Test UT has fewer params than standard with same depth."""
        from src.model import ArithmeticModel

        # Standard: 8 layers
        standard = ArithmeticModel(
            d_model=256,
            n_layers=8,
            n_heads=4,
            d_ff=512,
        )

        # UT: 2 layers x 4 loops = 8 depth
        ut = UniversalTransformerModel(
            d_model=256,
            n_layers=2,
            n_loops=4,
            n_heads=4,
            d_ff=512,
        )

        standard_params = standard.count_parameters()
        ut_params = ut.count_parameters()

        # UT should have significantly fewer parameters
        # (roughly 1/4 for transformer blocks, plus some shared overhead)
        assert ut_params < standard_params * 0.5


class TestUniversalTransformerIntegration:
    """Integration tests with tokenizer."""

    def test_model_tokenizer_compatibility(self) -> None:
        """Test UT model works correctly with tokenizer."""
        model = create_model_from_config(UT_SMALL_CONFIG)

        expressions = ["1+2=", "9+8=", "0+0="]

        for expr in expressions:
            tokens = tokenizer.encode(expr)
            input_ids = torch.tensor([tokens])

            # Should not raise errors
            logits = model(input_ids)
            assert logits.shape == (1, len(tokens), VOCAB_SIZE)

            # Test generation
            generated = model.generate(
                input_ids, max_new_tokens=3, end_token_id=VOCAB["<end>"]
            )

            # Should be decodable
            decoded = tokenizer.decode(generated[0].tolist())
            assert isinstance(decoded, str)
            assert expr in decoded

    def test_batch_processing(self) -> None:
        """Test UT model with batch of inputs."""
        model = create_model_from_config(UT_SMALL_CONFIG)

        expressions = ["1+2=", "3+4=", "5+6="]
        tokens_list = [tokenizer.encode(expr) for expr in expressions]

        # Pad to same length
        max_len = max(len(tokens) for tokens in tokens_list)
        padded_tokens = []
        for tokens in tokens_list:
            padded = tokens + [0] * (max_len - len(tokens))
            padded_tokens.append(padded)

        input_ids = torch.tensor(padded_tokens)

        # Should handle batch correctly
        logits = model(input_ids)
        assert logits.shape == (3, max_len, VOCAB_SIZE)


class TestUniversalTransformerGeneration:
    """Test generation capabilities of Universal Transformer."""

    def test_generate_produces_valid_tokens(self) -> None:
        """Test generation produces tokens in valid range."""
        model = create_model_from_config(UT_SMALL_CONFIG)
        input_ids = torch.tensor([[1, 10, 2, 11]])  # "1+2="

        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=1.0,
            end_token_id=VOCAB["<end>"],
        )

        # All tokens should be in valid range
        assert (generated >= 0).all()
        assert (generated < VOCAB_SIZE).all()

        # Output should be longer than input
        assert generated.shape[1] >= input_ids.shape[1]

    def test_generate_stops_at_end_token(self) -> None:
        """Test generation stops at end token."""
        model = create_model_from_config(UT_SMALL_CONFIG)
        # Use trained model behavior - just verify end token handling
        input_ids = torch.tensor([[1, 10, 2, 11]])  # "1+2="

        # With very low temperature, generation should be deterministic
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.01,
            end_token_id=VOCAB["<end>"],
        )

        # If end token was generated, sequence should end there
        # (model might not generate end token if untrained, that's ok)
        end_positions = (generated == VOCAB["<end>"]).nonzero()
        if len(end_positions) > 0:
            first_end = end_positions[0, 1].item()
            # No tokens after end token
            assert generated.shape[1] == first_end + 1
