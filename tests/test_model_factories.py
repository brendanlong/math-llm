"""Unit tests for model creation from config."""

from src.config import ModelConfig
from src.model import (
    ArithmeticModel,
    create_model_from_config,
)
from src.tokenizer import VOCAB_SIZE

# Test configs matching the original factory functions
SMALL_CONFIG = ModelConfig(
    architecture="standard",
    d_model=256,
    n_layers=4,
    n_heads=4,
    d_ff=512,
    dropout=0.1,
)

MEDIUM_CONFIG = ModelConfig(
    architecture="standard",
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_ff=1024,
    dropout=0.1,
)

LARGE_CONFIG = ModelConfig(
    architecture="standard",
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
)


class TestModelFromConfig:
    """Test model creation from config."""

    def test_create_small_model(self) -> None:
        """Test small model creation."""
        model = create_model_from_config(SMALL_CONFIG)

        assert isinstance(model, ArithmeticModel)
        assert model.d_model == 256
        assert len(model.layers) == 4

        # Should be around 1-3M parameters
        param_count = model.count_parameters()
        assert 500_000 < param_count < 5_000_000

    def test_create_medium_model(self) -> None:
        """Test medium model creation."""
        model = create_model_from_config(MEDIUM_CONFIG)

        assert isinstance(model, ArithmeticModel)
        assert model.d_model == 512
        assert len(model.layers) == 6

        # Should be around 5-15M parameters
        param_count = model.count_parameters()
        assert 3_000_000 < param_count < 20_000_000

    def test_create_large_model(self) -> None:
        """Test large model creation."""
        model = create_model_from_config(LARGE_CONFIG)

        assert isinstance(model, ArithmeticModel)
        assert model.d_model == 512
        assert len(model.layers) == 8

        # Should be around 10-30M parameters
        param_count = model.count_parameters()
        assert 10_000_000 < param_count < 50_000_000

    def test_all_models_same_vocab_size(self) -> None:
        """Test that all model configs use correct vocab size."""
        models = [
            create_model_from_config(SMALL_CONFIG),
            create_model_from_config(MEDIUM_CONFIG),
            create_model_from_config(LARGE_CONFIG),
        ]

        for model in models:
            assert model.token_embedding.num_embeddings == VOCAB_SIZE
            assert model.lm_head.out_features == VOCAB_SIZE
