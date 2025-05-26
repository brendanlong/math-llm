"""Unit tests for model architecture."""

import torch

from src.model import (
    ArithmeticModel,
    PositionalEncoding,
    TransformerBlock,
    create_large_model,
    create_medium_model,
    create_small_model,
)
from src.tokenizer import MAX_SEQUENCE_LENGTH, VOCAB_SIZE, ArithmeticTokenizer


class TestPositionalEncoding:
    """Test positional encoding module."""

    def test_init(self):
        """Test positional encoding initialization."""
        pe = PositionalEncoding(d_model=256, max_len=32)
        assert pe.pe.shape == (32, 1, 256)

    def test_forward(self):
        """Test positional encoding forward pass."""
        pe = PositionalEncoding(d_model=256, max_len=32)
        x = torch.randn(10, 2, 256)  # (seq_len, batch_size, d_model)

        output = pe(x)
        assert output.shape == x.shape

        # Check that positional encoding is added
        assert not torch.equal(output, x)


class TestTransformerBlock:
    """Test transformer block module."""

    def test_init(self):
        """Test transformer block initialization."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)
        assert block.self_attn.embed_dim == 256
        assert block.self_attn.num_heads == 4

    def test_forward_shape(self):
        """Test transformer block forward pass shapes."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)
        x = torch.randn(2, 10, 256)  # (batch_size, seq_len, d_model)

        output = block(x)
        assert output.shape == x.shape

    def test_forward_with_mask(self):
        """Test transformer block with attention mask."""
        block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)
        x = torch.randn(2, 10, 256)
        mask = torch.triu(torch.ones(10, 10), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))

        output = block(x, mask=mask)
        assert output.shape == x.shape


class TestArithmeticModel:
    """Test main arithmetic model."""

    def test_init_default(self):
        """Test model initialization with default parameters."""

        model = ArithmeticModel()
        assert model.d_model == 256
        assert model.max_seq_len == MAX_SEQUENCE_LENGTH
        assert model.token_embedding.num_embeddings == VOCAB_SIZE
        assert len(model.layers) == 4

    def test_init_custom(self):
        """Test model initialization with custom parameters."""
        model = ArithmeticModel(
            vocab_size=20, d_model=512, n_layers=6, n_heads=8, max_seq_len=64
        )
        assert model.d_model == 512
        assert model.max_seq_len == 64
        assert len(model.layers) == 6
        assert model.token_embedding.num_embeddings == 20

    def test_forward_shape(self):
        """Test model forward pass output shapes."""
        model = ArithmeticModel(vocab_size=13, d_model=256)
        input_ids = torch.randint(0, 13, (2, 10))  # (batch_size, seq_len)

        logits = model(input_ids)
        assert logits.shape == (2, 10, 13)

    def test_forward_with_tokenizer(self):
        """Test model forward pass with real tokenizer input."""
        tokenizer = ArithmeticTokenizer()
        model = ArithmeticModel(vocab_size=tokenizer.vocab_size)

        text = "3+5=8<end>"
        input_ids = torch.tensor([tokenizer.encode(text)])

        logits = model(input_ids)
        expected_shape = (1, len(tokenizer.encode(text)), tokenizer.vocab_size)
        assert logits.shape == expected_shape

    def test_causal_mask(self):
        """Test causal mask generation."""
        model = ArithmeticModel()
        mask = model._get_causal_mask(5)  # type: ignore[reportPrivateUsage]

        # Check shape
        assert mask.shape == (5, 5)

        # Check that it's upper triangular with -inf above diagonal
        expected = torch.tensor(
            [
                [0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                [0.0, 0.0, float("-inf"), float("-inf"), float("-inf")],
                [0.0, 0.0, 0.0, float("-inf"), float("-inf")],
                [0.0, 0.0, 0.0, 0.0, float("-inf")],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert torch.equal(mask, expected)

    def test_generate_shape(self):
        """Test generation output shape."""
        model = ArithmeticModel(vocab_size=13)
        input_ids = torch.tensor([[3, 10, 5, 11]])  # "3+5="

        generated = model.generate(input_ids, max_new_tokens=5, temperature=1.0)

        # Should have original tokens plus up to 5 new ones
        assert generated.shape[0] == 1
        assert generated.shape[1] >= input_ids.shape[1]
        assert generated.shape[1] <= input_ids.shape[1] + 5

    def test_generate_ends_with_end_token(self):
        """Test that generation can end with end token."""
        model = ArithmeticModel(vocab_size=13)
        input_ids = torch.tensor([[3, 10, 5, 11]])  # "3+5="

        # Set temperature very low to make generation more deterministic
        generated = model.generate(
            input_ids, max_new_tokens=10, temperature=0.1, end_token_id=12
        )

        # Check that sequence is longer than input
        assert generated.shape[1] > input_ids.shape[1]

    def test_count_parameters(self):
        """Test parameter counting."""
        model = ArithmeticModel(vocab_size=13, d_model=128, n_layers=2)
        param_count = model.count_parameters()

        # Should be > 0 and reasonable for a small model
        assert param_count > 0
        assert param_count < 10_000_000  # Less than 10M parameters


class TestModelFactories:
    """Test model factory functions."""

    def test_create_small_model(self):
        """Test small model creation."""
        model = create_small_model()

        assert isinstance(model, ArithmeticModel)
        assert model.d_model == 256
        assert len(model.layers) == 4

        # Should be around 1-3M parameters
        param_count = model.count_parameters()
        assert 500_000 < param_count < 5_000_000

    def test_create_medium_model(self):
        """Test medium model creation."""
        model = create_medium_model()

        assert isinstance(model, ArithmeticModel)
        assert model.d_model == 512
        assert len(model.layers) == 6

        # Should be around 5-15M parameters
        param_count = model.count_parameters()
        assert 3_000_000 < param_count < 20_000_000

    def test_create_large_model(self):
        """Test large model creation."""
        model = create_large_model()

        assert isinstance(model, ArithmeticModel)
        assert model.d_model == 512
        assert len(model.layers) == 8

        # Should be around 10-30M parameters
        param_count = model.count_parameters()
        assert 10_000_000 < param_count < 50_000_000

    def test_all_models_same_vocab_size(self):
        """Test that all model factories use correct vocab size."""
        from src.tokenizer import VOCAB_SIZE

        models = [create_small_model(), create_medium_model(), create_large_model()]

        for model in models:
            assert model.token_embedding.num_embeddings == VOCAB_SIZE
            assert model.lm_head.out_features == VOCAB_SIZE


class TestModelIntegration:
    """Integration tests with tokenizer."""

    def test_model_tokenizer_compatibility(self):
        """Test model works correctly with tokenizer."""
        tokenizer = ArithmeticTokenizer()
        model = create_small_model()

        # Test expressions
        expressions = ["1+2=", "9+8=", "0+0="]

        for expr in expressions:
            tokens = tokenizer.encode(expr)
            input_ids = torch.tensor([tokens])

            # Should not raise errors
            logits = model(input_ids)
            assert logits.shape == (1, len(tokens), tokenizer.vocab_size)

            # Test generation
            generated = model.generate(
                input_ids, max_new_tokens=3, end_token_id=tokenizer.end_token_id
            )

            # Should be decodable
            decoded = tokenizer.decode(generated[0].tolist())
            assert isinstance(decoded, str)
            assert expr in decoded

    def test_batch_processing(self):
        """Test model with batch of inputs."""
        tokenizer = ArithmeticTokenizer()
        model = create_small_model()

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
        assert logits.shape == (3, max_len, tokenizer.vocab_size)
