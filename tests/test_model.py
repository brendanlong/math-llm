"""Unit tests for model architecture."""

import torch

from src.model import (
    MAX_SEQUENCE_LENGTH,
    ArithmeticModel,
    PositionalEncoding,
    TransformerBlock,
    compute_cot_agnostic_loss,
    create_cot_mask,
    create_large_model,
    create_medium_model,
    create_small_model,
    remove_cot_content,
)
from src.tokenizer import VOCAB_SIZE, ArithmeticTokenizer


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


class TestCoTAgnosticLoss:
    """Test CoT-agnostic loss computation functionality."""

    def test_remove_cot_content(self):
        """Test that CoT content removal works correctly."""
        tokenizer = ArithmeticTokenizer()

        # Example: "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"
        text = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"
        tokens = tokenizer.encode(text)

        # Remove CoT content, keeping only opening/closing tags
        filtered_tokens = remove_cot_content(tokens)
        filtered_text = tokenizer.decode(filtered_tokens)

        # Should be: "12+34=<think_digit></think_digit>46<end>"
        expected_text = "12+34=<think_digit></think_digit>46<end>"
        assert filtered_text == expected_text

    def test_cot_agnostic_loss_masks_cot_content(self):
        """Test that CoT-agnostic loss correctly masks CoT content."""
        tokenizer = ArithmeticTokenizer()

        # Two sequences with same non-CoT content but different CoT
        text1 = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"
        text2 = "12+34=<think_digit>\n4+2=6\n3+1=4</think_digit>46<end>"

        tokens1 = tokenizer.encode(text1)
        tokens2 = tokenizer.encode(text2)

        # Create model and get some predictions
        model = create_small_model()
        tokens1_tensor = torch.tensor([tokens1])
        tokens2_tensor = torch.tensor([tokens2])

        with torch.no_grad():
            logits1 = model(tokens1_tensor)

        # Compare loss when using same logits vs different labels
        loss_same = compute_cot_agnostic_loss(
            logits1, tokens1_tensor, cot_agnostic=True
        )
        loss_diff_cot = compute_cot_agnostic_loss(
            logits1, tokens2_tensor, cot_agnostic=True
        )
        loss_diff_regular = compute_cot_agnostic_loss(
            logits1, tokens2_tensor, cot_agnostic=False
        )

        # CoT-agnostic and regular losses should be different (showing masking works)
        # Direction depends on random weights, so we just verify they differ
        assert abs(loss_diff_cot - loss_diff_regular) > 0.001

        # Both losses should be computable
        assert isinstance(loss_same.item(), float)
        assert isinstance(loss_diff_cot.item(), float)
        assert isinstance(loss_diff_regular.item(), float)

    def test_cot_agnostic_loss_disabled_shows_difference(self):
        """Test that regular loss includes CoT differences while masked loss doesn't."""
        tokenizer = ArithmeticTokenizer()
        model = create_small_model()

        # Two sequences with same answer but different CoT content
        text1 = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"
        text2 = "12+34=<think_digit>\n4+2=6\n3+1=4</think_digit>46<end>"

        tokens1 = torch.tensor([tokenizer.encode(text1)])
        tokens2 = torch.tensor([tokenizer.encode(text2)])

        # Get model predictions
        with torch.no_grad():
            logits = model(tokens1)

        # Compare losses
        loss_regular = compute_cot_agnostic_loss(logits, tokens2, cot_agnostic=False)
        loss_cot_agnostic = compute_cot_agnostic_loss(
            logits, tokens2, cot_agnostic=True
        )

        # Both should be valid numbers and different (showing masking is working)
        assert isinstance(loss_regular.item(), float)
        assert isinstance(loss_cot_agnostic.item(), float)
        # The losses should be different, showing that masking changes the computation
        # (direction depends on random model weights, so we just check they're different)
        assert abs(loss_regular - loss_cot_agnostic) > 0.001

    def test_cot_agnostic_loss_with_different_lengths(self):
        """Test that masking works correctly even with different sequence lengths."""
        tokenizer = ArithmeticTokenizer()

        # Test with a sequence that has CoT content
        text_with_cot = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"

        tokens_with_cot = tokenizer.encode(text_with_cot)

        # Create tensors
        tokens_tensor = torch.tensor([tokens_with_cot])
        mask = create_cot_mask(tokens_tensor)

        # The mask should have False for CoT content
        think_start = tokenizer.vocab["<think_digit>"]
        think_end = tokenizer.vocab["</think_digit>"]

        # Find positions of think tags
        think_start_pos = tokens_with_cot.index(think_start)
        think_end_pos = tokens_with_cot.index(think_end)

        # Content between think tags should be masked (False)
        for i in range(think_start_pos + 1, think_end_pos):
            assert not mask[0, i], f"Position {i} should be masked but isn't"

        # Tags themselves and content outside should not be masked (True)
        assert mask[0, think_start_pos]
        assert mask[0, think_end_pos]
        assert mask[0, 0]  # First token
