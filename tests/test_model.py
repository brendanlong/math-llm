"""Unit tests for model architecture."""

import torch

from src.model import (
    MAX_SEQUENCE_LENGTH,
    ArithmeticModel,
    TransformerBlock,
    build_alibi_bias,
    create_large_model,
    create_medium_model,
    create_reasoning_mask,
    create_small_model,
)
from src.tokenizer import VOCAB, VOCAB_SIZE, tokenizer


class TestAliBi:
    """Test ALiBi bias computation."""

    def test_build_alibi_bias_basic(self):
        """Test basic ALiBi bias functionality."""
        n_heads = 4
        seq_len = 8
        device = torch.device("cpu")

        alibi_bias = build_alibi_bias(n_heads, seq_len, device)

        assert alibi_bias.shape == (n_heads, seq_len, seq_len)
        assert (alibi_bias <= 0).all()

        # Diagonal should be zero
        for h in range(n_heads):
            diagonal = torch.diagonal(alibi_bias[h])
            assert torch.allclose(diagonal, torch.zeros_like(diagonal))


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

    def test_forward_with_tokenizer(
        self,
    ):
        """Test model forward pass with real tokenizer input."""
        model = ArithmeticModel(vocab_size=VOCAB_SIZE)

        text = "3+5=8<end>"
        input_ids = torch.tensor([tokenizer.encode(text)])

        logits = model(input_ids)
        expected_shape = (1, len(tokenizer.encode(text)), VOCAB_SIZE)
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
        models = [create_small_model(), create_medium_model(), create_large_model()]

        for model in models:
            assert model.token_embedding.num_embeddings == VOCAB_SIZE
            assert model.lm_head.out_features == VOCAB_SIZE


class TestModelIntegration:
    """Integration tests with tokenizer."""

    def test_model_tokenizer_compatibility(
        self,
    ):
        """Test model works correctly with tokenizer."""
        model = create_small_model()

        # Test expressions
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

    def test_batch_processing(
        self,
    ):
        """Test model with batch of inputs."""
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
        assert logits.shape == (3, max_len, VOCAB_SIZE)


class TestReasoningMask:
    """Test reasoning mask functionality."""

    def test_visual_mask_verification(self):
        """Test reasoning mask with literal comparison for visual verification."""
        # Example: "1+2=<think>1+2=3</think>3<end>"
        text = "1+2=<think>1+2=3</think>3<end>"
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens])

        mask = create_reasoning_mask(input_ids)

        # Print for visual verification during development
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"Decoded tokens: {[tokenizer.decode([t]) for t in tokens]}")
        print(f"Mask: {mask[0].tolist()}")

        # Create expected mask manually
        # Tokens should be: [1, 10, 2, 11, 13, 1, 10, 2, 11, 3, 14, 3, 12]
        # Positions:        [0,  1, 2,  3,  4, 5,  6, 7,  8, 9, 10,11,12]
        # Expected mask:    [F,  F, F,  F,  F, T,  T, T,  T, T,  F, F, F]
        # Where:
        # - Position 4 is <think> (not masked)
        # - Positions 5-9 are "1+2=3" (masked)
        # - Position 10 is </think> (not masked)

        expected_mask = torch.tensor(
            [
                [
                    False,
                    False,
                    False,
                    False,
                    False,  # "1+2=<think>"
                    True,
                    True,
                    True,
                    True,
                    True,  # "1+2=3" (content between think tags)
                    False,
                    False,
                    False,
                ]  # "</think>3<end>"
            ]
        )

        assert torch.equal(mask, expected_mask), (
            f"Mask {mask[0].tolist()} != expected {expected_mask[0].tolist()}"
        )

    def test_basic_reasoning_mask(self):
        """Test basic reasoning mask with example from conversation."""
        # Example: "1+2=<think>1+2=3</think>3"
        text = "1+2=<think>1+2=3</think>3"
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens])

        mask = create_reasoning_mask(input_ids)

        # Check shape
        assert mask.shape == input_ids.shape

        # Check that <think> and </think> tokens are NOT masked
        think_positions = (input_ids == VOCAB["<think>"]).nonzero(as_tuple=True)
        think_end_positions = (input_ids == VOCAB["</think>"]).nonzero(as_tuple=True)

        for batch_idx, pos in zip(think_positions[0], think_positions[1]):
            assert not mask[batch_idx, pos].item(), "<think> token should not be masked"

        for batch_idx, pos in zip(think_end_positions[0], think_end_positions[1]):
            assert not mask[batch_idx, pos].item(), (
                "</think> token should not be masked"
            )

        # Check that content between think tags IS masked
        # Find the positions between <think> and </think>
        think_start = int(think_positions[1][0].item())
        think_end = int(think_end_positions[1][0].item())

        # Positions between the tags should be masked
        for pos in range(think_start + 1, think_end):
            assert mask[0, pos].item(), (
                f"Position {pos} between think tags should be masked"
            )

        # Positions outside think tags should NOT be masked
        for pos in range(0, think_start):
            assert not mask[0, pos].item(), (
                f"Position {pos} before think tags should not be masked"
            )
        for pos in range(think_end + 1, input_ids.shape[1]):
            assert not mask[0, pos].item(), (
                f"Position {pos} after think tags should not be masked"
            )

    def test_no_reasoning_tags(self):
        """Test mask when no reasoning tags are present."""
        text = "3+5=8<end>"
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens])

        mask = create_reasoning_mask(input_ids)

        # Should be all False (no masking)
        assert not mask.any().item(), (
            "No positions should be masked when no reasoning tags present"
        )

    def test_batch_reasoning_mask(self):
        """Test reasoning mask with batch of sequences."""
        texts = [
            "1+2=<think>1+2=3</think>3",  # Has reasoning
            "4+5=9<end>",  # No reasoning
            "6+7=<think>6+7=13</think>13",  # Has reasoning
        ]

        # Tokenize and pad to same length
        tokens_list = [tokenizer.encode(text) for text in texts]
        max_len = max(len(tokens) for tokens in tokens_list)

        padded_tokens = []
        for tokens in tokens_list:
            padded = tokens + [VOCAB["<end>"]] * (max_len - len(tokens))
            padded_tokens.append(padded)

        input_ids = torch.tensor(padded_tokens)
        mask = create_reasoning_mask(input_ids)

        # Check batch dimension
        assert mask.shape[0] == 3
        assert mask.shape[1] == max_len

        # First sequence should have some masked positions
        assert mask[0].any().item(), "First sequence should have masked positions"

        # Second sequence should have no masked positions
        assert not mask[1].any().item(), (
            "Second sequence should have no masked positions"
        )

        # Third sequence should have some masked positions
        assert mask[2].any().item(), "Third sequence should have masked positions"

    def test_edge_case_only_start_tag(self):
        """Test edge case with only <think> tag but no </think>."""
        text = "1+2=<think>1+3=4<end>"
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens])

        mask = create_reasoning_mask(input_ids)

        # Should be all False since no valid think pair
        assert not mask.any().item(), (
            "No positions should be masked with incomplete think tags"
        )

    def test_edge_case_only_end_tag(self):
        """Test edge case with only </think> tag but no <think>."""
        text = "1+2=4+5=9</think>3"
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens])

        mask = create_reasoning_mask(input_ids)

        # Should be all False since no valid think pair
        assert not mask.any().item(), (
            "No positions should be masked with incomplete think tags"
        )

    def test_empty_reasoning_block(self):
        """Test reasoning block with no content between tags."""
        text = "1+2=<think></think>3"
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens])

        mask = create_reasoning_mask(input_ids)

        # Tags should not be masked
        think_positions = (input_ids == VOCAB["<think>"]).nonzero(as_tuple=True)
        think_end_positions = (input_ids == VOCAB["</think>"]).nonzero(as_tuple=True)

        for batch_idx, pos in zip(think_positions[0], think_positions[1]):
            assert not mask[batch_idx, pos].item()

        for batch_idx, pos in zip(think_end_positions[0], think_end_positions[1]):
            assert not mask[batch_idx, pos].item()

        # Since tags are adjacent, no content between them to mask
        # Should have minimal masking
        think_start = int(think_positions[1][0].item())
        think_end = int(think_end_positions[1][0].item())

        # If tags are adjacent, no positions between them
        if think_end == think_start + 1:
            assert not mask.any().item(), "No content to mask between adjacent tags"
