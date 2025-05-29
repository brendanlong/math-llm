"""Unit tests for model architecture."""

import torch

from src.model import (
    MAX_SEQUENCE_LENGTH,
    ArithmeticModel,
    PositionalEncoding,
    TransformerBlock,
    compute_loss,
    create_cot_mask,
    create_large_model,
    create_medium_model,
    create_small_model,
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

    def test_loss_zero_with_perfect_match(self):
        """Test that loss is zero when logits perfectly predict the target sequence."""
        tokenizer = ArithmeticTokenizer()
        text = "1+2=3<end>"
        tokens = torch.tensor([tokenizer.encode(text)])

        vocab_size = len(tokenizer.vocab)
        seq_len = tokens.shape[1]
        batch_size = tokens.shape[0]

        # Create perfect logits: very high for correct tokens, very low for others
        perfect_logits = torch.full((batch_size, seq_len, vocab_size), -1000.0)

        # For next-token prediction: logits[i] should predict tokens[i+1]
        for batch_idx in range(batch_size):
            for pos in range(seq_len - 1):  # -1 because we predict next token
                correct_next_token = tokens[batch_idx, pos + 1]
                perfect_logits[batch_idx, pos, correct_next_token] = 1000.0

        loss = compute_loss(perfect_logits, tokens)

        # Should be essentially zero (allowing for floating point precision)
        assert loss.item() < 1e-6

    def test_cot_agnostic_loss_ignores_reasoning_differences(self):
        """Test that CoT-agnostic loss is ~0 for sequences with same answer but different reasoning."""
        tokenizer = ArithmeticTokenizer()

        # Two sequences with same answer but different CoT content
        text1 = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"
        text2 = "12+34=<think_digit>\n4+2=6\n3+1=4</think_digit>46<end>"

        tokens1 = torch.tensor([tokenizer.encode(text1)])
        tokens2 = torch.tensor([tokenizer.encode(text2)])

        vocab_size = len(tokenizer.vocab)

        # Create perfect logits for both sequences
        def create_perfect_logits(tokens: torch.Tensor):
            seq_len = tokens.shape[1]
            perfect_logits = torch.full((1, seq_len, vocab_size), -1000.0)

            for pos in range(seq_len - 1):
                correct_next_token = tokens[0, pos + 1]
                perfect_logits[0, pos, correct_next_token] = 1000.0

            return perfect_logits

        perfect_logits1 = create_perfect_logits(tokens1)
        perfect_logits2 = create_perfect_logits(tokens2)

        # Additional test: cross-sequence loss with CoT masking
        # Logits from sequence 1 should have low loss on sequence 2 when CoT is masked
        # and vice-versa
        cross_loss = compute_loss(perfect_logits1, tokens2, cot_agnostic=True)
        assert cross_loss.item() < 1e-6  # Should be low since non-CoT parts match
        cross_loss = compute_loss(perfect_logits2, tokens1, cot_agnostic=True)
        assert cross_loss.item() < 1e-6

    def test_cot_mask_creation(self):
        """Test that CoT mask correctly identifies content to keep vs mask."""

        tokenizer = ArithmeticTokenizer()

        # Test sequence with CoT content
        text = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"
        tokens = torch.tensor([tokenizer.encode(text)])

        mask = create_cot_mask(tokens)

        # Verify mask shape
        assert mask.shape == tokens.shape

        # Find positions of CoT tags
        think_open_pos = None
        think_close_pos = None
        for i, token in enumerate(tokens[0]):
            if token.item() == tokenizer.vocab["<think_digit>"]:
                think_open_pos = i
            elif token.item() == tokenizer.vocab["</think_digit>"]:
                think_close_pos = i
                break

        assert think_open_pos is not None
        assert think_close_pos is not None

        # Check that opening and closing tags are kept (True)
        assert mask[0, think_open_pos]
        assert mask[0, think_close_pos]

        # Check that content between tags is masked (False)
        for i in range(think_open_pos + 1, think_close_pos):
            assert not mask[0, i]

        # Check that content outside CoT is kept
        for i in range(0, think_open_pos):
            assert mask[0, i]
        for i in range(think_close_pos + 1, len(tokens[0])):
            assert mask[0, i]

    def test_cot_agnostic_vs_regular_loss_difference(self):
        """Test that CoT-agnostic loss differs from regular loss when CoT content varies."""
        tokenizer = ArithmeticTokenizer()

        # Two sequences: same non-CoT parts, different CoT content
        text1 = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"
        text2 = "12+34=<think_digit>\n9+9=6\n7+7=4</think_digit>46<end>"

        tokens1 = torch.tensor([tokenizer.encode(text1)])
        tokens2 = torch.tensor([tokenizer.encode(text2)])

        # Ensure same length for this test
        assert tokens1.shape[1] == tokens2.shape[1], (
            f"Lengths differ: {tokens1.shape[1]} vs {tokens2.shape[1]}"
        )

        vocab_size = len(tokenizer.vocab)
        seq_len = tokens1.shape[1]

        # Create logits that are perfect for text1
        perfect_logits1 = torch.full((1, seq_len, vocab_size), -1000.0)
        for pos in range(seq_len - 1):
            correct_next_token = tokens1[0, pos + 1]
            perfect_logits1[0, pos, correct_next_token] = 1000.0
        perfect_logits1.requires_grad_(True)

        # Regular loss: should be high when predicting text2 with logits trained on text1
        regular_loss = compute_loss(perfect_logits1, tokens2, cot_agnostic=False)

        # CoT-agnostic loss: should be low since non-CoT parts match
        cot_agnostic_loss = compute_loss(perfect_logits1, tokens2, cot_agnostic=True)

        # CoT-agnostic loss should be much lower than regular loss
        assert cot_agnostic_loss.item() < regular_loss.item()
        assert cot_agnostic_loss.item() < 1e-6  # Should be near zero
        assert regular_loss.item() > 1.0  # Should be substantial

    def test_sequence_length_mismatch_handling(self):
        """Test that loss computation handles different sequence lengths correctly."""
        tokenizer = ArithmeticTokenizer()

        # Create sequences of different lengths
        short_text = "1+2=3<end>"
        long_text = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"

        short_tokens = torch.tensor([tokenizer.encode(short_text)])
        long_tokens = torch.tensor([tokenizer.encode(long_text)])

        vocab_size = len(tokenizer.vocab)

        # Create logits for longer sequence with gradient tracking
        long_seq_len = long_tokens.shape[1]
        logits = torch.randn(1, long_seq_len, vocab_size, requires_grad=True)

        # Test with shorter labels - should not crash
        loss1 = compute_loss(logits, short_tokens)
        assert isinstance(loss1, torch.Tensor)
        assert loss1.requires_grad

        # Test with CoT-agnostic mode
        loss2 = compute_loss(logits, short_tokens, cot_agnostic=True)
        assert isinstance(loss2, torch.Tensor)
        assert loss2.requires_grad

    def test_next_token_prediction_alignment(self):
        """Test that the shifting for next-token prediction is correct."""
        tokenizer = ArithmeticTokenizer()
        text = "1+2=3<end>"
        tokens = torch.tensor([tokenizer.encode(text)])

        vocab_size = len(tokenizer.vocab)
        seq_len = tokens.shape[1]

        # Create logits where each position predicts a specific wrong token
        # except position i predicts the correct token i+1
        wrong_token_id = 0  # Use token '0' as wrong prediction
        logits = torch.full((1, seq_len, vocab_size), -1000.0)

        # Make wrong token very likely everywhere
        logits[:, :, wrong_token_id] = 1000.0

        # Now make correct next-token predictions even more likely
        for pos in range(seq_len - 1):
            correct_next_token = tokens[0, pos + 1]
            logits[0, pos, correct_next_token] = 2000.0  # Higher than wrong token

        loss = compute_loss(logits, tokens)

        # Loss should be low since we predict next tokens correctly
        assert loss.item() < 1e-6

    def test_empty_cot_mask_handling(self):
        """Test loss computation when CoT mask results in very few valid tokens."""
        tokenizer = ArithmeticTokenizer()

        # Create a sequence that's mostly CoT content
        # The issue is that after shifting for next-token prediction,
        # the mask still keeps the closing tag, so it's not actually empty
        think_open = tokenizer.vocab["<think_digit>"]
        think_close = tokenizer.vocab["</think_digit>"]
        digit_2 = tokenizer.vocab["2"]

        # Sequence: <think_digit>2</think_digit>
        tokens = torch.tensor([[think_open, digit_2, think_close]])

        vocab_size = len(tokenizer.vocab)
        seq_len = tokens.shape[1]
        logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)

        # Should handle case gracefully
        loss = compute_loss(logits, tokens, cot_agnostic=True)

        # Should return a valid tensor (not necessarily zero since closing tag is kept)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_multi_cot_blocks_masking(self):
        """Test masking works correctly with multiple CoT blocks."""
        tokenizer = ArithmeticTokenizer()

        # Create sequence with both think_digit and think_multi blocks
        text = "12+34+56=<think_multi>\n<think_digit>\n2+4=6\n</think_digit>\n<think_digit>\n1+3+5=9\n</think_digit>\n</think_multi>102<end>"
        tokens = torch.tensor([tokenizer.encode(text)])

        vocab_size = len(tokenizer.vocab)
        seq_len = tokens.shape[1]

        # Create perfect logits
        perfect_logits = torch.full((1, seq_len, vocab_size), -1000.0)
        for pos in range(seq_len - 1):
            correct_next_token = tokens[0, pos + 1]
            perfect_logits[0, pos, correct_next_token] = 1000.0

        # CoT-agnostic loss should be low (only final answer matters)
        loss = compute_loss(perfect_logits, tokens, cot_agnostic=True)
        assert loss.item() < 1e-6
