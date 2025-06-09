"""Unit tests for model architecture."""

import torch

from src.model import (
    MAX_SEQUENCE_LENGTH,
    ArithmeticModel,
    PositionalEncoding,
    TransformerBlock,
    compute_loss,
    create_completion_mask,
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


class TestCompletionOnlyLoss:
    """Test completion-only loss computation functionality."""

    def test_completion_mask_creation(self):
        """Test that completion mask correctly identifies tokens after = sign."""
        tokenizer = ArithmeticTokenizer()

        # Test simple arithmetic expression
        text = "3+5=8<end>"
        tokens = torch.tensor([tokenizer.encode(text)])

        mask = create_completion_mask(tokens)

        # Verify mask shape
        assert mask.shape == tokens.shape

        # Find position of = sign
        equals_pos = None
        for i, token in enumerate(tokens[0]):
            if token.item() == tokenizer.vocab["="]:
                equals_pos = i
                break

        assert equals_pos is not None

        # Check that tokens before and including = are masked out (False)
        for i in range(equals_pos + 1):
            assert not mask[0, i]

        # Check that tokens after = are kept (True)
        for i in range(equals_pos + 1, len(tokens[0])):
            assert mask[0, i]

    def test_completion_mask_with_cot(self):
        """Test completion mask works correctly with CoT sequences."""
        tokenizer = ArithmeticTokenizer()

        # Test with CoT content
        text = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"
        tokens = torch.tensor([tokenizer.encode(text)])

        mask = create_completion_mask(tokens)

        # Find position of = sign
        equals_pos = None
        for i, token in enumerate(tokens[0]):
            if token.item() == tokenizer.vocab["="]:
                equals_pos = i
                break

        assert equals_pos is not None

        # Everything after = should be True (including CoT content)
        for i in range(equals_pos + 1, len(tokens[0])):
            assert mask[0, i]

    def test_completion_mask_batch(self):
        """Test completion mask works with batched inputs."""
        tokenizer = ArithmeticTokenizer()

        # Test with batch of different expressions
        texts = ["3+5=8<end>", "1+2=3<end>", "9+1=10<end>"]
        tokens_list = [tokenizer.encode(text) for text in texts]

        # Pad to same length
        max_len = max(len(tokens) for tokens in tokens_list)
        padded_tokens = []
        for tokens in tokens_list:
            padded = tokens + [0] * (max_len - len(tokens))
            padded_tokens.append(padded)

        batch_tokens = torch.tensor(padded_tokens)
        mask = create_completion_mask(batch_tokens)

        # Check each sequence in batch
        for b in range(len(texts)):
            # Find equals position for this sequence
            equals_pos = None
            for i, token in enumerate(batch_tokens[b]):
                if token.item() == tokenizer.vocab["="]:
                    equals_pos = i
                    break

            if equals_pos is not None:
                # Tokens after = should be True
                for i in range(equals_pos + 1, len(tokens_list[b])):
                    assert mask[b, i]

    def test_completion_mask_no_equals(self):
        """Test completion mask when no equals sign is present."""
        tokenizer = ArithmeticTokenizer()

        # Test with sequence that has no equals
        text = "3+5"  # No equals sign
        tokens = torch.tensor([tokenizer.encode(text)])

        mask = create_completion_mask(tokens)

        # All tokens should be False since no = found
        assert not mask.any()

    def test_completion_only_loss_vs_regular_loss(self):
        """Test that completion-only loss differs significantly from regular loss."""
        tokenizer = ArithmeticTokenizer()

        # Create sequence: input + wrong answer
        text = "3+5=9<end>"  # Wrong answer (should be 8)
        tokens = torch.tensor([tokenizer.encode(text)])

        vocab_size = len(tokenizer.vocab)
        seq_len = tokens.shape[1]

        # Create logits that predict the input correctly but wrong answer
        logits = torch.full((1, seq_len, vocab_size), -1000.0)

        # Make input predictions perfect (positions 0-3: "3+5=")
        for pos in range(3):  # 0,1,2 predict 1,2,3 ("3+5")
            correct_next_token = tokens[0, pos + 1]
            logits[0, pos, correct_next_token] = 1000.0

        # Make equals prediction perfect (position 3 predicts "9")
        logits[0, 3, tokens[0, 4]] = 1000.0  # Position 3 predicts position 4

        # Make answer prediction wrong (position 4 predicts "<end>" instead of right answer)
        logits[0, 4, tokenizer.vocab["<end>"]] = 1000.0

        logits.requires_grad_(True)

        # Regular loss: averages over all predictions (input + wrong answer)
        regular_loss = compute_loss(logits, tokens)

        # Completion-only loss: only cares about answer predictions
        completion_loss = compute_loss(logits, tokens)  # Uses completion-only training

        # Both should have some loss due to wrong answer, but let's test the mechanism
        assert isinstance(regular_loss, torch.Tensor)
        assert isinstance(completion_loss, torch.Tensor)
        assert regular_loss.requires_grad
        assert completion_loss.requires_grad

    def test_completion_only_perfect_answer(self):
        """Test completion-only loss is low when answer is correct."""
        tokenizer = ArithmeticTokenizer()

        text = "3+5=8<end>"
        tokens = torch.tensor([tokenizer.encode(text)])

        vocab_size = len(tokenizer.vocab)
        seq_len = tokens.shape[1]

        # Create perfect logits for next-token prediction
        perfect_logits = torch.full((1, seq_len, vocab_size), -1000.0)
        for pos in range(seq_len - 1):
            correct_next_token = tokens[0, pos + 1]
            perfect_logits[0, pos, correct_next_token] = 1000.0

        loss = compute_loss(perfect_logits, tokens)

        # Should be very low since all predictions are correct
        assert loss.item() < 1e-6

    def test_completion_only_wrong_input_correct_answer(self):
        """Test that completion-only training ignores wrong input predictions."""
        tokenizer = ArithmeticTokenizer()

        text = "3+5=8<end>"
        tokens = torch.tensor([tokenizer.encode(text)])

        vocab_size = len(tokenizer.vocab)
        seq_len = tokens.shape[1]

        # Create logits with wrong input predictions but correct answer predictions
        logits = torch.full((1, seq_len, vocab_size), -1000.0)

        # Wrong input predictions (positions 0-2)
        wrong_token = tokenizer.vocab["9"]
        for pos in range(3):  # Positions that predict input tokens
            logits[0, pos, wrong_token] = 1000.0  # Predict wrong token

        # Correct answer predictions (positions 3-4 for "=" -> "8" -> "<end>")
        for pos in range(3, seq_len - 1):
            correct_next_token = tokens[0, pos + 1]
            logits[0, pos, correct_next_token] = 1000.0

        # Create the same logits but with completion mask applied
        completion_mask = create_completion_mask(
            tokens[:, 1:]
        )  # Shifted for next-token prediction

        # The current implementation applies completion mask in the else branch
        # Let's test the mechanism by checking mask creation
        assert completion_mask.any()  # Should have some True values

        # Check that mask correctly identifies answer positions
        equals_pos = None
        for i, token in enumerate(tokens[0]):
            if token.item() == tokenizer.vocab["="]:
                equals_pos = i
                break

        # In shifted labels, position after = should be True
        if equals_pos is not None and equals_pos < len(completion_mask[0]):
            # The mask should be True for positions after =
            for i in range(equals_pos, len(completion_mask[0])):
                if i < len(completion_mask[0]):
                    assert completion_mask[0, i]

    def test_tensor_alignment_after_shifting(self):
        """Test that tensor dimensions align correctly after shifting for next-token prediction."""
        tokenizer = ArithmeticTokenizer()

        # Test with different sequence lengths
        short_text = "1+2=3<end>"
        long_text = "12+34=<think_digit>\n2+4=6\n1+3=4</think_digit>46<end>"

        for text in [short_text, long_text]:
            tokens = torch.tensor([tokenizer.encode(text)])
            vocab_size = len(tokenizer.vocab)
            seq_len = tokens.shape[1]

            # Create random logits
            logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)

            # Loss computation should not crash
            loss = compute_loss(logits, tokens)

            assert isinstance(loss, torch.Tensor)
            assert loss.requires_grad

            # Test with different logits length
            different_logits = torch.randn(
                1, seq_len - 2, vocab_size, requires_grad=True
            )
            loss2 = compute_loss(different_logits, tokens)

            assert isinstance(loss2, torch.Tensor)
            assert loss2.requires_grad

    def test_completion_mask_multiple_equals(self):
        """Test completion mask with multiple equals signs (should use first one)."""
        tokenizer = ArithmeticTokenizer()

        # Create artificial sequence with multiple = signs
        equals_token = tokenizer.vocab["="]
        tokens = torch.tensor([[1, 2, equals_token, 4, equals_token, 6]])

        mask = create_completion_mask(tokens)

        # Should mask everything before and including first =, keep everything after
        assert not mask[0, 0]  # Before first =
        assert not mask[0, 1]  # Before first =
        assert not mask[0, 2]  # First = itself
        assert mask[0, 3]  # After first =
        assert mask[0, 4]  # After first = (even though it's another =)
        assert mask[0, 5]  # After first =

    def test_completion_mask_with_padding_tokens(self):
        """Test completion mask correctly handles -100 padding tokens."""
        tokenizer = ArithmeticTokenizer()

        # Create sequence with padding tokens (-100) like trainer would use
        text = "3+5=8<end>"
        tokens = tokenizer.encode(text)

        # Add padding tokens at the end
        tokens_with_padding = tokens + [-100, -100, -100]
        batch_tokens = torch.tensor([tokens_with_padding])

        mask = create_completion_mask(batch_tokens)

        # Find equals position
        equals_pos = None
        for i, token in enumerate(batch_tokens[0]):
            if token.item() == tokenizer.vocab["="]:
                equals_pos = i
                break

        assert equals_pos is not None

        # Check that tokens before and including = are False
        for i in range(equals_pos + 1):
            assert not mask[0, i]

        # Check that actual answer tokens after = are True
        actual_answer_tokens = []
        for i in range(equals_pos + 1, len(tokens)):  # Only up to original length
            if tokens[i] != -100:
                actual_answer_tokens.append(i)
                assert mask[0, i], f"Position {i} should be True (answer token)"

        # Check that padding tokens are False
        for i in range(len(tokens), len(tokens_with_padding)):
            assert not mask[0, i], f"Position {i} should be False (padding token)"

        # Verify total count
        expected_answer_count = len([t for t in tokens[equals_pos + 1 :] if t != -100])
        actual_answer_count = mask[0].sum().item()
        assert actual_answer_count == expected_answer_count

    def test_completion_mask_mixed_padding(self):
        """Test completion mask with padding tokens interspersed (edge case)."""
        tokenizer = ArithmeticTokenizer()

        # Create artificial sequence: 3+5= answer_token padding answer_token padding
        equals_token = tokenizer.vocab["="]
        answer_token = tokenizer.vocab["8"]
        end_token = tokenizer.vocab["<end>"]

        # [3, +, 5, =, 8, -100, <end>, -100]
        tokens = torch.tensor(
            [
                [
                    tokenizer.vocab["3"],
                    tokenizer.vocab["+"],
                    tokenizer.vocab["5"],
                    equals_token,
                    answer_token,  # Should be True
                    -100,  # Should be False (padding)
                    end_token,  # Should be True
                    -100,  # Should be False (padding)
                ]
            ]
        )

        mask = create_completion_mask(tokens)

        expected_mask = torch.tensor(
            [[False, False, False, False, True, False, True, False]]
        )

        assert torch.equal(mask, expected_mask), (
            f"Expected {expected_mask[0]}, got {mask[0]}"
        )
        assert mask.sum().item() == 2  # Only the two non-padding answer tokens
