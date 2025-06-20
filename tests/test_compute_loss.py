"""Unit tests for compute_loss function with reasoning masking."""

import torch

from src.model import compute_loss
from src.tokenizer import VOCAB


class TestComputeLoss:
    """Test compute_loss functionality with various scenarios."""

    def test_simple_loss_without_reasoning(self):
        """Test basic loss computation without reasoning tags."""
        # Example: "3+5=8<end>"
        # Tokens: [3, 10, 5, 11, 8, 12]
        # Labels predict next token, so for position i, we predict token at i+1

        batch_size = 1
        seq_len = 6
        vocab_size = len(VOCAB)

        # Create dummy logits - make them match labels perfectly for easy calculation
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[3, 10, 5, 11, 8, 12]])

        # Set logits to predict the correct next token with high confidence
        # Position 0 predicts token 10 (index 1)
        logits[0, 0, 10] = 10.0
        # Position 1 predicts token 5 (index 2)
        logits[0, 1, 5] = 10.0
        # Position 2 predicts token 11 (index 3)
        logits[0, 2, 11] = 10.0
        # Position 3 predicts token 8 (index 4)
        logits[0, 3, 8] = 10.0
        # Position 4 predicts token 12 (index 5)
        logits[0, 4, 12] = 10.0

        loss = compute_loss(logits, labels, mask_reasoning=False)

        # With perfect predictions and high confidence, loss should be very close to 0
        assert loss.item() < 0.001, f"Expected near-zero loss, got {loss.item()}"

    def test_all_correct_predictions(self):
        """Test that all correct predictions result in ~0 loss with or without masking."""
        # Example: "1+2=<think>calc</think>3<end>"
        # Tokens: [1, 10, 2, 11, 13, 15, 14, 3, 12]

        batch_size = 1
        seq_len = 9
        vocab_size = len(VOCAB)

        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 10, 2, 11, 13, 15, 14, 3, 12]])

        # Make all predictions correct
        # Position 0 -> 10
        logits[0, 0, 10] = 10.0
        # Position 1 -> 2
        logits[0, 1, 2] = 10.0
        # Position 2 -> 11
        logits[0, 2, 11] = 10.0
        # Position 3 -> 13 (<think>)
        logits[0, 3, 13] = 10.0
        # Position 4 -> 15 (content in reasoning)
        logits[0, 4, 15] = 10.0
        # Position 5 -> 14 (</think>)
        logits[0, 5, 14] = 10.0
        # Position 6 -> 3
        logits[0, 6, 3] = 10.0
        # Position 7 -> 12 (<end>)
        logits[0, 7, 12] = 10.0

        # Both should be near 0
        loss_no_mask = compute_loss(logits, labels, mask_reasoning=False)
        loss_with_mask = compute_loss(logits, labels, mask_reasoning=True)

        assert loss_no_mask.item() < 0.001, (
            f"Expected near-zero loss without mask, got {loss_no_mask.item()}"
        )
        assert loss_with_mask.item() < 0.001, (
            f"Expected near-zero loss with mask, got {loss_with_mask.item()}"
        )

    def test_wrong_reasoning_correct_elsewhere(self):
        """Test that wrong predictions in reasoning are ignored when masked."""
        # Example: "1+2=<think>XXXX</think>3<end>"
        # Tokens: [1, 10, 2, 11, 13, 5, 6, 7, 8, 14, 3, 12]
        # where 5,6,7,8 are wrong tokens in reasoning

        batch_size = 1
        seq_len = 12
        vocab_size = len(VOCAB)

        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 10, 2, 11, 13, 5, 6, 7, 8, 14, 3, 12]])

        # Make all predictions correct EXCEPT in reasoning section
        logits[0, 0, 10] = 10.0  # Correct (but before =, so ignored)
        logits[0, 1, 2] = 10.0  # Correct (but before =, so ignored)
        logits[0, 2, 11] = 10.0  # Correct (but before =, so ignored)
        logits[0, 3, 13] = 10.0  # Correct: <think>

        # Wrong predictions in reasoning (positions 4-7 predict wrong)
        logits[0, 4, 0] = 10.0  # Wrong: should be 6
        logits[0, 5, 0] = 10.0  # Wrong: should be 7
        logits[0, 6, 0] = 10.0  # Wrong: should be 8
        logits[0, 7, 0] = 10.0  # Wrong: should be 14

        # Correct predictions after reasoning
        logits[0, 8, 14] = 10.0  # Correct: </think>
        logits[0, 9, 3] = 10.0  # Correct: 3
        logits[0, 10, 12] = 10.0  # Correct: <end>

        loss_no_mask = compute_loss(logits, labels, mask_reasoning=False)
        loss_with_mask = compute_loss(logits, labels, mask_reasoning=True)

        # Without mask: Wrong predictions in reasoning contribute to loss
        # With mask: Those wrong predictions are ignored
        assert loss_with_mask.item() < loss_no_mask.item(), (
            f"Masked loss {loss_with_mask.item()} should be less than "
            f"unmasked loss {loss_no_mask.item()} when masking wrong predictions"
        )

        # With masking and all non-masked completion predictions correct,
        # loss should be near 0
        assert loss_with_mask.item() < 0.001, (
            f"Expected near-zero loss with masking, got {loss_with_mask.item()}"
        )

    def test_empty_reasoning_block_loss(self):
        """Test loss computation with empty reasoning block."""
        # "1+2=<think></think>3<end>"
        # Tokens: [1, 10, 2, 11, 13, 14, 3, 12]

        batch_size = 1
        seq_len = 8
        vocab_size = len(VOCAB)

        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 10, 2, 11, 13, 14, 3, 12]])

        # All correct predictions
        logits[0, 0, 10] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2, 11] = 10.0
        logits[0, 3, 13] = 10.0
        logits[0, 4, 14] = 10.0
        logits[0, 5, 3] = 10.0
        logits[0, 6, 12] = 10.0

        loss_no_mask = compute_loss(logits, labels, mask_reasoning=False)
        loss_with_mask = compute_loss(logits, labels, mask_reasoning=True)

        # Should be the same since there's no content between tags
        assert abs(loss_no_mask.item() - loss_with_mask.item()) < 0.001, (
            f"Losses should be equal with empty reasoning block: "
            f"{loss_no_mask.item()} vs {loss_with_mask.item()}"
        )
