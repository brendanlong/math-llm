"""Unit tests for compute_loss function."""

import torch

from src.model import compute_loss
from src.tokenizer import VOCAB


class TestComputeLoss:
    """Test compute_loss functionality."""

    def test_simple_loss(self):
        """Test basic loss computation."""
        # Example: "3+5=8<end>"
        # Tokens: [3, 10, 5, 11, 8, 12]

        batch_size = 1
        seq_len = 6
        vocab_size = len(VOCAB)

        # Create dummy logits - make them match labels perfectly
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[3, 10, 5, 11, 8, 12]])

        # Set logits to predict the correct next token with high confidence
        logits[0, 0, 10] = 10.0  # Position 0 predicts token 10
        logits[0, 1, 5] = 10.0  # Position 1 predicts token 5
        logits[0, 2, 11] = 10.0  # Position 2 predicts token 11
        logits[0, 3, 8] = 10.0  # Position 3 predicts token 8
        logits[0, 4, 12] = 10.0  # Position 4 predicts token 12

        loss = compute_loss(logits, labels)

        # With perfect predictions and high confidence, loss should be very close to 0
        assert loss.item() < 0.001, f"Expected near-zero loss, got {loss.item()}"

    def test_all_correct_predictions(self):
        """Test that all correct predictions result in ~0 loss."""
        # Example: "1+2=<think>calc</think>3<end>"
        # Tokens: [1, 10, 2, 11, 13, 15, 14, 3, 12]

        batch_size = 1
        seq_len = 9
        vocab_size = len(VOCAB)

        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 10, 2, 11, 13, 15, 14, 3, 12]])

        # Make all predictions correct
        logits[0, 0, 10] = 10.0  # 1 -> +
        logits[0, 1, 2] = 10.0  # + -> 2
        logits[0, 2, 11] = 10.0  # 2 -> =
        logits[0, 3, 13] = 10.0  # = -> <think>
        logits[0, 4, 15] = 10.0  # <think> -> content
        logits[0, 5, 14] = 10.0  # content -> </think>
        logits[0, 6, 3] = 10.0  # </think> -> 3
        logits[0, 7, 12] = 10.0  # 3 -> <end>

        loss = compute_loss(logits, labels)

        assert loss.item() < 0.001, f"Expected near-zero loss, got {loss.item()}"

    def test_wrong_predictions_high_loss(self):
        """Test that wrong predictions result in higher loss."""
        batch_size = 1
        seq_len = 6
        vocab_size = len(VOCAB)

        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[3, 10, 5, 11, 8, 12]])

        # Make all predictions wrong
        logits[0, 0, 0] = 10.0  # Wrong: should predict 10
        logits[0, 1, 0] = 10.0  # Wrong: should predict 5
        logits[0, 2, 0] = 10.0  # Wrong: should predict 11
        logits[0, 3, 0] = 10.0  # Wrong: should predict 8
        logits[0, 4, 0] = 10.0  # Wrong: should predict 12

        loss = compute_loss(logits, labels)

        # Loss should be substantial with all wrong predictions
        assert loss.item() > 1.0, (
            f"Expected high loss with wrong predictions, got {loss.item()}"
        )

    def test_partial_correct_predictions(self):
        """Test loss with mix of correct and incorrect predictions."""
        batch_size = 1
        seq_len = 6
        vocab_size = len(VOCAB)

        logits_all_correct = torch.zeros(batch_size, seq_len, vocab_size)
        logits_half_correct = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[3, 10, 5, 11, 8, 12]])

        # All correct predictions
        logits_all_correct[0, 0, 10] = 10.0
        logits_all_correct[0, 1, 5] = 10.0
        logits_all_correct[0, 2, 11] = 10.0
        logits_all_correct[0, 3, 8] = 10.0
        logits_all_correct[0, 4, 12] = 10.0

        # Half correct predictions
        logits_half_correct[0, 0, 10] = 10.0  # Correct
        logits_half_correct[0, 1, 0] = 10.0  # Wrong
        logits_half_correct[0, 2, 11] = 10.0  # Correct
        logits_half_correct[0, 3, 0] = 10.0  # Wrong
        logits_half_correct[0, 4, 12] = 10.0  # Correct

        loss_correct = compute_loss(logits_all_correct, labels)
        loss_partial = compute_loss(logits_half_correct, labels)

        # Partial correct should have higher loss than all correct
        assert loss_partial.item() > loss_correct.item(), (
            f"Partial loss {loss_partial.item()} should be higher than correct loss {loss_correct.item()}"
        )
