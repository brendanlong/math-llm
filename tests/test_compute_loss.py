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

    def test_special_tokens_never_masked_in_reasoning(self):
        """Test that special tokens inside reasoning blocks are never masked."""
        # Example: "1+2=<think>5<end>6</think>3<end>"
        # Tokens: [1, 10, 2, 11, 13, 5, 12, 6, 14, 3, 12]
        # The 5 and 6 are regular content inside reasoning that should be masked
        # But the 12 (<end>) inside reasoning should NOT be masked

        batch_size = 1
        seq_len = 11
        vocab_size = len(VOCAB)

        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 10, 2, 11, 13, 5, 12, 6, 14, 3, 12]])

        # Make predictions
        logits[0, 0, 10] = 10.0  # Correct: 1 -> 10
        logits[0, 1, 2] = 10.0  # Correct: 10 -> 2
        logits[0, 2, 11] = 10.0  # Correct: 2 -> 11
        logits[0, 3, 13] = 10.0  # Correct: 11 -> 13 (<think>)

        # Inside reasoning block:
        logits[0, 4, 5] = (
            10.0  # Correct: 13 -> 5 (but should be masked since 5 is regular content)
        )
        logits[0, 5, 12] = 10.0  # Correct: 5 -> 12 (<end>) - should NOT be masked!
        logits[0, 6, 0] = 10.0  # Wrong: should be 6, but 6 should be masked anyway
        logits[0, 7, 14] = 10.0  # Correct: 6 -> 14 (</think>) - should NOT be masked!

        # After reasoning block:
        logits[0, 8, 3] = 10.0  # Correct: 14 -> 3
        logits[0, 9, 12] = 10.0  # Correct: 3 -> 12 (<end>)

        loss_no_mask = compute_loss(logits, labels, mask_reasoning=False)
        loss_with_mask = compute_loss(logits, labels, mask_reasoning=True)

        # Without mask: All predictions contribute to loss (1 wrong prediction)
        # With mask: Regular reasoning content is masked, but special tokens are not
        # So we train on: <think>, <end> (inside), </think>, and post-reasoning content
        # One correct reasoning token (5) gets masked, one wrong reasoning token (6) gets masked
        # But <end> and </think> inside reasoning are still trained on
        assert loss_with_mask.item() < loss_no_mask.item(), (
            f"Masked loss {loss_with_mask.item()} should be less than "
            f"unmasked loss {loss_no_mask.item()} due to masking some wrong predictions"
        )

        # The masked loss should be very low since all unmasked predictions are correct
        assert loss_with_mask.item() < 0.001, (
            f"Expected near-zero masked loss with correct special token predictions, "
            f"got {loss_with_mask.item()}"
        )

    def test_repeated_end_think_tokens_high_loss(self):
        """Test that repeated </think> tokens result in high loss."""
        # Ground truth: "1+1=<think><noop><noop>...</think>2<end>"
        # Model prediction: "1+1=</think></think></think>...2<end>"
        # This should result in very high loss since almost every token after = is wrong

        batch_size = 1
        seq_len = 10
        vocab_size = len(VOCAB)

        # Ground truth: 1+1=<think><noop><noop><noop></think>2<end>
        labels = torch.tensor([[1, 10, 1, 11, 13, 15, 15, 15, 14, 2, 12]])

        # Model prediction logits: 1+1=</think></think></think></think></think>...2<end>
        # Position predictions: [10, 1, 11, 14, 14, 14, 14, 14, 2, 12]
        logits = torch.zeros(batch_size, seq_len, vocab_size)

        # Set up logits to predict the wrong sequence
        logits[0, 0, 10] = 10.0  # 1 -> + (correct)
        logits[0, 1, 1] = 10.0  # + -> 1 (correct)
        logits[0, 2, 11] = 10.0  # 1 -> = (correct)

        # Wrong predictions: = -> </think> instead of <think>
        logits[0, 3, 14] = 10.0  # = -> </think> (WRONG! should be <think>)
        logits[0, 4, 14] = 10.0  # <think> -> </think> (WRONG! should be <noop>)
        logits[0, 5, 14] = 10.0  # <noop> -> </think> (WRONG! should be <noop>)
        logits[0, 6, 14] = 10.0  # <noop> -> </think> (WRONG! should be <noop>)
        logits[0, 7, 14] = 10.0  # <noop> -> </think> (WRONG! should be </think>)
        logits[0, 8, 2] = 10.0  # 2 -> 2 (correct)
        logits[0, 9, 12] = 10.0  # <end> -> <end> (correct)

        loss_no_mask = compute_loss(logits, labels, mask_reasoning=False)
        loss_with_mask = compute_loss(logits, labels, mask_reasoning=True)

        print(f"Labels: {labels}")
        print(f"Loss without mask: {loss_no_mask.item():.4f}")
        print(f"Loss with mask: {loss_with_mask.item():.4f}")

        # Both losses should be substantial (> 1.0) due to many wrong predictions
        assert loss_no_mask.item() > 1.0, (
            f"Expected high loss without mask due to wrong predictions, got {loss_no_mask.item()}"
        )
        assert loss_with_mask.item() == loss_no_mask.item(), (
            f"Expected same loss with mask, got {loss_with_mask.item()}"
        )

    def test_learning_reasoning_structure(self):
        """Test that <think>...</think>answer is always better than anything random, when masking is enabled."""
        # Ground truth: "1+1=<think><noop><noop>...</think>2<end>"
        # Model prediction: "1+1=</think></think></think>2<end>"
        # This should result in very high loss since almost every token after = is wrong
        # Model prediction 2: "1+1=<think>[random tokens]</think>[random token]<end>"
        # This should result in much lower loss since most of tokens are masked

        batch_size = 1
        seq_len = 10
        vocab_size = len(VOCAB)

        # Ground truth: 1+1=<think><noop><noop><noop></think>2<end>
        labels = torch.tensor([[1, 10, 1, 11, 13, 15, 15, 15, 14, 2, 12]])

        # Model prediction logits: 1+1=</think></think></think></think></think>2<end>
        # Position predictions: [10, 1, 11, 14, 14, 14, 14, 14, 2, 12]
        logits = torch.zeros(batch_size, seq_len, vocab_size)

        # Set up logits to predict the wrong sequence
        logits[0, 0, 10] = 10.0  # 1 -> + (correct)
        logits[0, 1, 1] = 10.0  # + -> 1 (correct)
        logits[0, 2, 11] = 10.0  # 1 -> = (correct)

        # Wrong predictions: = -> </think> instead of <think>
        logits[0, 3, 14] = 10.0  # = -> </think> (WRONG! should be <think>)
        logits[0, 4, 14] = 10.0  # <think> -> </think> (WRONG! should be <noop>)
        logits[0, 5, 14] = 10.0  # <noop> -> </think> (WRONG! should be <noop>)
        logits[0, 6, 14] = 10.0  # <noop> -> </think> (WRONG! should be <noop>)
        logits[0, 7, 14] = 10.0  # <noop> -> </think> (WRONG! should be </think>)
        logits[0, 8, 2] = 10.0  # 2 -> 2 (correct)
        logits[0, 9, 12] = 10.0  # <end> -> <end> (correct)

        loss_no_mask = compute_loss(logits, labels, mask_reasoning=False)
        loss_with_mask = compute_loss(logits, labels, mask_reasoning=True)

        print(f"Labels: {labels}")
        print(f"Loss without mask: {loss_no_mask.item():.4f}")
        print(f"Loss with mask: {loss_with_mask.item():.4f}")

        # Both losses should be substantial (> 1.0) due to many wrong predictions
        assert loss_no_mask.item() > 1.0, (
            f"Expected high loss without mask due to wrong predictions, got {loss_no_mask.item()}"
        )
        assert loss_with_mask.item() == loss_no_mask.item(), (
            f"Expected same loss with mask, got {loss_with_mask.item()}"
        )

        # Now test "good" structure: <think>[random]</think>answer<end>
        # This should have much lower loss with masking because reasoning content gets masked
        good_logits = torch.zeros(batch_size, seq_len, vocab_size)

        # Set up correct structure but wrong reasoning content
        good_logits[0, 0, 10] = 10.0  # 1 -> + (correct)
        good_logits[0, 1, 1] = 10.0  # + -> 1 (correct)
        good_logits[0, 2, 11] = 10.0  # 1 -> = (correct)
        good_logits[0, 3, 13] = 10.0  # = -> <think> (CORRECT!)

        # Wrong content inside reasoning (should be masked)
        good_logits[0, 4, 7] = 10.0  # <think> -> 7 (WRONG but masked)
        good_logits[0, 5, 9] = 10.0  # <noop> -> 9 (WRONG but masked)
        good_logits[0, 6, 3] = 10.0  # <noop> -> 3 (WRONG but masked)

        good_logits[0, 7, 14] = 10.0  # <noop> -> </think> (CORRECT!)
        good_logits[0, 8, 2] = 10.0  # </think> -> 2 (CORRECT!)
        good_logits[0, 9, 12] = 10.0  # 2 -> <end> (CORRECT!)

        good_loss_no_mask = compute_loss(good_logits, labels, mask_reasoning=False)
        good_loss_with_mask = compute_loss(good_logits, labels, mask_reasoning=True)

        print("\nGood structure predictions:")
        print(f"Good loss without mask: {good_loss_no_mask.item():.4f}")
        print(f"Good loss with mask: {good_loss_with_mask.item():.4f}")

        # The key test: good structure with masking should be much better than bad structure
        assert good_loss_with_mask.item() < loss_with_mask.item(), (
            f"Good structure with mask ({good_loss_with_mask.item():.4f}) should have lower loss "
            f"than bad structure with mask ({loss_with_mask.item():.4f})"
        )

        # Good structure with masking should be much better than without masking
        assert good_loss_with_mask.item() < good_loss_no_mask.item(), (
            f"Good structure should benefit from masking: "
            f"with mask {good_loss_with_mask.item():.4f} < without mask {good_loss_no_mask.item():.4f}"
        )

        # Verify that this incentivizes learning proper structure
        print("\nLoss comparison (lower is better):")
        print(f"Bad structure (</think></think>...): {loss_with_mask.item():.4f}")
        print(
            f"Good structure (<think>random</think>answer): {good_loss_with_mask.item():.4f}"
        )
        print(f"Improvement: {loss_with_mask.item() - good_loss_with_mask.item():.4f}")

        # The improvement should be substantial
        improvement = loss_with_mask.item() - good_loss_with_mask.item()
        assert improvement > 0.5, (
            f"Good structure should provide substantial improvement, got {improvement:.4f}"
        )
