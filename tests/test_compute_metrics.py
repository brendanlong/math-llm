"""Unit tests for compute_metrics function from training.py."""

import numpy as np
from transformers.trainer_utils import EvalPrediction

from src.training import compute_metrics


class TestComputeMetrics:
    """Test compute_metrics functionality."""

    def test_label_shifting(self):
        """Test that label shifting is correctly applied."""
        # Verify that predictions at position i are compared with labels at position i+1
        batch_size = 1
        seq_len = 4
        vocab_size = 16

        predictions = np.zeros((batch_size, seq_len, vocab_size))
        # Original labels: [5, 10, 3, 12]
        # After shifting in compute_metrics: [10, 3, 12, -100]
        labels = np.array([[5, 10, 3, 12]])

        # Make predictions match the shifted labels
        predictions[0, 0, 10] = 10.0  # Predicting 10 (matches shifted label[0])
        predictions[0, 1, 3] = 10.0  # Predicting 3 (matches shifted label[1])
        predictions[0, 2, 12] = 10.0  # Predicting 12 (matches shifted label[2])
        # Position 3 shifted label is -100 (ignored)

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_pred)

        # Should be 1.0 accuracy with correct shifted predictions
        assert abs(metrics["token_accuracy"] - 1.0) < 0.001, (
            f"Expected accuracy 1.0 with correct shifting, got {metrics['token_accuracy']}"
        )

    def test_masked_tokens_ignored(self):
        """Test that -100 masked tokens are ignored."""
        batch_size = 1
        seq_len = 6
        vocab_size = 16

        predictions = np.zeros((batch_size, seq_len, vocab_size))
        # Labels with some positions masked (simulating data loader masking)
        # Original: [-100, -100, -100, -100, 3, 12]
        # Shifted:  [-100, -100, -100, 3, 12, -100]
        labels = np.array([[-100, -100, -100, -100, 3, 12]])

        # Make predictions wrong for masked positions (should be ignored)
        predictions[0, 0, 0] = 10.0  # Wrong but masked
        predictions[0, 1, 0] = 10.0  # Wrong but masked
        predictions[0, 2, 0] = 10.0  # Wrong but masked

        # Make predictions correct for unmasked positions after shifting
        predictions[0, 3, 3] = 10.0  # Correct: matches shifted[3] = 3
        predictions[0, 4, 12] = 10.0  # Correct: matches shifted[4] = 12
        # Position 5 shifted label is -100 (ignored)

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_pred)

        # Should be 1.0 accuracy since only the unmasked predictions matter
        assert abs(metrics["token_accuracy"] - 1.0) < 0.001, (
            f"Expected accuracy 1.0 with masked tokens ignored, got {metrics['token_accuracy']}"
        )

    def test_partial_correct_predictions(self):
        """Test accuracy calculation with some correct and some wrong predictions."""
        batch_size = 1
        seq_len = 6
        vocab_size = 16

        predictions = np.zeros((batch_size, seq_len, vocab_size))
        labels = np.array([[1, 10, 2, 11, 3, 12]])

        # Make some predictions correct, some wrong
        predictions[0, 0, 10] = 10.0  # Correct: 1 -> 10
        predictions[0, 1, 0] = 10.0  # Wrong: should be 2
        predictions[0, 2, 11] = 10.0  # Correct: 2 -> 11
        predictions[0, 3, 0] = 10.0  # Wrong: should be 3
        predictions[0, 4, 12] = 10.0  # Correct: 3 -> 12
        # Position 5 doesn't predict anything (last position)

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_pred)

        # 3 correct out of 5 predictions = 0.6 accuracy
        expected_accuracy = 3.0 / 5.0
        assert abs(metrics["token_accuracy"] - expected_accuracy) < 0.001, (
            f"Expected accuracy {expected_accuracy}, got {metrics['token_accuracy']}"
        )

    def test_batch_computation(self):
        """Test metrics computation with multiple examples in a batch."""
        batch_size = 2
        seq_len = 4
        vocab_size = 16

        predictions = np.zeros((batch_size, seq_len, vocab_size))
        labels = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        # First example: all correct
        predictions[0, 0, 2] = 10.0  # 1 -> 2
        predictions[0, 1, 3] = 10.0  # 2 -> 3
        predictions[0, 2, 4] = 10.0  # 3 -> 4

        # Second example: all wrong
        predictions[1, 0, 0] = 10.0  # Wrong: should be 6
        predictions[1, 1, 0] = 10.0  # Wrong: should be 7
        predictions[1, 2, 0] = 10.0  # Wrong: should be 8

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_pred)

        # 3 correct out of 6 total predictions = 0.5 accuracy
        expected_accuracy = 3.0 / 6.0
        assert abs(metrics["token_accuracy"] - expected_accuracy) < 0.001, (
            f"Expected accuracy {expected_accuracy}, got {metrics['token_accuracy']}"
        )

    def test_no_valid_tokens_assertion(self):
        """Test that assertion fires when no valid tokens are found."""
        batch_size = 1
        seq_len = 4
        vocab_size = 16

        predictions = np.zeros((batch_size, seq_len, vocab_size))
        # All tokens masked
        labels = np.array([[-100, -100, -100, -100]])

        predictions[0, 0, 5] = 10.0  # Some prediction (should be ignored)

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Should raise assertion error
        try:
            compute_metrics(eval_pred)
            assert False, "Expected assertion error for no valid tokens"
        except AssertionError as e:
            assert "No valid tokens found for evaluation" in str(e)

    def test_mask_reasoning_enabled(self):
        """Test that mask_reasoning=True properly masks reasoning content."""
        batch_size = 1
        seq_len = 12
        vocab_size = 16

        predictions = np.zeros((batch_size, seq_len, vocab_size))
        # Create a sequence: 3 + 5 = <think> 3 + 5 = 8 </think> 8 <end>
        # Token IDs:        3  10  5  11   13    3  10  5  11  8    14   8   12
        # Positions:        0   1  2   3    4    5   6  7   8  9    10  11   12
        # Full sequence: 3 + 5 = <think> 3 + 5 = 8 </think> 8 <end>
        labels = np.array(
            [[3, 10, 5, 11, 13, 3, 10, 5, 11, 8, 14, 8]]
        )  # Complete sequence with </think>

        # Make predictions that would be wrong for reasoning tokens but right for others
        # After shifting, we're predicting position i+1 from position i
        predictions[0, 0, 10] = 10.0  # Predict + (correct: 3->10)
        predictions[0, 1, 5] = 10.0  # Predict 5 (correct: 10->5)
        predictions[0, 2, 11] = 10.0  # Predict = (correct: 5->11)
        predictions[0, 3, 13] = 10.0  # Predict <think> (correct: 11->13)

        # These should be masked out when mask_reasoning=True (between <think> and </think>)
        predictions[0, 4, 0] = (
            10.0  # Wrong prediction for reasoning content (should be masked)
        )
        predictions[0, 5, 0] = (
            10.0  # Wrong prediction for reasoning content (should be masked)
        )
        predictions[0, 6, 0] = (
            10.0  # Wrong prediction for reasoning content (should be masked)
        )
        predictions[0, 7, 0] = (
            10.0  # Wrong prediction for reasoning content (should be masked)
        )
        predictions[0, 8, 0] = (
            10.0  # Wrong prediction for reasoning content (should be masked)
        )
        predictions[0, 9, 14] = 10.0  # Predict </think> (correct: 8->14)

        # After </think> - should be evaluated
        predictions[0, 10, 8] = 10.0  # Predict 8 (correct: 14->8)

        # Last position doesn't predict anything (gets shifted out)

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Test without mask_reasoning (should get low accuracy due to wrong reasoning predictions)
        metrics_no_mask = compute_metrics(eval_pred, mask_reasoning=False)

        # Test with mask_reasoning (should get higher accuracy as reasoning tokens are ignored)
        metrics_with_mask = compute_metrics(eval_pred, mask_reasoning=True)

        # With mask_reasoning=False: 6 correct out of 11 = ~0.55 (4 before <think> + 1 </think> + 1 after)
        # With mask_reasoning=True: 6 correct out of 6 non-reasoning = 1.0 (reasoning content masked)
        # (The reasoning tokens between <think> and </think> should be masked)

        assert (
            metrics_with_mask["token_accuracy"] > metrics_no_mask["token_accuracy"]
        ), (
            f"Expected higher accuracy with mask_reasoning=True. "
            f"No mask: {metrics_no_mask['token_accuracy']:.3f}, "
            f"With mask: {metrics_with_mask['token_accuracy']:.3f}"
        )

        # The masked version should have perfect accuracy since only non-reasoning tokens are evaluated
        assert abs(metrics_with_mask["token_accuracy"] - 1.0) < 0.001, (
            f"Expected perfect accuracy with mask_reasoning=True, got {metrics_with_mask['token_accuracy']}"
        )

    def test_mask_reasoning_disabled(self):
        """Test that mask_reasoning=False behaves the same as the default."""
        batch_size = 1
        seq_len = 4
        vocab_size = 16

        predictions = np.zeros((batch_size, seq_len, vocab_size))
        labels = np.array([[1, 2, 3, 4]])

        # Make all predictions correct
        predictions[0, 0, 2] = 10.0  # 1 -> 2
        predictions[0, 1, 3] = 10.0  # 2 -> 3
        predictions[0, 2, 4] = 10.0  # 3 -> 4

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Both should give same result
        metrics_default = compute_metrics(eval_pred)
        metrics_no_mask = compute_metrics(eval_pred, mask_reasoning=False)

        assert (
            abs(metrics_default["token_accuracy"] - metrics_no_mask["token_accuracy"])
            < 0.001
        ), (
            f"Expected same accuracy. Default: {metrics_default['token_accuracy']}, "
            f"No mask: {metrics_no_mask['token_accuracy']}"
        )
