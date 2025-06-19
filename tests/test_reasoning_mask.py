"""Unit tests for reasoning mask functionality."""

import torch

from src.model import create_reasoning_mask
from src.tokenizer import VOCAB, tokenizer


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
