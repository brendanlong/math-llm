"""Integration tests with tokenizer."""

import torch

from src.config import ModelConfig
from src.model import create_model_from_config
from src.tokenizer import VOCAB, VOCAB_SIZE, tokenizer

# Test config matching the original small model
SMALL_CONFIG = ModelConfig(
    architecture="standard",
    d_model=256,
    n_layers=4,
    n_heads=4,
    d_ff=512,
    dropout=0.1,
)


class TestModelIntegration:
    """Integration tests with tokenizer."""

    def test_model_tokenizer_compatibility(
        self,
    ) -> None:
        """Test model works correctly with tokenizer."""
        model = create_model_from_config(SMALL_CONFIG)

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
    ) -> None:
        """Test model with batch of inputs."""
        model = create_model_from_config(SMALL_CONFIG)

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
