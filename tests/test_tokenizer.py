"""Tests for the arithmetic tokenizer."""

import pytest

from src.tokenizer import tokenizer


class TestArithmeticTokenizer:
    """Test cases for ArithmeticTokenizer."""

    def test_simple_encoding(
        self,
    ):
        """Test encoding of simple expressions."""
        # Single digit addition
        text = "3+5=8<end>"
        expected = [3, 10, 5, 11, 8, 12]
        assert tokenizer.encode(text) == expected

        # Multi-digit addition
        text = "12+34=46<end>"
        expected = [1, 2, 10, 3, 4, 11, 4, 6, 12]
        assert tokenizer.encode(text) == expected

    def test_simple_decoding(
        self,
    ):
        """Test decoding of token sequences."""
        # Single digit
        token_ids = [3, 10, 5, 11, 8, 12]
        expected = "3+5=8<end>"
        assert tokenizer.decode(token_ids) == expected

        # Multi-digit
        token_ids = [1, 2, 10, 3, 4, 11, 4, 6, 12]
        expected = "12+34=46<end>"
        assert tokenizer.decode(token_ids) == expected

    def test_batch_decoding(
        self,
    ):
        """Test decoding of multiple sequences individually."""
        batch_ids = [[3, 10, 5, 11, 8, 12], [1, 2, 10, 3, 4, 11, 4, 6, 12]]
        expected = ["3+5=8<end>", "12+34=46<end>"]
        results = [tokenizer.decode(seq) for seq in batch_ids]
        assert results == expected

    def test_roundtrip_consistency(
        self,
    ):
        """Test encode/decode roundtrip maintains original text."""
        texts = ["0+0=0<end>", "9+9=18<end>", "123+456=579<end>", "1000+2000=3000<end>"]

        for text in texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert decoded == text

    def test_invalid_character_encoding(
        self,
    ):
        """Test encoding raises error for invalid characters."""
        with pytest.raises(Exception, match="UNK"):
            tokenizer.encode("3+5=8a")

        with pytest.raises(Exception, match="UNK"):
            tokenizer.encode("3-5=8<end>")  # subtraction not supported

    def test_invalid_token_id_decoding(
        self,
    ):
        """Test decoding raises error for invalid token IDs."""
        # Ignores invalid token
        assert tokenizer.decode([3, 10, 5, 99]) == "3+5"

    def test_reasoning_tokens(
        self,
    ):
        """Test encoding and decoding of reasoning tokens."""
        # Test individual reasoning tokens
        assert tokenizer.encode("<think>") == [13]
        assert tokenizer.encode("</think>") == [14]

        # Test simple reasoning expression
        text = "3+5=<think>3+5=8</think>8<end>"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        # Test multi-operand reasoning expression
        text = "3+5+2=<think>3+5=8+2=10</think>10<end>"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_chain_of_thought_example(
        self,
    ):
        """Test full chain-of-thought example."""
        text = "658+189=<think>958+981=7471</think>847<end>"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_edge_cases(
        self,
    ):
        """Test edge cases and boundary conditions."""
        # Empty string
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""

        # Only special token
        assert tokenizer.encode("<end>") == [12]
        assert tokenizer.decode([12]) == "<end>"

        # All digits
        assert tokenizer.encode("0123456789") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Multiple end tokens
        assert tokenizer.encode("<end><end>") == [12, 12]

        # Only reasoning tokens
        assert tokenizer.encode("<think></think>") == [13, 14]

    def test_large_numbers(
        self,
    ):
        """Test with larger arithmetic expressions."""
        text = "9999+1=10000<end>"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        # Very long expression
        text = "123456789+987654321=1111111110<end>"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text
