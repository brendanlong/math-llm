"""Tests for the arithmetic tokenizer."""

import pytest

from src.tokenizer import ArithmeticTokenizer


class TestArithmeticTokenizer:
    """Test cases for ArithmeticTokenizer."""

    def setup_method(self):
        """Set up tokenizer for each test."""
        self.tokenizer = ArithmeticTokenizer()

    def test_simple_encoding(self):
        """Test encoding of simple expressions."""
        # Single digit addition
        text = "3+5=8<end>"
        expected = [3, 10, 5, 11, 8, 12]
        assert self.tokenizer.encode(text) == expected

        # Multi-digit addition
        text = "12+34=46<end>"
        expected = [1, 2, 10, 3, 4, 11, 4, 6, 12]
        assert self.tokenizer.encode(text) == expected

    def test_simple_decoding(self):
        """Test decoding of token sequences."""
        # Single digit
        token_ids = [3, 10, 5, 11, 8, 12]
        expected = "3+5=8<end>"
        assert self.tokenizer.decode(token_ids) == expected

        # Multi-digit
        token_ids = [1, 2, 10, 3, 4, 11, 4, 6, 12]
        expected = "12+34=46<end>"
        assert self.tokenizer.decode(token_ids) == expected

    def test_batch_decoding(self):
        """Test decoding of multiple sequences individually."""
        batch_ids = [[3, 10, 5, 11, 8, 12], [1, 2, 10, 3, 4, 11, 4, 6, 12]]
        expected = ["3+5=8<end>", "12+34=46<end>"]
        results = [self.tokenizer.decode(seq) for seq in batch_ids]
        assert results == expected

    def test_tokenize(self):
        """Test tokenization into string tokens."""
        text = "3+5=8<end>"
        expected = ["3", "+", "5", "=", "8", "<end>"]
        assert self.tokenizer.tokenize(text) == expected

        text = "12+34=46<end>"
        expected = ["1", "2", "+", "3", "4", "=", "4", "6", "<end>"]
        assert self.tokenizer.tokenize(text) == expected

    def test_callable_interface(self):
        """Test tokenizer is callable for HuggingFace compatibility."""
        # Single text
        text = "3+5=8<end>"
        expected = [3, 10, 5, 11, 8, 12]
        assert self.tokenizer(text) == expected

        # Batch of texts
        texts = ["3+5=8<end>", "1+2=3<end>"]
        expected = [[3, 10, 5, 11, 8, 12], [1, 10, 2, 11, 3, 12]]
        assert self.tokenizer(texts) == expected

    def test_roundtrip_consistency(self):
        """Test encode/decode roundtrip maintains original text."""
        texts = ["0+0=0<end>", "9+9=18<end>", "123+456=579<end>", "1000+2000=3000<end>"]

        for text in texts:
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded)
            assert decoded == text

    def test_invalid_character_encoding(self):
        """Test encoding raises error for invalid characters."""
        with pytest.raises(ValueError, match="Unknown character"):
            self.tokenizer.encode("3+5=8a")

        with pytest.raises(ValueError, match="Unknown character"):
            self.tokenizer.encode("3-5=8<end>")  # subtraction not supported

    def test_invalid_token_id_decoding(self):
        """Test decoding raises error for invalid token IDs."""
        with pytest.raises(ValueError, match="Unknown token ID"):
            self.tokenizer.decode([3, 10, 5, 99])  # 99 is out of range

        with pytest.raises(ValueError, match="Unknown token ID"):
            self.tokenizer.decode([3, 10, 5, -1])  # negative ID

    def test_reasoning_tokens(self):
        """Test encoding and decoding of reasoning tokens."""
        # Test individual reasoning tokens
        assert self.tokenizer.encode("<think>") == [13]
        assert self.tokenizer.encode("</think>") == [14]
        assert self.tokenizer.encode("\n") == [15]

        # Test simple reasoning expression
        text = "3+5=<think>\n3+5=8\n</think>8<end>"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        assert decoded == text

    def test_chain_of_thought_example(self):
        """Test full chain-of-thought example."""
        text = "658+189=<think>\n8+9=17\n5+8=14\n6+1=8</think>847<end>"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        assert decoded == text

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty string
        assert self.tokenizer.encode("") == []
        assert self.tokenizer.decode([]) == ""

        # Only special token
        assert self.tokenizer.encode("<end>") == [12]
        assert self.tokenizer.decode([12]) == "<end>"

        # All digits
        assert self.tokenizer.encode("0123456789") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Multiple end tokens
        assert self.tokenizer.encode("<end><end>") == [12, 12]

        # Only reasoning tokens
        assert self.tokenizer.encode("<think></think>") == [13, 14]

    def test_large_numbers(self):
        """Test with larger arithmetic expressions."""
        text = "9999+1=10000<end>"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        assert decoded == text

        # Very long expression
        text = "123456789+987654321=1111111110<end>"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        assert decoded == text
