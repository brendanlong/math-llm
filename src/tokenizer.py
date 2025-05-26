"""Custom character-level tokenizer for arithmetic expressions with reasoning.

This tokenizer handles a vocabulary of 16 tokens:
- Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Operators: +, =
- Special: <end>
- Reasoning: <think>, </think>
- Formatting: \n (newline)
"""

from typing import Any, Union

# Vocabulary mapping for arithmetic expressions with reasoning
V = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "=": 11,
    "<end>": 12,
    "<think>": 13,
    "</think>": 14,
    "\n": 15,
}

# Constants derived from vocabulary
VOCAB_SIZE = len(V)
MAX_SEQUENCE_LENGTH = 128


class ArithmeticTokenizer:
    """Character-level tokenizer for arithmetic expressions."""

    vocab = V

    def __init__(self):
        """Initialize the tokenizer with arithmetic vocabulary."""
        # Create reverse mapping for decoding
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Special token IDs
        self.end_token_id = self.vocab["<end>"]

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text (e.g., "3+5=8<end>")

        Returns:
            List of token IDs

        Raises:
            ValueError: If text contains unknown characters
        """
        tokens = []
        i = 0
        while i < len(text):
            # Check for multi-character special tokens first
            if text[i : i + 8] == "</think>":
                tokens.append(self.vocab["</think>"])
                i += 8
            elif text[i : i + 7] == "<think>":
                tokens.append(self.vocab["<think>"])
                i += 7
            elif text[i : i + 5] == "<end>":
                tokens.append(self.vocab["<end>"])
                i += 5
            elif text[i] in self.vocab:
                tokens.append(self.vocab[text[i]])
                i += 1
            else:
                raise ValueError(f"Unknown character: '{text[i]}'")

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ID(s) to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string

        Raises:
            ValueError: If token ID is out of vocabulary range
        """
        tokens = []
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id}")
            tokens.append(self.id_to_token[token_id])

        return "".join(tokens)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into string tokens.

        Args:
            text: Input text

        Returns:
            List of string tokens
        """
        tokens = []
        i = 0
        while i < len(text):
            # Check for multi-character special tokens first
            if text[i : i + 8] == "</think>":
                tokens.append("</think>")
                i += 8
            elif text[i : i + 7] == "<think>":
                tokens.append("<think>")
                i += 7
            elif text[i : i + 5] == "<end>":
                tokens.append("<end>")
                i += 5
            elif text[i] in self.vocab:
                tokens.append(text[i])
                i += 1
            else:
                raise ValueError(f"Unknown character: '{text[i]}'")

        return tokens

    def __call__(
        self, text: Union[str, list[str]], **_kwargs: Any
    ) -> Union[list[int], list[list[int]]]:
        """Make tokenizer callable for compatibility with HuggingFace.

        Args:
            text: Input text or list of texts
            **_kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Token IDs or batch of token IDs
        """
        if isinstance(text, str):
            return self.encode(text)
        else:
            return [self.encode(t) for t in text]
