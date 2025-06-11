"""Custom character-level tokenizer for arithmetic expressions with reasoning.

This tokenizer handles a vocabulary of 19 tokens:
- Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Operators: +, =
- Special: <end>, <noop>
- Multi-operand reasoning: <think_multi>, </think_multi>
- Multi-digit reasoning: <think_digit>, </think_digit>
- Formatting: \n (newline)
"""

import tempfile

from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Vocabulary mapping for arithmetic expressions with reasoning
VOCAB = {
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
    "<think_multi>": 13,
    "</think_multi>": 14,
    "<think_digit>": 15,
    "</think_digit>": 16,
    "\n": 17,
    "<noop>": 18,
}

TOKEN_PATTERN = (
    r"</think_multi>|</think_digit>|<think_multi>|<think_digit>|<noop>|<end>|\n|[0-9+=]"
)

# Constants derived from vocabulary
VOCAB_SIZE = len(VOCAB)

tokenizer = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token=None))
tokenizer.pre_tokenizer = pre_tokenizers.Split(
    pattern=Regex(TOKEN_PATTERN), behavior="isolated"
)  # type: ignore
tokenizer.decoder = decoders.Fuse()  # type: ignore

end_token_id = VOCAB["<end>"]
with tempfile.NamedTemporaryFile() as f:
    tokenizer.save(f.name)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f.name,
        unk_token=None,
        pad_token="<end>",
        eos_token="<end>",
        clean_up_tokenization_spaces=False,
    )


class ArithmeticTokenizer:
    """Character-level tokenizer for arithmetic expressions."""

    vocab = VOCAB

    def __init__(self):
        """Initialize the tokenizer with arithmetic vocabulary."""
        self.tokenizer = fast_tokenizer

        self.end_token_id = VOCAB["<end>"]

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
            Exception: If text contains unknown characters
        """
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ID(s) to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids)
