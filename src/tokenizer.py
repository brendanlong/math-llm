"""Custom character-level tokenizer for arithmetic expressions with reasoning.

This tokenizer handles a vocabulary of 17 tokens:
- Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Operators: +, =
- Special: <end>, <noop>, <begin>
- Reasoning: <think>, </think>
"""

import tempfile
from typing import Literal

from tokenizers import (
    Regex,
    decoders,
    models,
    pre_tokenizers,
)
from tokenizers import (
    Tokenizer as HFTokenizer,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Type for valid vocabulary tokens
VocabToken = Literal[
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",  # Digits
    "+",
    "=",  # Operators
    "<end>",
    "<think>",
    "</think>",
    "<noop>",
    "<begin>",  # Special tokens
]

# Vocabulary mapping for arithmetic expressions with reasoning
VOCAB: dict[VocabToken, int] = {
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
    "<noop>": 15,
    "<begin>": 16,
}

TOKEN_PATTERN = r"<begin>|</think>|<think>|<noop>|<end>|[0-9+=]"

# Constants derived from vocabulary
VOCAB_SIZE = len(VOCAB)
END_TOKEN_ID = VOCAB["<end>"]
THINK_TOKEN_ID = VOCAB["<think>"]
END_THINK_TOKEN_ID = VOCAB["</think>"]
EQUALS_TOKEN_ID = VOCAB["="]


def _create_tokenizer() -> PreTrainedTokenizerFast:
    """Create and configure the tokenizer.

    Returns:
        Configured PreTrainedTokenizerFast instance
    """
    base_tokenizer = HFTokenizer(models.WordLevel(vocab=VOCAB, unk_token=None))
    base_tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(TOKEN_PATTERN), behavior="isolated"
    )  # type: ignore
    base_tokenizer.decoder = decoders.Fuse()  # type: ignore

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        base_tokenizer.save(f.name)
        return PreTrainedTokenizerFast(
            tokenizer_file=f.name,
            unk_token=None,
            pad_token="<end>",
            eos_token="<end>",
            clean_up_tokenization_spaces=False,
        )


# Create the tokenizer instance
tokenizer = _create_tokenizer()
