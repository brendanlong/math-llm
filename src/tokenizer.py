"""Custom character-level tokenizer for arithmetic expressions with reasoning.

This tokenizer handles a vocabulary of 16 tokens:
- Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Operators: +, =
- Special: <end>, <noop>
- Reasoning: <think>, </think>
"""

import tempfile
from typing import Literal

from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
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

TOKEN_PATTERN = r"</think>|<think>|<noop>|<end>|[0-9+=]"

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

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f.name,
        unk_token=None,
        pad_token="<end>",
        eos_token="<end>",
        clean_up_tokenization_spaces=False,
    )
