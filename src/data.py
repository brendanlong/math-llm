"""Data loading utilities for arithmetic expression datasets."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .model import MAX_SEQUENCE_LENGTH
from .tokenizer import ArithmeticTokenizer
from .types import DatasetDict


class ArithmeticDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for arithmetic expressions.

    Loads expressions from JSON files and tokenizes them for training.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: ArithmeticTokenizer,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSON file containing expressions
            tokenizer: Tokenizer instance for encoding expressions
            max_length: Maximum sequence length (sequences will be padded/truncated)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        with open(self.data_path, "r") as f:
            dataset: DatasetDict = json.load(f)
            self.data = dataset["examples"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single tokenized example.

        Args:
            idx: Index of the example

        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        expression = self.data[idx]["text"]

        # Tokenize the expression
        tokens = self.tokenizer.encode(expression)

        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            # Pad with end token
            pad_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.vocab["<end>"]] * pad_length

        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # For causal language modeling, labels are the same as input_ids
        # but shifted by one position (next token prediction)
        labels = input_ids.clone()

        # Mask padding tokens in labels (loss should ignore them)
        # Find first natural end token (after result)
        expression_tokens = self.tokenizer.encode(expression)
        if len(expression_tokens) < self.max_length:
            # Mask all padding tokens except the first natural end
            for i in range(len(expression_tokens), self.max_length):
                labels[i] = -100

        return {"input_ids": input_ids, "labels": labels}


def create_dataloader(
    data_path: str | Path,
    tokenizer: ArithmeticTokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: int = MAX_SEQUENCE_LENGTH,
    num_workers: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Create a DataLoader for arithmetic expressions.

    Args:
        data_path: Path to JSON file containing expressions
        tokenizer: Tokenizer instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader instance ready for training/evaluation
    """
    dataset = ArithmeticDataset(
        data_path=data_path, tokenizer=tokenizer, max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_splits(
    data_dir: str | Path,
    tokenizer: ArithmeticTokenizer,
    batch_size: int = 32,
    max_length: int = MAX_SEQUENCE_LENGTH,
    num_workers: int = 0,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    """Load train, validation, and test DataLoaders.

    Args:
        data_dir: Directory containing train.json, val.json, test.json
        tokenizer: Tokenizer instance
        batch_size: Batch size for all splits
        max_length: Maximum sequence length
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_loader = create_dataloader(
        data_path=data_dir / "train.json",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_length=max_length,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        data_path=data_dir / "val.json",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
        num_workers=num_workers,
    )

    test_loader = create_dataloader(
        data_path=data_dir / "test.json",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
