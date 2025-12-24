"""Data loading utilities for arithmetic expression datasets."""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import VOCAB, tokenizer


class ArithmeticDataset(Dataset[dict[str, torch.Tensor]]):
    """Streaming dataset for arithmetic expressions.

    Uses line indexing to read JSONL files on demand, avoiding loading all data
    into memory at once. Only byte offsets are stored, not the actual data.
    """

    def __init__(
        self,
        data_path: str | Path,
        max_length: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSONL file containing expressions (one per line)
            max_length: Maximum sequence length (sequences will be padded/truncated).
                       If None, reads from metadata.json in same directory.
        """
        self.data_path = Path(data_path)
        self.end_token_id = VOCAB["<end>"]
        self.equals_token_id = VOCAB["="]

        # Build line index (stores byte offsets for each line)
        self.line_offsets: list[int] = []
        self._build_line_index()

        # Set max_length from metadata if not provided
        if max_length is None:
            metadata_path = self.data_path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    # Use longest example length in tokens (estimate from chars + buffer)
                    self.max_length = metadata.get("longest_example_length", 128) + 10
            else:
                # Fall back to a reasonable default
                self.max_length = 128
        else:
            self.max_length = max_length

    def _build_line_index(self) -> None:
        """Build an index of byte offsets for each line in the file."""
        with open(self.data_path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():  # Skip empty lines
                    self.line_offsets.append(offset)
                offset += len(line)

    def _read_line(self, idx: int) -> str:
        """Read a single line from the file by index.

        Args:
            idx: Index of the line to read

        Returns:
            The line content as a string (without newline)
        """
        with open(self.data_path, "rb") as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()
            return line.decode("utf-8").rstrip("\n")

    def __len__(self) -> int:
        return len(self.line_offsets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single tokenized example.

        Args:
            idx: Index of the example

        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        expression = self._read_line(idx)

        # Tokenize the entire expression
        full_tokens = tokenizer.encode(expression)
        original_length = len(full_tokens)

        # Convert to tensor and pad/truncate in one step
        input_ids = torch.full((self.max_length,), self.end_token_id, dtype=torch.long)
        seq_len = min(original_length, self.max_length)
        input_ids[:seq_len] = torch.tensor(full_tokens, dtype=torch.long)[:seq_len]
        labels = input_ids.clone()

        # Find the "=" token and mask everything before it (including "=")
        equals_position = torch.where(input_ids == self.equals_token_id)[0][0]

        # Mask from start up to and including the first "=" token
        mask_end = equals_position + 1
        labels[:mask_end] = -100

        # Mask padding tokens
        if original_length < self.max_length:
            labels[original_length:] = -100

        return {"input_ids": input_ids, "labels": labels}


def create_dataloader(
    data_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: Optional[int] = None,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = 4,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Create a DataLoader for arithmetic expressions with prefetching.

    Args:
        data_path: Path to JSONL file containing expressions (one per line)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading. Set to 0 to
            disable multiprocessing (useful for debugging).
        prefetch_factor: Number of batches to prefetch per worker. Only used
            when num_workers > 0. Higher values use more memory but can hide
            I/O latency better.

    Returns:
        DataLoader instance ready for training/evaluation
    """
    dataset = ArithmeticDataset(data_path=data_path, max_length=max_length)

    # prefetch_factor and persistent_workers are only valid when num_workers > 0
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )


def load_splits(
    data_dir: str | Path,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = 4,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    """Load train, validation, and test DataLoaders with streaming.

    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
        batch_size: Batch size for all splits
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_loader = create_dataloader(
        data_path=data_dir / "train.jsonl",
        batch_size=batch_size,
        shuffle=True,
        max_length=max_length,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    val_loader = create_dataloader(
        data_path=data_dir / "val.jsonl",
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    test_loader = create_dataloader(
        data_path=data_dir / "test.jsonl",
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, test_loader
