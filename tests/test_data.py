"""Tests for data loading utilities."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import SequentialSampler

from src.data import ArithmeticDataset, create_dataloader, load_splits
from src.tokenizer import ArithmeticTokenizer


@pytest.fixture
def tokenizer() -> ArithmeticTokenizer:
    """Create a tokenizer instance for testing."""
    return ArithmeticTokenizer()


@pytest.fixture
def sample_data() -> list[str]:
    """Create sample arithmetic expressions for testing."""
    return [
        "3+5=8<end>",
        "1+2=3<end>",
        "7+1=8<end>",
        "4+6=10<end>",
        "9+0=9<end>",
        "12+34=<think_digit>\n2+4=6\n6\n1+3=4\n4\n</think_digit>46<end>",
    ]


@pytest.fixture
def temp_data_file(sample_data: list[str]) -> Path:
    """Create a temporary JSON file with sample data in new format."""
    tokenizer = ArithmeticTokenizer()
    dataset = {
        "examples": sample_data,
        "metadata": {
            "split": "test",
            "num_examples": len(sample_data),
            "vocab_size": tokenizer.vocab_size,
            "format": "operand1+operand2=<think_digit>...</think_digit>result<end>",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dataset, f)
        return Path(f.name)


class TestArithmeticDataset:
    """Tests for ArithmeticDataset class."""

    def test_dataset_length(
        self,
        temp_data_file: Path,
        tokenizer: ArithmeticTokenizer,
        sample_data: list[dict[str, str]],
    ) -> None:
        """Test that dataset returns correct length."""
        dataset = ArithmeticDataset(temp_data_file, tokenizer, max_length=32)
        assert len(dataset) == len(sample_data)

    def test_dataset_getitem(
        self, temp_data_file: Path, tokenizer: ArithmeticTokenizer
    ) -> None:
        """Test that dataset returns correctly formatted items."""
        dataset = ArithmeticDataset(temp_data_file, tokenizer, max_length=32)
        item = dataset[0]

        # Check return format
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "labels" in item

        # Check tensor properties
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long

        # Check sequence length
        assert item["input_ids"].shape == (32,)
        assert item["labels"].shape == (32,)

    def test_tokenization(
        self, temp_data_file: Path, tokenizer: ArithmeticTokenizer
    ) -> None:
        """Test that expressions are properly tokenized."""
        dataset = ArithmeticDataset(temp_data_file, tokenizer, max_length=32)
        item = dataset[0]  # "3+5=8<end>"

        # Manually tokenize to compare
        expected_tokens = tokenizer.encode("3+5=8<end>")
        input_tokens = item["input_ids"][: len(expected_tokens)].tolist()

        assert input_tokens == expected_tokens

    def test_padding(
        self, temp_data_file: Path, tokenizer: ArithmeticTokenizer
    ) -> None:
        """Test that short sequences are properly padded."""
        dataset = ArithmeticDataset(temp_data_file, tokenizer, max_length=32)
        item = dataset[0]  # "3+5=8<end>" - should be much shorter than 32

        # Check that sequence is padded to max_length
        assert item["input_ids"].shape[0] == 32

        # Check that padding uses end token
        end_token_id = tokenizer.vocab["<end>"]
        original_length = len(tokenizer.encode("3+5=8<end>"))

        # After original content, should be padded with end tokens
        for i in range(original_length, 32):
            assert item["input_ids"][i].item() == end_token_id

    def test_label_masking(
        self, temp_data_file: Path, tokenizer: ArithmeticTokenizer
    ) -> None:
        """Test that prompt tokens and padding are masked in labels for completion-style training."""
        dataset = ArithmeticDataset(temp_data_file, tokenizer, max_length=32)
        item = dataset[0]  # "3+5=8<end>"

        # For completion-style training, we mask prompt tokens (everything before and including "=")
        prompt_tokens = tokenizer.encode("3+5=")
        completion_tokens = tokenizer.encode("8<end>")

        # Prompt tokens should be masked
        for i in range(len(prompt_tokens)):
            assert item["labels"][i].item() == -100

        # Completion tokens should not be masked
        for i in range(len(prompt_tokens), len(prompt_tokens) + len(completion_tokens)):
            assert item["labels"][i].item() != -100

        # Padding tokens should be masked
        original_length = len(prompt_tokens) + len(completion_tokens)
        for i in range(original_length, 32):
            assert item["labels"][i].item() == -100

    def test_truncation(self, tokenizer: ArithmeticTokenizer) -> None:
        """Test that long sequences are properly truncated."""
        # Create a very long expression
        long_expression = "1+2=3<end>" * 10  # Much longer than 32 tokens

        dataset = {
            "examples": [long_expression],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dataset, f)
            temp_file = Path(f.name)

        dataset = ArithmeticDataset(temp_file, tokenizer, max_length=32)
        item = dataset[0]

        # Should be truncated to max_length
        assert item["input_ids"].shape[0] == 32
        assert item["labels"].shape[0] == 32


class TestDataLoader:
    """Tests for DataLoader creation functions."""

    def test_create_dataloader(
        self, temp_data_file: Path, tokenizer: ArithmeticTokenizer
    ) -> None:
        """Test creating a single DataLoader."""
        dataloader = create_dataloader(
            data_path=temp_data_file,
            tokenizer=tokenizer,
            batch_size=2,
            shuffle=False,
            max_length=16,
        )

        # Test basic properties
        assert dataloader.batch_size == 2
        # Note: shuffle=False creates a SequentialSampler which doesn't have shuffle attribute
        # We can verify non-shuffling by checking sampler type instead
        assert isinstance(dataloader.sampler, SequentialSampler)

        # Test that we can iterate through it
        batch = next(iter(dataloader))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape == (2, 16)  # (batch_size, max_length)
        assert batch["labels"].shape == (2, 16)

    def test_load_splits(
        self, sample_data: list[dict[str, str]], tokenizer: ArithmeticTokenizer
    ) -> None:
        """Test loading train/val/test splits."""
        # Create temporary directory with split files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create split files in new format
            for split in ["train", "val", "test"]:
                dataset = {"examples": sample_data}
                with open(temp_path / f"{split}.json", "w") as f:
                    json.dump(dataset, f)

            # Load splits
            train_loader, val_loader, test_loader = load_splits(
                data_dir=temp_path, tokenizer=tokenizer, batch_size=2, max_length=16
            )

            # Test that all loaders work
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            test_batch = next(iter(test_loader))

            for batch in [train_batch, val_batch, test_batch]:
                assert "input_ids" in batch
                assert "labels" in batch
                assert batch["input_ids"].shape == (2, 16)

    def test_dataloader_cuda_pinning(
        self, temp_data_file: Path, tokenizer: ArithmeticTokenizer
    ) -> None:
        """Test that CUDA memory pinning is set correctly."""
        dataloader = create_dataloader(
            data_path=temp_data_file, tokenizer=tokenizer, batch_size=1
        )

        # pin_memory should be True if CUDA is available
        expected_pin_memory = torch.cuda.is_available()
        assert dataloader.pin_memory == expected_pin_memory


class TestDataIntegration:
    """Integration tests for data loading with tokenizer."""

    def test_roundtrip_tokenization(
        self, temp_data_file: Path, tokenizer: ArithmeticTokenizer
    ) -> None:
        """Test that data can be tokenized and decoded back."""
        dataset = ArithmeticDataset(temp_data_file, tokenizer, max_length=32)
        item = dataset[0]

        # Get the tokens up to the first natural end
        tokens = item["input_ids"].tolist()
        end_token_id = tokenizer.vocab["<end>"]

        # Find first end token (should be the natural one)
        try:
            first_end_idx = tokens.index(end_token_id)
            original_tokens = tokens[: first_end_idx + 1]
        except ValueError:
            # If no end token found, take all tokens
            original_tokens = tokens

        # Decode back to string
        decoded = tokenizer.decode(original_tokens)

        # Should be a valid arithmetic expression
        assert "=" in decoded
        assert "+" in decoded
        assert decoded.endswith("<end>")

    def test_batch_consistency(
        self, temp_data_file: Path, tokenizer: ArithmeticTokenizer
    ) -> None:
        """Test that batched data maintains consistency."""
        dataloader = create_dataloader(
            data_path=temp_data_file, tokenizer=tokenizer, batch_size=3, shuffle=False
        )

        batch = next(iter(dataloader))

        # All sequences in batch should have same length
        assert batch["input_ids"].shape[0] == 3  # batch size
        assert batch["labels"].shape[0] == 3

        # Check that each item in batch is valid
        for i in range(3):
            input_ids = batch["input_ids"][i]
            labels = batch["labels"][i]

            # Should be proper tensors
            assert input_ids.dtype == torch.long
            assert labels.dtype == torch.long

            # Labels should match input_ids for non-masked positions
            non_masked = labels != -100
            assert torch.equal(input_ids[non_masked], labels[non_masked])
