"""Shared pytest fixtures for all tests."""

import pytest

from src.tokenizer import ArithmeticTokenizer


@pytest.fixture
def tokenizer() -> ArithmeticTokenizer:
    """Create a tokenizer instance for testing."""
    return ArithmeticTokenizer()
