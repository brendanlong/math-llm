"""Shared pytest fixtures for all tests."""

import pytest

from src.tokenizer import tokenizer


@pytest.fixture
def tokenizer_fixture():
    """Create a tokenizer instance for testing."""
    return tokenizer
