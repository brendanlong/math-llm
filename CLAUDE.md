# Claude AI Assistant Instructions

## Project Context

This is a machine learning project training a small transformer model to perform basic arithmetic (starting with addition). The codebase uses Python with PyTorch and HuggingFace Transformers.

## Key Documentation Files

- @README.md Project overview, quick start, and usage instructions
- @DESIGN.md Architecture decisions, model specifications, and technical details
- @CONTRIBUTING.md Development setup, code quality standards, and ML guidelines
- @TODO.md Implementation roadmap and current priorities

IMPORTANT: Make sure to update TODO.md before checking anything in!

## Development Guidelines

### Code Quality Commands

- **Formatting**: `ruff format .`
- **Linting**: `ruff check .` (use `--fix` for auto-fixes)
- **Type checking**: `pyright`
- **Pre-commit**: `pre-commit run` (runs all checks)
- **Tests**: `pytest` (prefer `pytest` over `python -m pytest`)

### Project Structure

- `src/`: Core model and utilities
- `scripts/`: Training, data generation, and evaluation scripts
- `data/`: Generated datasets
- `checkpoints/`: Model checkpoints
- `tests/`: Test files

### ML-Specific Guidelines

- Always set random seeds for reproducibility
- Log experiments to Weights & Biases
- Save checkpoints every 1000 steps
- Document tensor shapes in function docstrings
- Use type hints for all torch.Tensor parameters

### Dependencies

- Use `uv` for package management
- Key packages: torch, transformers, datasets, wandb
- Dev tools: ruff, pyright, pytest, pre-commit

### Code Style

- Follow Google-style docstrings
- Use type hints for all function signatures
- Format with ruff (similar to Black)
- Keep functions focused and well-documented
- Only add comments when code is non-obvious
- Don't catch exceptions unless you plan to handle them meaningfully
- Use `raise NewException(...) from e` instead of logging and re-raising

### Common Type Issues

- When pyright complains about PyTorch Dataset not being Sized, use `cast(Sized, dataset)` from typing

## Project Goals

1. Start with single-digit addition: "3+5=8<end>"
2. Progress to multi-digit: "12+34=46<end>"
3. Custom character-level tokenizer (16 tokens total)
4. Small model (~1M-10M parameters) for RTX3060
5. Track training with W&B visualization
