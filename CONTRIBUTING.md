# Contributing to Math LLM

## Development Setup

```bash
# Clone and setup
git clone <repo-url>
cd math-llm
./setup.sh
```

Development tools are configured in `pyproject.toml` for consistency and easy management.

### Bash Scripts

All bash scripts should be run with `set -e` to exit on any error, ensuring failures are caught early.

## Code Quality

### Formatting

We use `ruff` for code formatting:

```bash
# Format code
ruff format .

# Check formatting
ruff format --check .
```

### Type Checking

We use `pyright` for static type analysis:

```bash
# Run type checking
pyright
```

### Linting

Additional linting with `ruff`:

```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .
```

## Code Standards

### Type Hints

- All function signatures must include type hints
- Use `torch.Tensor` for tensor types
- Prefer built-in types (`list`, `dict`, `tuple`) over `typing` imports when available (Python 3.9+)
- Use `typing` module only for complex types like `Union`, `Optional`, `Callable`
- For complex dictionary structures, use `TypedDict` instead of `dict[str, Any]` to provide explicit type safety

### Documentation

- Docstrings for all public functions and classes
- Use Google-style docstrings
- Document tensor shapes in docstrings: `(batch_size, seq_len, hidden_size)`

### Testing

- Write tests for all data generation and model utilities
- Use `pytest` for testing framework
- Place tests in `tests/` directory

## ML Development Guidelines

### Experiment Tracking

- Log all experiments to Weights & Biases
- Use descriptive experiment names with hyperparameters
- Tag experiments with model version and data version

### Reproducibility

- Set random seeds in all scripts
- Save hyperparameters with each checkpoint
- Version control data generation scripts

### Model Checkpoints

- Save checkpoints every 1000 training steps
- Include optimizer state for resuming training
- Save best model based on validation accuracy

### Data Management

- Keep generated datasets under version control metadata
- Document data generation parameters
- Use consistent train/val/test splits

## Pre-commit Checks

We use `pre-commit` to ensure code quality. Run checks with:

```bash
pre-commit run
```

## Performance Guidelines

### Memory Management

- Use gradient checkpointing for large models
- Clear cache between experiments: `torch.cuda.empty_cache()`
- Monitor GPU memory usage during training

### Training Efficiency

- Use mixed precision training when possible
- Optimize data loading with appropriate num_workers
- Profile training loops to identify bottlenecks
