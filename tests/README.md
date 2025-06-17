# Tests

## Running Tests

### Regular Unit Tests

Run the normal test suite (excludes integration tests):

```bash
pytest
```

### Integration Tests

Integration tests verify end-to-end functionality including data generation and training.
They are excluded from normal test runs because they take longer to execute.

To run integration tests:

```bash
# Run only integration tests
pytest -m integration tests/test_training_integration.py

# Run all tests including integration
pytest -m ""

# Run specific integration test
pytest -m integration -k test_training_achieves_high_accuracy
```

## Test Structure

- `test_*.py` - Unit tests that run quickly
- `test_training_integration.py` - Integration tests that generate data and train models

## Integration Test Details

The main integration test (`test_training_achieves_high_accuracy`) verifies that:

1. Data generation works correctly with 3-digit numbers and up to 3 operands
2. The model can achieve >99% token accuracy after 100 epochs
3. Training is deterministic with fixed seeds

Settings optimized for fast convergence:

- 10,000 training examples
- Batch size: 512
- Learning rate: 0.001
- Max sequence length: 64
