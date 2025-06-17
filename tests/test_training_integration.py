"""Integration test for data generation and training.

This test verifies that the model can achieve >99% token accuracy
on a small dataset with specific settings. It is excluded from the
normal test suite and should be run manually when needed.

Run with: pytest -m integration tests/test_training_integration.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
def test_training_achieves_high_accuracy():
    """Test that model can achieve >99% token accuracy with optimal settings."""
    # Create temporary directory for test data and checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "data"
        checkpoint_dir = temp_path / "checkpoints"

        # Create directories
        data_dir.mkdir()
        checkpoint_dir.mkdir()

        # Step 1: Generate data with specific parameters
        print("Generating test data...")
        generate_cmd = [
            "python",
            "scripts/generate_data.py",
            "--max-digits",
            "3",
            "--max-operands",
            "3",
            "--num-examples",
            "10000",
            "--seed",
            "42",
            "--output-dir",
            str(data_dir),
        ]

        result = subprocess.run(generate_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Data generation failed: {result.stderr}"

        # Verify data was generated
        assert (data_dir / "train.json").exists()
        assert (data_dir / "val.json").exists()
        assert (data_dir / "test.json").exists()

        # Step 2: Run training with optimized parameters
        print("Running training...")
        train_cmd = [
            "python",
            "scripts/train.py",
            "--model-size",
            "small",
            "--num-epochs",
            "100",
            "--batch-size",
            "512",
            "--max-length",
            "64",
            "--learning-rate",
            "0.001",
            "--warmup-steps",
            "100",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(checkpoint_dir),
            "--seed",
            "42",
            "--no-wandb",  # Disable W&B for testing
            "--logging-steps",
            "500",  # Less frequent logging
            "--eval-steps",
            "5000",  # Less frequent evaluation
            "--save-steps",
            "10000",  # Less frequent saving
        ]

        result = subprocess.run(train_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Training failed: {result.stderr}"

        # Step 3: Check final accuracy
        test_results_path = checkpoint_dir / "test_results.json"
        assert test_results_path.exists(), "Test results file not found"

        with open(test_results_path) as f:
            test_results = json.load(f)

        token_accuracy = test_results.get("eval_token_accuracy", 0)
        print(f"Final token accuracy: {token_accuracy:.4f}")

        # Assert high accuracy
        assert token_accuracy > 0.99, (
            f"Token accuracy {token_accuracy:.4f} is below required 99%. "
            "Model may need more training or hyperparameter tuning."
        )

        # Optional: Check loss is low
        eval_loss = test_results.get("eval_loss", float("inf"))
        assert eval_loss < 0.1, f"Eval loss {eval_loss:.4f} is too high"


if __name__ == "__main__":
    # Allow running directly with python
    test_training_achieves_high_accuracy()
    print("✅ All tests passed!")
