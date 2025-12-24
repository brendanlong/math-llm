"""Integration tests for data generation and training.

These tests verify that different training modes and model configurations
work without crashing. They are excluded from the normal test suite and
should be run manually when needed.

Fast integration tests (run quickly, just test for crashes):
    pytest -m integration tests/test_training_integration.py

Slow integration tests (run full training to convergence):
    pytest -m slow_integration tests/test_training_integration.py

Run all integration tests:
    pytest -m "integration or slow_integration" tests/test_training_integration.py

Run specific tests:
    pytest -m integration tests/test_training_integration.py::test_standard_training_runs
    pytest -m integration tests/test_training_integration.py::test_different_model_configs_train
    pytest -m integration tests/test_training_integration.py::test_model_generation_after_training
    pytest -m slow_integration tests/test_training_integration.py::test_training_achieves_high_accuracy

Test coverage:
- Standard training mode (teacher forcing)
- Different model configs (xsmall, small)
- Model generation after training
- Training convergence to high accuracy (slow test)
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.slow_integration
def test_training_achieves_high_accuracy() -> None:
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
            "--config",
            "config/standard-small.yaml",
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


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_file", ["config/standard-xsmall.yaml", "config/standard-small.yaml"]
)
def test_different_model_configs_train(config_file: str) -> None:
    """Test that different model configs can train without crashing."""
    # Create temporary directory for test data and checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "data"
        checkpoint_dir = temp_path / "checkpoints"

        # Create directories
        data_dir.mkdir()
        checkpoint_dir.mkdir()

        # Step 1: Generate small dataset
        print(f"Generating test data for {config_file}...")
        generate_cmd = [
            "python",
            "scripts/generate_data.py",
            "--max-digits",
            "1",
            "--max-operands",
            "2",
            "--num-examples",
            "50",
            "--seed",
            "42",
            "--output-dir",
            str(data_dir),
        ]

        result = subprocess.run(generate_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Data generation failed: {result.stderr}"

        # Determine batch size based on config
        batch_size = "4" if "small" in config_file else "8"

        # Step 2: Run training with specified config
        print(f"Running training with {config_file}...")
        train_cmd = [
            "python",
            "scripts/train.py",
            "--config",
            config_file,
            "--num-epochs",
            "1",
            "--batch-size",
            batch_size,
            "--max-length",
            "32",
            "--learning-rate",
            "0.01",
            "--warmup-steps",
            "5",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(checkpoint_dir),
            "--seed",
            "42",
            "--no-wandb",
            "--logging-steps",
            "5",
            "--eval-steps",
            "25",
            "--save-steps",
            "100",
        ]

        result = subprocess.run(train_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"{config_file} training failed: {result.stderr}"

        # Just verify training completed without crash
        print(f"✅ {config_file} training completed successfully")


@pytest.mark.integration
def test_standard_training_runs() -> None:
    """Test that standard training mode runs without crashing."""
    # Create temporary directory for test data and checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "data"
        checkpoint_dir = temp_path / "checkpoints"

        # Create directories
        data_dir.mkdir()
        checkpoint_dir.mkdir()

        # Step 1: Generate small dataset
        print("Generating test data for standard training...")
        generate_cmd = [
            "python",
            "scripts/generate_data.py",
            "--max-digits",
            "1",
            "--max-operands",
            "2",
            "--num-examples",
            "100",
            "--seed",
            "42",
            "--output-dir",
            str(data_dir),
        ]

        result = subprocess.run(generate_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Data generation failed: {result.stderr}"

        # Step 2: Run standard training (very short training)
        print("Running standard training...")
        train_cmd = [
            "python",
            "scripts/train.py",
            "--config",
            "config/standard-xsmall.yaml",
            "--num-epochs",
            "1",
            "--batch-size",
            "8",
            "--max-length",
            "32",
            "--learning-rate",
            "0.01",
            "--warmup-steps",
            "10",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(checkpoint_dir),
            "--seed",
            "42",
            "--no-wandb",
            "--logging-steps",
            "5",
            "--eval-steps",
            "50",
            "--save-steps",
            "100",
        ]

        result = subprocess.run(train_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Standard training failed: {result.stderr}"

        # Just verify training completed without crash
        print("✅ Standard training completed successfully")


@pytest.mark.integration
def test_model_generation_after_training() -> None:
    """Test that model can generate completions after training."""
    # Create temporary directory for test data and checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "data"
        checkpoint_dir = temp_path / "checkpoints"

        # Create directories
        data_dir.mkdir()
        checkpoint_dir.mkdir()

        # Step 1: Generate small dataset
        print("Generating test data for generation test...")
        generate_cmd = [
            "python",
            "scripts/generate_data.py",
            "--max-digits",
            "1",
            "--max-operands",
            "2",
            "--num-examples",
            "100",
            "--seed",
            "42",
            "--output-dir",
            str(data_dir),
        ]

        result = subprocess.run(generate_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Data generation failed: {result.stderr}"

        # Step 2: Run short training to get a model
        print("Running training to create model...")
        train_cmd = [
            "python",
            "scripts/train.py",
            "--config",
            "config/standard-xsmall.yaml",
            "--num-epochs",
            "5",
            "--batch-size",
            "16",
            "--max-length",
            "32",
            "--learning-rate",
            "0.01",
            "--warmup-steps",
            "10",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(checkpoint_dir),
            "--seed",
            "42",
            "--no-wandb",
            "--logging-steps",
            "10",
            "--eval-steps",
            "50",
            "--save-steps",
            "100",
        ]

        result = subprocess.run(train_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Training failed: {result.stderr}"

        # Step 3: Test that the model can generate (interactive script with single input)
        print("Testing model generation...")

        # Find the final model file
        model_files = list(checkpoint_dir.glob("*.safetensors"))
        if not model_files:
            model_files = list(checkpoint_dir.glob("**/pytorch_model.bin"))

        assert model_files, "No model checkpoint found after training"
        model_path = model_files[0]

        # Test with evaluate script - config is auto-detected from checkpoint dir
        eval_cmd = [
            "python",
            "scripts/evaluate.py",
            "--checkpoint",
            str(model_path),
            "--data-path",
            str(data_dir / "test.json"),
            "--batch-size",
            "8",
        ]

        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Model evaluation failed: {result.stderr}"

        print("✅ Model generation test completed successfully")


if __name__ == "__main__":
    # Allow running directly with python
    test_standard_training_runs()
    test_model_generation_after_training()
    test_training_achieves_high_accuracy()
    print("✅ All tests passed!")
