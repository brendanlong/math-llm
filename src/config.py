"""Model configuration management using YAML files and pydantic."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Configuration for model architecture.

    Supports three transformer architectures:
    - standard: Traditional transformer with N unique layers
    - universal: Weight-shared layers applied in loops
    - feedback: Layers attend to shared memory of all previous layer outputs
    """

    architecture: Literal["standard", "universal", "feedback"] = "standard"

    # Core architecture parameters
    d_model: int = Field(gt=0, description="Model dimension")
    n_layers: int = Field(gt=0, description="Number of transformer layers")
    n_heads: int = Field(gt=0, description="Number of attention heads")
    d_ff: int = Field(gt=0, description="Feed-forward dimension")
    dropout: float = Field(
        ge=0.0, le=1.0, default=0.1, description="Dropout probability"
    )

    # Attention variants
    positional_encoding: Literal["learned", "sinusoidal", "pope", "rope"] = Field(
        default="learned",
        description="Positional encoding type: learned, sinusoidal, pope, or rope",
    )
    softmax_variant: Literal["standard", "softmax1"] = Field(
        default="standard",
        description="Softmax variant: standard or softmax1 (+1 in denominator)",
    )

    # Universal transformer specific
    n_loops: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of loop iterations (universal transformer only)",
    )
    use_loop_embeddings: bool = Field(
        default=True,
        description="Whether to use loop position embeddings (universal transformer)",
    )

    @model_validator(mode="after")
    def validate_universal_config(self) -> "ModelConfig":
        """Validate that universal transformer has required n_loops parameter."""
        if self.architecture == "universal" and self.n_loops is None:
            raise ValueError("Universal transformer requires n_loops parameter")
        return self


def load_config(path: Path) -> ModelConfig:
    """Load model configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Validated ModelConfig instance

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    return ModelConfig.model_validate(data)


def save_config(config: ModelConfig, path: Path) -> None:
    """Save model configuration to a YAML file.

    Args:
        config: ModelConfig instance to save
        path: Destination path for the YAML file
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Exclude None values and defaults for cleaner output
    data = config.model_dump(exclude_none=True)

    with path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def find_config_in_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    """Find model_config.yaml in a checkpoint directory.

    Searches for config in these locations (in order):
    1. Same directory as checkpoint file (model_config.yaml)
    2. Parent directory of checkpoint file (for nested checkpoint dirs)

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Path to config file if found, None otherwise
    """
    # Get the directory containing the checkpoint
    if checkpoint_path.is_file():
        checkpoint_dir = checkpoint_path.parent
    else:
        checkpoint_dir = checkpoint_path

    # Check for config in same directory
    config_path = checkpoint_dir / "model_config.yaml"
    if config_path.exists():
        return config_path

    # Check parent directory (for checkpoint-N subdirectories)
    parent_config = checkpoint_dir.parent / "model_config.yaml"
    if parent_config.exists():
        return parent_config

    return None
