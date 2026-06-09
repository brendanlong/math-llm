"""Model configuration management using YAML files and pydantic."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Configuration for model architecture.

    Supports four architectures:
    - standard: Traditional transformer with N unique layers
    - universal: Weight-shared layers applied in loops
    - feedback: Layers attend to shared memory of all previous layer outputs
    - ssm: Mamba-style selective state space model (no attention)
    """

    architecture: Literal["standard", "universal", "feedback", "ssm"] = "standard"

    # Core architecture parameters
    d_model: int = Field(gt=0, description="Model dimension")
    n_layers: int = Field(gt=0, description="Number of layers")
    n_heads: Optional[int] = Field(
        default=None, gt=0, description="Number of attention heads (not used by SSM)"
    )
    d_ff: Optional[int] = Field(
        default=None,
        gt=0,
        description="Feed-forward dimension (not used by SSM)",
    )
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
    layer_norm_type: Literal["pre", "post"] = Field(
        default="post",
        description="Layer norm placement: pre-LN (norm before sublayer) or post-LN (norm after residual add)",
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

    # SSM specific
    d_state: Optional[int] = Field(
        default=None,
        gt=0,
        description="SSM state dimension (ssm architecture only, default: 16)",
    )
    d_conv: Optional[int] = Field(
        default=None,
        gt=0,
        description="SSM convolution kernel size (ssm architecture only, default: 4)",
    )
    expand: Optional[int] = Field(
        default=None,
        gt=0,
        description="SSM expansion factor for inner dimension (ssm architecture only, default: 2)",
    )

    @model_validator(mode="after")
    def validate_architecture_config(self) -> "ModelConfig":
        """Validate architecture-specific parameters."""
        if self.architecture == "universal" and self.n_loops is None:
            raise ValueError("Universal transformer requires n_loops parameter")
        if self.architecture == "ssm":
            if self.n_heads is not None:
                raise ValueError("SSM architecture does not use n_heads")
            if self.d_ff is not None:
                raise ValueError("SSM architecture does not use d_ff")
        else:
            if self.n_heads is None:
                raise ValueError(
                    f"{self.architecture} architecture requires n_heads parameter"
                )
            if self.d_ff is None:
                raise ValueError(
                    f"{self.architecture} architecture requires d_ff parameter"
                )
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


def find_checkpoint_in_output_dir(output_dir: Path) -> Optional[Path]:
    """Find model checkpoint file in an output directory.

    Searches for checkpoint files in this order:
    1. model.safetensors (preferred format)
    2. pytorch_model.bin (legacy format)

    Args:
        output_dir: Path to the output/checkpoint directory

    Returns:
        Path to checkpoint file if found, None otherwise
    """
    # Check for safetensors format (preferred)
    safetensors_path = output_dir / "model.safetensors"
    if safetensors_path.exists():
        return safetensors_path

    # Check for PyTorch format (legacy)
    pytorch_path = output_dir / "pytorch_model.bin"
    if pytorch_path.exists():
        return pytorch_path

    return None


def resolve_checkpoint(
    path: Path,
    config_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Resolve a checkpoint argument to (checkpoint_file, config_file).

    Accepts either a checkpoint file (model.safetensors / pytorch_model.bin)
    or a directory containing one. The model config is auto-detected from the
    checkpoint directory unless explicitly provided.

    Args:
        path: Path to a checkpoint file or a directory containing one
        config_path: Optional explicit path to model_config.yaml

    Returns:
        Tuple of (checkpoint_file, config_file)

    Raises:
        FileNotFoundError: If no checkpoint or config can be found
    """
    if path.is_dir():
        checkpoint_path = find_checkpoint_in_output_dir(path)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin found in {path}"
            )
    elif path.is_file():
        checkpoint_path = path
    else:
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if config_path is None:
        config_path = find_config_in_checkpoint(checkpoint_path)
        if config_path is None:
            raise FileNotFoundError(
                f"No model_config.yaml found near {checkpoint_path}. "
                "Specify --config explicitly."
            )

    return checkpoint_path, config_path
