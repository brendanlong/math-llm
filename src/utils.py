"""Shared utility functions for the math LLM project."""

import logging
from pathlib import Path

import colorlog
import torch
from safetensors.torch import load_file

from .config import load_config
from .model import Model, create_model_from_config


def setup_logging(include_file_handler: bool = False) -> None:
    """Setup colored logging configuration.

    Args:
        include_file_handler: Whether to include a file handler for logs/training.log
    """
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    if include_file_handler:
        import os

        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/training.log")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)-8s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)


def load_model(
    checkpoint_path: Path,
    config_path: Path,
) -> Model:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model configuration YAML file

    Returns:
        Loaded model

    Raises:
        FileNotFoundError: If checkpoint or config file doesn't exist
    """
    config = load_config(config_path)
    model = create_model_from_config(config)

    # Load checkpoint - handle different formats
    if checkpoint_path.suffix == ".safetensors":
        # Load safetensors format
        state_dict = load_file(str(checkpoint_path))
        model.load_state_dict(state_dict)
    else:
        # Load PyTorch format
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # Assume checkpoint is the state dict directly
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

    return model


def get_device(device_str: str = "auto") -> torch.device:
    """Get the appropriate torch device.

    Args:
        device_str: Device specification ("cuda", "cpu", or "auto")

    Returns:
        Configured torch.device
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
