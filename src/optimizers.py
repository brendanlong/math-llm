"""Optimizer factory for training.

Provides a unified interface for creating optimizers:
- AdamW: PyTorch built-in (torch.optim.AdamW)
- Muon: From pytorch-optimizer (handles 2D weights + AdamW fallback)
- ADOPT: From pytorch-optimizer package
"""

from pytorch_optimizer import (
    ADOPT,  # type: ignore[import-untyped]
    Muon,  # type: ignore[import-untyped]
)
from torch import Tensor
from torch.optim import AdamW, Optimizer

from .model import BaseModel


def create_optimizer(
    model: BaseModel,
    optimizer_name: str,
    lr: float,
    weight_decay: float = 0.01,
) -> Optimizer:
    """Create an optimizer for the model.

    Args:
        model: The model to optimize
        optimizer_name: One of "adamw", "adopt", or "muon"
        lr: Learning rate
        weight_decay: Weight decay coefficient

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer_name is not recognized
    """
    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif optimizer_name == "adopt":
        # Use weight_decouple=True for AdamW-style decoupled weight decay
        # as recommended by ADOPT authors when replacing AdamW
        return ADOPT(
            model.parameters(), lr=lr, weight_decay=weight_decay, weight_decouple=True
        )

    elif optimizer_name == "muon":
        # Muon works best on 2D+ weight matrices
        # Use AdamW for embeddings, LayerNorm, biases (via use_muon=False)
        muon_params: list[Tensor] = []
        adamw_params: list[Tensor] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Use AdamW for embeddings, LayerNorm, biases, and 1D params
            is_embedding = "embedding" in name.lower()
            is_norm = "norm" in name.lower() or "ln" in name.lower()
            is_bias = "bias" in name.lower() or param.ndim < 2

            if is_embedding or is_norm or is_bias:
                adamw_params.append(param)
            else:
                muon_params.append(param)

        # Create param groups with use_muon flag
        param_groups = [
            {"params": muon_params, "use_muon": True},
            {"params": adamw_params, "use_muon": False},
        ]

        return Muon(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            adamw_lr=lr,
            adamw_wd=weight_decay,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Choose from: adamw, adopt, muon"
        )
