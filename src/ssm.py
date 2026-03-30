"""Selective State Space Model (SSM) for arithmetic tasks.

Implements a Mamba-style architecture with selective state spaces instead of
attention. Each SSM block uses input-dependent (selective) state transitions,
enabling the model to filter information based on content.

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
(Gu & Dao, 2023)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig
from .model import BaseModel, _init_weights, compute_loss
from .tokenizer import VOCAB_SIZE

# Architecture type for SSM
SSM_ARCHITECTURE = "ssm"


class SelectiveSSM(nn.Module):
    """Selective state space model core computation.

    Implements the selective scan: given input x, computes output y through
    a discretized state space with input-dependent parameters.

    State equation: h_t = A_bar * h_{t-1} + B_bar * x_t
    Output equation: y_t = C_t * h_t + D * x_t

    Where A_bar, B_bar are discretized using dt (input-dependent step size),
    and B, C are also input-dependent (selective).
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = 0):
        """Initialize selective SSM.

        Args:
            d_inner: Inner dimension (expansion of d_model)
            d_state: SSM state dimension (N in the paper)
            dt_rank: Rank for dt projection (0 = auto = ceil(d_inner / 16))
        """
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank > 0 else math.ceil(d_inner / 16)

        # Input-dependent projections for dt, B, C
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt projection from low rank to full dimension
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)

        # Initialize A as a diagonal matrix (log-spaced for stability)
        # A is (d_inner, d_state) - each channel has its own state dynamics
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D is a skip connection parameter
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run selective scan over input sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, d_inner)

        Returns:
            Output tensor of shape (batch, seq_len, d_inner)
        """
        batch, seq_len, _ = x.shape

        # Project input to get dt, B, C (all input-dependent / selective)
        x_proj = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)

        dt = x_proj[:, :, : self.dt_rank]  # (batch, seq_len, dt_rank)
        B = x_proj[:, :, self.dt_rank : self.dt_rank + self.d_state]  # (batch, seq_len, d_state)
        C = x_proj[:, :, self.dt_rank + self.d_state :]  # (batch, seq_len, d_state)

        # Project dt to full dimension and apply softplus for positivity
        dt = F.softplus(self.dt_proj(dt))  # (batch, seq_len, d_inner)

        # Get A from log parameterization
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        # For efficiency, we compute element-wise
        # dt: (batch, seq_len, d_inner) -> (batch, seq_len, d_inner, 1)
        dt_expanded = dt.unsqueeze(-1)  # (batch, seq_len, d_inner, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)

        A_bar = torch.exp(dt_expanded * A_expanded)  # (batch, seq_len, d_inner, d_state)
        # B: (batch, seq_len, d_state) -> (batch, seq_len, 1, d_state)
        B_bar = dt_expanded * B.unsqueeze(2)  # (batch, seq_len, d_inner, d_state)

        # Sequential scan
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # y_t = C_t * h_t (sum over state dimension)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_inner)

        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y


class MambaBlock(nn.Module):
    """Single Mamba block: norm -> SSM layer with gating.

    Architecture:
    1. LayerNorm
    2. Linear projection to 2*d_inner (split into x and gate z)
    3. Conv1d on x path
    4. SiLU activation
    5. Selective SSM on x
    6. Multiply by SiLU(z) (gating)
    7. Linear projection back to d_model
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize Mamba block.

        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model

        # Pre-norm
        self.norm = nn.LayerNorm(d_model)

        # Input projection: d_model -> 2*d_inner (x path + gate path)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal conv1d on x path
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal padding
            groups=self.d_inner,  # depthwise
            bias=True,
        )

        # Selective SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state)

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Project to 2*d_inner and split
        xz = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)  # each (batch, seq_len, d_inner)

        # Causal conv1d on x path (transpose for conv1d: batch, channels, seq_len)
        x_path = x_path.transpose(1, 2)
        x_path = self.conv1d(x_path)[:, :, : x.shape[1]]  # trim causal padding
        x_path = x_path.transpose(1, 2)

        # SiLU activation after conv
        x_path = F.silu(x_path)

        # Selective SSM
        x_path = self.ssm(x_path)

        # Gate with SiLU(z)
        x_path = x_path * F.silu(z)

        # Output projection
        out = self.out_proj(x_path)

        return residual + self.dropout(out)


class SSMModel(BaseModel):
    """Mamba-style SSM model for arithmetic tasks.

    Uses selective state space layers instead of attention. Each layer
    processes the sequence with input-dependent state transitions, allowing
    the model to selectively remember or forget information.
    """

    architecture = "ssm"  # type: ignore[assignment]

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 5,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        """Initialize SSM model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_layers: Number of Mamba blocks
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_heads = 1  # SSMs don't have heads, but BaseModel expects this
        self.n_layers = n_layers
        self.embed_scale = math.sqrt(d_model)

        # Embedding (no positional encoding needed - SSMs are inherently sequential)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Stack of Mamba blocks
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(_init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_attention_scores: bool = False,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]] | torch.Tensor:
        """Forward pass through the SSM model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            labels: Optional labels for computing loss
            output_attentions: Ignored (SSMs don't have attention)
            output_attention_scores: Ignored (SSMs don't have attention)

        Returns:
            If labels provided: dict with 'loss' and 'logits'
            Otherwise: logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        del output_attentions, output_attention_scores

        # Token embeddings with scaling
        x = self.token_embedding(input_ids)
        x.mul_(self.embed_scale)

        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)

        # Final norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if labels is not None:
            result: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {
                "loss": compute_loss(logits, labels),
                "logits": logits,
            }
            return result

        return logits


def create_ssm_from_config(config: ModelConfig) -> SSMModel:
    """Create an SSM model from config.

    Args:
        config: ModelConfig with SSM-specific parameters

    Returns:
        SSMModel instance
    """
    return SSMModel(
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state or 16,
        d_conv=config.d_conv or 4,
        expand=config.expand or 2,
        dropout=config.dropout,
    )
