"""Small transformer model for arithmetic tasks.

This module implements a decoder-only transformer model designed for
learning basic arithmetic operations like addition.
"""

import math
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .tokenizer import VOCAB_SIZE

MAX_SEQUENCE_LENGTH = 1024


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute next-token prediction loss.

    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        labels: Target labels of shape (batch_size, seq_len)

    Returns:
        Computed loss tensor
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Handle sequence length differences after shifting
    min_len = min(shift_logits.shape[1], shift_labels.shape[1])
    shift_logits = shift_logits[:, :min_len, :]
    shift_labels = shift_labels[:, :min_len]

    # Flatten tensors
    shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels_flat = shift_labels.view(-1)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits_flat, shift_labels_flat)


def build_alibi_bias(
    n_heads: int, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build ALiBi (Attention with Linear Biases) position bias matrix.

    Args:
        n_heads: Number of attention heads
        seq_len: Sequence length
        device: Device to create tensor on
        dtype: Data type of the tensor

    Returns:
        ALiBi bias tensor of shape (n_heads, seq_len, seq_len)
    """
    # Create position indices
    positions = torch.arange(seq_len, device=device, dtype=dtype)

    # Calculate slopes for each head
    # For n heads, we want slopes that are geometric sequence of 2^(-8/n), 2^(-16/n), ...
    slopes = torch.tensor(
        [2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)],
        device=device,
        dtype=dtype,
    )

    # Create relative position matrix (j - i for all i, j)
    # This gives negative values for future positions (j > i) and positive for past
    relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)

    # Apply slopes to get biases for each head
    # We want to penalize attention to future positions, so multiply by negative slopes
    # Shape: (n_heads, seq_len, seq_len)
    alibi = -slopes.unsqueeze(1).unsqueeze(2) * relative_positions.unsqueeze(0).abs()

    return alibi


class TransformerBlock(nn.Module):
    """Single transformer decoder block with masked self-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        # Use separate linear layers for Q, K, V to have more control
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        alibi_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            alibi_bias: ALiBi bias tensor of shape (n_heads, seq_len, seq_len)

        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = (
            self.q_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))

        # Combine ALiBi bias with causal mask
        # ALiBi bias shape: (n_heads, seq_len, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        attn_mask = alibi_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # Causal mask shape: (seq_len, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        causal_mask = (
            causal_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, self.n_heads, -1, -1)
        )
        attn_mask = attn_mask + causal_mask

        # Use optimized scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,  # Always False since we're using explicit mask with ALiBi
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        # Residual connection and layer norm
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class ArithmeticModel(nn.Module):
    """Small transformer model for arithmetic tasks."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        dropout: float = 0.1,
    ):
        """Initialize the arithmetic model.

        Args:
            vocab_size: Size of vocabulary (default from tokenizer)
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.n_heads = n_heads

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            bias = getattr(module, "bias", None)
            if bias is not None:
                torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (unused)
            labels: Optional labels for computing loss

        Returns:
            If labels provided: dict with 'loss' and 'logits'
            Otherwise: logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        _, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        x = x * math.sqrt(self.d_model)  # Scale embeddings

        # Build ALiBi bias
        alibi_bias = build_alibi_bias(
            self.n_heads, seq_len, input_ids.device, dtype=x.dtype
        )

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, alibi_bias=alibi_bias)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        # Compute loss if labels are provided
        if labels is not None:
            loss = compute_loss(logits, labels)
            return {"loss": loss, "logits": logits}

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        end_token_id: int = 12,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Initial input tokens of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k sampling
            end_token_id: Token ID for end-of-sequence

        Returns:
            Generated tokens of shape (batch_size, seq_len + num_generated)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get predictions for current sequence
            with torch.no_grad():
                outputs = self.forward(input_ids)
                # Extract logits (forward returns dict when labels provided, tensor otherwise)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Get logits for last token
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop if end token is generated
                if next_token.item() == end_token_id:
                    break

        return input_ids

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


def create_extra_small_model() -> ArithmeticModel:
    """Create an extra-small model configuration."""
    return ArithmeticModel(
        d_model=32,
        n_layers=4,
        n_heads=4,
        d_ff=128,
        dropout=0.1,
    )


def create_small_model() -> ArithmeticModel:
    """Create a small model configuration (~1M parameters)."""
    return ArithmeticModel(
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
    )


def create_medium_model() -> ArithmeticModel:
    """Create a medium model configuration (~5M parameters)."""
    return ArithmeticModel(
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
    )


def create_large_model() -> ArithmeticModel:
    """Create a large model configuration (~10M parameters)."""
    return ArithmeticModel(
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
    )


ModelSizeStr = Literal["xsmall", "small", "medium", "large"]


def create_model(
    model_size: ModelSizeStr,
) -> ArithmeticModel:
    if model_size == "xsmall":
        return create_extra_small_model()
    elif model_size == "small":
        return create_small_model()
    elif model_size == "medium":
        return create_medium_model()
    elif model_size == "large":
        return create_large_model()
    else:
        raise ValueError(f"Unknown model size: {model_size}")
