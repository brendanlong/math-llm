"""Small transformer model for arithmetic tasks.

This module implements a decoder-only transformer model designed for
learning basic arithmetic operations like addition.
"""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .tokenizer import VOCAB_SIZE

MAX_SEQUENCE_LENGTH = 1024


def create_cot_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Create mask for CoT content (True = keep, False = ignore in loss).

    Args:
        tokens: Token tensor of shape (batch_size, seq_len)

    Returns:
        Boolean mask of same shape where True means keep for loss computation
    """
    from .tokenizer import ArithmeticTokenizer

    tokenizer = ArithmeticTokenizer()

    # Token IDs for CoT tags
    think_digit_open = tokenizer.vocab["<think_digit>"]
    think_digit_close = tokenizer.vocab["</think_digit>"]
    think_multi_open = tokenizer.vocab["<think_multi>"]
    think_multi_close = tokenizer.vocab["</think_multi>"]

    batch_size, _ = tokens.shape
    mask = torch.ones_like(tokens, dtype=torch.bool)

    for b in range(batch_size):
        # Find all tag positions for this batch
        sequence = tokens[b]

        # Process digit CoT blocks
        digit_opens = (sequence == think_digit_open).nonzero(as_tuple=False).squeeze(-1)
        digit_closes = (
            (sequence == think_digit_close).nonzero(as_tuple=False).squeeze(-1)
        )
        _mask_cot_blocks(mask, b, digit_opens, digit_closes)

        # Process multi CoT blocks
        multi_opens = (sequence == think_multi_open).nonzero(as_tuple=False).squeeze(-1)
        multi_closes = (
            (sequence == think_multi_close).nonzero(as_tuple=False).squeeze(-1)
        )
        _mask_cot_blocks(mask, b, multi_opens, multi_closes)

    return mask


def _mask_cot_blocks(
    mask: torch.Tensor, batch_idx: int, opens: torch.Tensor, closes: torch.Tensor
) -> None:
    """Mask content between matched opening and closing tags.

    Args:
        mask: Mask tensor to update
        batch_idx: Batch index
        opens: Positions of opening tags
        closes: Positions of closing tags
    """
    for open_pos in opens:
        open_pos = open_pos.item()

        # Find the first closing tag after this opening tag
        close_idx = torch.searchsorted(closes, open_pos + 1)

        if close_idx < len(closes):
            close_pos = closes[close_idx].item()
            # Mask content between tags (excluding the tags themselves)
            mask[batch_idx, open_pos + 1 : close_pos] = False


def compute_loss(
    logits: torch.Tensor, labels: torch.Tensor, cot_agnostic: bool = False
) -> torch.Tensor:
    """Compute loss with optional CoT content masking.
    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        labels: Target labels of shape (batch_size, seq_len)
        cot_agnostic: If True, mask CoT content in loss computation
    Returns:
        Computed loss tensor
    """
    # Handle sequence length differences
    _, logits_seq_len, _ = logits.shape  # Fixed syntax
    labels_seq_len = labels.shape[1]
    min_seq_len = min(logits_seq_len, labels_seq_len)

    # Shift for next-token prediction
    shift_logits = logits[:, : min_seq_len - 1, :].contiguous()
    shift_labels = labels[:, 1:min_seq_len].contiguous()

    if cot_agnostic:
        # The mask should correspond to shift_labels, not original labels
        shifted_labels_for_mask = labels[:, 1:min_seq_len]
        shift_mask = create_cot_mask(shifted_labels_for_mask).contiguous()

        # Flatten tensors
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)
        shift_mask_flat = shift_mask.view(-1)

        # Apply mask - only compute loss on non-masked tokens
        if shift_mask_flat.any():
            masked_logits = shift_logits_flat[shift_mask_flat]
            masked_labels = shift_labels_flat[shift_mask_flat]
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(masked_logits, masked_labels)
        else:
            # Return zero tensor with proper device and gradient tracking
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )
    else:
        # Standard loss computation
        loss_fct = nn.CrossEntropyLoss()
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)
        return loss_fct(shift_logits_flat, shift_labels_flat)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 32):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encoding added
        """
        pe_slice = self.pe[: x.size(0), :]
        return x + pe_slice


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

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
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
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

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

    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal (lower triangular) attention mask.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask tensor
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        cot_agnostic: bool = False,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (unused)
            labels: Optional labels for computing loss
            cot_agnostic: Whether to use CoT-agnostic loss computation

        Returns:
            If labels provided: dict with 'loss' and 'logits'
            Otherwise: logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        _, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        x = x * math.sqrt(self.d_model)  # Scale embeddings

        # Positional encoding (convert to batch_first format)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # Create causal mask
        causal_mask = self._get_causal_mask(seq_len).to(input_ids.device)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        # Compute loss if labels are provided
        if labels is not None:
            loss = compute_loss(logits, labels, cot_agnostic=cot_agnostic)
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
