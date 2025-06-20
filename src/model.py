"""Small transformer model for arithmetic tasks.

This module implements a decoder-only transformer model designed for
learning basic arithmetic operations like addition.
"""

import math
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .tokenizer import VOCAB, VOCAB_SIZE

MAX_SEQUENCE_LENGTH = 1024


def create_reasoning_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Create mask for content between <think> and </think> tags (vectorized).

    Assumes only one <think>...</think> pair per sequence for performance.

    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)

    Returns:
        Boolean mask of shape (batch_size, seq_len) where True indicates
        positions that should be masked (content between think tags,
        excluding the tags themselves)
    """
    _, seq_len = input_ids.shape
    think_start_id = VOCAB["<think>"]
    think_end_id = VOCAB["</think>"]

    # Find positions of <think> and </think> tokens
    think_start_mask = input_ids == think_start_id
    think_end_mask = input_ids == think_end_id

    # Find first <think> position in each sequence (-1 if not found)
    start_positions = think_start_mask.float().argmax(dim=1)  # (batch_size,)
    has_start = think_start_mask.any(dim=1)  # (batch_size,)
    start_positions = torch.where(has_start, start_positions, -1)

    # Find first </think> position in each sequence (-1 if not found)
    end_positions = think_end_mask.float().argmax(dim=1)  # (batch_size,)
    has_end = think_end_mask.any(dim=1)  # (batch_size,)
    end_positions = torch.where(has_end, end_positions, -1)

    # Create position indices
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(
        0
    )  # (1, seq_len)

    # Create mask: True for positions between start+1 and end (exclusive)
    # Mask is True where: start_pos < position < end_pos AND both start/end exist
    valid_pairs = has_start & has_end & (start_positions < end_positions)

    # Expand dimensions for broadcasting
    start_expanded = start_positions.unsqueeze(1)  # (batch_size, 1)
    end_expanded = end_positions.unsqueeze(1)  # (batch_size, 1)
    valid_expanded = valid_pairs.unsqueeze(1)  # (batch_size, 1)

    # Create the mask: position > start AND position < end AND valid_pair
    mask = (positions > start_expanded) & (positions < end_expanded) & valid_expanded

    return mask


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_reasoning: bool = False,
) -> torch.Tensor:
    """Compute completion-only loss (only on tokens after = sign).

    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        labels: Target labels of shape (batch_size, seq_len)
        mask_reasoning: If True, mask reasoning content between <think> and </think>

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

    # Apply reasoning masking if requested
    if mask_reasoning:
        # Create reasoning mask from labels (not input_ids) to mask based on ground truth
        # We need the original labels before shifting to find <think>...</think> positions
        reasoning_mask = create_reasoning_mask(labels)

        # Shift the reasoning mask to align with shifted labels
        reasoning_mask_shifted = reasoning_mask[:, 1 : min_len + 1]

        # Never mask special tokens - always train on them to prevent malformed reasoning
        special_tokens = torch.tensor(
            [VOCAB["<end>"], VOCAB["<think>"], VOCAB["</think>"]], device=labels.device
        )
        original_labels_shifted = labels[:, 1 : min_len + 1]
        # Check if any position contains a special token (broadcasting)
        is_special_token = (
            original_labels_shifted.unsqueeze(-1) == special_tokens
        ).any(dim=-1)
        reasoning_mask_shifted = reasoning_mask_shifted & ~is_special_token

        # Set masked positions to ignore index (-100)
        shift_labels = shift_labels.masked_fill(reasoning_mask_shifted, -100)

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

    def _gumbel_softmax(
        self, logits: torch.Tensor, temperature: float = 1.0, hard: bool = False
    ) -> torch.Tensor:
        """Apply Gumbel-Softmax to logits.

        Args:
            logits: Logits tensor of shape (..., vocab_size)
            temperature: Temperature for Gumbel-Softmax
            hard: If True, returns one-hot vectors (straight-through estimator)

        Returns:
            Soft (or hard) token probabilities
        """
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)

    def _forward_with_gumbel(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        temperature: float = 1.0,
        mask_reasoning: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Forward pass with Gumbel-Softmax generation.

        Instead of teacher forcing, generate the full sequence using Gumbel-Softmax
        sampling to maintain differentiability.

        Args:
            input_ids: Input token IDs
            labels: Full target sequence including the answer
            temperature: Gumbel-Softmax temperature
            mask_reasoning: Whether to mask reasoning content between <think> and </think>

        Returns:
            Dictionary with loss and generated logits
        """
        batch_size, seq_len = input_ids.shape

        # Use input_ids as-is and compute logits for the full sequence
        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        x = x * math.sqrt(self.d_model)  # Scale embeddings

        # For the generation part, we'll replace some tokens with Gumbel-sampled ones
        # Find the equals sign to determine where generation should start
        equals_token_id = VOCAB["="]

        # Find positions of equals signs
        equals_mask = input_ids == equals_token_id

        # Check that each sequence has at least one equals sign
        equals_per_sequence = equals_mask.sum(dim=1)
        assert (equals_per_sequence >= 1).all(), (
            "Each sequence must have at least one equals sign"
        )

        # Find the position of the FIRST equals sign in each sequence
        # argmax on a boolean tensor gives the first True position
        generation_starts = (
            equals_mask.float().argmax(dim=1) + 1
        )  # +1 to start after equals

        # Process sequences with differentiable generation after equals
        all_logits = []

        # Initialize embeddings tensor (more efficient than list operations)
        current_embeddings = x.clone()  # (batch_size, seq_len, d_model)

        for pos in range(seq_len):
            # Get embeddings up to current position (avoid tensor stacking)
            pos_embeddings = current_embeddings[:, : pos + 1, :]

            # Build ALiBi bias for current position
            alibi_bias = build_alibi_bias(
                self.n_heads, pos + 1, input_ids.device, dtype=pos_embeddings.dtype
            )

            # Apply transformer layers
            hidden = pos_embeddings
            for layer in self.layers:
                hidden = layer(hidden, alibi_bias=alibi_bias)

            # Final layer norm and projection
            hidden = self.ln_f(hidden)
            logits = self.lm_head(hidden)  # (batch_size, pos+1, vocab_size)

            # Get logits for current position
            current_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            all_logits.append(current_logits)

            # For positions after equals sign, use Gumbel-Softmax
            should_generate = pos >= generation_starts  # Boolean mask (batch_size,)

            if should_generate.any() and pos + 1 < seq_len:
                # Apply Gumbel-Softmax to get soft token probabilities
                soft_probs = self._gumbel_softmax(
                    current_logits,
                    temperature=temperature,
                    hard=False,
                )  # (batch_size, vocab_size)

                # Compute soft embeddings for next position
                soft_embeddings = (
                    soft_probs @ self.token_embedding.weight
                )  # (batch_size, d_model)

                # Update embeddings for next position (avoid in-place to preserve gradients)
                next_pos_embeddings = torch.where(
                    should_generate.unsqueeze(1),
                    soft_embeddings,
                    current_embeddings[:, pos + 1, :],
                )
                # Create new tensor with updated embeddings
                current_embeddings = torch.cat(
                    [
                        current_embeddings[:, : pos + 1, :],
                        next_pos_embeddings.unsqueeze(1),
                        current_embeddings[:, pos + 2 :, :]
                        if pos + 2 < seq_len
                        else torch.empty(
                            batch_size,
                            0,
                            self.d_model,
                            device=current_embeddings.device,
                        ),
                    ],
                    dim=1,
                )

        # Stack all logits
        logits = torch.stack(all_logits, dim=1)  # (batch_size, seq_len, vocab_size)

        # Compute loss using standard method
        if labels is not None:
            loss = compute_loss(logits, labels, mask_reasoning)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_gumbel: bool = False,
        gumbel_temperature: float = 1.0,
        mask_reasoning: bool = False,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (unused)
            labels: Optional labels for computing loss
            use_gumbel: Whether to use Gumbel-Softmax for differentiable generation
            gumbel_temperature: Temperature for Gumbel-Softmax (lower = more discrete)
            mask_reasoning: Whether to mask reasoning content between <think> and </think>

        Returns:
            If labels provided: dict with 'loss' and 'logits'
            Otherwise: logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        if use_gumbel and labels is not None and self.training:
            # Generate full sequence using Gumbel-Softmax (only during training)
            return self._forward_with_gumbel(
                input_ids, labels, gumbel_temperature, mask_reasoning
            )

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
            loss = compute_loss(logits, labels, mask_reasoning)
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
