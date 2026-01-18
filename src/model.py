"""Small transformer model for arithmetic tasks.

This module implements a decoder-only transformer model designed for
learning basic arithmetic operations like addition.
"""

import math
from typing import Any, Optional, cast

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig
from .tokenizer import VOCAB_SIZE

MAX_SEQUENCE_LENGTH = 1024


def softmax1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with +1 in denominator, allowing attention heads to output nothing.

    Formula: softmax1(x)_i = exp(x_i) / (1 + Σ_j exp(x_j))

    This modification adds an implicit "zero" option that attention heads can use
    when they have no meaningful information to contribute.

    Args:
        x: Input tensor
        dim: Dimension to apply softmax over

    Returns:
        Tensor with same shape as input, values sum to less than 1
    """
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return exp_x / (1.0 + exp_x.sum(dim=dim, keepdim=True))


class PoPE(nn.Module):
    """Polar Positional Encoding (PoPE).

    PoPE separates "what" (content) from "where" (position) in attention by using
    polar coordinates. Magnitudes encode content information via softplus activation,
    while phases encode positional information.

    Attention score: Σ μ_q * μ_k * cos((s - t) * θ + δ)

    Where:
    - μ = softplus(projection) are content-based magnitudes
    - θ are frequency components (like RoPE but doubled)
    - δ are learnable phase biases
    """

    def __init__(self, d_model: int, n_heads: int, base: float = 10000.0):
        """Initialize PoPE.

        Args:
            d_model: Model dimension (must equal n_heads * head_dim)
            n_heads: Number of attention heads
            base: Base for frequency computation (like RoPE theta)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Compute frequencies for each dimension
        # Unlike RoPE which uses d/2 frequencies, PoPE uses d frequencies
        freqs = 1.0 / (base ** (torch.arange(0, self.head_dim).float() / self.head_dim))
        self.register_buffer("freqs", freqs)

        # Learnable phase biases δ, initialized uniformly in [-2π, 0]
        # This allows the model to tune optimal relative offsets
        self.phase_bias = nn.Parameter(
            torch.empty(n_heads, self.head_dim).uniform_(-2 * math.pi, 0)
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply PoPE transformation to queries and keys.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim)
            positions: Position indices of shape (seq_len,)

        Returns:
            Tuple of (transformed_q, transformed_k) ready for attention
        """
        # Compute magnitudes via softplus (ensures non-negative)
        # Shape: (batch, n_heads, seq_len, head_dim)
        mu_q = F.softplus(q)
        mu_k = F.softplus(k)

        # Get frequencies buffer as tensor
        freqs = cast(torch.Tensor, self.freqs)

        # Compute position-based phases: position * frequency
        # Shape: (seq_len, head_dim)
        pos_phases = positions.unsqueeze(-1).float() * freqs.unsqueeze(0)

        # Add learnable phase bias
        # Shape: (1, n_heads, seq_len, head_dim)
        phases = pos_phases.unsqueeze(0).unsqueeze(0) + self.phase_bias.unsqueeze(
            0
        ).unsqueeze(2)

        # Convert polar to Cartesian coordinates for efficient computation:
        # q_cart = μ_q * (cos(φ), sin(φ))
        # k_cart = μ_k * (cos(φ), sin(φ))
        # Then real part of q_cart^H @ k_cart gives the attention score

        cos_phases = torch.cos(phases)
        sin_phases = torch.sin(phases)

        # Real and imaginary parts
        q_real = mu_q * cos_phases
        q_imag = mu_q * sin_phases
        k_real = mu_k * cos_phases
        k_imag = mu_k * sin_phases

        # Pack as interleaved real/imag for matrix multiplication
        # We'll compute: sum(q_real * k_real + q_imag * k_imag) per head
        # which equals: sum(μ_q * μ_k * cos(phase_diff))
        # This is the real part of the complex dot product

        # For efficient computation with standard attention, we need to
        # return modified Q and K that when multiplied give the PoPE score.
        # We concatenate real and imaginary parts and scale appropriately.
        q_pope = torch.cat([q_real, q_imag], dim=-1)  # (batch, heads, seq, 2*head_dim)
        k_pope = torch.cat([k_real, k_imag], dim=-1)  # (batch, heads, seq, 2*head_dim)

        return q_pope, k_pope


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


class TransformerBlock(nn.Module):
    """Single transformer decoder block with masked self-attention.

    Supports configurable positional encoding (learned or PoPE) and
    softmax variants (standard or softmax1).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        positional_encoding: str = "learned",
        softmax_variant: str = "standard",
    ):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            positional_encoding: "learned" or "pope"
            softmax_variant: "standard" or "softmax1"
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.positional_encoding = positional_encoding
        self.softmax_variant = softmax_variant

        # Use separate linear layers for Q, K, V to have more control
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # PoPE for polar positional encoding
        if positional_encoding == "pope":
            self.pope = PoPE(d_model, n_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Precompute scaling factor
        self._scale = 1.0 / math.sqrt(self.head_dim)

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        use_softmax1: bool,
    ) -> torch.Tensor:
        """Manual attention implementation supporting softmax variants.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim or 2*head_dim for PoPE)
            k: Key tensor (batch, n_heads, seq_len, head_dim or 2*head_dim for PoPE)
            v: Value tensor (batch, n_heads, seq_len, head_dim)
            use_softmax1: Whether to use softmax1 instead of standard softmax

        Returns:
            Attention output (batch, n_heads, seq_len, head_dim)
        """
        seq_len = q.shape[2]

        # Compute attention scores
        # For PoPE, q and k are 2*head_dim, so scaling should be adjusted
        scale = 1.0 / math.sqrt(q.shape[-1])
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Apply softmax variant
        if use_softmax1:
            attn_weights = softmax1(attn_scores, dim=-1)
        else:
            attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if self.training:
            attn_weights = self.attn_dropout(attn_weights)

        # Compute weighted sum
        return torch.matmul(attn_weights, v)

    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Position indices of shape (seq_len,), required for PoPE

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

        # Apply PoPE if configured
        if self.positional_encoding == "pope":
            if positions is None:
                positions = torch.arange(seq_len, device=x.device)
            q, k = self.pope(q, k, positions)

        # Choose attention implementation
        use_manual = (
            self.softmax_variant == "softmax1" or self.positional_encoding == "pope"
        )

        if use_manual:
            attn_out = self._manual_attention(
                q, k, v, use_softmax1=(self.softmax_variant == "softmax1")
            )
        else:
            # Use optimized scaled dot-product attention with Flash Attention
            attn_out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
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

    # Class attribute to identify architecture type
    architecture: str = "standard"

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        dropout: float = 0.1,
        positional_encoding: str = "learned",
        softmax_variant: str = "standard",
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
            positional_encoding: "learned" for learned embeddings, "pope" for PoPE
            softmax_variant: "standard" or "softmax1" (+1 in denominator)
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.positional_encoding = positional_encoding
        self.softmax_variant = softmax_variant
        self._embed_scale = math.sqrt(d_model)

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Only use learned position embeddings when not using PoPE
        if positional_encoding == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    positional_encoding=positional_encoding,
                    softmax_variant=softmax_variant,
                )
                for _ in range(n_layers)
            ]
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
        _batch_size, seq_len = input_ids.shape

        # Token embeddings with scaling
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        x.mul_(self._embed_scale)

        # Position indices for PoPE or learned embeddings
        positions = torch.arange(seq_len, device=input_ids.device)

        # Add learned positional embeddings only when not using PoPE
        if self.positional_encoding == "learned":
            x = x + self.position_embedding(positions)

        # Apply transformer layers (pass positions for PoPE)
        for layer in self.layers:
            x = layer(x, positions)

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


class UniversalTransformerModel(nn.Module):
    """Universal Transformer model with weight sharing across depth.

    Unlike standard transformers with N unique layers, Universal Transformers
    apply a smaller set of layers repeatedly in a loop. This allows the model
    to learn iterative algorithms where the same computation is applied
    multiple times, potentially specializing at each loop iteration.

    For example, UT-2L-4loop has 2 unique layers applied 4 times each,
    giving the same sequential depth (8) as an 8-layer standard transformer
    but with 1/4 the parameters.
    """

    # Class attribute to identify architecture type
    architecture: str = "universal"

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 2,
        n_loops: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        dropout: float = 0.1,
        use_loop_embeddings: bool = True,
        positional_encoding: str = "learned",
        softmax_variant: str = "standard",
    ):
        """Initialize the Universal Transformer model.

        Args:
            vocab_size: Size of vocabulary (default from tokenizer)
            d_model: Model dimension
            n_layers: Number of unique transformer layers (weight-shared blocks)
            n_loops: Number of times to apply the layer stack
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_loop_embeddings: Whether to add learnable loop position embeddings
            positional_encoding: "learned" for learned embeddings, "pope" for PoPE
            softmax_variant: "standard" or "softmax1" (+1 in denominator)
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_loops = n_loops
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_loop_embeddings = use_loop_embeddings
        self.positional_encoding = positional_encoding
        self.softmax_variant = softmax_variant
        self._embed_scale = math.sqrt(d_model)

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Only use learned position embeddings when not using PoPE
        if positional_encoding == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Optional loop embeddings to help model distinguish iterations
        if use_loop_embeddings:
            self.loop_embeddings = nn.Embedding(n_loops, d_model)

        # Transformer layers (shared across loops)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    positional_encoding=positional_encoding,
                    softmax_variant=softmax_variant,
                )
                for _ in range(n_layers)
            ]
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
        _batch_size, seq_len = input_ids.shape

        # Token embeddings with scaling
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        x.mul_(self._embed_scale)

        # Position indices for PoPE or learned embeddings
        positions = torch.arange(seq_len, device=input_ids.device)

        # Add learned positional embeddings only when not using PoPE
        if self.positional_encoding == "learned":
            x = x + self.position_embedding(positions)

        # Pre-fetch all loop embeddings for efficiency
        loop_embs = self.loop_embeddings.weight if self.use_loop_embeddings else None

        # Apply transformer layers repeatedly (Universal Transformer loop)
        for loop_idx in range(self.n_loops):
            # Optionally add loop embedding to help model track iteration
            if loop_embs is not None:
                x = x + loop_embs[loop_idx]

            # Apply all layers in this loop iteration (pass positions for PoPE)
            for layer in self.layers:
                x = layer(x, positions)

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

    @property
    def sequential_depth(self) -> int:
        """Return effective sequential depth (n_layers * n_loops)."""
        return self.n_layers * self.n_loops


class FeedbackTransformerBlock(nn.Module):
    """Transformer block that attends to shared memory instead of self-attention.

    In the Feedback Transformer, all layers attend to a shared memory containing
    weighted sums of all layer outputs from previous timesteps. This block receives
    pre-computed K/V from the memory and only computes Q from its input.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        softmax_variant: str = "standard",
    ):
        """Initialize feedback transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            softmax_variant: "standard" or "softmax1"
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.softmax_variant = softmax_variant

        # Only Q projection is per-layer; K/V are shared and passed in
        self.q_proj = nn.Linear(d_model, d_model)
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

        self._scale = 1.0 / math.sqrt(self.head_dim)

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Manual attention for softmax1 variant.

        Args:
            q: Query tensor (batch, n_heads, 1, head_dim)
            k: Key tensor (batch, n_heads, mem_len, head_dim)
            v: Value tensor (batch, n_heads, mem_len, head_dim)

        Returns:
            Attention output (batch, n_heads, 1, head_dim)
        """
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale

        if self.softmax_variant == "softmax1":
            attn_weights = softmax1(attn_scores, dim=-1)
        else:
            attn_weights = F.softmax(attn_scores, dim=-1)

        if self.training:
            attn_weights = self.attn_dropout(attn_weights)

        return torch.matmul(attn_weights, v)

    def forward(
        self,
        x: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through feedback transformer block.

        Args:
            x: Input tensor of shape (batch_size, d_model) - single position
            memory_keys: Keys from memory of shape (batch_size, mem_len, d_model)
            memory_values: Values from memory of shape (batch_size, mem_len, d_model)

        Returns:
            Output tensor of shape (batch_size, d_model)
        """
        batch_size = x.shape[0]
        mem_len = memory_keys.shape[1]

        # Compute Q from current input: (batch_size, 1, n_heads, head_dim)
        q = (
            self.q_proj(x)
            .reshape(batch_size, 1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, n_heads, 1, head_dim)

        # Reshape pre-computed K/V from memory
        k = memory_keys.reshape(
            batch_size, mem_len, self.n_heads, self.head_dim
        ).transpose(1, 2)  # (batch_size, n_heads, mem_len, head_dim)
        v = memory_values.reshape(
            batch_size, mem_len, self.n_heads, self.head_dim
        ).transpose(1, 2)  # (batch_size, n_heads, mem_len, head_dim)

        if self.softmax_variant == "softmax1":
            attn_out = self._manual_attention(q, k, v)
        else:
            # Scaled dot-product attention (no mask needed - only attending to past memory)
            attn_out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, self.d_model)
        attn_out = self.out_proj(attn_out)

        # Residual connection and layer norm
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class FeedbackTransformerModel(nn.Module):
    """Feedback Transformer model with shared memory attention.

    Unlike standard transformers where each layer attends to same-layer representations,
    the Feedback Transformer has all layers attend to a shared memory containing
    weighted sums of ALL layer outputs from previous timesteps. This enables
    recursive computation where even the bottom layer can access top-layer
    abstractions from previous positions.

    Key features:
    - Memory vector m_t = Σ softmax(w^l) * x^l_t (weighted sum of all layers)
    - All layers attend to memory from previous timesteps
    - Shared K/V projections across all layers
    - Requires sequential processing during training

    Note: PoPE is not supported with feedback architecture due to the
    memory-based attention pattern. Use learned positional embeddings instead.
    """

    # Class attribute to identify architecture type
    architecture: str = "feedback"

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        dropout: float = 0.1,
        positional_encoding: str = "learned",
        softmax_variant: str = "standard",
    ):
        """Initialize the Feedback Transformer model.

        Args:
            vocab_size: Size of vocabulary (default from tokenizer)
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            positional_encoding: Only "learned" is supported for feedback architecture
            softmax_variant: "standard" or "softmax1" (+1 in denominator)

        Raises:
            ValueError: If positional_encoding is "pope" (not supported)
        """
        super().__init__()

        if positional_encoding == "pope":
            raise ValueError(
                "PoPE is not supported with feedback architecture. "
                "Use positional_encoding='learned' instead."
            )

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.positional_encoding = positional_encoding
        self.softmax_variant = softmax_variant
        self._embed_scale = math.sqrt(d_model)

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Layer weights for memory composition (L+1 scalars: embedding + L layers)
        # Initialized to zero so softmax gives uniform weights initially
        self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))

        # Shared K/V projections for memory (all layers use same K/V)
        self.shared_k_proj = nn.Linear(d_model, d_model)
        self.shared_v_proj = nn.Linear(d_model, d_model)

        # Transformer layers (each has own Q projection)
        self.layers = nn.ModuleList(
            [
                FeedbackTransformerBlock(
                    d_model, n_heads, d_ff, dropout, softmax_variant=softmax_variant
                )
                for _ in range(n_layers)
            ]
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
        """Forward pass through the model (sequential over sequence).

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (unused)
            labels: Optional labels for computing loss

        Returns:
            If labels provided: dict with 'loss' and 'logits'
            Otherwise: logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        _batch_size, seq_len = input_ids.shape

        # Token embeddings with scaling
        embeddings = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        embeddings.mul_(self._embed_scale)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        embeddings = embeddings + self.position_embedding(positions)

        # Compute softmax weights for layer combination
        weights = F.softmax(self.layer_weights, dim=0)

        # Initialize memory storage for K/V
        # We'll accumulate memory as we process each position
        memory_keys_list: list[torch.Tensor] = []
        memory_values_list: list[torch.Tensor] = []
        outputs_list: list[torch.Tensor] = []

        # Process sequence position by position (sequential)
        for t in range(seq_len):
            x = embeddings[:, t]  # (batch_size, d_model)
            layer_outputs = [x]  # Start with embedding (layer 0)

            if t > 0:
                # Stack accumulated memory
                memory_keys = torch.stack(memory_keys_list, dim=1)
                memory_values = torch.stack(memory_values_list, dim=1)

                # Pass through transformer layers
                for layer in self.layers:
                    x = layer(x, memory_keys, memory_values)
                    layer_outputs.append(x)
            else:
                # First position: no memory to attend to, just pass through layers
                # with no attention (layers still do feed-forward)
                for layer in self.layers:
                    # Cast to FeedbackTransformerBlock for type checker
                    block = cast(FeedbackTransformerBlock, layer)
                    # For first position, do a simplified forward without attention
                    # Just apply feed-forward with residuals
                    ff_out = block.feed_forward(x)
                    x = block.norm2(x + block.dropout(ff_out))
                    layer_outputs.append(x)

            # Compute memory vector as weighted sum of all layer outputs
            m_t = sum(w * out for w, out in zip(weights, layer_outputs))

            # Compute and store K/V for this timestep
            memory_keys_list.append(self.shared_k_proj(m_t))
            memory_values_list.append(self.shared_v_proj(m_t))

            outputs_list.append(x)

        # Stack outputs: (batch_size, seq_len, d_model)
        hidden_states = torch.stack(outputs_list, dim=1)

        # Final layer norm and projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)

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


# Type alias for any model type
Model = ArithmeticModel | UniversalTransformerModel | FeedbackTransformerModel


def create_model_from_config(config: ModelConfig) -> Model:
    """Create a model from a configuration object.

    Args:
        config: ModelConfig with architecture and hyperparameters

    Returns:
        Model instance (ArithmeticModel, UniversalTransformerModel, or
        FeedbackTransformerModel)
    """
    if config.architecture == "standard":
        return ArithmeticModel(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            positional_encoding=config.positional_encoding,
            softmax_variant=config.softmax_variant,
        )
    elif config.architecture == "universal":
        if config.n_loops is None:
            raise ValueError("Universal transformer requires n_loops parameter")
        return UniversalTransformerModel(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_loops=config.n_loops,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            use_loop_embeddings=config.use_loop_embeddings,
            positional_encoding=config.positional_encoding,
            softmax_variant=config.softmax_variant,
        )
    elif config.architecture == "feedback":
        return FeedbackTransformerModel(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            positional_encoding=config.positional_encoding,
            softmax_variant=config.softmax_variant,
        )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")
