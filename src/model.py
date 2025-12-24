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
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_loops = n_loops
        self.n_layers = n_layers
        self.use_loop_embeddings = use_loop_embeddings

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.n_heads = n_heads

        # Optional loop embeddings to help model distinguish iterations
        if use_loop_embeddings:
            self.loop_embeddings = nn.Embedding(n_loops, d_model)

        # Transformer layers (shared across loops)
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

        # Apply transformer layers repeatedly (Universal Transformer loop)
        for loop_idx in range(self.n_loops):
            # Optionally add loop embedding to help model track iteration
            if self.use_loop_embeddings:
                loop_emb = self.loop_embeddings.weight[loop_idx]
                x = x + loop_emb

            # Apply all layers in this loop iteration
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

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize feedback transformer block.

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

    def forward(
        self,
        x: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        alibi_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through feedback transformer block.

        Args:
            x: Input tensor of shape (batch_size, d_model) - single position
            memory_keys: Keys from memory of shape (batch_size, mem_len, d_model)
            memory_values: Values from memory of shape (batch_size, mem_len, d_model)
            alibi_bias: ALiBi bias tensor of shape (n_heads, 1, mem_len)

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

        # ALiBi bias: (n_heads, 1, mem_len) -> (batch_size, n_heads, 1, mem_len)
        attn_mask = alibi_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,  # Causality handled by only attending to past memory
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
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Layer weights for memory composition (L+1 scalars: embedding + L layers)
        # Initialized to zero so softmax gives uniform weights initially
        self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))

        # Shared K/V projections for memory (all layers use same K/V)
        self.shared_k_proj = nn.Linear(d_model, d_model)
        self.shared_v_proj = nn.Linear(d_model, d_model)

        # Transformer layers (each has own Q projection)
        self.layers = nn.ModuleList(
            [
                FeedbackTransformerBlock(d_model, n_heads, d_ff, dropout)
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
        device = input_ids.device
        dtype = self.token_embedding.weight.dtype

        # Token embeddings with scaling
        embeddings = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        embeddings = embeddings * math.sqrt(self.d_model)

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

                # Build ALiBi bias for current position attending to past
                alibi_bias = build_alibi_bias(self.n_heads, t, device, dtype=dtype)[
                    :, -1:, :
                ]  # (n_heads, 1, t)

                # Pass through transformer layers
                for layer in self.layers:
                    x = layer(x, memory_keys, memory_values, alibi_bias)
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
        )
    elif config.architecture == "feedback":
        return FeedbackTransformerModel(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")
