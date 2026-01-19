"""Activation statistics computation and collection for transformer models.

This module provides utilities for tracking activation distributions during
training and evaluation, including kurtosis, outlier detection, and attention
statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


def compute_kurtosis(x: Tensor) -> float:
    """Compute excess kurtosis of a tensor.

    Excess kurtosis measures the "tailedness" of a distribution relative to a
    normal distribution. Normal distribution has excess kurtosis of 0, while
    heavy-tailed distributions have positive excess kurtosis.

    Args:
        x: Input tensor of any shape

    Returns:
        Excess kurtosis value (normal distribution = 0, heavy tails > 0)
    """
    x_flat = x.flatten().float()
    if x_flat.numel() < 4:
        return 0.0
    mean = x_flat.mean()
    std = x_flat.std(unbiased=False)
    if std < 1e-8:
        return 0.0
    centered = (x_flat - mean) / std
    return (centered**4).mean().item() - 3.0


def compute_outlier_fraction(x: Tensor, threshold_std: float = 5.0) -> float:
    """Compute fraction of values exceeding threshold standard deviations.

    Args:
        x: Input tensor of any shape
        threshold_std: Number of standard deviations to consider as outlier

    Returns:
        Fraction of values that are outliers (0.0 to 1.0)
    """
    x_flat = x.flatten().float()
    if x_flat.numel() < 2:
        return 0.0
    mean = x_flat.mean()
    std = x_flat.std(unbiased=False)
    if std < 1e-8:
        return 0.0
    return ((x_flat - mean).abs() > threshold_std * std).float().mean().item()


def compute_attention_entropy(attn_weights: Tensor) -> float:
    """Compute mean entropy of attention distributions.

    Higher entropy indicates more uniform attention, lower entropy indicates
    more focused/sparse attention.

    Args:
        attn_weights: Attention weights of shape (batch, heads, seq, seq)

    Returns:
        Mean entropy across all attention distributions
    """
    # Clamp to avoid log(0)
    attn_clamped = attn_weights.clamp(min=1e-10)
    entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # (B, H, S)
    return entropy.mean().item()


def compute_attention_sparsity(attn_weights: Tensor, threshold: float = 0.01) -> float:
    """Compute fraction of attention weights below threshold.

    Higher sparsity indicates more focused attention patterns.

    Args:
        attn_weights: Attention weights of shape (batch, heads, seq, seq)
        threshold: Threshold below which weights are considered "zero"

    Returns:
        Fraction of attention weights below threshold (0.0 to 1.0)
    """
    return (attn_weights < threshold).float().mean().item()


def compute_softmax1_abstention(attn_weights: Tensor) -> float:
    """Compute mean abstention rate for softmax1 attention.

    For softmax1, attention weights sum to less than 1.0, with the "missing"
    probability mass representing abstention. This measures how much the
    attention heads are choosing to abstain.

    Args:
        attn_weights: Attention weights of shape (batch, heads, seq, seq)

    Returns:
        Mean abstention rate (1.0 - sum of weights) across all positions
    """
    sums = attn_weights.sum(dim=-1)  # (B, H, S)
    abstention = 1.0 - sums
    return abstention.mean().item()


@dataclass
class LayerStats:
    """Statistics for a single layer's activations."""

    kurtosis: float = 0.0
    max_abs: float = 0.0
    mean_abs: float = 0.0
    outlier_fraction: float = 0.0


@dataclass
class AttentionStats:
    """Statistics for attention weights and scores."""

    entropy: float = 0.0
    sparsity: float = 0.0
    abstention: float = 0.0  # Only meaningful for softmax1
    score_kurtosis: float = 0.0
    score_max_abs: float = 0.0


@dataclass
class ActivationStatsSummary:
    """Complete summary of activation statistics across all layers."""

    hidden_states: list[LayerStats] = field(default_factory=list)
    ffn_intermediate: list[LayerStats] = field(default_factory=list)
    attention: list[AttentionStats] = field(default_factory=list)
    num_batches: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "hidden_states": [
                {
                    "layer": i,
                    "kurtosis": s.kurtosis,
                    "max_abs": s.max_abs,
                    "mean_abs": s.mean_abs,
                    "outlier_fraction": s.outlier_fraction,
                }
                for i, s in enumerate(self.hidden_states)
            ],
            "ffn_intermediate": [
                {
                    "layer": i,
                    "kurtosis": s.kurtosis,
                    "max_abs": s.max_abs,
                    "mean_abs": s.mean_abs,
                    "outlier_fraction": s.outlier_fraction,
                }
                for i, s in enumerate(self.ffn_intermediate)
            ],
            "attention": [
                {
                    "layer": i,
                    "entropy": s.entropy,
                    "sparsity": s.sparsity,
                    "abstention": s.abstention,
                    "score_kurtosis": s.score_kurtosis,
                    "score_max_abs": s.score_max_abs,
                }
                for i, s in enumerate(self.attention)
            ],
            "aggregate": {
                "hidden_kurtosis_mean": (
                    sum(s.kurtosis for s in self.hidden_states)
                    / len(self.hidden_states)
                    if self.hidden_states
                    else 0.0
                ),
                "hidden_max_abs": (
                    max(s.max_abs for s in self.hidden_states)
                    if self.hidden_states
                    else 0.0
                ),
                "hidden_outlier_fraction_mean": (
                    sum(s.outlier_fraction for s in self.hidden_states)
                    / len(self.hidden_states)
                    if self.hidden_states
                    else 0.0
                ),
                "attention_entropy_mean": (
                    sum(s.entropy for s in self.attention) / len(self.attention)
                    if self.attention
                    else 0.0
                ),
                "attention_sparsity_mean": (
                    sum(s.sparsity for s in self.attention) / len(self.attention)
                    if self.attention
                    else 0.0
                ),
                "attention_abstention_mean": (
                    sum(s.abstention for s in self.attention) / len(self.attention)
                    if self.attention
                    else 0.0
                ),
            },
            "num_batches": self.num_batches,
        }

    def save(self, path: Path) -> None:
        """Save statistics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def format_stats_summary(stats: ActivationStatsSummary) -> str:
    """Format activation statistics as a human-readable summary.

    Args:
        stats: Statistics summary to format

    Returns:
        Formatted string for console output
    """
    lines = ["Activation Statistics Summary", "=" * 40]

    if stats.hidden_states:
        lines.append("\nHidden States (per layer):")
        for i, s in enumerate(stats.hidden_states):
            lines.append(
                f"  Layer {i}: kurtosis={s.kurtosis:.2f}, "
                f"max_abs={s.max_abs:.2f}, outliers={s.outlier_fraction:.4f}"
            )

    if stats.attention:
        lines.append("\nAttention (per layer):")
        for i, s in enumerate(stats.attention):
            abstention_str = (
                f", abstention={s.abstention:.4f}" if s.abstention > 0 else ""
            )
            lines.append(
                f"  Layer {i}: entropy={s.entropy:.2f}, "
                f"sparsity={s.sparsity:.4f}{abstention_str}"
            )

    if stats.ffn_intermediate:
        lines.append("\nFFN Intermediate (per layer):")
        for i, s in enumerate(stats.ffn_intermediate):
            lines.append(
                f"  Layer {i}: kurtosis={s.kurtosis:.2f}, "
                f"max_abs={s.max_abs:.2f}, outliers={s.outlier_fraction:.4f}"
            )

    # Aggregate stats
    stats_dict = stats.to_dict()
    agg = stats_dict["aggregate"]
    lines.append("\nAggregate Statistics:")
    lines.append(f"  Hidden kurtosis (mean): {agg['hidden_kurtosis_mean']:.2f}")
    lines.append(f"  Hidden max abs: {agg['hidden_max_abs']:.2f}")
    lines.append(
        f"  Hidden outlier fraction (mean): {agg['hidden_outlier_fraction_mean']:.4f}"
    )
    lines.append(f"  Attention entropy (mean): {agg['attention_entropy_mean']:.2f}")
    lines.append(f"  Attention sparsity (mean): {agg['attention_sparsity_mean']:.4f}")
    if agg["attention_abstention_mean"] > 0:
        lines.append(
            f"  Attention abstention (mean): {agg['attention_abstention_mean']:.4f}"
        )

    lines.append(f"\nBatches processed: {stats.num_batches}")

    return "\n".join(lines)


class ActivationStatsCollector:
    """Collects activation statistics using forward hooks.

    This collector registers hooks on transformer layers to capture hidden
    states and FFN intermediate activations during forward passes.
    """

    def __init__(self, model: nn.Module, use_softmax1: bool = False):
        """Initialize the activation stats collector.

        Args:
            model: The transformer model to collect stats from
            use_softmax1: Whether the model uses softmax1 (affects abstention tracking)
        """
        self.model = model
        self.use_softmax1 = use_softmax1
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Accumulator lists for batch statistics
        self._hidden_states: list[list[Tensor]] = []  # [layer][batch_samples]
        self._ffn_intermediate: list[list[Tensor]] = []  # [layer][batch_samples]
        self._attention_weights: list[list[Tensor]] = []  # [layer][batch_samples]
        self._attention_scores: list[list[Tensor]] = []  # [layer][batch_samples]
        self._num_batches = 0

    def _create_hidden_hook(
        self, layer_idx: int
    ) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        """Create a hook for capturing hidden states after a layer."""

        def hook(
            _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
        ) -> None:
            # output can be a tuple (hidden_states, attn_weights) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Ensure we have enough slots
            while len(self._hidden_states) <= layer_idx:
                self._hidden_states.append([])

            self._hidden_states[layer_idx].append(hidden.detach())

        return hook

    def _create_ffn_hook(
        self, layer_idx: int
    ) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        """Create a hook for capturing FFN intermediate activations."""

        def hook(
            _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
        ) -> None:
            # Ensure we have enough slots
            while len(self._ffn_intermediate) <= layer_idx:
                self._ffn_intermediate.append([])

            self._ffn_intermediate[layer_idx].append(output.detach())

        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on the model's layers."""
        # Find transformer blocks
        layers_attr = getattr(self.model, "layers", None)
        if layers_attr is None:
            return

        for layer_idx, layer in enumerate(layers_attr):
            # Hook after the full transformer block for hidden states
            hook = layer.register_forward_hook(self._create_hidden_hook(layer_idx))
            self._hooks.append(hook)

            # Hook on FFN intermediate (first linear in feed_forward)
            ff = getattr(layer, "feed_forward", None)
            if ff is not None and isinstance(ff, nn.Sequential) and len(ff) > 0:
                ffn_hook = ff[0].register_forward_hook(self._create_ffn_hook(layer_idx))
                self._hooks.append(ffn_hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def process_attention_outputs(
        self,
        attention_weights: Optional[tuple[Tensor, ...]],
        attention_scores: Optional[tuple[Tensor, ...]],
    ) -> None:
        """Process attention weights and scores from model output.

        Args:
            attention_weights: Tuple of attention weight tensors per layer
            attention_scores: Tuple of pre-softmax attention score tensors per layer
        """
        if attention_weights is not None:
            for layer_idx, weights in enumerate(attention_weights):
                while len(self._attention_weights) <= layer_idx:
                    self._attention_weights.append([])
                self._attention_weights[layer_idx].append(weights.detach())

        if attention_scores is not None:
            for layer_idx, scores in enumerate(attention_scores):
                while len(self._attention_scores) <= layer_idx:
                    self._attention_scores.append([])
                self._attention_scores[layer_idx].append(scores.detach())

        self._num_batches += 1

    def compute_statistics(self) -> ActivationStatsSummary:
        """Compute aggregate statistics from collected activations.

        Returns:
            Summary of activation statistics across all layers
        """
        summary = ActivationStatsSummary(num_batches=self._num_batches)

        # Process hidden states
        for layer_activations in self._hidden_states:
            if not layer_activations:
                continue

            # Concatenate all batches for this layer
            all_hidden = torch.cat(layer_activations, dim=0)

            summary.hidden_states.append(
                LayerStats(
                    kurtosis=compute_kurtosis(all_hidden),
                    max_abs=all_hidden.abs().max().item(),
                    mean_abs=all_hidden.abs().mean().item(),
                    outlier_fraction=compute_outlier_fraction(all_hidden),
                )
            )

        # Process FFN intermediate
        for layer_activations in self._ffn_intermediate:
            if not layer_activations:
                continue

            all_ffn = torch.cat(layer_activations, dim=0)

            summary.ffn_intermediate.append(
                LayerStats(
                    kurtosis=compute_kurtosis(all_ffn),
                    max_abs=all_ffn.abs().max().item(),
                    mean_abs=all_ffn.abs().mean().item(),
                    outlier_fraction=compute_outlier_fraction(all_ffn),
                )
            )

        # Process attention
        for layer_idx in range(
            max(len(self._attention_weights), len(self._attention_scores))
        ):
            attn_stats = AttentionStats()

            # Attention weights (post-softmax)
            if (
                layer_idx < len(self._attention_weights)
                and self._attention_weights[layer_idx]
            ):
                all_weights = torch.cat(self._attention_weights[layer_idx], dim=0)
                attn_stats.entropy = compute_attention_entropy(all_weights)
                attn_stats.sparsity = compute_attention_sparsity(all_weights)
                if self.use_softmax1:
                    attn_stats.abstention = compute_softmax1_abstention(all_weights)

            # Attention scores (pre-softmax)
            if (
                layer_idx < len(self._attention_scores)
                and self._attention_scores[layer_idx]
            ):
                all_scores = torch.cat(self._attention_scores[layer_idx], dim=0)
                # Mask out -inf values for kurtosis computation
                finite_scores = all_scores[all_scores.isfinite()]
                if finite_scores.numel() > 0:
                    attn_stats.score_kurtosis = compute_kurtosis(finite_scores)
                    attn_stats.score_max_abs = finite_scores.abs().max().item()

            summary.attention.append(attn_stats)

        return summary

    def clear(self) -> None:
        """Clear all accumulated statistics."""
        self._hidden_states.clear()
        self._ffn_intermediate.clear()
        self._attention_weights.clear()
        self._attention_scores.clear()
        self._num_batches = 0

    def __enter__(self) -> "ActivationStatsCollector":
        """Context manager entry: register hooks."""
        self.register_hooks()
        return self

    def __exit__(
        self,
        _exc_type: Optional[type],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[object],
    ) -> None:
        """Context manager exit: remove hooks."""
        self.remove_hooks()
