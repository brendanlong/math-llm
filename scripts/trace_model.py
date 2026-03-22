"""Mechanistic interpretability analysis for arithmetic transformer models.

Performs three analyses on a trained model:
1. Logit Lens: How predictions evolve through layers
2. Attention Patterns: What each head attends to for key predictions
3. Causal Tracing: Which (layer, position) pairs are critical

Generates matplotlib visualizations and a text summary.

Usage:
    python scripts/trace_model.py \
        --checkpoint-dir checkpoints/standard-small-rope-preln/checkpoint-19000/ \
        --prompt "<begin>10+9=" \
        --output-dir traces/
"""

import argparse
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch

from src.config import ModelConfig, find_config_in_checkpoint, load_config
from src.model import ArithmeticModel, create_model_from_config
from src.tokenizer import END_THINK_TOKEN_ID, END_TOKEN_ID, VOCAB

matplotlib.use("Agg")

INV_VOCAB = {v: k for k, v in VOCAB.items()}


def tokenize(text: str) -> list[int]:
    """Tokenize text manually using regex."""
    pattern = r"<begin>|</think>|<think>|<noop>|<end>|[0-9+=]"
    return [VOCAB[t] for t in re.findall(pattern, text)]


def decode_tokens(ids: list[int]) -> list[str]:
    """Decode token IDs to display strings."""
    return [INV_VOCAB[i] for i in ids]


def load_model(
    checkpoint_dir: Path,
) -> tuple[ArithmeticModel, ModelConfig]:
    """Load model from checkpoint directory."""
    config_path = find_config_in_checkpoint(checkpoint_dir)
    if config_path is None:
        raise FileNotFoundError(f"No model_config.yaml found near {checkpoint_dir}")
    config = load_config(config_path)
    model = create_model_from_config(config)
    state = safetensors.torch.load_file(str(checkpoint_dir / "model.safetensors"))
    model.load_state_dict(state)
    model.eval()
    assert isinstance(model, ArithmeticModel), "Only standard architecture supported"
    return model, config


# ── Analysis 1: Logit Lens ──────────────────────────────────────────


def logit_lens(
    model: ArithmeticModel, input_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Project residual stream at each layer through ln_f + lm_head.

    Returns dict mapping layer name to logits (seq_len, vocab_size).
    """
    layer_outputs: dict[int, torch.Tensor] = {}

    def capture(idx: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            layer_outputs[idx] = out[0].detach()

        return hook

    hooks = [
        layer.register_forward_hook(capture(i)) for i, layer in enumerate(model.layers)
    ]
    with torch.no_grad():
        final = model(input_ids)
    for h in hooks:
        h.remove()

    if isinstance(final, dict):
        final = final["logits"]
    assert isinstance(final, torch.Tensor)

    results: dict[str, torch.Tensor] = {}

    # Embedding (before any layer)
    with torch.no_grad():
        emb = model.token_embedding(input_ids) * model.embed_scale
        results["embed"] = model.lm_head(model.ln_f(emb))[0].cpu()

    for idx in sorted(layer_outputs):
        with torch.no_grad():
            results[f"layer_{idx}"] = model.lm_head(model.ln_f(layer_outputs[idx]))[
                0
            ].cpu()

    results["final"] = final[0].detach().cpu()
    return results


def plot_logit_lens(
    ll_results: dict[str, torch.Tensor],
    token_labels: list[str],
    target_positions: list[int],
    target_tokens: list[int],
    output_path: Path,
) -> None:
    """Plot logit lens: how the prediction evolves through layers.

    Creates two subplots:
    1. Top-1 prediction at each (layer, position) with confidence
    2. Probability of correct next token at target positions across layers
    """
    layer_names = list(ll_results.keys())
    n_layers = len(layer_names)
    seq_len = ll_results["final"].shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, seq_len * 0.8), 10))

    # ── Subplot 1: Top prediction at each (layer, position) ──
    pred_grid = np.zeros((n_layers, seq_len), dtype=int)
    prob_grid = np.zeros((n_layers, seq_len))

    for li, name in enumerate(layer_names):
        logits = ll_results[name]
        probs = torch.softmax(logits, dim=-1)
        pred_grid[li] = logits.argmax(dim=-1).numpy()
        prob_grid[li] = probs.max(dim=-1).values.numpy()

    im1 = ax1.imshow(prob_grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax1.set_yticks(range(n_layers))
    ax1.set_yticklabels(layer_names)
    ax1.set_xticks(range(seq_len))
    ax1.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_title("Logit Lens: Top-1 Prediction (color = confidence)")

    # Annotate each cell with predicted token
    for li in range(n_layers):
        for pos in range(seq_len):
            tok = INV_VOCAB[pred_grid[li, pos]]
            if len(tok) > 3:
                tok = tok[:3]  # Truncate long tokens for display
            color = "white" if prob_grid[li, pos] > 0.6 else "black"
            ax1.text(pos, li, tok, ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im1, ax=ax1, label="Max probability")

    # ── Subplot 2: Correct-token probability at target positions ──
    for tp, tt in zip(target_positions, target_tokens):
        probs_across_layers = []
        for name in layer_names:
            logits = ll_results[name]
            prob = torch.softmax(logits[tp], dim=-1)[tt].item()
            probs_across_layers.append(prob)
        label = f"pos {tp} ({token_labels[tp]}) → {INV_VOCAB[tt]}"
        ax2.plot(range(n_layers), probs_across_layers, "o-", label=label)

    ax2.set_xticks(range(n_layers))
    ax2.set_xticklabels(layer_names, rotation=45, ha="right")
    ax2.set_ylabel("P(correct next token)")
    ax2.set_title("Correct Token Probability Across Layers")
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved logit lens plot to {output_path}")


# ── Analysis 2: Attention Patterns ──────────────────────────────────


def get_attention_weights(
    model: ArithmeticModel, input_ids: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Get attention weights from all layers.

    Returns tuple of (batch, heads, seq, seq) tensors.
    """
    # Model only returns dict (with attentions) when labels are provided
    with torch.no_grad():
        out = model(input_ids, labels=input_ids, output_attentions=True)
    assert isinstance(out, dict)
    return tuple(a.detach().cpu() for a in out["attentions"])


def plot_attention_patterns(
    attentions: tuple[torch.Tensor, ...],
    token_labels: list[str],
    target_positions: list[int],
    target_tokens: list[int],
    output_path: Path,
) -> None:
    """Plot attention patterns for target positions.

    For each target position, shows what each head attends to across layers.
    """
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    n_targets = len(target_positions)

    fig, axes = plt.subplots(
        n_targets,
        n_layers,
        figsize=(n_layers * 4, n_targets * 3),
        squeeze=False,
    )

    for ti, (tp, tt) in enumerate(zip(target_positions, target_tokens)):
        for li in range(n_layers):
            ax = axes[ti, li]
            # attention shape: (batch, heads, seq, seq)
            # We want attention FROM position tp TO all other positions
            attn = attentions[li][0, :, tp, : tp + 1]  # (heads, tp+1)
            ax.imshow(attn.numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)

            ax.set_yticks(range(n_heads))
            ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)

            # Only show x labels for positions that exist
            ax.set_xticks(range(tp + 1))
            ax.set_xticklabels(
                token_labels[: tp + 1], rotation=90, ha="center", fontsize=6
            )

            if ti == 0:
                ax.set_title(f"Layer {li}", fontsize=10)
            if li == 0:
                target_label = INV_VOCAB[tt]
                ax.set_ylabel(
                    f"pos {tp} → {target_label}", fontsize=9, fontweight="bold"
                )

    fig.suptitle(
        "Attention Patterns: What each head attends to at answer positions",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved attention patterns to {output_path}")


def plot_attention_summary(
    attentions: tuple[torch.Tensor, ...],
    token_labels: list[str],
    target_positions: list[int],
    target_tokens: list[int],
    output_path: Path,
) -> None:
    """Plot a summary: for each answer position, max attention across all heads per source."""
    n_targets = len(target_positions)

    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 3 * n_targets), squeeze=False)

    for ti, (tp, tt) in enumerate(zip(target_positions, target_tokens)):
        ax = axes[ti, 0]

        # Collect max attention from any head at any layer to each source position
        max_attn = np.zeros(tp + 1)

        for attn_layer in attentions:
            for hi in range(attn_layer.shape[1]):
                attn_vals = attn_layer[0, hi, tp, : tp + 1].numpy()
                for pos in range(tp + 1):
                    if attn_vals[pos] > max_attn[pos]:
                        max_attn[pos] = attn_vals[pos]

        bars = ax.bar(range(tp + 1), max_attn, color="steelblue", alpha=0.8)
        ax.set_xticks(range(tp + 1))
        ax.set_xticklabels(token_labels[: tp + 1], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Max attention weight")
        target_label = INV_VOCAB[tt]
        ax.set_title(
            f"Position {tp} ({token_labels[tp]}) predicting '{target_label}': "
            f"max attention to each source"
        )
        ax.set_ylim(0, 1)

        # Highlight top-3 source positions
        top_3 = np.argsort(max_attn)[-3:]
        for idx in top_3:
            bars[idx].set_color("coral")
            bars[idx].set_alpha(1.0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved attention summary to {output_path}")


# ── Analysis 3: Causal Tracing ──────────────────────────────────────


def causal_trace(
    model: ArithmeticModel,
    clean_ids: torch.Tensor,
    target_pos: int,
    target_token: int,
    noise_std: Optional[float] = None,
    n_samples: int = 10,
) -> tuple[Any, float, float]:
    """Denoising activation patching for causal tracing.

    For each (layer, position), restores the clean activation in an otherwise
    noisy forward pass and measures recovery of the target logit.

    Args:
        model: The model to trace
        clean_ids: Clean input token IDs (1, seq_len)
        target_pos: Position where we're predicting
        target_token: Token ID we want to predict
        noise_std: Std of Gaussian noise (auto-calibrated if None)
        n_samples: Number of noise samples to average

    Returns:
        results: (n_layers+1, seq_len) array of recovery fractions
        clean_logit: logit for target token in clean run
        corrupt_logit: average logit for target token in noisy runs
    """
    n_layers = len(model.layers)
    seq_len = clean_ids.shape[1]

    # ── Clean run: capture layer outputs ──
    clean_outputs: dict[int, torch.Tensor] = {}

    def capture(idx: int) -> Any:
        def hook(_m: Any, _i: Any, out: Any) -> None:
            clean_outputs[idx] = out[0].detach().clone()

        return hook

    hooks = [
        layer.register_forward_hook(capture(i)) for i, layer in enumerate(model.layers)
    ]
    with torch.no_grad():
        clean_logits = model(clean_ids)
    for h in hooks:
        h.remove()

    if isinstance(clean_logits, dict):
        clean_logits = clean_logits["logits"]
    assert isinstance(clean_logits, torch.Tensor)
    clean_logit = clean_logits[0, target_pos, target_token].item()

    # Clean embedding
    with torch.no_grad():
        clean_embed = model.token_embedding(clean_ids) * model.embed_scale

    # Auto-calibrate noise
    effective_noise_std: float = (
        noise_std if noise_std is not None else 3.0 * clean_embed.std().item()
    )
    print(f"  Using noise_std={effective_noise_std:.3f}")

    # ── Generate noise samples and get corrupt baselines ──
    noises: list[torch.Tensor] = []
    corrupt_logit_sum = 0.0

    for _ in range(n_samples):
        noise = torch.randn_like(clean_embed) * effective_noise_std
        noises.append(noise)

        nh = model.layers[0].register_forward_pre_hook(
            lambda _m, args, n=noise: (args[0] + n,) + args[1:]
        )
        with torch.no_grad():
            c_out = model(clean_ids)
        nh.remove()
        if isinstance(c_out, dict):
            c_out = c_out["logits"]
        assert isinstance(c_out, torch.Tensor)
        corrupt_logit_sum += c_out[0, target_pos, target_token].item()

    corrupt_logit = corrupt_logit_sum / n_samples
    logit_diff = clean_logit - corrupt_logit

    if abs(logit_diff) < 0.01:
        print(
            f"  WARNING: clean/corrupt logit diff is tiny ({logit_diff:.4f}), "
            "noise may be too small"
        )

    # ── Patch each (layer, position) ──
    results = np.zeros((n_layers + 1, seq_len))

    total_patches = n_samples * (n_layers + 1) * seq_len
    print(f"  Running {total_patches} patched forward passes...")

    for noise in noises:
        # Embedding-level patching (row 0)
        for pos in range(seq_len):

            def embed_patch_hook(
                _m: Any,
                args: Any,
                n: torch.Tensor = noise,
                p: int = pos,
            ) -> Any:
                x = args[0] + n
                x = x.clone()
                x[0, p] = clean_embed[0, p]
                return (x,) + args[1:]

            nh = model.layers[0].register_forward_pre_hook(embed_patch_hook)
            with torch.no_grad():
                p_out = model(clean_ids)
            nh.remove()
            if isinstance(p_out, dict):
                p_out = p_out["logits"]
            assert isinstance(p_out, torch.Tensor)
            p_logit = p_out[0, target_pos, target_token].item()
            results[0, pos] += (p_logit - corrupt_logit) / (logit_diff + 1e-10)

        # Layer-level patching (rows 1..n_layers)
        for li in range(n_layers):
            for pos in range(seq_len):

                def restore_hook(
                    _m: Any,
                    _inp: Any,
                    out: Any,
                    p: int = pos,
                    clean_out: torch.Tensor = clean_outputs[li],
                ) -> Any:
                    x, aw, asc = out
                    x = x.clone()
                    x[0, p] = clean_out[0, p]
                    return (x, aw, asc)

                nh = model.layers[0].register_forward_pre_hook(
                    lambda _m, args, n=noise: (args[0] + n,) + args[1:]
                )
                rh = model.layers[li].register_forward_hook(restore_hook)
                with torch.no_grad():
                    p_out = model(clean_ids)
                nh.remove()
                rh.remove()
                if isinstance(p_out, dict):
                    p_out = p_out["logits"]
                assert isinstance(p_out, torch.Tensor)
                p_logit = p_out[0, target_pos, target_token].item()
                results[li + 1, pos] += (p_logit - corrupt_logit) / (logit_diff + 1e-10)

    results /= n_samples
    return results, clean_logit, corrupt_logit


def plot_causal_trace(
    results: Any,
    token_labels: list[str],
    target_pos: int,
    target_token: int,
    clean_logit: float,
    corrupt_logit: float,
    output_path: Path,
) -> None:
    """Plot causal trace heatmap."""
    n_layers_plus = results.shape[0]
    seq_len = results.shape[1]

    fig, ax = plt.subplots(figsize=(max(14, seq_len * 0.8), 5))

    im = ax.imshow(
        results,
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=max(1.0, results.max()),
    )

    layer_labels = ["embed"] + [f"layer_{i}" for i in range(n_layers_plus - 1)]
    ax.set_yticks(range(n_layers_plus))
    ax.set_yticklabels(layer_labels)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels[:seq_len], rotation=45, ha="right", fontsize=8)

    # Annotate cells with values
    for li in range(n_layers_plus):
        for pos in range(seq_len):
            val = results[li, pos]
            if abs(val) > 0.1:
                color = "white" if val > 0.5 else "black"
                ax.text(
                    pos,
                    li,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )

    target_label = INV_VOCAB[target_token]
    ax.set_title(
        f"Causal Trace: Predicting '{target_label}' at position {target_pos}\n"
        f"(clean logit={clean_logit:.2f}, corrupt logit={corrupt_logit:.2f}, "
        f"recovery=1.0 means fully restored)",
        fontsize=10,
    )

    fig.colorbar(im, ax=ax, label="Logit recovery fraction")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved causal trace to {output_path}")


# ── Summary Generation ──────────────────────────────────────────────


def generate_summary(
    token_labels: list[str],
    target_positions: list[int],
    target_tokens: list[int],
    ll_results: dict[str, torch.Tensor],
    attentions: tuple[torch.Tensor, ...],
    causal_results: dict[int, tuple[Any, float, float]],
) -> str:
    """Generate a text summary of the findings."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("MECHANISTIC INTERPRETABILITY ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Sequence: {''.join(token_labels)}")
    lines.append(f"Tokens: {token_labels}")
    lines.append("")

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    layer_names = list(ll_results.keys())

    for tp, tt in zip(target_positions, target_tokens):
        target_label = INV_VOCAB[tt]
        lines.append("-" * 70)
        lines.append(
            f"PREDICTION: position {tp} ({token_labels[tp]}) → '{target_label}'"
        )
        lines.append("-" * 70)

        # Logit lens: at which layer does the correct prediction appear?
        lines.append("\n  Logit Lens (P(correct) at each layer):")
        for name in layer_names:
            logits = ll_results[name]
            prob = torch.softmax(logits[tp], dim=-1)[tt].item()
            top_id = int(logits[tp].argmax().item())
            top_tok = INV_VOCAB[top_id]
            marker = " ◄ CORRECT" if top_id == tt else ""
            lines.append(f"    {name:12s}: P={prob:.4f}, top='{top_tok}'{marker}")

        # Attention: which positions does the model attend to?
        lines.append("\n  Top attention sources (by head):")
        for li in range(n_layers):
            for hi in range(n_heads):
                attn = attentions[li][0, hi, tp, : tp + 1].numpy()
                top_positions = np.argsort(attn)[-3:][::-1]
                top_info = ", ".join(
                    f"pos {p} ({token_labels[p]})={attn[p]:.3f}" for p in top_positions
                )
                lines.append(f"    L{li}H{hi}: {top_info}")

        # Causal trace: which positions matter?
        if tp in causal_results:
            trace, cl, crl = causal_results[tp]
            lines.append(f"\n  Causal Trace (clean={cl:.2f}, corrupt={crl:.2f}):")
            lines.append("  Top-5 critical (layer, position) for this prediction:")
            flat_indices = np.argsort(trace.ravel())[-5:][::-1]
            for flat_idx in flat_indices:
                li, pos = np.unravel_index(flat_idx, trace.shape)
                layer_name = "embed" if li == 0 else f"layer_{li - 1}"
                recovery = trace[li, pos]
                lines.append(
                    f"    ({layer_name}, pos {pos} '{token_labels[pos]}'): "
                    f"recovery={recovery:.3f}"
                )

        lines.append("")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace model decisions using mechanistic interpretability"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<begin>10+9=",
        help="Prompt to generate from (default: '<begin>10+9=')",
    )
    parser.add_argument(
        "--full-sequence",
        type=str,
        default=None,
        help="Full sequence to analyze (skips generation)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("traces"),
        help="Output directory for plots and summary",
    )
    parser.add_argument(
        "--noise-samples",
        type=int,
        default=10,
        help="Number of noise samples for causal tracing",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint_dir)
    print(f"  Architecture: {config.architecture}")
    print(
        f"  d_model={config.d_model}, n_layers={config.n_layers}, "
        f"n_heads={config.n_heads}"
    )
    print(f"  Positional encoding: {config.positional_encoding}")
    print(f"  Layer norm: {config.layer_norm_type}")

    # Get the full sequence (generate if needed)
    if args.full_sequence:
        full_text = args.full_sequence
        print(f"\nUsing provided sequence: {full_text}")
    else:
        prompt_ids = tokenize(args.prompt)
        input_ids = torch.tensor([prompt_ids])
        print(f"\nGenerating from: {args.prompt}")
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=100, temperature=0.01)
        gen_tokens = generated[0].tolist()
        full_text = "".join(INV_VOCAB[t] for t in gen_tokens)
        print(f"Generated: {full_text}")

    # Tokenize full sequence
    all_ids = tokenize(full_text)
    token_labels = decode_tokens(all_ids)
    input_tensor = torch.tensor([all_ids])
    print(f"Tokens ({len(all_ids)}): {token_labels}")

    # Find answer positions: tokens after </think> that predict answer digits
    # The model predicts next token at each position, so we analyze position i
    # where position i is </think> or an answer digit, predicting the next token
    target_positions: list[int] = []
    target_tokens: list[int] = []

    end_think_idx = None
    for i, tid in enumerate(all_ids):
        if tid == END_THINK_TOKEN_ID:
            end_think_idx = i
            break

    if end_think_idx is not None:
        # Positions from </think> onwards (until <end>), each predicting next token
        for i in range(end_think_idx, len(all_ids) - 1):
            if all_ids[i + 1] == END_TOKEN_ID:
                # Also include the position predicting <end>
                target_positions.append(i)
                target_tokens.append(all_ids[i + 1])
                break
            target_positions.append(i)
            target_tokens.append(all_ids[i + 1])

    print("\nTarget predictions:")
    for tp, tt in zip(target_positions, target_tokens):
        print(f"  Position {tp} ({token_labels[tp]}) → '{INV_VOCAB[tt]}'")

    # ── Run analyses ──
    print("\n1. Running logit lens analysis...")
    ll_results = logit_lens(model, input_tensor)
    plot_logit_lens(
        ll_results,
        token_labels,
        target_positions,
        target_tokens,
        args.output_dir / "logit_lens.png",
    )

    print("\n2. Running attention pattern analysis...")
    attentions = get_attention_weights(model, input_tensor)
    plot_attention_patterns(
        attentions,
        token_labels,
        target_positions,
        target_tokens,
        args.output_dir / "attention_patterns.png",
    )
    plot_attention_summary(
        attentions,
        token_labels,
        target_positions,
        target_tokens,
        args.output_dir / "attention_summary.png",
    )

    print("\n3. Running causal tracing...")
    causal_results: dict[int, tuple[Any, float, float]] = {}
    for tp, tt in zip(target_positions, target_tokens):
        print(f"  Tracing position {tp} ({token_labels[tp]}) → '{INV_VOCAB[tt]}'...")
        trace, cl, crl = causal_trace(
            model, input_tensor, tp, tt, n_samples=args.noise_samples
        )
        causal_results[tp] = (trace, cl, crl)
        plot_causal_trace(
            trace,
            token_labels,
            tp,
            tt,
            cl,
            crl,
            args.output_dir / f"causal_trace_pos{tp}.png",
        )

    # ── Generate summary ──
    print("\n4. Generating summary...")
    summary = generate_summary(
        token_labels,
        target_positions,
        target_tokens,
        ll_results,
        attentions,
        causal_results,
    )
    print(summary)

    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text(summary)
    print(f"\nSaved summary to {summary_path}")
    print(f"All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
