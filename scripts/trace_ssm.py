"""Mechanistic interpretability analysis for SSM (Mamba) models.

Performs four analyses on a trained SSM model:
1. Logit Lens: How predictions evolve through layers
2. Selective Gate Analysis: Where the model "opens the gate" (dt values)
3. Effective Influence Matrix: The SSM's analogue of attention weights
4. Causal Tracing: Which (layer, position) pairs are critical

The key insight is that SSMs have direct analogues to attention:
- dt (step size) controls how much the model "attends" to the current input
  vs. relying on state memory. Large dt = open gate, small dt = rely on memory.
- The effective influence from position j on output at position i is:
  C_i * (prod_{k=j+1}^{i} A_bar_k) * B_bar_j, which gives us an
  "attention-like" matrix showing information flow.

Usage:
    python scripts/trace_ssm.py \
        --checkpoint-dir checkpoints/ssm-small/checkpoint-19000/ \
        --prompt "<begin>10+9=" \
        --output-dir traces/ssm/
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
from torch.nn import functional as F

from src.config import ModelConfig, find_config_in_checkpoint, load_config
from src.model import create_model_from_config
from src.ssm import MambaBlock, SelectiveSSM, SSMModel
from src.tokenizer import END_THINK_TOKEN_ID, END_TOKEN_ID, VOCAB

matplotlib.use("Agg")

INV_VOCAB = {v: k for k, v in VOCAB.items()}


def tokenize(text: str) -> list[int]:
    """Tokenize text manually using regex."""
    pattern = r"<begin>|</think>|<think>|<noop>|<end>|[0-9+=\s]"
    return [VOCAB[t] for t in re.findall(pattern, text)]


def decode_tokens(ids: list[int]) -> list[str]:
    """Decode token IDs to display strings."""
    return [INV_VOCAB[i] for i in ids]


def load_model(checkpoint_dir: Path) -> tuple[SSMModel, ModelConfig]:
    """Load SSM model from checkpoint directory."""
    config_path = find_config_in_checkpoint(checkpoint_dir)
    if config_path is None:
        raise FileNotFoundError(f"No model_config.yaml found near {checkpoint_dir}")
    config = load_config(config_path)
    model = create_model_from_config(config)
    state = safetensors.torch.load_file(str(checkpoint_dir / "model.safetensors"))
    model.load_state_dict(state)
    model.eval()
    if not isinstance(model, SSMModel):
        raise TypeError(f"Expected SSMModel, got {type(model).__name__}")
    return model, config


# ── SSM Internals Computation ────────────────────────────────────────


class SSMDiagnostics:
    """Captured internals from a single SelectiveSSM layer."""

    def __init__(
        self,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        h_states: torch.Tensor,
        x_input: torch.Tensor,
    ):
        self.dt = dt  # (batch, seq_len, d_inner)
        self.B = B  # (batch, seq_len, d_state)
        self.C = C  # (batch, seq_len, d_state)
        self.A_bar = A_bar  # (batch, seq_len, d_inner, d_state)
        self.B_bar = B_bar  # (batch, seq_len, d_inner, d_state)
        self.h_states = h_states  # (batch, seq_len, d_inner, d_state)
        self.x_input = x_input  # (batch, seq_len, d_inner)


def compute_ssm_internals(ssm: SelectiveSSM, x: torch.Tensor) -> SSMDiagnostics:
    """Recompute SSM internals for analysis without modifying the model.

    Given the input to SelectiveSSM and its parameters, recomputes all
    intermediate values (dt, B, C, A_bar, B_bar, hidden states).
    """
    with torch.no_grad():
        x_proj = ssm.x_proj(x)
        dt_raw = x_proj[:, :, : ssm.dt_rank]
        B = x_proj[:, :, ssm.dt_rank : ssm.dt_rank + ssm.d_state]
        C = x_proj[:, :, ssm.dt_rank + ssm.d_state :]

        dt = F.softplus(ssm.dt_proj(dt_raw))

        A = -torch.exp(ssm.A_log)
        dt_expanded = dt.unsqueeze(-1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)

        A_bar = torch.exp(dt_expanded * A_expanded)
        B_bar = dt_expanded * B.unsqueeze(2)

        # Rerun the sequential scan to capture h at each timestep
        batch = x.shape[0]
        seq_len = x.shape[1]
        h = torch.zeros(batch, ssm.d_inner, ssm.d_state, device=x.device, dtype=x.dtype)
        h_states = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            h_states.append(h.clone())

        h_states_tensor = torch.stack(h_states, dim=1)

    return SSMDiagnostics(
        dt=dt,
        B=B,
        C=C,
        A_bar=A_bar,
        B_bar=B_bar,
        h_states=h_states_tensor,
        x_input=x,
    )


def capture_all_ssm_internals(
    model: SSMModel, input_ids: torch.Tensor
) -> list[SSMDiagnostics]:
    """Capture SSM internals from all layers via hooks.

    Hooks capture the input to each MambaBlock's SSM submodule, then
    recomputes internals from those inputs and the SSM's parameters.
    """
    ssm_inputs: dict[int, torch.Tensor] = {}

    def capture_ssm_input(idx: int) -> Any:
        def hook(_mod: Any, args: tuple[torch.Tensor, ...]) -> None:
            ssm_inputs[idx] = args[0].detach()

        return hook

    hooks = []
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, MambaBlock)
        hooks.append(layer.ssm.register_forward_pre_hook(capture_ssm_input(i)))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    diagnostics = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        assert isinstance(layer, MambaBlock)
        diagnostics.append(compute_ssm_internals(layer.ssm, ssm_inputs[i]))

    return diagnostics


# ── Analysis 1: Logit Lens ──────────────────────────────────────────


def logit_lens(model: SSMModel, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Project residual stream at each layer through ln_f + lm_head."""
    layer_outputs: dict[int, torch.Tensor] = {}

    def capture(idx: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: torch.Tensor) -> None:
            # MambaBlock returns a plain tensor (not a tuple)
            layer_outputs[idx] = out.detach()

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
    """Plot logit lens: how the prediction evolves through layers."""
    layer_names = list(ll_results.keys())
    n_layers = len(layer_names)
    seq_len = ll_results["final"].shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, seq_len * 0.8), 10))

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

    for li in range(n_layers):
        for pos in range(seq_len):
            tok = INV_VOCAB[pred_grid[li, pos]]
            if len(tok) > 3:
                tok = tok[:3]
            color = "white" if prob_grid[li, pos] > 0.6 else "black"
            ax1.text(pos, li, tok, ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im1, ax=ax1, label="Max probability")

    for tp, tt in zip(target_positions, target_tokens):
        probs_across_layers = []
        for name in layer_names:
            logits = ll_results[name]
            prob = torch.softmax(logits[tp], dim=-1)[tt].item()
            probs_across_layers.append(prob)
        label = f"pos {tp} ({token_labels[tp]}) -> {INV_VOCAB[tt]}"
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


# ── Analysis 2: Selective Gate (dt) Analysis ────────────────────────


def plot_selective_gates(
    diagnostics: list[SSMDiagnostics],
    token_labels: list[str],
    output_path: Path,
) -> None:
    """Plot dt (step size) values across layers and positions.

    dt controls how much the model "opens the gate" to the current input.
    Large dt = model is paying attention to this position.
    Small dt = model is relying on state memory (ignoring current input).
    """
    n_layers = len(diagnostics)
    seq_len = diagnostics[0].dt.shape[1]

    # Average dt over d_inner channels for a position-level summary
    dt_grid = np.zeros((n_layers, seq_len))
    for li, diag in enumerate(diagnostics):
        dt_grid[li] = diag.dt[0].mean(dim=-1).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, seq_len * 0.8), 8))

    # Heatmap of mean dt per (layer, position)
    im = ax1.imshow(dt_grid, aspect="auto", cmap="YlOrRd")
    ax1.set_yticks(range(n_layers))
    ax1.set_yticklabels([f"layer_{i}" for i in range(n_layers)])
    ax1.set_xticks(range(seq_len))
    ax1.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_title("Selective Gate (dt): Mean step size per (layer, position)")
    fig.colorbar(im, ax=ax1, label="Mean dt")

    for li in range(n_layers):
        for pos in range(seq_len):
            val = dt_grid[li, pos]
            color = "white" if val > dt_grid.max() * 0.6 else "black"
            ax1.text(
                pos,
                li,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=5,
                color=color,
            )

    # Line plot: dt per position for each layer
    for li in range(n_layers):
        ax2.plot(range(seq_len), dt_grid[li], "o-", markersize=3, label=f"layer_{li}")
    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean dt")
    ax2.set_title("Gate Opening Per Position")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved selective gate plot to {output_path}")


# ── Analysis 3: Effective Influence Matrix ──────────────────────────


def compute_effective_influence(
    diag: SSMDiagnostics,
) -> np.ndarray:
    """Compute the effective influence matrix for one SSM layer.

    This is the SSM's analogue of attention weights. For output at position i,
    the contribution from input at position j (j <= i) flows through:

        influence(i, j) = C_i * (prod_{k=j+1}^{i} A_bar_k) * B_bar_j

    We compute the L2 norm over state dimensions and average over channels
    to get a scalar influence weight for each (i, j) pair.

    Returns:
        influence: (seq_len, seq_len) numpy array, lower-triangular
    """
    seq_len = diag.A_bar.shape[1]

    # Work with batch=0
    A_bar = diag.A_bar[0]  # (seq_len, d_inner, d_state)
    B_bar = diag.B_bar[0]  # (seq_len, d_inner, d_state)
    C = diag.C[0]  # (seq_len, d_state)
    # Compute cumulative log of A_bar for efficient decay computation
    # log(A_bar) is negative (since A is negative and exp preserves sign structure)
    log_A_bar = torch.log(A_bar + 1e-10)  # (seq_len, d_inner, d_state)
    cum_log = torch.cumsum(log_A_bar, dim=0)  # (seq_len, d_inner, d_state)

    influence = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        for j in range(i + 1):
            if j == i:
                # Direct contribution: C_i * B_bar_i (no decay)
                contrib = C[i].unsqueeze(0) * B_bar[j]  # (d_inner, d_state)
            else:
                # Decay from j+1 to i: exp(cum_log[i] - cum_log[j])
                decay = torch.exp(cum_log[i] - cum_log[j])  # (d_inner, d_state)
                contrib = C[i].unsqueeze(0) * decay * B_bar[j]  # (d_inner, d_state)

            # Sum over state dims, then take abs and average over channels
            per_channel = contrib.sum(dim=-1).abs()  # (d_inner,)
            influence[i, j] = per_channel.mean().item()

    return influence


def plot_effective_influence(
    diagnostics: list[SSMDiagnostics],
    token_labels: list[str],
    target_positions: list[int],
    output_path: Path,
) -> None:
    """Plot effective influence matrices (SSM's "attention") for each layer.

    Shows what each output position is "attending to" in the input.
    """
    n_layers = len(diagnostics)
    seq_len = diagnostics[0].dt.shape[1]

    # Compute influence matrices for all layers
    influences = [compute_effective_influence(diag) for diag in diagnostics]

    fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 5, 5), squeeze=False)

    im = None
    for li, inf_matrix in enumerate(influences):
        ax = axes[0, li]
        # Normalize per-row for better visibility
        row_max = inf_matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        normalized = inf_matrix / row_max

        im = ax.imshow(normalized, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Layer {li}", fontsize=10)
        ax.set_xlabel("Source position")
        ax.set_ylabel("Output position")

        # Mark target positions
        for tp in target_positions:
            ax.axhline(y=tp, color="red", linewidth=0.5, alpha=0.5)

        if seq_len <= 30:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=5)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(token_labels, fontsize=5)

    fig.suptitle(
        "Effective Influence Matrix (SSM's 'Attention')\n"
        "Row-normalized: for each output position, what inputs contribute most",
        fontsize=12,
        fontweight="bold",
    )
    assert im is not None
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Normalized influence")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved effective influence plot to {output_path}")


def plot_influence_at_targets(
    diagnostics: list[SSMDiagnostics],
    token_labels: list[str],
    target_positions: list[int],
    target_tokens: list[int],
    output_path: Path,
) -> None:
    """For each target prediction, plot influence from all source positions per layer.

    This is the SSM analogue of "what does the model attend to when predicting
    each answer digit?"
    """
    n_layers = len(diagnostics)
    n_targets = len(target_positions)

    influences = [compute_effective_influence(diag) for diag in diagnostics]

    fig, axes = plt.subplots(
        n_targets,
        n_layers,
        figsize=(n_layers * 4, n_targets * 3),
        squeeze=False,
    )

    for ti, (tp, tt) in enumerate(zip(target_positions, target_tokens)):
        for li in range(n_layers):
            ax = axes[ti, li]
            inf_row = influences[li][tp, : tp + 1]
            # Normalize
            row_max = inf_row.max()
            if row_max > 0:
                inf_row = inf_row / row_max

            bars = ax.bar(range(tp + 1), inf_row, color="steelblue", alpha=0.8)

            # Highlight top-3
            top_3 = np.argsort(inf_row)[-3:]
            for idx in top_3:
                bars[idx].set_color("coral")
                bars[idx].set_alpha(1.0)

            ax.set_xticks(range(tp + 1))
            ax.set_xticklabels(
                token_labels[: tp + 1], rotation=90, ha="center", fontsize=5
            )
            ax.set_ylim(0, 1.1)

            if ti == 0:
                ax.set_title(f"Layer {li}", fontsize=10)
            if li == 0:
                target_label = INV_VOCAB[tt]
                ax.set_ylabel(
                    f"pos {tp} -> {target_label}", fontsize=9, fontweight="bold"
                )

    fig.suptitle(
        "Source Influence at Target Positions (SSM's 'Attention Pattern')\n"
        "Top-3 sources highlighted in coral",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved influence-at-targets plot to {output_path}")


# ── Analysis 4: State Dynamics ──────────────────────────────────────


def plot_state_dynamics(
    diagnostics: list[SSMDiagnostics],
    token_labels: list[str],
    output_path: Path,
) -> None:
    """Plot hidden state norm evolution across positions for each layer.

    Shows where information accumulates or decays in the SSM state.
    """
    seq_len = diagnostics[0].h_states.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, seq_len * 0.8), 5))

    # Plot 1: State norm per position per layer
    ax1 = axes[0]
    for li, diag in enumerate(diagnostics):
        # h_states: (batch, seq_len, d_inner, d_state)
        h_norm = diag.h_states[0].norm(dim=(-2, -1)).cpu().numpy()
        ax1.plot(range(seq_len), h_norm, "o-", markersize=3, label=f"layer_{li}")

    ax1.set_xticks(range(seq_len))
    ax1.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("||h_t|| (Frobenius norm)")
    ax1.set_title("Hidden State Norm Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: State change (delta h norm) per position per layer
    ax2 = axes[1]
    for li, diag in enumerate(diagnostics):
        h = diag.h_states[0]  # (seq_len, d_inner, d_state)
        delta_h = torch.zeros(seq_len)
        delta_h[0] = h[0].norm().item()
        for t in range(1, seq_len):
            delta_h[t] = (h[t] - h[t - 1]).norm().item()
        ax2.plot(
            range(seq_len), delta_h.numpy(), "o-", markersize=3, label=f"layer_{li}"
        )

    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("||h_t - h_{t-1}||")
    ax2.set_title("State Change Magnitude (where info is written)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved state dynamics plot to {output_path}")


# ── Analysis 5: Causal Tracing ──────────────────────────────────────


def causal_trace(
    model: SSMModel,
    clean_ids: torch.Tensor,
    target_pos: int,
    target_token: int,
    noise_std: Optional[float] = None,
    n_samples: int = 10,
) -> tuple[Any, float, float]:
    """Denoising activation patching for causal tracing.

    For each (layer, position), restores the clean activation in an otherwise
    noisy forward pass and measures recovery of the target logit.
    """
    n_layers = len(model.layers)
    seq_len = clean_ids.shape[1]

    # Clean run: capture layer outputs
    clean_outputs: dict[int, torch.Tensor] = {}

    def capture(idx: int) -> Any:
        def hook(_m: Any, _i: Any, out: torch.Tensor) -> None:
            clean_outputs[idx] = out.detach().clone()

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

    with torch.no_grad():
        clean_embed = model.token_embedding(clean_ids) * model.embed_scale

    effective_noise_std: float = (
        noise_std if noise_std is not None else 3.0 * clean_embed.std().item()
    )
    print(f"  Using noise_std={effective_noise_std:.3f}")

    # Generate noise samples and get corrupt baselines
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

    # Patch each (layer, position)
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
                    out: torch.Tensor,
                    p: int = pos,
                    clean_out: torch.Tensor = clean_outputs[li],
                ) -> torch.Tensor:
                    # MambaBlock returns a plain tensor
                    x = out.clone()
                    x[0, p] = clean_out[0, p]
                    return x

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
    diagnostics: list[SSMDiagnostics],
    causal_results: dict[int, tuple[Any, float, float]],
) -> str:
    """Generate a text summary of the findings."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("MECHANISTIC INTERPRETABILITY ANALYSIS (SSM)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Sequence: {''.join(token_labels)}")
    lines.append(f"Tokens: {token_labels}")
    lines.append("")

    n_layers = len(diagnostics)
    layer_names = list(ll_results.keys())

    # Precompute influence matrices
    influences = [compute_effective_influence(diag) for diag in diagnostics]

    for tp, tt in zip(target_positions, target_tokens):
        target_label = INV_VOCAB[tt]
        lines.append("-" * 70)
        lines.append(
            f"PREDICTION: position {tp} ({token_labels[tp]}) -> '{target_label}'"
        )
        lines.append("-" * 70)

        # Logit lens
        lines.append("\n  Logit Lens (P(correct) at each layer):")
        for name in layer_names:
            logits = ll_results[name]
            prob = torch.softmax(logits[tp], dim=-1)[tt].item()
            top_id = int(logits[tp].argmax().item())
            top_tok = INV_VOCAB[top_id]
            marker = " << CORRECT" if top_id == tt else ""
            lines.append(f"    {name:12s}: P={prob:.4f}, top='{top_tok}'{marker}")

        # Selective gate analysis
        lines.append("\n  Selective Gate (dt) - where the model opens the gate:")
        for li in range(n_layers):
            dt_at_pos = diagnostics[li].dt[0, tp].mean().item()
            dt_all_mean = diagnostics[li].dt[0].mean().item()
            ratio = dt_at_pos / (dt_all_mean + 1e-10)
            lines.append(
                f"    layer_{li}: dt={dt_at_pos:.3f} "
                f"(mean={dt_all_mean:.3f}, ratio={ratio:.2f}x)"
            )

        # Effective influence: top sources
        lines.append("\n  Top influence sources (SSM's 'attention'):")
        for li in range(n_layers):
            inf_row = influences[li][tp, : tp + 1]
            row_max = inf_row.max()
            if row_max > 0:
                inf_row = inf_row / row_max
            top_positions = np.argsort(inf_row)[-3:][::-1]
            top_info = ", ".join(
                f"pos {p} ({token_labels[p]})={inf_row[p]:.3f}" for p in top_positions
            )
            lines.append(f"    layer_{li}: {top_info}")

        # State dynamics at this position
        lines.append("\n  State dynamics:")
        for li in range(n_layers):
            h_norm = diagnostics[li].h_states[0, tp].norm().item()
            if tp > 0:
                delta = (
                    (
                        diagnostics[li].h_states[0, tp]
                        - diagnostics[li].h_states[0, tp - 1]
                    )
                    .norm()
                    .item()
                )
            else:
                delta = h_norm
            lines.append(f"    layer_{li}: ||h||={h_norm:.3f}, ||delta_h||={delta:.3f}")

        # Causal trace
        if tp in causal_results:
            trace, cl, crl = causal_results[tp]
            lines.append(f"\n  Causal Trace (clean={cl:.2f}, corrupt={crl:.2f}):")
            lines.append("  Top-5 critical (layer, position):")
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
        description="Trace SSM model decisions using mechanistic interpretability"
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
        default=Path("traces/ssm"),
        help="Output directory for plots and summary",
    )
    parser.add_argument(
        "--noise-samples",
        type=int,
        default=10,
        help="Number of noise samples for causal tracing",
    )
    parser.add_argument(
        "--skip-causal-trace",
        action="store_true",
        help="Skip causal tracing (slow for long sequences)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint_dir)
    print(f"  Architecture: {config.architecture}")
    print(
        f"  d_model={config.d_model}, n_layers={config.n_layers}, "
        f"d_state={config.d_state}, expand={config.expand}"
    )

    # Get the full sequence
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

    # Find answer positions
    target_positions: list[int] = []
    target_tokens: list[int] = []

    end_think_idx = None
    for i, tid in enumerate(all_ids):
        if tid == END_THINK_TOKEN_ID:
            end_think_idx = i
            break

    if end_think_idx is not None:
        for i in range(end_think_idx, len(all_ids) - 1):
            if all_ids[i + 1] == END_TOKEN_ID:
                target_positions.append(i)
                target_tokens.append(all_ids[i + 1])
                break
            target_positions.append(i)
            target_tokens.append(all_ids[i + 1])

    if not target_positions:
        # No </think> found -- look for = sign and use positions after it
        for i, tid in enumerate(all_ids):
            if INV_VOCAB[tid] == "=" and i < len(all_ids) - 1:
                for j in range(i, len(all_ids) - 1):
                    if all_ids[j + 1] == END_TOKEN_ID:
                        target_positions.append(j)
                        target_tokens.append(all_ids[j + 1])
                        break
                    target_positions.append(j)
                    target_tokens.append(all_ids[j + 1])
                break

    print("\nTarget predictions:")
    for tp, tt in zip(target_positions, target_tokens):
        print(f"  Position {tp} ({token_labels[tp]}) -> '{INV_VOCAB[tt]}'")

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

    print("\n2. Capturing SSM internals...")
    diagnostics = capture_all_ssm_internals(model, input_tensor)

    print("\n3. Plotting selective gate (dt) analysis...")
    plot_selective_gates(
        diagnostics, token_labels, args.output_dir / "selective_gates.png"
    )

    print("\n4. Computing effective influence matrices...")
    plot_effective_influence(
        diagnostics,
        token_labels,
        target_positions,
        args.output_dir / "effective_influence.png",
    )
    plot_influence_at_targets(
        diagnostics,
        token_labels,
        target_positions,
        target_tokens,
        args.output_dir / "influence_at_targets.png",
    )

    print("\n5. Plotting state dynamics...")
    plot_state_dynamics(
        diagnostics, token_labels, args.output_dir / "state_dynamics.png"
    )

    # Causal tracing
    causal_results: dict[int, tuple[Any, float, float]] = {}
    if not args.skip_causal_trace:
        print("\n6. Running causal tracing...")
        for tp, tt in zip(target_positions, target_tokens):
            print(
                f"  Tracing position {tp} ({token_labels[tp]}) -> '{INV_VOCAB[tt]}'..."
            )
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
    else:
        print("\n6. Skipping causal tracing (use --skip-causal-trace to skip)")

    # Generate summary
    print("\n7. Generating summary...")
    summary = generate_summary(
        token_labels,
        target_positions,
        target_tokens,
        ll_results,
        diagnostics,
        causal_results,
    )
    print(summary)

    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text(summary)
    print(f"\nSaved summary to {summary_path}")
    print(f"All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
