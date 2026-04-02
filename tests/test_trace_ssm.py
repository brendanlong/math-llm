"""Tests for SSM mechanistic interpretability tracing."""

import numpy as np
import torch

from scripts.trace_ssm import (
    SSMDiagnostics,
    capture_all_ssm_internals,
    compute_effective_influence,
    compute_ssm_internals,
    logit_lens,
)
from src.config import ModelConfig
from src.ssm import SelectiveSSM, SSMModel

SSM_TEST_CONFIG = ModelConfig(
    architecture="ssm",
    d_model=32,
    n_layers=2,
    d_state=4,
    d_conv=4,
    expand=2,
    dropout=0.0,
)


def make_test_model() -> SSMModel:
    """Create a small SSM model for testing."""
    model = SSMModel(
        d_model=32,
        n_layers=2,
        d_state=4,
        d_conv=4,
        expand=2,
        dropout=0.0,
    )
    model.eval()
    return model


class TestComputeSSMInternals:
    """Tests for recomputing SSM internals."""

    def test_output_types(self) -> None:
        """Test that diagnostics contain expected tensor shapes."""
        d_inner = 64
        ssm = SelectiveSSM(d_inner=d_inner, d_state=4)
        ssm.eval()
        x = torch.randn(1, 5, d_inner)
        diag = compute_ssm_internals(ssm, x)

        assert isinstance(diag, SSMDiagnostics)
        assert diag.dt.shape == (1, 5, d_inner)
        assert diag.B.shape == (1, 5, 4)
        assert diag.C.shape == (1, 5, 4)
        assert diag.A_bar.shape == (1, 5, d_inner, 4)
        assert diag.B_bar.shape == (1, 5, d_inner, 4)
        assert diag.h_states.shape == (1, 5, d_inner, 4)

    def test_dt_positive(self) -> None:
        """Test that dt values are positive (softplus output)."""
        d_inner = 64
        ssm = SelectiveSSM(d_inner=d_inner, d_state=4)
        ssm.eval()
        x = torch.randn(1, 5, d_inner)
        diag = compute_ssm_internals(ssm, x)
        assert (diag.dt > 0).all()

    def test_a_bar_range(self) -> None:
        """Test that A_bar values are in (0, 1) range (decay factors)."""
        d_inner = 64
        ssm = SelectiveSSM(d_inner=d_inner, d_state=4)
        ssm.eval()
        x = torch.randn(1, 5, d_inner)
        diag = compute_ssm_internals(ssm, x)
        assert (diag.A_bar > 0).all()
        assert (diag.A_bar <= 1).all()


class TestCaptureAllInternals:
    """Tests for capturing internals from full model."""

    def test_captures_all_layers(self) -> None:
        """Test that diagnostics are captured for each layer."""
        model = make_test_model()
        input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        diagnostics = capture_all_ssm_internals(model, input_ids)
        assert len(diagnostics) == 2  # n_layers=2

    def test_diagnostic_shapes(self) -> None:
        """Test that captured diagnostics have correct shapes."""
        model = make_test_model()
        input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        diagnostics = capture_all_ssm_internals(model, input_ids)

        d_inner = 32 * 2  # d_model * expand
        for diag in diagnostics:
            assert diag.dt.shape == (1, 5, d_inner)
            assert diag.h_states.shape == (1, 5, d_inner, 4)


class TestLogitLens:
    """Tests for logit lens on SSM models."""

    def test_returns_all_layers(self) -> None:
        """Test that logit lens returns results for embed + each layer + final."""
        model = make_test_model()
        input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        results = logit_lens(model, input_ids)

        assert "embed" in results
        assert "layer_0" in results
        assert "layer_1" in results
        assert "final" in results
        assert len(results) == 4  # embed + 2 layers + final

    def test_output_shape(self) -> None:
        """Test that logit lens outputs have correct shape."""
        model = make_test_model()
        input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        results = logit_lens(model, input_ids)

        from src.tokenizer import VOCAB_SIZE

        for name, logits in results.items():
            assert logits.shape == (5, VOCAB_SIZE), f"Wrong shape for {name}"


class TestEffectiveInfluence:
    """Tests for the effective influence matrix computation."""

    def test_shape(self) -> None:
        """Test that influence matrix is (seq_len, seq_len)."""
        d_inner = 64
        ssm = SelectiveSSM(d_inner=d_inner, d_state=4)
        ssm.eval()
        x = torch.randn(1, 8, d_inner)
        diag = compute_ssm_internals(ssm, x)
        influence = compute_effective_influence(diag)

        assert influence.shape == (8, 8)

    def test_causal(self) -> None:
        """Test that influence matrix is lower-triangular (causal)."""
        d_inner = 64
        ssm = SelectiveSSM(d_inner=d_inner, d_state=4)
        ssm.eval()
        x = torch.randn(1, 8, d_inner)
        diag = compute_ssm_internals(ssm, x)
        influence = compute_effective_influence(diag)

        # Upper triangle (excluding diagonal) should be zero
        for i in range(8):
            for j in range(i + 1, 8):
                assert influence[i, j] == 0.0, f"Non-causal at ({i}, {j})"

    def test_nonnegative(self) -> None:
        """Test that influence values are non-negative (we take abs)."""
        d_inner = 64
        ssm = SelectiveSSM(d_inner=d_inner, d_state=4)
        ssm.eval()
        x = torch.randn(1, 8, d_inner)
        diag = compute_ssm_internals(ssm, x)
        influence = compute_effective_influence(diag)

        assert (influence >= 0).all()

    def test_recent_positions_have_more_influence(self) -> None:
        """Test that more recent positions generally have more influence.

        Due to the decay factor A_bar < 1, information from distant positions
        should generally decay, so the diagonal and near-diagonal entries
        should tend to be larger.
        """
        d_inner = 64
        ssm = SelectiveSSM(d_inner=d_inner, d_state=4)
        ssm.eval()
        x = torch.randn(1, 20, d_inner)
        diag = compute_ssm_internals(ssm, x)
        influence = compute_effective_influence(diag)

        # For the last row, the self-influence (diagonal) should be among the largest
        last_row = influence[-1, :]
        assert last_row[-1] > np.median(last_row[last_row > 0])
