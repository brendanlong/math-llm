"""Unit tests for activation statistics computation."""

import torch

from src.activation_stats import (
    ActivationStatsCollector,
    ActivationStatsSummary,
    LayerStats,
    compute_attention_entropy,
    compute_attention_sparsity,
    compute_kurtosis,
    compute_outlier_fraction,
    compute_softmax1_abstention,
    format_stats_summary,
)


class TestComputeKurtosis:
    """Tests for compute_kurtosis function."""

    def test_normal_distribution_kurtosis(self) -> None:
        """Normal distribution should have excess kurtosis close to 0."""
        torch.manual_seed(42)
        # Generate a large normal distribution sample
        x = torch.randn(10000)
        kurtosis = compute_kurtosis(x)
        # Excess kurtosis of normal distribution is 0
        # Allow some variance due to sampling
        assert abs(kurtosis) < 0.3, (
            f"Expected kurtosis ~0 for normal dist, got {kurtosis}"
        )

    def test_uniform_distribution_kurtosis(self) -> None:
        """Uniform distribution should have negative excess kurtosis (~-1.2)."""
        torch.manual_seed(42)
        # Generate uniform distribution
        x = torch.rand(10000)
        kurtosis = compute_kurtosis(x)
        # Excess kurtosis of uniform distribution is -1.2
        assert -1.5 < kurtosis < -0.9, (
            f"Expected kurtosis ~-1.2 for uniform dist, got {kurtosis}"
        )

    def test_heavy_tailed_distribution_kurtosis(self) -> None:
        """Distribution with heavy tails should have positive excess kurtosis."""
        torch.manual_seed(42)
        # Create a distribution with outliers (simulating heavy tails)
        x = torch.randn(10000)
        # Add some extreme outliers
        x[::100] *= 10  # Make every 100th value an outlier
        kurtosis = compute_kurtosis(x)
        # Should have positive excess kurtosis due to outliers
        assert kurtosis > 1.0, (
            f"Expected positive kurtosis for heavy-tailed dist, got {kurtosis}"
        )

    def test_constant_tensor_kurtosis(self) -> None:
        """Constant tensor should return 0 (no variation)."""
        x = torch.ones(100)
        kurtosis = compute_kurtosis(x)
        assert kurtosis == 0.0

    def test_small_tensor_kurtosis(self) -> None:
        """Small tensor (< 4 elements) should return 0."""
        x = torch.tensor([1.0, 2.0, 3.0])
        kurtosis = compute_kurtosis(x)
        assert kurtosis == 0.0

    def test_multidimensional_tensor_kurtosis(self) -> None:
        """Kurtosis should flatten multidimensional tensors."""
        torch.manual_seed(42)
        x = torch.randn(10, 10, 100)
        kurtosis = compute_kurtosis(x)
        # Should compute kurtosis over all 10000 values
        assert isinstance(kurtosis, float)


class TestComputeOutlierFraction:
    """Tests for compute_outlier_fraction function."""

    def test_normal_distribution_outliers(self) -> None:
        """Normal distribution should have few outliers at 5 std threshold."""
        torch.manual_seed(42)
        x = torch.randn(10000)
        outlier_frac = compute_outlier_fraction(x, threshold_std=5.0)
        # Very few samples should be > 5 std in a normal distribution
        assert outlier_frac < 0.001, f"Expected <0.1% outliers, got {outlier_frac}"

    def test_some_outliers(self) -> None:
        """Test distribution with some outliers."""
        # Create a distribution where outliers are clearly beyond the threshold
        # Most values are 0, a few are extreme outliers
        x = torch.zeros(1000)
        x[:10] = 100.0  # 1% extreme outliers
        outlier_frac = compute_outlier_fraction(x, threshold_std=3.0)
        # The mean is ~1, std is ~10, so outliers at 100 are ~10 stds away
        # Should detect the 10 outliers out of 1000 (1%)
        assert outlier_frac > 0.005, (
            f"Expected outliers to be detected, got {outlier_frac}"
        )

    def test_no_outliers(self) -> None:
        """Test distribution with no outliers."""
        x = torch.linspace(-2, 2, 100)  # Values between -2 and 2
        outlier_frac = compute_outlier_fraction(x, threshold_std=5.0)
        # All values are within 5 std (uniform distribution spread)
        assert outlier_frac == 0.0

    def test_constant_tensor_outliers(self) -> None:
        """Constant tensor should have 0 outliers."""
        x = torch.ones(100)
        outlier_frac = compute_outlier_fraction(x)
        assert outlier_frac == 0.0


class TestComputeAttentionEntropy:
    """Tests for compute_attention_entropy function."""

    def test_uniform_attention_entropy(self) -> None:
        """Uniform attention should have maximum entropy."""
        # Uniform distribution over 10 items
        attn = torch.ones(1, 4, 8, 10) / 10.0
        entropy = compute_attention_entropy(attn)
        # Max entropy for 10 items is ln(10) ~= 2.303
        expected_max_entropy = torch.log(torch.tensor(10.0)).item()
        assert abs(entropy - expected_max_entropy) < 0.01

    def test_focused_attention_entropy(self) -> None:
        """Focused attention (all on one position) should have low entropy."""
        # Attention focused on single position
        attn = torch.zeros(1, 4, 8, 10)
        attn[:, :, :, 0] = 1.0  # All attention on first position
        entropy = compute_attention_entropy(attn)
        # Entropy should be close to 0
        assert entropy < 0.01

    def test_intermediate_attention_entropy(self) -> None:
        """Test attention with intermediate focus."""
        # 50% on one position, 50% on another
        attn = torch.zeros(1, 4, 8, 10)
        attn[:, :, :, 0] = 0.5
        attn[:, :, :, 1] = 0.5
        entropy = compute_attention_entropy(attn)
        # Entropy of binary uniform is ln(2) ~= 0.693
        expected_entropy = torch.log(torch.tensor(2.0)).item()
        assert abs(entropy - expected_entropy) < 0.01


class TestComputeAttentionSparsity:
    """Tests for compute_attention_sparsity function."""

    def test_uniform_attention_sparsity(self) -> None:
        """Uniform attention over many positions should have low sparsity."""
        # 10 positions with uniform attention
        attn = torch.ones(1, 4, 8, 10) / 10.0
        sparsity = compute_attention_sparsity(attn, threshold=0.01)
        # All weights are 0.1, above threshold of 0.01
        assert sparsity == 0.0

    def test_sparse_attention(self) -> None:
        """Attention with many zero weights should have high sparsity."""
        attn = torch.zeros(1, 4, 8, 10)
        attn[:, :, :, 0] = 1.0  # Only attend to first position
        sparsity = compute_attention_sparsity(attn, threshold=0.01)
        # 9 out of 10 positions have weight 0, so 90% sparse
        assert abs(sparsity - 0.9) < 0.01


class TestComputeSoftmax1Abstention:
    """Tests for compute_softmax1_abstention function."""

    def test_no_abstention(self) -> None:
        """Standard softmax (sum=1) should have no abstention."""
        attn = torch.ones(1, 4, 8, 10) / 10.0  # Sum to 1
        abstention = compute_softmax1_abstention(attn)
        assert abs(abstention) < 1e-6

    def test_full_abstention(self) -> None:
        """All-zero weights should have full abstention."""
        attn = torch.zeros(1, 4, 8, 10)
        abstention = compute_softmax1_abstention(attn)
        assert abs(abstention - 1.0) < 1e-6

    def test_partial_abstention(self) -> None:
        """Partial sum should have proportional abstention."""
        attn = torch.ones(1, 4, 8, 10) * 0.05  # Sum to 0.5
        abstention = compute_softmax1_abstention(attn)
        assert abs(abstention - 0.5) < 1e-6


class TestActivationStatsSummary:
    """Tests for ActivationStatsSummary class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        summary = ActivationStatsSummary(
            hidden_states=[
                LayerStats(
                    kurtosis=1.5, max_abs=10.0, mean_abs=2.0, outlier_fraction=0.01
                )
            ],
            num_batches=5,
        )
        d = summary.to_dict()
        assert "hidden_states" in d
        assert "aggregate" in d
        assert d["num_batches"] == 5
        assert d["hidden_states"][0]["kurtosis"] == 1.5

    def test_format_stats_summary(self) -> None:
        """Test formatting to string."""
        summary = ActivationStatsSummary(
            hidden_states=[
                LayerStats(
                    kurtosis=1.5, max_abs=10.0, mean_abs=2.0, outlier_fraction=0.01
                )
            ],
            num_batches=5,
        )
        formatted = format_stats_summary(summary)
        assert "Activation Statistics Summary" in formatted
        assert "kurtosis=1.50" in formatted
        assert "max_abs=10.00" in formatted


class TestActivationStatsCollector:
    """Tests for ActivationStatsCollector class."""

    def test_context_manager(self) -> None:
        """Test collector works as context manager."""
        model = torch.nn.Linear(10, 10)
        with ActivationStatsCollector(model):
            pass  # Just verify no exception is raised

    def test_clear(self) -> None:
        """Test clearing accumulated statistics."""
        model = torch.nn.Linear(10, 10)
        collector = ActivationStatsCollector(model)
        # Process some fake attention data
        collector.process_attention_outputs(
            (torch.randn(2, 4, 8, 8),), (torch.randn(2, 4, 8, 8),)
        )
        collector.clear()
        # After clear, computing statistics should return empty
        stats = collector.compute_statistics()
        assert stats.num_batches == 0

    def test_compute_empty_statistics(self) -> None:
        """Test computing statistics with no data."""
        model = torch.nn.Linear(10, 10)
        collector = ActivationStatsCollector(model)
        stats = collector.compute_statistics()
        assert stats.num_batches == 0
        assert len(stats.hidden_states) == 0
