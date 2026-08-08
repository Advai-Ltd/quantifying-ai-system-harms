"""
Unit tests for Option A: Variable Interbank Exposure.

Tests the interbank_assets_fraction field in BankGroupConfig.
"""

import numpy as np
from financial_contagion_networks.config.models import BankGroupConfig, DistributionConfig, PortfolioConfig


def create_test_config(interbank_fraction=None):
    """Helper to create a valid BankGroupConfig for testing."""
    return BankGroupConfig(
        capital_ratio=DistributionConfig(distribution='fixed', value=0.10),
        total_assets=1000.0,  # Must be float, not DistributionConfig
        portfolio=PortfolioConfig(
            mortgage=DistributionConfig(distribution='fixed', value=0.50),
            remaining={'government_bond': 0.3, 'corporate_bond': 0.4, 'stock': 0.3}  # Must sum to 1.0
        ),
        external_assets_fraction=0.85,
        interbank_assets_fraction=interbank_fraction
    )


class TestOptionABasicConfig:
    """Test basic configuration options for Option A."""

    def test_interbank_fraction_none_accepted(self):
        """Test that None is accepted (backward compatibility)."""
        config = create_test_config(interbank_fraction=None)
        assert config.interbank_assets_fraction is None

    def test_interbank_fraction_float_accepted(self):
        """Test that float values are accepted."""
        config = create_test_config(interbank_fraction=0.15)
        assert config.interbank_assets_fraction == 0.15

    def test_interbank_fraction_distribution_accepted(self):
        """Test that DistributionConfig is accepted."""
        ib_dist = DistributionConfig(
            distribution='normal',
            mean=0.15,
            std=0.05,
            min=0.08,
            max=0.30
        )
        config = create_test_config(interbank_fraction=ib_dist)

        assert isinstance(config.interbank_assets_fraction, DistributionConfig)
        assert config.interbank_assets_fraction.mean == 0.15

    def test_interbank_fraction_various_floats(self):
        """Test various valid float values."""
        test_values = [0.0, 0.05, 0.15, 0.30, 0.50, 1.0]

        for value in test_values:
            config = create_test_config(interbank_fraction=value)
            assert config.interbank_assets_fraction == value


class TestOptionASampling:
    """Test sampling behavior for DistributionConfig."""

    def test_sampling_respects_bounds(self):
        """Test that sampling respects min/max bounds."""
        ib_dist = DistributionConfig(
            distribution='normal',
            mean=0.15,
            std=0.05,
            min=0.08,
            max=0.30
        )

        rng = np.random.RandomState(42)
        samples = [ib_dist.sample(rng) for _ in range(100)]

        assert all(0.08 <= s <= 0.30 for s in samples)

    def test_sampling_produces_heterogeneity(self):
        """Test that sampling produces different values (not all the same)."""
        ib_dist = DistributionConfig(
            distribution='normal',
            mean=0.15,
            std=0.05,
            min=0.08,
            max=0.30
        )

        rng = np.random.RandomState(42)
        samples = [ib_dist.sample(rng) for _ in range(50)]

        # Should have variety
        unique_samples = len(set(samples))
        assert unique_samples > 40, f"Only {unique_samples} unique values"

        # Should have reasonable mean
        mean_sample = np.mean(samples)
        assert 0.12 <= mean_sample <= 0.18

    def test_sampling_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        ib_dist = DistributionConfig(
            distribution='normal',
            mean=0.15,
            std=0.05,
            min=0.08,
            max=0.30
        )

        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(123)

        samples1 = [ib_dist.sample(rng1) for _ in range(10)]
        samples2 = [ib_dist.sample(rng2) for _ in range(10)]

        assert samples1 != samples2


class TestOptionARealisticRanges:
    """Test realistic interbank ranges from research."""

    def test_pre_2008_core_ranges(self):
        """Test pre-2008 core bank realistic ranges (15% ± 5%)."""
        ib_dist = DistributionConfig(
            distribution='normal',
            mean=0.15,
            std=0.05,
            min=0.08,
            max=0.30
        )

        rng = np.random.RandomState(42)
        samples = [ib_dist.sample(rng) for _ in range(200)]

        # Check properties
        mean = np.mean(samples)
        assert 0.13 <= mean <= 0.17

        # All in range
        assert all(0.08 <= s <= 0.30 for s in samples)

        # Some at extremes
        assert any(s < 0.10 for s in samples)
        assert any(s > 0.25 for s in samples)

    def test_post_2008_reduced_ranges(self):
        """Test post-2008 ranges (10% ± 3%, reduced from pre-2008)."""
        ib_dist = DistributionConfig(
            distribution='normal',
            mean=0.10,
            std=0.03,
            min=0.05,
            max=0.18
        )

        rng = np.random.RandomState(42)
        samples = [ib_dist.sample(rng) for _ in range(200)]

        mean = np.mean(samples)
        assert 0.09 <= mean <= 0.11
        assert all(0.05 <= s <= 0.18 for s in samples)
