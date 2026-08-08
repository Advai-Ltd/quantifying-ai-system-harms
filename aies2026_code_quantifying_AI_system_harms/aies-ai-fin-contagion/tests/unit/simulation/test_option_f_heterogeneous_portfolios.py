"""
Unit tests for Option F: Heterogeneous External Assets

Tests portfolio generation with heterogeneous asset allocations where each bank
can have different portfolio compositions (e.g., some specialize in mortgages,
others in government bonds).
"""

import pytest
import numpy as np
from financial_contagion_networks.config.models import (
    DistributionConfig,
    PortfolioConfig,
)
from financial_contagion_networks.core.assets import AssetType
from financial_contagion_networks.simulation.generators import NetworkGenerator


def test_heterogeneous_portfolio_config_valid():
    """Test that heterogeneous portfolio config is valid."""
    portfolio = PortfolioConfig(
        mortgage=DistributionConfig(distribution='uniform', min=0.2, max=0.5),
        government_bond=DistributionConfig(distribution='uniform', min=0.1, max=0.4),
        corporate_bond=DistributionConfig(distribution='uniform', min=0.1, max=0.4),
        stock=DistributionConfig(distribution='uniform', min=0.1, max=0.4)
    )

    assert portfolio.is_heterogeneous()
    assert portfolio.mortgage is not None
    assert portfolio.government_bond is not None


def test_legacy_portfolio_config_valid():
    """Test that legacy portfolio config is still valid."""
    portfolio = PortfolioConfig(
        mortgage=DistributionConfig(distribution='uniform', min=0.4, max=0.55),
        remaining={'government_bond': 0.3, 'corporate_bond': 0.4, 'stock': 0.3}
    )

    assert not portfolio.is_heterogeneous()
    assert portfolio.remaining is not None


def test_heterogeneous_portfolio_missing_asset_fails():
    """Test that heterogeneous mode requires all three assets."""
    with pytest.raises(ValueError, match="all three assets"):
        PortfolioConfig(
            mortgage=DistributionConfig(distribution='uniform', min=0.2, max=0.5),
            government_bond=DistributionConfig(distribution='uniform', min=0.1, max=0.4),
            # Missing corporate_bond and stock
        )


def test_portfolio_neither_mode_fails():
    """Test that portfolio must specify either remaining or individual assets."""
    with pytest.raises(ValueError, match="Either 'remaining'"):
        PortfolioConfig(
            mortgage=DistributionConfig(distribution='uniform', min=0.2, max=0.5),
            # Missing both remaining and individual assets
        )


def test_heterogeneous_portfolio_weights_sum_to_one():
    """Test that heterogeneous portfolios are normalized to sum to 1.0."""
    portfolio = PortfolioConfig(
        mortgage=DistributionConfig(distribution='fixed', value=0.5),
        government_bond=DistributionConfig(distribution='fixed', value=0.3),
        corporate_bond=DistributionConfig(distribution='fixed', value=0.2),
        stock=DistributionConfig(distribution='fixed', value=0.1)
    )

    # Use NetworkGenerator static method
    rng = np.random.default_rng(42)
    weights = NetworkGenerator._calculate_portfolio_weights(portfolio, rng)

    # Weights should be normalized
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6

    # Check proportions (should be normalized from 0.5:0.3:0.2:0.1 = 1.1 total)
    # Expected: 0.5/1.1 ≈ 0.4545, 0.3/1.1 ≈ 0.2727, etc.
    assert abs(weights[AssetType.MORTGAGE] - 0.4545) < 0.01
    assert abs(weights[AssetType.GOVERNMENT_BOND] - 0.2727) < 0.01


def test_heterogeneous_portfolio_produces_variation():
    """Test that heterogeneous portfolios produce different compositions per bank."""
    portfolio = PortfolioConfig(
        mortgage=DistributionConfig(distribution='uniform', min=0.2, max=0.6),
        government_bond=DistributionConfig(distribution='uniform', min=0.1, max=0.4),
        corporate_bond=DistributionConfig(distribution='uniform', min=0.1, max=0.4),
        stock=DistributionConfig(distribution='uniform', min=0.1, max=0.4)
    )

    # Sample 10 different portfolios
    mortgage_weights = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        weights = NetworkGenerator._calculate_portfolio_weights(portfolio, rng)
        mortgage_weights.append(weights[AssetType.MORTGAGE])

    # Check variation (at least 8 unique values out of 10)
    unique_values = len(set(round(w, 4) for w in mortgage_weights))
    assert unique_values >= 8, f"Expected at least 8 unique mortgage weights, got {unique_values}"

    # Check range (should span reasonable range)
    assert max(mortgage_weights) - min(mortgage_weights) > 0.1


def test_heterogeneous_portfolio_realistic_ranges():
    """Test realistic portfolio ranges for heterogeneous assets."""
    # Mortgage specialist (high mortgage exposure)
    mortgage_specialist = PortfolioConfig(
        mortgage=DistributionConfig(distribution='uniform', min=0.45, max=0.65),
        government_bond=DistributionConfig(distribution='uniform', min=0.10, max=0.25),
        corporate_bond=DistributionConfig(distribution='uniform', min=0.10, max=0.25),
        stock=DistributionConfig(distribution='uniform', min=0.05, max=0.20)
    )

    rng = np.random.default_rng(42)
    weights = NetworkGenerator._calculate_portfolio_weights(mortgage_specialist, rng)

    # Should have high mortgage weight after normalization
    assert weights[AssetType.MORTGAGE] > 0.40  # At least 40% mortgages

    # Conservative bank (high government bonds)
    conservative = PortfolioConfig(
        mortgage=DistributionConfig(distribution='uniform', min=0.15, max=0.30),
        government_bond=DistributionConfig(distribution='uniform', min=0.40, max=0.60),
        corporate_bond=DistributionConfig(distribution='uniform', min=0.15, max=0.30),
        stock=DistributionConfig(distribution='uniform', min=0.05, max=0.20)
    )

    rng = np.random.default_rng(100)
    weights = NetworkGenerator._calculate_portfolio_weights(conservative, rng)

    # Should have high government bond weight
    assert weights[AssetType.GOVERNMENT_BOND] > 0.35  # At least 35% gov bonds


def test_heterogeneous_vs_legacy_portfolio_difference():
    """Test that heterogeneous mode creates more diversity than legacy mode."""
    # Legacy mode: All banks in tier get same gov/corp/stock split
    legacy = PortfolioConfig(
        mortgage=DistributionConfig(distribution='uniform', min=0.4, max=0.5),
        remaining={'government_bond': 0.3, 'corporate_bond': 0.4, 'stock': 0.3}
    )

    # Heterogeneous mode: Each bank gets different splits
    heterogeneous = PortfolioConfig(
        mortgage=DistributionConfig(distribution='uniform', min=0.4, max=0.5),
        government_bond=DistributionConfig(distribution='uniform', min=0.1, max=0.4),
        corporate_bond=DistributionConfig(distribution='uniform', min=0.1, max=0.4),
        stock=DistributionConfig(distribution='uniform', min=0.1, max=0.4)
    )

    # Sample 10 portfolios with each mode
    legacy_gov_weights = []
    heterogeneous_gov_weights = []

    for seed in range(10):
        rng = np.random.default_rng(seed)
        legacy_weights = NetworkGenerator._calculate_portfolio_weights(legacy, rng)
        legacy_gov_weights.append(legacy_weights[AssetType.GOVERNMENT_BOND])

        rng = np.random.default_rng(seed)
        hetero_weights = NetworkGenerator._calculate_portfolio_weights(heterogeneous, rng)
        heterogeneous_gov_weights.append(hetero_weights[AssetType.GOVERNMENT_BOND])

    # Legacy: All gov bond weights should be very similar (only vary due to mortgage sampling)
    legacy_range = max(legacy_gov_weights) - min(legacy_gov_weights)

    # Heterogeneous: Gov bond weights should vary significantly
    hetero_range = max(heterogeneous_gov_weights) - min(heterogeneous_gov_weights)

    # Heterogeneous should have at least 2x more variation
    assert hetero_range > legacy_range * 2
