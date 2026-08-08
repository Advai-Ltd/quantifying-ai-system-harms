"""
Unit tests for shocks module.

Tests the three shock modes:
- deterministic: Exact base shocks (no randomness)
- correlated: Correlated random shocks (traditional model)
- uncorrelated: Independent random shocks per asset
"""

import pytest
import numpy as np
from financial_contagion_networks.core.shocks import (
    ShockMode,
    ShockScenario,
    ShockGenerator
)
from financial_contagion_networks.core.assets import AssetType


# ============================================================================
# Test ShockMode Enum
# ============================================================================

class TestShockMode:
    """Test ShockMode enum."""

    def test_shock_mode_values(self):
        """Test ShockMode enum values."""
        assert ShockMode.DETERMINISTIC.value == "deterministic"
        assert ShockMode.CORRELATED.value == "correlated"
        assert ShockMode.UNCORRELATED.value == "uncorrelated"

    def test_shock_mode_from_string(self):
        """Test creating ShockMode from string."""
        assert ShockMode("deterministic") == ShockMode.DETERMINISTIC
        assert ShockMode("correlated") == ShockMode.CORRELATED
        assert ShockMode("uncorrelated") == ShockMode.UNCORRELATED


# ============================================================================
# Test ShockScenario
# ============================================================================

class TestShockScenario:
    """Test ShockScenario dataclass."""

    def test_minimal_scenario(self):
        """Test minimal scenario creation."""
        scenario = ShockScenario(
            name="Test Scenario",
            asset_shocks={
                AssetType.MORTGAGE: -0.20,
                AssetType.STOCK: -0.15
            }
        )
        assert scenario.name == "Test Scenario"
        assert scenario.fire_sale_intensity == 0.1  # Default
        assert scenario.correlation == 0.5  # Default
        assert scenario.mode == ShockMode.CORRELATED  # Default
        assert scenario.shock_volatility == 0.03  # Default

    def test_scenario_with_all_parameters(self):
        """Test scenario with all parameters specified."""
        scenario = ShockScenario(
            name="Full Scenario",
            asset_shocks={AssetType.MORTGAGE: -0.20},
            fire_sale_intensity=0.25,
            correlation=0.9,
            mode=ShockMode.DETERMINISTIC,
            shock_volatility=0.05,
            description="Test description"
        )
        assert scenario.fire_sale_intensity == 0.25
        assert scenario.correlation == 0.9
        assert scenario.mode == ShockMode.DETERMINISTIC
        assert scenario.shock_volatility == 0.05
        assert scenario.description == "Test description"

    def test_scenario_mode_string_conversion(self):
        """Test that string mode is converted to enum."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={AssetType.MORTGAGE: -0.20},
            mode="deterministic"  # String
        )
        assert scenario.mode == ShockMode.DETERMINISTIC
        assert isinstance(scenario.mode, ShockMode)

    def test_get_shock(self):
        """Test getting shock for specific asset type."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={
                AssetType.MORTGAGE: -0.20,
                AssetType.STOCK: -0.15
            }
        )
        assert scenario.get_shock(AssetType.MORTGAGE) == -0.20
        assert scenario.get_shock(AssetType.STOCK) == -0.15
        assert scenario.get_shock(AssetType.GOVERNMENT_BOND) == 0.0  # Not specified


# ============================================================================
# Test ShockGenerator - Deterministic Mode
# ============================================================================

class TestDeterministicMode:
    """Test deterministic shock generation."""

    def test_deterministic_returns_exact_shocks(self):
        """Test that deterministic mode returns exact base shocks."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={
                AssetType.MORTGAGE: -0.20,
                AssetType.CORPORATE_BOND: -0.10,
                AssetType.STOCK: -0.15
            },
            mode=ShockMode.DETERMINISTIC
        )

        generator = ShockGenerator(seed=42)

        # Generate shocks multiple times
        shocks1 = generator.generate_correlated_shocks_from_scenario(scenario)
        shocks2 = generator.generate_correlated_shocks_from_scenario(scenario)
        shocks3 = generator.generate_correlated_shocks_from_scenario(scenario)

        # All should be identical to base shocks
        assert shocks1[AssetType.MORTGAGE] == -0.20
        assert shocks1[AssetType.CORPORATE_BOND] == -0.10
        assert shocks1[AssetType.STOCK] == -0.15

        assert shocks1 == shocks2 == shocks3

    def test_deterministic_ignores_volatility(self):
        """Test that deterministic mode ignores volatility parameter."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={AssetType.MORTGAGE: -0.20},
            mode=ShockMode.DETERMINISTIC,
            shock_volatility=0.10  # Large volatility, should be ignored
        )

        generator = ShockGenerator(seed=42)
        shocks = generator.generate_correlated_shocks_from_scenario(scenario)

        # Should still return exact base shock
        assert shocks[AssetType.MORTGAGE] == -0.20

    def test_deterministic_ignores_correlation(self):
        """Test that deterministic mode ignores correlation parameter."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={
                AssetType.MORTGAGE: -0.20,
                AssetType.STOCK: -0.15
            },
            mode=ShockMode.DETERMINISTIC,
            correlation=1.0  # Perfect correlation, should be ignored
        )

        generator = ShockGenerator(seed=42)
        shocks = generator.generate_correlated_shocks_from_scenario(scenario)

        assert shocks[AssetType.MORTGAGE] == -0.20
        assert shocks[AssetType.STOCK] == -0.15


# ============================================================================
# Test ShockGenerator - Correlated Mode
# ============================================================================

class TestCorrelatedMode:
    """Test correlated shock generation."""

    def test_correlated_returns_randomized_shocks(self):
        """Test that correlated mode returns randomized shocks."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={
                AssetType.MORTGAGE: -0.20,
                AssetType.STOCK: -0.15
            },
            mode=ShockMode.CORRELATED,
            correlation=0.8
        )

        generator = ShockGenerator(seed=42)

        # Generate shocks multiple times
        shocks1 = generator.generate_correlated_shocks_from_scenario(scenario)
        shocks2 = generator.generate_correlated_shocks_from_scenario(scenario)

        # Should be different (random)
        assert shocks1[AssetType.MORTGAGE] != shocks2[AssetType.MORTGAGE]
        assert shocks1[AssetType.STOCK] != shocks2[AssetType.STOCK]

        # But should be close to base values
        assert abs(shocks1[AssetType.MORTGAGE] - (-0.20)) < 0.1
        assert abs(shocks1[AssetType.STOCK] - (-0.15)) < 0.1

    def test_correlated_respects_seed(self):
        """Test that same seed produces same results."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={AssetType.MORTGAGE: -0.20},
            mode=ShockMode.CORRELATED,
            correlation=0.8
        )

        generator1 = ShockGenerator(seed=42)
        generator2 = ShockGenerator(seed=42)

        shocks1 = generator1.generate_correlated_shocks_from_scenario(scenario)
        shocks2 = generator2.generate_correlated_shocks_from_scenario(scenario)

        assert shocks1[AssetType.MORTGAGE] == shocks2[AssetType.MORTGAGE]

    def test_correlated_high_correlation(self):
        """Test that high correlation makes shocks move together."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={
                AssetType.MORTGAGE: -0.20,
                AssetType.STOCK: -0.15
            },
            mode=ShockMode.CORRELATED,
            correlation=0.99,  # Very high correlation
            shock_volatility=0.05
        )

        generator = ShockGenerator(seed=42)

        # Generate many samples
        mortgage_deviations = []
        stock_deviations = []

        for i in range(100):
            gen = ShockGenerator(seed=i)
            shocks = gen.generate_correlated_shocks_from_scenario(scenario)
            mortgage_deviations.append(shocks[AssetType.MORTGAGE] - (-0.20))
            stock_deviations.append(shocks[AssetType.STOCK] - (-0.15))

        # Calculate correlation of deviations
        correlation = np.corrcoef(mortgage_deviations, stock_deviations)[0, 1]

        # Should be highly correlated (close to 0.99)
        assert correlation > 0.8, f"Expected high correlation, got {correlation}"

    def test_correlated_low_correlation(self):
        """Test that low correlation makes shocks more independent."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={
                AssetType.MORTGAGE: -0.20,
                AssetType.STOCK: -0.15
            },
            mode=ShockMode.CORRELATED,
            correlation=0.1,  # Low correlation
            shock_volatility=0.05
        )

        # Generate many samples
        mortgage_deviations = []
        stock_deviations = []

        for i in range(100):
            gen = ShockGenerator(seed=i)
            shocks = gen.generate_correlated_shocks_from_scenario(scenario)
            mortgage_deviations.append(shocks[AssetType.MORTGAGE] - (-0.20))
            stock_deviations.append(shocks[AssetType.STOCK] - (-0.15))

        # Calculate correlation of deviations
        correlation = np.corrcoef(mortgage_deviations, stock_deviations)[0, 1]

        # Should be weakly correlated (close to 0.1)
        assert abs(correlation) < 0.5, f"Expected low correlation, got {correlation}"


# ============================================================================
# Test ShockGenerator - Uncorrelated Mode
# ============================================================================

class TestUncorrelatedMode:
    """Test uncorrelated shock generation."""

    def test_uncorrelated_returns_randomized_shocks(self):
        """Test that uncorrelated mode returns randomized shocks."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={
                AssetType.MORTGAGE: -0.20,
                AssetType.STOCK: -0.15
            },
            mode=ShockMode.UNCORRELATED
        )

        generator = ShockGenerator(seed=42)

        # Generate shocks multiple times
        shocks1 = generator.generate_correlated_shocks_from_scenario(scenario)
        shocks2 = generator.generate_correlated_shocks_from_scenario(scenario)

        # Should be different (random)
        assert shocks1[AssetType.MORTGAGE] != shocks2[AssetType.MORTGAGE]
        assert shocks1[AssetType.STOCK] != shocks2[AssetType.STOCK]

    def test_uncorrelated_independence(self):
        """Test that uncorrelated mode produces independent shocks."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={
                AssetType.MORTGAGE: -0.20,
                AssetType.STOCK: -0.15
            },
            mode=ShockMode.UNCORRELATED,
            shock_volatility=0.05
        )

        # Generate many samples
        mortgage_deviations = []
        stock_deviations = []

        for i in range(100):
            gen = ShockGenerator(seed=i)
            shocks = gen.generate_correlated_shocks_from_scenario(scenario)
            mortgage_deviations.append(shocks[AssetType.MORTGAGE] - (-0.20))
            stock_deviations.append(shocks[AssetType.STOCK] - (-0.15))

        # Calculate correlation of deviations
        correlation = np.corrcoef(mortgage_deviations, stock_deviations)[0, 1]

        # Should be nearly uncorrelated (close to 0)
        assert abs(correlation) < 0.3, f"Expected near-zero correlation, got {correlation}"

    def test_uncorrelated_respects_volatility(self):
        """Test that uncorrelated mode respects volatility parameter."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={AssetType.MORTGAGE: -0.20},
            mode=ShockMode.UNCORRELATED,
            shock_volatility=0.01  # Small volatility
        )

        generator = ShockGenerator(seed=42)

        # Generate many samples
        shocks_list = []
        for i in range(50):
            gen = ShockGenerator(seed=i)
            shocks = gen.generate_correlated_shocks_from_scenario(scenario)
            shocks_list.append(shocks[AssetType.MORTGAGE])

        # Standard deviation should be close to volatility
        std_dev = np.std(shocks_list)
        assert 0.005 < std_dev < 0.02, f"Expected std ~0.01, got {std_dev}"


# ============================================================================
# Test ShockGenerator - Volatility Override
# ============================================================================

class TestVolatilityOverride:
    """Test volatility parameter override."""

    def test_volatility_override_in_correlated_mode(self):
        """Test overriding volatility in correlated mode."""
        scenario = ShockScenario(
            name="Test",
            asset_shocks={AssetType.MORTGAGE: -0.20},
            mode=ShockMode.CORRELATED,
            shock_volatility=0.01  # Base volatility
        )

        # Generate with override
        generator = ShockGenerator(seed=42)
        shocks = generator.generate_correlated_shocks_from_scenario(
            scenario,
            volatility=0.10  # Override to higher value
        )

        # With higher volatility, shock could deviate more
        # Just check it's different from base
        assert shocks[AssetType.MORTGAGE] != -0.20


# ============================================================================
# Test Mode Comparison
# ============================================================================

class TestModeComparison:
    """Test comparing different modes."""

    def test_mode_variance_ordering(self):
        """Test that deterministic < uncorrelated < correlated variance (generally)."""
        base_shocks = {AssetType.MORTGAGE: -0.20, AssetType.STOCK: -0.15}

        # Deterministic
        scenario_det = ShockScenario(
            name="Det",
            asset_shocks=base_shocks,
            mode=ShockMode.DETERMINISTIC
        )

        # Uncorrelated
        scenario_uncorr = ShockScenario(
            name="Uncorr",
            asset_shocks=base_shocks,
            mode=ShockMode.UNCORRELATED,
            shock_volatility=0.05
        )

        # Correlated
        scenario_corr = ShockScenario(
            name="Corr",
            asset_shocks=base_shocks,
            mode=ShockMode.CORRELATED,
            correlation=0.9,
            shock_volatility=0.05
        )

        # Generate samples for each mode
        det_samples = []
        uncorr_samples = []
        corr_samples = []

        for i in range(50):
            gen = ShockGenerator(seed=i)
            det_shocks = gen.generate_correlated_shocks_from_scenario(scenario_det)
            det_samples.append(det_shocks[AssetType.MORTGAGE])

            gen = ShockGenerator(seed=i)
            uncorr_shocks = gen.generate_correlated_shocks_from_scenario(scenario_uncorr)
            uncorr_samples.append(uncorr_shocks[AssetType.MORTGAGE])

            gen = ShockGenerator(seed=i)
            corr_shocks = gen.generate_correlated_shocks_from_scenario(scenario_corr)
            corr_samples.append(corr_shocks[AssetType.MORTGAGE])

        # Calculate variances
        det_var = np.var(det_samples)
        uncorr_var = np.var(uncorr_samples)
        corr_var = np.var(corr_samples)

        # Deterministic should have zero variance (within floating point tolerance)
        assert abs(det_var) < 1e-10, f"Deterministic variance should be ~0, got {det_var}"

        # Uncorrelated and correlated should have non-zero variance
        assert uncorr_var > 1e-6, f"Uncorrelated variance should be > 0, got {uncorr_var}"
        assert corr_var > 1e-6, f"Correlated variance should be > 0, got {corr_var}"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises error."""
        # Create scenario with manual mode bypass
        scenario = ShockScenario(
            name="Test",
            asset_shocks={AssetType.MORTGAGE: -0.20}
        )
        # Manually set invalid mode (bypass enum validation)
        scenario.mode = "invalid"

        generator = ShockGenerator(seed=42)

        with pytest.raises(ValueError, match="Unknown shock mode"):
            generator.generate_correlated_shocks_from_scenario(scenario)

    def test_empty_asset_shocks(self):
        """Test scenario with empty asset shocks."""
        scenario = ShockScenario(
            name="Empty",
            asset_shocks={},
            mode=ShockMode.DETERMINISTIC
        )

        generator = ShockGenerator(seed=42)
        shocks = generator.generate_correlated_shocks_from_scenario(scenario)

        # Should still work, just return empty or defaults
        assert isinstance(shocks, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
