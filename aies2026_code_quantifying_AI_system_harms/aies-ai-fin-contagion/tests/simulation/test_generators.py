"""
Unit tests for network generators module.

Tests the NetworkGenerator class and all three network modes:
- fixed: Identical network every time
- template: Fixed structure, varying parameters
- stochastic: New network structure and parameters each time
"""

import pytest
import numpy as np

from financial_contagion_networks.simulation.generators import (
    NetworkGenerator,
    NetworkTemplate,
    NetworkTopology,
    generate_network
)
from financial_contagion_networks.config import (
    NetworkConfig,
    NetworkMode,
    BankGroupConfig,
    PortfolioConfig,
    ConnectivityConfig,
    DistributionConfig
)
from financial_contagion_networks.core.assets import AssetType


# ============================================================================
# Test NetworkTopology
# ============================================================================

class TestNetworkTopology:
    """Test NetworkTopology class."""

    def test_valid_topology(self):
        """Test creating valid topology."""
        topology = NetworkTopology(
            num_banks=20,
            num_core_banks=5,
            connections=[(0, 1), (1, 2), (2, 3)]
        )
        assert topology.num_banks == 20
        assert topology.num_core_banks == 5
        assert len(topology.connections) == 3

    def test_invalid_topology_core_exceeds_total(self):
        """Test topology with more core banks than total."""
        with pytest.raises(ValueError, match="num_core_banks .* > num_banks"):
            NetworkTopology(
                num_banks=20,
                num_core_banks=25,
                connections=[]
            )


# ============================================================================
# Test NetworkGenerator - Portfolio Weights
# ============================================================================

class TestPortfolioWeights:
    """Test portfolio weight calculation."""

    def create_simple_portfolio(self) -> PortfolioConfig:
        """Create simple portfolio with fixed weights."""
        return PortfolioConfig(
            mortgage=DistributionConfig(distribution='fixed', value=0.2),
            remaining={
                'government_bond': 0.5,  # 0.4 / 0.8 = 0.5
                'corporate_bond': 0.25,  # 0.2 / 0.8 = 0.25
                'stock': 0.25  # 0.2 / 0.8 = 0.25
            }
        )

    def test_fixed_portfolio_weights(self):
        """Test portfolio with all fixed weights."""
        portfolio = self.create_simple_portfolio()
        rng = np.random.RandomState(42)

        weights = NetworkGenerator._calculate_portfolio_weights(portfolio, rng)

        assert len(weights) == 4
        assert weights[AssetType.GOVERNMENT_BOND] == 0.4
        assert weights[AssetType.CORPORATE_BOND] == 0.2
        assert weights[AssetType.MORTGAGE] == 0.2
        assert weights[AssetType.STOCK] == 0.2
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_portfolio_with_distribution(self):
        """Test portfolio with distribution + remaining (required pattern)."""
        portfolio = PortfolioConfig(
            mortgage=DistributionConfig(distribution='uniform', min=0.15, max=0.25),
            remaining={
                'government_bond': 0.50,
                'corporate_bond': 0.25,
                'stock': 0.25
            }
        )
        rng = np.random.RandomState(42)

        weights = NetworkGenerator._calculate_portfolio_weights(portfolio, rng)

        # Mortgage should be in sampled range
        mortgage_weight = weights[AssetType.MORTGAGE]
        assert 0.15 <= mortgage_weight <= 0.25

        # Remaining should be distributed proportionally
        remaining = 1.0 - mortgage_weight
        assert abs(weights[AssetType.GOVERNMENT_BOND] - remaining * 0.50) < 1e-6
        assert abs(weights[AssetType.CORPORATE_BOND] - remaining * 0.25) < 1e-6
        assert abs(weights[AssetType.STOCK] - remaining * 0.25) < 1e-6

        # Total should be exactly 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_portfolio_distribution_without_remaining_fails(self):
        """Test that distributions without 'remaining' are rejected."""
        # This should fail at construction time: PortfolioConfig requires 'remaining' field
        with pytest.raises(Exception):  # Will raise ValidationError from Pydantic
            portfolio = PortfolioConfig(
                mortgage=DistributionConfig(distribution='uniform', min=0.15, max=0.25)
                # Missing required 'remaining' field
            )

    def test_portfolio_with_remaining_specification(self):
        """Test portfolio with 'remaining' specification."""
        portfolio = PortfolioConfig(
            mortgage=DistributionConfig(distribution='uniform', min=0.15, max=0.25),
            remaining={
                'government_bond': 0.50,
                'corporate_bond': 0.25,
                'stock': 0.25
            }
        )
        rng = np.random.RandomState(42)

        weights = NetworkGenerator._calculate_portfolio_weights(portfolio, rng)

        # Check mortgage is in range
        mortgage_weight = weights[AssetType.MORTGAGE]
        assert 0.15 <= mortgage_weight <= 0.25

        # Check remaining are allocated proportionally
        remaining = 1.0 - mortgage_weight
        assert abs(weights[AssetType.GOVERNMENT_BOND] - remaining * 0.50) < 1e-6
        assert abs(weights[AssetType.CORPORATE_BOND] - remaining * 0.25) < 1e-6
        assert abs(weights[AssetType.STOCK] - remaining * 0.25) < 1e-6

        # Total should be 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_portfolio_weights_invalid_sum(self):
        """Test portfolio with weights that don't sum to 1.0."""
        # The 'remaining' dict must sum to 1.0, so this should fail at construction
        with pytest.raises(Exception):  # Will raise ValidationError from Pydantic
            portfolio = PortfolioConfig(
                mortgage=DistributionConfig(distribution='fixed', value=0.2),
                remaining={
                    'government_bond': 0.4,
                    'corporate_bond': 0.2,
                    'stock': 0.5  # Sum = 1.1, should fail
                }
            )

    def test_portfolio_duplicate_specification(self):
        """Test portfolio with asset specified both explicitly and in remaining."""
        # With new structure, mortgage is always explicit, and remaining can't contain it
        # The validator in network_generators should catch this
        portfolio = PortfolioConfig(
            mortgage=DistributionConfig(distribution='fixed', value=0.2),
            remaining={
                'government_bond': 0.25,
                'corporate_bond': 0.25,
                'mortgage': 0.25,  # Duplicate! Should fail in _calculate_portfolio_weights
                'stock': 0.25
            }
        )
        rng = np.random.RandomState(42)

        with pytest.raises(ValueError, match="specified both in mortgage and remaining"):
            NetworkGenerator._calculate_portfolio_weights(portfolio, rng)


# ============================================================================
# Test NetworkGenerator - Config Creation Helpers
# ============================================================================

class TestNetworkGeneratorHelpers:
    """Helper methods for creating test configurations."""

    @staticmethod
    def create_minimal_network_config(mode: str = 'stochastic') -> NetworkConfig:
        """Create minimal valid network configuration with all required fields."""
        from tests.conftest import create_minimal_test_config
        # Get full config from conftest helper, then extract network with mode override
        # Override to use 20 banks (5 core, 15 periphery) as expected by these tests
        full_config = create_minimal_test_config(
            network={
                'mode': mode,
                'num_banks': 20,
                'num_core_banks': 5
            }
        )
        return full_config.network

    @staticmethod
    def create_stochastic_config_with_distributions() -> NetworkConfig:
        """Create config with distributions for testing stochastic mode."""
        from tests.conftest import create_minimal_test_config
        # Use conftest helper with actual distributions (not fixed values)
        # Override to use 20 banks (5 core, 15 periphery) as expected by these tests
        full_config = create_minimal_test_config(
            network={
                'mode': 'stochastic',
                'num_banks': 20,
                'num_core_banks': 5,
                'core_banks': {
                    'capital_ratio': {'distribution': 'uniform', 'min': 0.12, 'max': 0.18},
                    'portfolio': {
                        'mortgage': {'distribution': 'uniform', 'min': 0.15, 'max': 0.25},
                        'remaining': {
                            'government_bond': 0.35,
                            'corporate_bond': 0.40,
                            'stock': 0.25
                        }
                    },
                    'total_assets': 500.0,
                    'external_assets_fraction': 0.9
                },
                'periphery_banks': {
                    'capital_ratio': {'distribution': 'uniform', 'min': 0.08, 'max': 0.14},
                    'portfolio': {
                        'mortgage': {'distribution': 'uniform', 'min': 0.20, 'max': 0.30},
                        'remaining': {
                            'government_bond': 0.30,
                            'corporate_bond': 0.35,
                            'stock': 0.35
                        }
                    },
                    'total_assets': 100.0,
                    'external_assets_fraction': 0.92
                }
            }
        )
        return full_config.network


# ============================================================================
# Test NetworkGenerator - Fixed Mode
# ============================================================================

class TestFixedMode:
    """Test fixed mode network generation."""

    def test_fixed_mode_generates_identical_networks(self):
        """Test that fixed mode always generates identical network."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config('fixed')

        # Generate network multiple times
        network1 = NetworkGenerator.from_config(config, run_id=0)
        network2 = NetworkGenerator.from_config(config, run_id=1)
        network3 = NetworkGenerator.from_config(config, run_id=999)

        # Check all have same structure
        assert len(network1.banks) == len(network2.banks) == len(network3.banks) == 20

        # Check core banks have same capital (fixed mode should be identical)
        bank1_0 = network1.get_bank(0)
        bank2_0 = network2.get_bank(0)
        bank3_0 = network3.get_bank(0)

        assert bank1_0.get_equity() == bank2_0.get_equity() == bank3_0.get_equity()
        assert bank1_0.get_total_assets() == bank2_0.get_total_assets() == bank3_0.get_total_assets()

    def test_fixed_mode_deterministic_with_seed(self):
        """Test that same seed produces identical network."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config('fixed')

        network1 = NetworkGenerator.from_config(config)
        network2 = NetworkGenerator.from_config(config)

        # Should be identical
        assert len(network1.banks) == len(network2.banks)

        for bank_id in range(len(network1.banks)):
            bank1 = network1.get_bank(bank_id)
            bank2 = network2.get_bank(bank_id)
            assert abs(bank1.get_equity() - bank2.get_equity()) < 1e-10
            assert abs(bank1.get_total_assets() - bank2.get_total_assets()) < 1e-10


# ============================================================================
# Test NetworkGenerator - Template Mode
# ============================================================================

class TestTemplateMode:
    """Test template mode network generation."""

    def test_create_template(self):
        """Test creating a network template."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config('template')

        template = NetworkGenerator.create_template(config)

        assert template is not None
        assert template.topology.num_banks == 20
        assert template.topology.num_core_banks == 5
        assert isinstance(template.topology.connections, list)

    def test_template_mode_fixed_structure(self):
        """Test that template mode has fixed structure."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config('template')

        template = NetworkGenerator.create_template(config)

        # Generate multiple networks from template
        network1 = template.generate(parameter_seed=42)
        network2 = template.generate(parameter_seed=43)

        # Structure should be identical
        assert len(network1.banks) == len(network2.banks) == 20

    def test_template_mode_varying_parameters(self):
        """Test that template mode varies parameters."""
        config = TestNetworkGeneratorHelpers.create_stochastic_config_with_distributions()
        config.mode = 'template'  # Use string

        template = NetworkGenerator.create_template(config)

        # Generate networks with different parameter seeds
        network1 = template.generate(parameter_seed=42)
        network2 = template.generate(parameter_seed=43)

        # Structure same
        assert len(network1.banks) == len(network2.banks)

        # Parameters should differ (due to distributions)
        bank1_0 = network1.get_bank(0)
        bank2_0 = network2.get_bank(0)

        # With distributions, capital should vary
        assert bank1_0.get_equity() != bank2_0.get_equity() or bank1_0.get_total_assets() != bank2_0.get_total_assets()

    def test_template_mode_from_config(self):
        """Test generating template mode network via from_config."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config('template')

        # Should create template internally
        network1 = NetworkGenerator.from_config(config, run_id=0)
        network2 = NetworkGenerator.from_config(config, run_id=1)

        assert len(network1.banks) == 20
        assert len(network2.banks) == 20

    def test_create_template_wrong_mode_fails(self):
        """Test that create_template fails for non-template mode."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config('stochastic')

        with pytest.raises(ValueError, match="Can only create template for template mode"):
            NetworkGenerator.create_template(config)


# ============================================================================
# Test NetworkGenerator - Stochastic Mode
# ============================================================================

class TestStochasticMode:
    """Test stochastic mode network generation."""

    def test_stochastic_mode_basic(self):
        """Test basic stochastic mode generation."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config('stochastic')

        network = NetworkGenerator.from_config(config, run_id=0)

        assert len(network.banks) == 20
        assert network.get_bank(0) is not None
        assert network.get_bank(19) is not None

    def test_stochastic_mode_different_seeds_differ(self):
        """Test that different run_ids produce different networks."""
        config = TestNetworkGeneratorHelpers.create_stochastic_config_with_distributions()

        network1 = NetworkGenerator.from_config(config, run_id=0)
        network2 = NetworkGenerator.from_config(config, run_id=1)

        # Both should have 20 banks
        assert len(network1.banks) == 20
        assert len(network2.banks) == 20

        # Due to stochastic mode with distributions, parameters should differ
        bank1_0 = network1.get_bank(0)
        bank2_0 = network2.get_bank(0)

        # At least one bank should differ
        assert bank1_0.get_equity() != bank2_0.get_equity() or bank1_0.get_total_assets() != bank2_0.get_total_assets()

    def test_stochastic_mode_same_run_id_same_network(self):
        """Test that same run_id produces same network."""
        config = TestNetworkGeneratorHelpers.create_stochastic_config_with_distributions()

        network1 = NetworkGenerator.from_config(config, run_id=42)
        network2 = NetworkGenerator.from_config(config, run_id=42)

        # Should be identical (same seeds)
        assert len(network1.banks) == len(network2.banks)

        for bank_id in range(len(network1.banks)):
            bank1 = network1.get_bank(bank_id)
            bank2 = network2.get_bank(bank_id)
            assert abs(bank1.get_equity() - bank2.get_equity()) < 1e-10
            assert abs(bank1.get_total_assets() - bank2.get_total_assets()) < 1e-10


# ============================================================================
# Test NetworkGenerator - Connectivity
# ============================================================================

class TestConnectivity:
    """Test network connectivity generation."""

    def test_custom_connectivity_probabilities(self):
        """Test network with custom connectivity probabilities."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config('fixed')

        # Set high connectivity with required exposure distributions
        config.connectivity = ConnectivityConfig(
            core_to_core=1.0,  # Always connect
            core_to_periphery=1.0,
            periphery_to_core=1.0,
            periphery_to_periphery=1.0,
            core_to_core_exposure=DistributionConfig(distribution='fixed', value=5.0),
            core_to_periphery_exposure=DistributionConfig(distribution='fixed', value=2.0),
            periphery_to_core_exposure=DistributionConfig(distribution='fixed', value=1.0),
            periphery_to_periphery_exposure=DistributionConfig(distribution='fixed', value=1.0)
        )

        network = NetworkGenerator.from_config(config)

        # With probability 1.0, we should have many connections
        assert len(network.banks) == 20

    def test_zero_connectivity(self):
        """Test network with no interbank connections."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config('fixed')

        # Zero connectivity (exposure amounts don't matter when probability is 0)
        config.connectivity = ConnectivityConfig(
            core_to_core=0.0,
            core_to_periphery=0.0,
            periphery_to_core=0.0,
            periphery_to_periphery=0.0,
            core_to_core_exposure=DistributionConfig(distribution='fixed', value=5.0),
            core_to_periphery_exposure=DistributionConfig(distribution='fixed', value=2.0),
            periphery_to_core_exposure=DistributionConfig(distribution='fixed', value=1.0),
            periphery_to_periphery_exposure=DistributionConfig(distribution='fixed', value=1.0)
        )

        network = NetworkGenerator.from_config(config)

        # Network should still have 20 banks
        assert len(network.banks) == 20


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_generate_network_function(self):
        """Test generate_network convenience function."""
        config = TestNetworkGeneratorHelpers.create_minimal_network_config()

        network = generate_network(config, run_id=0)

        assert network is not None
        assert len(network.banks) == 20


# ============================================================================
# Integration Tests
# ============================================================================

class TestNetworkGeneratorIntegration:
    """Integration tests for network generation."""

    def test_all_modes_produce_valid_networks(self):
        """Test that all three modes produce valid networks."""
        for mode in ['fixed', 'template', 'stochastic']:
            config = TestNetworkGeneratorHelpers.create_minimal_network_config(mode)

            network = NetworkGenerator.from_config(config, run_id=0)

            assert len(network.banks) == 20
            assert network.get_bank(0) is not None
            assert network.get_bank(19) is not None

            # Check core banks exist
            for i in range(5):
                bank = network.get_bank(i)
                assert bank is not None
                assert bank.get_total_assets() > 0
                assert bank.get_equity() > 0

            # Check periphery banks exist
            for i in range(5, 20):
                bank = network.get_bank(i)
                assert bank is not None
                assert bank.get_total_assets() > 0
                assert bank.get_equity() > 0

    def test_realistic_post_2008_network(self):
        """Test generating a realistic post-2008 network."""
        config = TestNetworkGeneratorHelpers.create_stochastic_config_with_distributions()

        # Generate network
        network = NetworkGenerator.from_config(config, run_id=0)

        # Verify structure
        assert len(network.banks) == 20

        # Collect capital ratios
        core_capital_ratios = []
        for i in range(5):
            bank = network.get_bank(i)
            capital_ratio = bank.get_capital_ratio()
            core_capital_ratios.append(capital_ratio)
            # Just verify it's positive and reasonable
            assert 0 < capital_ratio < 1, f"Core bank {i} has invalid capital ratio {capital_ratio}"

        periphery_capital_ratios = []
        for i in range(5, 20):
            bank = network.get_bank(i)
            capital_ratio = bank.get_capital_ratio()
            periphery_capital_ratios.append(capital_ratio)
            # Just verify it's positive and reasonable
            assert 0 < capital_ratio < 1, f"Periphery bank {i} has invalid capital ratio {capital_ratio}"

        # Core banks should have higher average capital
        assert np.mean(core_capital_ratios) > np.mean(periphery_capital_ratios), \
            f"Core avg capital {np.mean(core_capital_ratios):.3f} should be > periphery {np.mean(periphery_capital_ratios):.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
