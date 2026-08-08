"""
Tests for comprehensive data recording in simulation runner.

Ensures runner.py captures all critical data including bank_type (core/periphery)
and initial bank configurations for validation.
"""

import pytest
from financial_contagion_networks.core.network import ContagionNetwork
from financial_contagion_networks.core.shocks import ShockScenario, ShockMode
from financial_contagion_networks.core.assets import AssetType
from financial_contagion_networks.simulation.runner import simulate_with_portfolio_shocks
from financial_contagion_networks.simulation.generators import create_portfolio_bank


class TestRunnerDataRecording:
    """Test that runner captures all essential data."""

    def create_test_network(self, num_core=2, num_periphery=3):
        """Create a small test network with core and periphery banks."""
        network = ContagionNetwork()

        # Add core banks (ids 0, 1) with 15% interbank
        for i in range(num_core):
            bank = create_portfolio_bank(
                bank_id=i,
                total_assets=1000.0,
                capital_ratio=0.10,
                portfolio_weights={
                    AssetType.MORTGAGE: 0.50,
                    AssetType.GOVERNMENT_BOND: 0.30,
                    AssetType.CORPORATE_BOND: 0.20
                },
                interbank_fraction=0.15
            )
            network.add_bank(bank)

        # Add periphery banks with 12% interbank
        for i in range(num_core, num_core + num_periphery):
            bank = create_portfolio_bank(
                bank_id=i,
                total_assets=1000.0,
                capital_ratio=0.10,
                portfolio_weights={
                    AssetType.MORTGAGE: 0.60,
                    AssetType.GOVERNMENT_BOND: 0.25,
                    AssetType.CORPORATE_BOND: 0.15
                },
                interbank_fraction=0.12
            )
            network.add_bank(bank)

        # Add some interbank connections
        if num_core + num_periphery >= 2:
            network.add_exposure(0, 1, 50.0)
            if num_core + num_periphery >= 3:
                network.add_exposure(1, 2, 30.0)

        return network

    def test_initial_state_contains_bank_type(self):
        """Test that initial_state includes bank_type for each bank."""
        network = self.create_test_network(num_core=2, num_periphery=3)

        scenario = ShockScenario(
            name="test",
            mode=ShockMode.DETERMINISTIC,
            asset_shocks={AssetType.MORTGAGE: -0.05},
            correlation=0.0,
            fire_sale_intensity=0.0,
            shock_volatility=0.0
        )

        result = simulate_with_portfolio_shocks(
            network=network,
            scenario=scenario,
            fire_sales_enabled=False,
            num_core_banks=2  # Specify 2 core banks
        )

        assert 'initial_state' in result

        # Check core banks have bank_type='core'
        for bank_id in [0, 1]:
            assert str(bank_id) in result['initial_state'] or bank_id in result['initial_state']
            bank_data = result['initial_state'][str(bank_id)] if str(bank_id) in result['initial_state'] else result['initial_state'][bank_id]
            assert 'bank_type' in bank_data, f"Bank {bank_id} missing bank_type"
            assert bank_data['bank_type'] == 'core', f"Bank {bank_id} should be core"

        # Check periphery banks have bank_type='periphery'
        for bank_id in [2, 3, 4]:
            assert str(bank_id) in result['initial_state'] or bank_id in result['initial_state']
            bank_data = result['initial_state'][str(bank_id)] if str(bank_id) in result['initial_state'] else result['initial_state'][bank_id]
            assert 'bank_type' in bank_data, f"Bank {bank_id} missing bank_type"
            assert bank_data['bank_type'] == 'periphery', f"Bank {bank_id} should be periphery"

    def test_initial_state_contains_interbank_fraction(self):
        """Test that initial_state includes interbank_assets_fraction."""
        network = self.create_test_network(num_core=2, num_periphery=3)

        scenario = ShockScenario(
            name="test",
            mode=ShockMode.DETERMINISTIC,
            asset_shocks={AssetType.MORTGAGE: -0.05},
            correlation=0.0,
            fire_sale_intensity=0.0,
            shock_volatility=0.0
        )

        result = simulate_with_portfolio_shocks(
            network=network,
            scenario=scenario,
            fire_sales_enabled=False,
            num_core_banks=2
        )

        # Check all banks have interbank_assets_fraction
        for bank_id in range(5):
            bank_data = result['initial_state'][str(bank_id)] if str(bank_id) in result['initial_state'] else result['initial_state'][bank_id]
            assert 'interbank_assets_fraction' in bank_data

            # Core banks: 150 / 1000 = 0.15
            if bank_id < 2:
                assert bank_data['interbank_assets_fraction'] == pytest.approx(0.15, abs=1e-6)
            # Periphery banks: 120 / 1000 = 0.12
            else:
                assert bank_data['interbank_assets_fraction'] == pytest.approx(0.12, abs=1e-6)

    def test_initial_state_without_num_core_banks(self):
        """Test that initial_state works without num_core_banks (no bank_type field)."""
        network = self.create_test_network(num_core=2, num_periphery=3)

        scenario = ShockScenario(
            name="test",
            mode=ShockMode.DETERMINISTIC,
            asset_shocks={AssetType.MORTGAGE: -0.05},
            correlation=0.0,
            fire_sale_intensity=0.0,
            shock_volatility=0.0
        )

        result = simulate_with_portfolio_shocks(
            network=network,
            scenario=scenario,
            fire_sales_enabled=False
            # num_core_banks not provided
        )

        assert 'initial_state' in result

        # Bank type should not be present
        for bank_id in range(5):
            bank_data = result['initial_state'][str(bank_id)] if str(bank_id) in result['initial_state'] else result['initial_state'][bank_id]
            # bank_type should not be present
            assert 'bank_type' not in bank_data
            # But interbank_assets_fraction should still be present
            assert 'interbank_assets_fraction' in bank_data

    def test_all_critical_fields_in_initial_state(self):
        """Test that initial_state contains all critical fields for analysis."""
        network = self.create_test_network(num_core=2, num_periphery=1)

        scenario = ShockScenario(
            name="test",
            mode=ShockMode.DETERMINISTIC,
            asset_shocks={AssetType.MORTGAGE: -0.05},
            correlation=0.0,
            fire_sale_intensity=0.0,
            shock_volatility=0.0
        )

        result = simulate_with_portfolio_shocks(
            network=network,
            scenario=scenario,
            fire_sales_enabled=False,
            num_core_banks=2
        )

        # Required fields in each bank's initial state
        required_fields = [
            'bank_id',
            'external_assets',
            'interbank_assets',
            'interbank_liabilities',
            'external_liabilities',
            'total_assets',
            'equity',
            'capital_ratio',
            'interbank_assets_fraction',  # NEW
            'bank_type',  # NEW (when num_core_banks provided)
            'portfolio',  # For banks with portfolios
        ]

        for bank_id in range(3):
            bank_data = result['initial_state'][str(bank_id)] if str(bank_id) in result['initial_state'] else result['initial_state'][bank_id]

            for field in required_fields:
                assert field in bank_data, f"Bank {bank_id} missing field: {field}"

    def test_interbank_fraction_heterogeneity_recorded(self):
        """Test that heterogeneous interbank fractions are correctly recorded."""
        network = ContagionNetwork()

        # Create banks with varying interbank fractions
        interbank_fractions = [0.08, 0.15, 0.22, 0.30]  # Different fractions

        for i, ib_fraction in enumerate(interbank_fractions):
            bank = create_portfolio_bank(
                bank_id=i,
                total_assets=1000.0,
                capital_ratio=0.10,
                portfolio_weights={
                    AssetType.MORTGAGE: 0.50,
                    AssetType.GOVERNMENT_BOND: 0.30,
                    AssetType.CORPORATE_BOND: 0.20
                },
                interbank_fraction=ib_fraction
            )
            network.add_bank(bank)

        scenario = ShockScenario(
            name="test",
            mode=ShockMode.DETERMINISTIC,
            asset_shocks={AssetType.MORTGAGE: -0.01},
            correlation=0.0,
            fire_sale_intensity=0.0,
            shock_volatility=0.0
        )

        result = simulate_with_portfolio_shocks(
            network=network,
            scenario=scenario,
            fire_sales_enabled=False,
            num_core_banks=2
        )

        # Verify each bank has the correct interbank fraction recorded
        for i, expected_fraction in enumerate(interbank_fractions):
            bank_data = result['initial_state'][str(i)] if str(i) in result['initial_state'] else result['initial_state'][i]
            actual_fraction = bank_data['interbank_assets_fraction']
            assert actual_fraction == pytest.approx(expected_fraction, abs=1e-6), \
                f"Bank {i}: expected {expected_fraction}, got {actual_fraction}"
