"""
Integration tests for Option A: Variable Interbank Exposure.

Tests the full pipeline from config → network generation → simulation
to ensure variable interbank exposure works correctly end-to-end.
"""

import pytest
import numpy as np
from financial_contagion_networks.config.models import (
    NetworkConfig, BankGroupConfig, DistributionConfig, PortfolioConfig,
    ShockConfig, SimulationConfig, ExperimentConfig, MetadataConfig,
    ConnectivityConfig, OutputConfig
)
from financial_contagion_networks.simulation.generators import NetworkGenerator
from financial_contagion_networks.simulation.experiment import ExperimentRunner
from financial_contagion_networks.core.assets import AssetType


def create_test_connectivity() -> ConnectivityConfig:
    """Create standard connectivity config for tests."""
    return ConnectivityConfig(
        core_to_core=0.8,
        core_to_periphery=0.7,
        periphery_to_core=0.6,
        periphery_to_periphery=0.6,
        core_to_core_exposure=DistributionConfig(distribution='uniform', min=4.0, max=8.0),
        core_to_periphery_exposure=DistributionConfig(distribution='uniform', min=1.5, max=4.0),
        periphery_to_core_exposure=DistributionConfig(distribution='uniform', min=1.0, max=3.0),
        periphery_to_periphery_exposure=DistributionConfig(distribution='uniform', min=0.5, max=2.0)
    )


def create_test_output() -> OutputConfig:
    """Create standard output config for tests."""
    return OutputConfig(
        output_dir='tests/integration/output',
        save_summary=True,
        save_detailed_results=False,  # Don't save large files during tests
        save_network_snapshots=False,
        save_config_copy=False,
        generate_plots=False,
        plot_formats=['png'],
        verbose=False
    )


class TestOptionANetworkGeneration:
    """Test network generation with variable interbank exposure."""

    def create_test_config_with_variable_ib(self):
        """Create a test config with variable interbank exposure."""
        return NetworkConfig(
            mode='stochastic',
            topology='pre_2008',
            structure_seed=42,
            parameter_seed=100,
            num_banks=10,
            num_core_banks=3,
            connectivity=create_test_connectivity(),
            core_banks=BankGroupConfig(
                capital_ratio=DistributionConfig(distribution='fixed', value=0.10),
                total_assets=1000.0,
                portfolio=PortfolioConfig(
                    mortgage=DistributionConfig(distribution='fixed', value=0.50),
                    remaining={'government_bond': 0.3, 'corporate_bond': 0.4, 'stock': 0.3}
                ),
                external_assets_fraction=0.85,
                # Variable interbank: 15% ± 5%
                interbank_assets_fraction=DistributionConfig(
                    distribution='normal',
                    mean=0.15,
                    std=0.05,
                    min=0.08,
                    max=0.30
                )
            ),
            periphery_banks=BankGroupConfig(
                capital_ratio=DistributionConfig(distribution='fixed', value=0.10),
                total_assets=500.0,
                portfolio=PortfolioConfig(
                    mortgage=DistributionConfig(distribution='fixed', value=0.60),
                    remaining={'government_bond': 0.25, 'corporate_bond': 0.35, 'stock': 0.4}
                ),
                external_assets_fraction=0.88,
                # Variable interbank: 12% ± 4%
                interbank_assets_fraction=DistributionConfig(
                    distribution='normal',
                    mean=0.12,
                    std=0.04,
                    min=0.05,
                    max=0.22
                )
            )
        )

    def test_variable_interbank_creates_heterogeneous_banks(self):
        """Test that variable IB config creates banks with different IB fractions."""
        config = self.create_test_config_with_variable_ib()
        network = NetworkGenerator.from_config(config, run_id=0)

        # Extract interbank fractions
        ib_fractions = []
        for bank in network.banks.values():
            total_assets = bank.get_total_assets()
            ib_fraction = bank.interbank_assets / total_assets
            ib_fractions.append(ib_fraction)

        # Should have variety (not all the same)
        unique_fractions = len(set(ib_fractions))
        assert unique_fractions >= 8, f"Only {unique_fractions} unique IB fractions out of 10 banks"

        # Should be in expected ranges
        assert all(0.05 <= f <= 0.30 for f in ib_fractions), \
            f"Some fractions outside bounds: {ib_fractions}"

    def test_core_banks_have_higher_average_ib_than_periphery(self):
        """Test that core banks have higher average IB fraction than periphery."""
        config = self.create_test_config_with_variable_ib()
        network = NetworkGenerator.from_config(config, run_id=0)

        core_ib_fractions = []
        periphery_ib_fractions = []

        for bank_id, bank in network.banks.items():
            total_assets = bank.get_total_assets()
            ib_fraction = bank.interbank_assets / total_assets

            if bank_id < 3:  # Core banks
                core_ib_fractions.append(ib_fraction)
            else:  # Periphery banks
                periphery_ib_fractions.append(ib_fraction)

        core_mean = np.mean(core_ib_fractions)
        periphery_mean = np.mean(periphery_ib_fractions)

        # Core should have higher average (15% vs 12%)
        assert core_mean > periphery_mean, \
            f"Core mean {core_mean:.2%} should be > periphery mean {periphery_mean:.2%}"

    def test_market_clearing_with_variable_ib(self):
        """Test that total IB assets ≈ total IB liabilities (market clearing)."""
        config = self.create_test_config_with_variable_ib()
        network = NetworkGenerator.from_config(config, run_id=0)

        total_ib_assets = sum(bank.interbank_assets for bank in network.banks.values())
        total_ib_liab = sum(bank.interbank_liabilities for bank in network.banks.values())

        # Should be approximately equal (within 1%)
        if total_ib_assets > 0:
            deviation = abs(total_ib_assets - total_ib_liab) / total_ib_assets
            assert deviation < 0.01, \
                f"Market not clearing: IB assets={total_ib_assets:.1f}, IB liab={total_ib_liab:.1f}, deviation={deviation:.2%}"

    def test_different_runs_produce_different_ib_fractions(self):
        """Test that different runs produce different IB fractions (stochastic)."""
        config = self.create_test_config_with_variable_ib()

        network1 = NetworkGenerator.from_config(config, run_id=0)
        network2 = NetworkGenerator.from_config(config, run_id=1)

        # Extract IB fractions from both networks
        fractions1 = []
        fractions2 = []

        for bank_id in range(10):
            bank1 = network1.banks[bank_id]
            bank2 = network2.banks[bank_id]

            frac1 = bank1.interbank_assets / bank1.get_total_assets()
            frac2 = bank2.interbank_assets / bank2.get_total_assets()

            fractions1.append(frac1)
            fractions2.append(frac2)

        # Should differ between runs
        assert fractions1 != fractions2, "Different runs should produce different IB fractions"

    def test_realistic_pre_2008_ranges(self):
        """Test that pre-2008 realistic ranges produce expected distributions."""
        config = self.create_test_config_with_variable_ib()

        # Generate multiple networks to check distribution
        all_core_fractions = []
        all_periphery_fractions = []

        for run_id in range(20):
            network = NetworkGenerator.from_config(config, run_id=run_id)

            for bank_id, bank in network.banks.items():
                ib_fraction = bank.interbank_assets / bank.get_total_assets()

                if bank_id < 3:  # Core
                    all_core_fractions.append(ib_fraction)
                else:  # Periphery
                    all_periphery_fractions.append(ib_fraction)

        # Check core distribution (15% ± 5%)
        core_mean = np.mean(all_core_fractions)
        assert 0.12 <= core_mean <= 0.18, f"Core mean {core_mean:.2%} not near 15%"

        # Check periphery distribution (12% ± 4%)
        periphery_mean = np.mean(all_periphery_fractions)
        assert 0.10 <= periphery_mean <= 0.14, f"Periphery mean {periphery_mean:.2%} not near 12%"


class TestOptionABackwardCompatibility:
    """Test backward compatibility when interbank_assets_fraction is None."""

    def create_config_without_variable_ib(self):
        """Create config without variable IB (backward compatibility)."""
        return NetworkConfig(
            mode='stochastic',
            topology='pre_2008',
            structure_seed=42,
            parameter_seed=100,
            num_banks=10,
            num_core_banks=3,
            connectivity=create_test_connectivity(),
            core_banks=BankGroupConfig(
                capital_ratio=DistributionConfig(distribution='fixed', value=0.10),
                total_assets=1000.0,
                portfolio=PortfolioConfig(
                    mortgage=DistributionConfig(distribution='fixed', value=0.50),
                    remaining={'government_bond': 0.3, 'corporate_bond': 0.4, 'stock': 0.3}
                ),
                external_assets_fraction=0.85
                # interbank_assets_fraction NOT specified - should use original behavior
            ),
            periphery_banks=BankGroupConfig(
                capital_ratio=DistributionConfig(distribution='fixed', value=0.10),
                total_assets=500.0,
                portfolio=PortfolioConfig(
                    mortgage=DistributionConfig(distribution='fixed', value=0.60),
                    remaining={'government_bond': 0.25, 'corporate_bond': 0.35, 'stock': 0.4}
                ),
                external_assets_fraction=0.88
                # interbank_assets_fraction NOT specified
            )
        )

    def test_old_config_still_works(self):
        """Test that old configs without interbank_assets_fraction still work."""
        config = self.create_config_without_variable_ib()
        network = NetworkGenerator.from_config(config, run_id=0)

        # Should create network successfully
        assert len(network.banks) == 10

        # Should calculate IB fraction from external_assets_fraction
        for bank_id, bank in network.banks.items():
            ib_fraction = bank.interbank_assets / bank.get_total_assets()

            if bank_id < 3:  # Core: 1 - 0.85 = 0.15
                assert abs(ib_fraction - 0.15) < 0.001
            else:  # Periphery: 1 - 0.88 = 0.12
                assert abs(ib_fraction - 0.12) < 0.001

    def test_old_config_creates_uniform_ib_fractions(self):
        """Test that old config creates uniform IB fractions (no heterogeneity)."""
        config = self.create_config_without_variable_ib()
        network = NetworkGenerator.from_config(config, run_id=0)

        # All core banks should have same IB fraction
        core_fractions = []
        periphery_fractions = []

        for bank_id, bank in network.banks.items():
            ib_fraction = bank.interbank_assets / bank.get_total_assets()

            if bank_id < 3:
                core_fractions.append(ib_fraction)
            else:
                periphery_fractions.append(ib_fraction)

        # All core should be identical
        assert len(set(core_fractions)) == 1, "Core banks should have identical IB fractions"

        # All periphery should be identical
        assert len(set(periphery_fractions)) == 1, "Periphery banks should have identical IB fractions"


class TestOptionASimulationIntegration:
    """Test full simulation with variable interbank exposure."""

    def create_full_experiment_config(self):
        """Create a complete experiment config with variable IB."""
        return ExperimentConfig(
            metadata=MetadataConfig(
                experiment_id='test_option_a',
                scenario_id='test',
                scenario_name='Test Option A',
                hypothesis='Variable interbank exposure should create heterogeneous bank behavior',
                description='Test variable interbank exposure',
                tags=['test', 'option-a']
            ),
            network=NetworkConfig(
                mode='stochastic',
                topology='pre_2008',
                structure_seed=42,
                parameter_seed=100,
                num_banks=10,
                num_core_banks=3,
                connectivity=create_test_connectivity(),
                core_banks=BankGroupConfig(
                    capital_ratio=DistributionConfig(distribution='fixed', value=0.10),
                    total_assets=1000.0,
                    portfolio=PortfolioConfig(
                        mortgage=DistributionConfig(distribution='fixed', value=0.50),
                        remaining={'government_bond': 0.3, 'corporate_bond': 0.4, 'stock': 0.3}
                    ),
                    external_assets_fraction=0.85,
                    interbank_assets_fraction=DistributionConfig(
                        distribution='normal',
                        mean=0.15,
                        std=0.05,
                        min=0.08,
                        max=0.30
                    )
                ),
                periphery_banks=BankGroupConfig(
                    capital_ratio=DistributionConfig(distribution='fixed', value=0.10),
                    total_assets=500.0,
                    portfolio=PortfolioConfig(
                        mortgage=DistributionConfig(distribution='fixed', value=0.60),
                        remaining={'government_bond': 0.25, 'corporate_bond': 0.35, 'stock': 0.4}
                    ),
                    external_assets_fraction=0.88,
                    interbank_assets_fraction=DistributionConfig(
                        distribution='normal',
                        mean=0.12,
                        std=0.04,
                        min=0.05,
                        max=0.22
                    )
                )
            ),
            shock=ShockConfig(
                mode='correlated',
                asset_shocks={
                    AssetType.MORTGAGE: -0.10,
                    AssetType.GOVERNMENT_BOND: 0.02,
                    AssetType.CORPORATE_BOND: -0.05,
                    AssetType.STOCK: -0.08
                },
                correlation=0.6,
                fire_sale_intensity=0.15,
                shock_volatility=0.03
            ),
            simulation=SimulationConfig(
                num_runs=5,  # Small number for fast tests
                seed=42,
                fire_sales_enabled=True,
                use_priority_claims=False
            ),
            output=create_test_output()
        )

    def test_simulation_runs_with_variable_ib(self):
        """Test that simulation runs successfully with variable IB."""
        config = self.create_full_experiment_config()
        runner = ExperimentRunner(config)

        results = runner.run(verbose=False)

        # Should complete successfully
        assert 'summary_statistics' in results
        assert 'simulation_results' in results

        # Should have 5 simulations
        assert len(results['simulation_results']) == 5

    def test_results_contain_interbank_fraction_data(self):
        """Test that results contain interbank_assets_fraction in initial_state."""
        config = self.create_full_experiment_config()
        runner = ExperimentRunner(config)

        results = runner.run(verbose=False)

        # Check first simulation
        first_sim = results['simulation_results'][0]

        assert 'initial_state' in first_sim

        # Check each bank has interbank_assets_fraction
        for bank_id, bank_data in first_sim['initial_state'].items():
            assert 'interbank_assets_fraction' in bank_data, \
                f"Bank {bank_id} missing interbank_assets_fraction"

            # Value should be in expected range
            ib_fraction = bank_data['interbank_assets_fraction']
            assert 0.05 <= ib_fraction <= 0.30, \
                f"Bank {bank_id} has IB fraction {ib_fraction:.2%} outside expected range"

    def test_results_show_heterogeneous_ib_fractions(self):
        """Test that results show heterogeneous IB fractions across banks."""
        config = self.create_full_experiment_config()
        runner = ExperimentRunner(config)

        results = runner.run(verbose=False)

        # Check first simulation
        first_sim = results['simulation_results'][0]

        ib_fractions = []
        for bank_data in first_sim['initial_state'].values():
            ib_fractions.append(bank_data['interbank_assets_fraction'])

        # Should have variety
        unique_fractions = len(set(ib_fractions))
        assert unique_fractions >= 8, \
            f"Only {unique_fractions} unique IB fractions - expected heterogeneity"

    def test_results_include_bank_type(self):
        """Test that results include bank_type (core/periphery)."""
        config = self.create_full_experiment_config()
        runner = ExperimentRunner(config)

        results = runner.run(verbose=False)

        first_sim = results['simulation_results'][0]

        # Check bank types are present
        for bank_id, bank_data in first_sim['initial_state'].items():
            assert 'bank_type' in bank_data, f"Bank {bank_id} missing bank_type"

            bank_type = bank_data['bank_type']
            bank_id_int = int(bank_id) if isinstance(bank_id, str) else bank_id

            if bank_id_int < 3:
                assert bank_type == 'core', f"Bank {bank_id} should be core"
            else:
                assert bank_type == 'periphery', f"Bank {bank_id} should be periphery"
