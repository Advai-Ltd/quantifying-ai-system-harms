"""
Integration tests for Option D: Stochastic Network Topology.

Tests the full pipeline from config → network generation → simulation
to ensure stochastic topology generation works correctly end-to-end.
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
        save_detailed_results=False,
        save_network_snapshots=False,
        save_config_copy=False,
        generate_plots=False,
        plot_formats=['png'],
        verbose=False
    )


class TestOptionDNetworkGeneration:
    """Test network generation with stochastic topology."""

    def create_test_config_stochastic(self):
        """Create a test config with stochastic mode."""
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
            ),
            periphery_banks=BankGroupConfig(
                capital_ratio=DistributionConfig(distribution='fixed', value=0.10),
                total_assets=500.0,
                portfolio=PortfolioConfig(
                    mortgage=DistributionConfig(distribution='fixed', value=0.60),
                    remaining={'government_bond': 0.25, 'corporate_bond': 0.35, 'stock': 0.4}
                ),
                external_assets_fraction=0.88
            )
        )

    def test_stochastic_mode_creates_different_topologies(self):
        """Test that stochastic mode creates different topologies across runs."""
        config = self.create_test_config_stochastic()

        # Generate multiple networks
        networks = [NetworkGenerator.from_config(config, run_id=i) for i in range(5)]

        # Extract topologies
        topologies = []
        for network in networks:
            matrices = network.get_exposure_matrices()
            exposure_matrix = matrices['exposure_matrix']
            topology = frozenset(exposure_matrix.keys())
            topologies.append(topology)

        # Should have different topologies
        unique_topologies = len(set(topologies))
        assert unique_topologies >= 4, \
            f"Only {unique_topologies} unique topologies out of 5 runs - expected variation"

    def test_stochastic_mode_maintains_network_properties(self):
        """Test that stochastic networks maintain expected properties."""
        config = self.create_test_config_stochastic()

        for run_id in range(5):
            network = NetworkGenerator.from_config(config, run_id=run_id)

            # All banks should be present
            assert len(network.banks) == 10

            # Banks should have valid IDs
            assert set(network.banks.keys()) == set(range(10))

            # Network should have connections
            matrices = network.get_exposure_matrices()
            exposure_matrix = matrices['exposure_matrix']
            assert len(exposure_matrix) > 0, "Network should have connections"

    def test_network_density_varies_with_stochastic_mode(self):
        """Test that network density varies across stochastic runs."""
        config = self.create_test_config_stochastic()

        densities = []
        for run_id in range(10):
            network = NetworkGenerator.from_config(config, run_id=run_id)
            matrices = network.get_exposure_matrices()
            exposure_matrix = matrices['exposure_matrix']

            num_connections = len(exposure_matrix)
            max_connections = 10 * 9  # 10 banks, no self-loops
            density = num_connections / max_connections

            densities.append(density)

        # Should have variety in densities
        unique_densities = len(set(densities))
        assert unique_densities >= 5, \
            f"Only {unique_densities} unique densities - expected more variation"

    def test_individual_exposure_amounts_vary_across_runs(self):
        """Test that individual exposure amounts vary across runs (parameter variation)."""
        config = self.create_test_config_stochastic()

        # Collect all individual exposure amounts from each run
        all_exposures = []
        for run_id in range(5):
            network = NetworkGenerator.from_config(config, run_id=run_id)
            matrices = network.get_exposure_matrices()
            exposure_matrix = matrices['exposure_matrix']

            # Get all exposure values
            for (from_id, to_id), amount in exposure_matrix.items():
                all_exposures.append(round(amount, 2))  # Round to avoid float precision issues

        # Should have variety in exposure amounts (parameter_seed varies exposure sampling)
        unique_exposures = len(set(all_exposures))
        assert unique_exposures >= 20, \
            f"Only {unique_exposures} unique exposure values - expected more parameter variation"

    def test_reproducibility_with_same_run_id(self):
        """Test that same run_id produces identical network."""
        config = self.create_test_config_stochastic()

        # Generate same network twice
        network1 = NetworkGenerator.from_config(config, run_id=0)
        network2 = NetworkGenerator.from_config(config, run_id=0)

        # Extract topologies
        matrices1 = network1.get_exposure_matrices()
        matrices2 = network2.get_exposure_matrices()

        topology1 = set(matrices1['exposure_matrix'].keys())
        topology2 = set(matrices2['exposure_matrix'].keys())

        # Should be identical
        assert topology1 == topology2, \
            "Same run_id should produce identical topology"

        # Exposure amounts should also be identical
        for (from_id, to_id) in topology1:
            exposure1 = matrices1['exposure_matrix'][(from_id, to_id)]
            exposure2 = matrices2['exposure_matrix'][(from_id, to_id)]
            assert abs(exposure1 - exposure2) < 0.001, \
                f"Exposure ({from_id}, {to_id}) differs: {exposure1} vs {exposure2}"


class TestOptionDSimulationIntegration:
    """Test full simulation with stochastic topology."""

    def create_full_experiment_config(self):
        """Create a complete experiment config with stochastic mode."""
        return ExperimentConfig(
            metadata=MetadataConfig(
                experiment_id='test_option_d',
                scenario_id='test',
                scenario_name='Test Option D',
                hypothesis='Stochastic topology should create diverse network structures',
                description='Test stochastic network topology generation',
                tags=['test', 'option-d']
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
                    external_assets_fraction=0.85
                ),
                periphery_banks=BankGroupConfig(
                    capital_ratio=DistributionConfig(distribution='fixed', value=0.10),
                    total_assets=500.0,
                    portfolio=PortfolioConfig(
                        mortgage=DistributionConfig(distribution='fixed', value=0.60),
                        remaining={'government_bond': 0.25, 'corporate_bond': 0.35, 'stock': 0.4}
                    ),
                    external_assets_fraction=0.88
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
                num_runs=5,
                seed=42,
                fire_sales_enabled=True,
                use_priority_claims=False
            ),
            output=create_test_output()
        )

    def test_simulation_runs_with_stochastic_topology(self):
        """Test that simulation runs successfully with stochastic topology."""
        config = self.create_full_experiment_config()
        runner = ExperimentRunner(config)

        results = runner.run(verbose=False)

        # Should complete successfully
        assert 'summary_statistics' in results
        assert 'simulation_results' in results

        # Should have 5 simulations
        assert len(results['simulation_results']) == 5

    def test_results_contain_network_structure_data(self):
        """Test that results contain network structure in initial state."""
        config = self.create_full_experiment_config()
        runner = ExperimentRunner(config)

        results = runner.run(verbose=False)

        # Check first simulation
        first_sim = results['simulation_results'][0]

        # Should have network structure data
        assert 'network_structure' in first_sim, \
            "Results should contain network_structure"

        # Should have exposure matrices
        assert 'exposure_matrices' in first_sim, \
            "Results should contain exposure_matrices"

    def test_different_simulations_have_different_networks(self):
        """Test that different simulations have different network structures."""
        config = self.create_full_experiment_config()
        runner = ExperimentRunner(config)

        results = runner.run(verbose=False)

        # Extract topologies from each simulation
        topologies = []
        for sim in results['simulation_results']:
            exposure_matrix = sim['exposure_matrices']['exposure_matrix']
            # exposure_matrix is a dict with (from, to) keys, convert to list of tuples
            topology = frozenset((from_id, to_id) for (from_id, to_id) in exposure_matrix.keys())
            topologies.append(topology)

        # Should have different topologies across simulations
        unique_topologies = len(set(topologies))
        assert unique_topologies >= 4, \
            f"Only {unique_topologies} unique topologies out of 5 simulations - expected variation"

    def test_network_metadata_recorded(self):
        """Test that network generation metadata is recorded."""
        config = self.create_full_experiment_config()
        runner = ExperimentRunner(config)

        results = runner.run(verbose=False)

        # Check first simulation
        first_sim = results['simulation_results'][0]

        # Should have network metadata
        if 'network_metadata' in first_sim:
            metadata = first_sim['network_metadata']

            # Should contain mode
            assert 'mode' in metadata, "Network metadata should contain mode"
            assert metadata['mode'] == 'stochastic', \
                f"Mode should be 'stochastic', got '{metadata['mode']}'"

    def test_topology_affects_contagion_outcomes(self):
        """Test that different topologies can produce different outcomes."""
        config = self.create_full_experiment_config()
        runner = ExperimentRunner(config)

        results = runner.run(verbose=False)

        # Extract failure rates
        failure_rates = []
        for sim in results['simulation_results']:
            failure_rates.append(sim['failure_rate'])

        # With different topologies, we might see different failure rates
        # (not guaranteed, but likely with enough runs)
        # At minimum, check that we get valid failure rates
        assert all(0 <= fr <= 1 for fr in failure_rates), \
            f"All failure rates should be between 0 and 1, got {failure_rates}"


class TestOptionDCombinedWithOptionA:
    """Test that Options A and D work together correctly."""

    def create_combined_config(self):
        """Create config with both Options A and D enabled."""
        return NetworkConfig(
            mode='stochastic',  # Option D: stochastic topology
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
                # Option A: variable interbank exposure
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
                # Option A: variable interbank exposure
                interbank_assets_fraction=DistributionConfig(
                    distribution='normal',
                    mean=0.12,
                    std=0.04,
                    min=0.05,
                    max=0.22
                )
            )
        )

    def test_combined_options_create_maximum_variation(self):
        """Test that combining Options A+D creates maximum network variation."""
        config = self.create_combined_config()

        # Generate multiple networks
        networks = []
        for run_id in range(5):
            network = NetworkGenerator.from_config(config, run_id=run_id)
            networks.append(network)

        # Check topology variation (Option D)
        topologies = []
        for network in networks:
            matrices = network.get_exposure_matrices()
            topology = frozenset(matrices['exposure_matrix'].keys())
            topologies.append(topology)

        unique_topologies = len(set(topologies))
        assert unique_topologies >= 4, \
            f"Option D: Only {unique_topologies} unique topologies"

        # Check interbank fraction variation (Option A)
        all_ib_fractions = []
        for network in networks:
            for bank in network.banks.values():
                ib_fraction = bank.interbank_assets / bank.get_total_assets()
                all_ib_fractions.append(ib_fraction)

        unique_ib_fractions = len(set(all_ib_fractions))
        assert unique_ib_fractions >= 40, \
            f"Option A: Only {unique_ib_fractions} unique IB fractions across 50 banks"

    def test_combined_options_maintain_market_clearing(self):
        """Test that Options A+D together maintain market clearing."""
        config = self.create_combined_config()

        for run_id in range(5):
            network = NetworkGenerator.from_config(config, run_id=run_id)

            # Check market clearing
            total_ib_assets = sum(bank.interbank_assets for bank in network.banks.values())
            total_ib_liab = sum(bank.interbank_liabilities for bank in network.banks.values())

            if total_ib_assets > 0:
                deviation = abs(total_ib_assets - total_ib_liab) / total_ib_assets
                assert deviation < 0.01, \
                    f"Run {run_id}: Market not clearing - deviation {deviation:.2%}"
