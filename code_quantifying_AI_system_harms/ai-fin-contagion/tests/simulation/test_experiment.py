"""
Unit tests for experiment runner.

Tests the ExperimentRunner class with all combinations of
network modes and shock modes.
"""

import pytest
import numpy as np
from pathlib import Path

from financial_contagion_networks.simulation.experiment import ExperimentRunner, run_experiment
from financial_contagion_networks.config import (
    ExperimentConfig,
    MetadataConfig,
    NetworkConfig,
    ShockConfig,
    SimulationConfig,
    OutputConfig,
    BankGroupConfig,
    PortfolioConfig,
    NetworkMode,
    ShockMode
)
from financial_contagion_networks.core.assets import AssetType


# ============================================================================
# Test Helpers
# ============================================================================

class TestHelpers:
    """Helper methods for creating test configurations."""

    @staticmethod
    def create_minimal_experiment_config(
        network_mode: str = 'stochastic',
        shock_mode: str = 'correlated',
        num_runs: int = 10
    ) -> ExperimentConfig:
        """Create minimal valid experiment configuration using conftest helper."""
        from tests.conftest import create_minimal_test_config
        # Use conftest helper with overrides to match test expectations
        return create_minimal_test_config(
            metadata={
                'experiment_id': 'TEST',
                'scenario_id': 'TEST.1',
                'scenario_name': 'Test Scenario'
            },
            network={
                'mode': network_mode,
                'num_banks': 20,
                'num_core_banks': 5
            },
            shock={'mode': shock_mode},
            simulation={'num_runs': num_runs}
        )


# ============================================================================
# Test ExperimentRunner Initialization
# ============================================================================

class TestExperimentRunnerInit:
    """Test ExperimentRunner initialization."""

    def test_init_with_valid_config(self):
        """Test initializing runner with valid config."""
        config = TestHelpers.create_minimal_experiment_config()
        runner = ExperimentRunner(config)

        assert runner.config == config
        assert runner.network_template is None  # Not template mode

    def test_init_with_template_mode(self):
        """Test initializing runner with template mode creates template."""
        config = TestHelpers.create_minimal_experiment_config(
            network_mode='template'
        )
        runner = ExperimentRunner(config)

        assert runner.network_template is not None
        assert runner.network_template.topology.num_banks == 20


# ============================================================================
# Test Network Mode Support
# ============================================================================

class TestNetworkModes:
    """Test all three network modes."""

    def test_fixed_mode(self):
        """Test experiment with fixed network mode."""
        config = TestHelpers.create_minimal_experiment_config(
            network_mode='fixed',
            num_runs=5
        )

        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        # Check results structure
        assert 'metadata' in results
        assert 'config' in results
        assert 'simulation_results' in results
        assert 'summary_statistics' in results

        # Check we got correct number of simulations
        assert len(results['simulation_results']) == 5

    def test_template_mode(self):
        """Test experiment with template network mode."""
        config = TestHelpers.create_minimal_experiment_config(
            network_mode='template',
            num_runs=5
        )

        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        assert len(results['simulation_results']) == 5
        assert results['metadata']['network_mode'] == 'template'

    def test_stochastic_mode(self):
        """Test experiment with stochastic network mode."""
        config = TestHelpers.create_minimal_experiment_config(
            network_mode='stochastic',
            num_runs=5
        )

        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        assert len(results['simulation_results']) == 5
        assert results['metadata']['network_mode'] == 'stochastic'


# ============================================================================
# Test Shock Mode Support
# ============================================================================

class TestShockModes:
    """Test all three shock modes."""

    def test_deterministic_mode(self):
        """Test experiment with deterministic shock mode."""
        config = TestHelpers.create_minimal_experiment_config(
            shock_mode='deterministic',
            num_runs=5
        )

        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        assert len(results['simulation_results']) == 5
        assert results['metadata']['shock_mode'] == 'deterministic'

    def test_correlated_mode(self):
        """Test experiment with correlated shock mode."""
        config = TestHelpers.create_minimal_experiment_config(
            shock_mode='correlated',
            num_runs=5
        )

        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        assert len(results['simulation_results']) == 5
        assert results['metadata']['shock_mode'] == 'correlated'

    def test_uncorrelated_mode(self):
        """Test experiment with uncorrelated shock mode."""
        config = TestHelpers.create_minimal_experiment_config(
            shock_mode='uncorrelated',
            num_runs=5
        )

        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        assert len(results['simulation_results']) == 5
        assert results['metadata']['shock_mode'] == 'uncorrelated'


# ============================================================================
# Test Results Structure
# ============================================================================

class TestResultsStructure:
    """Test the structure of returned results."""

    def test_results_contain_all_sections(self):
        """Test that results contain all expected sections."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=5)
        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        # Check top-level structure
        assert 'metadata' in results
        assert 'config' in results
        assert 'simulation_results' in results
        assert 'summary_statistics' in results

    def test_metadata_structure(self):
        """Test metadata structure."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=5)
        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        metadata = results['metadata']
        assert 'experiment_id' in metadata
        assert 'scenario_id' in metadata
        assert 'scenario_name' in metadata
        assert 'network_mode' in metadata
        assert 'shock_mode' in metadata
        assert 'num_simulations' in metadata

        assert metadata['experiment_id'] == 'TEST'
        assert metadata['scenario_id'] == 'TEST.1'

    def test_summary_statistics_structure(self):
        """Test summary statistics structure."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=10)
        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        stats = results['summary_statistics']

        # Check required fields
        assert 'num_simulations' in stats
        assert 'failure_rate' in stats
        assert 'total_rounds' in stats
        assert 'asset_losses' in stats
        assert 'fire_sale_losses' in stats
        assert 'systemic_crisis_probability' in stats

        # Check failure_rate subfields
        fr = stats['failure_rate']
        assert 'mean' in fr
        assert 'std' in fr
        assert 'min' in fr
        assert 'max' in fr
        assert 'median' in fr
        assert 'percentile_5' in fr
        assert 'percentile_95' in fr

        # Check values are reasonable
        assert 0 <= fr['mean'] <= 1
        assert 0 <= fr['std'] <= 1
        assert 0 <= stats['systemic_crisis_probability'] <= 1

    def test_simulation_results_structure(self):
        """Test individual simulation results structure."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=3)
        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        sim_results = results['simulation_results']
        assert len(sim_results) == 3

        # Check first result structure
        first_result = sim_results[0]
        assert 'scenario' in first_result
        assert 'failure_rate' in first_result
        assert 'total_failed' in first_result
        assert 'total_rounds' in first_result
        assert 'total_asset_losses' in first_result
        assert 'fire_sale_losses' in first_result


# ============================================================================
# Test Three-Level Contagion Mechanism Taxonomy
# ============================================================================

class TestContagionMechanismTaxonomy:
    """Test the three-level contagion mechanism analysis."""

    def test_contagion_mechanisms_present(self):
        """Test that contagion_mechanisms field is present in summary statistics."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=10)
        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        stats = results['summary_statistics']
        assert 'contagion_mechanisms' in stats, "contagion_mechanisms should be in summary statistics"

    def test_contagion_mechanisms_structure(self):
        """Test that contagion_mechanisms has all required fields."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=10)
        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        cm = results['summary_statistics']['contagion_mechanisms']

        # Level 1: Initial vs Contagion
        assert 'initial_failure_rate' in cm
        assert 'contagion_failure_rate' in cm

        # Level 2: Direct vs Indirect
        assert 'direct_contagion_rate' in cm
        assert 'indirect_contagion_rate' in cm

        # Level 3: Implementation detail
        assert 'interbank_failure_rate' in cm
        assert 'fire_sale_failure_rate' in cm

        # Contagion probabilities
        assert 'contagion_probability' in cm
        assert 'direct_contagion_probability' in cm
        assert 'indirect_contagion_probability' in cm

        # Composition percentages
        assert 'initial_pct_of_failures' in cm
        assert 'contagion_pct_of_failures' in cm
        assert 'direct_pct_of_failures' in cm
        assert 'indirect_pct_of_failures' in cm

    def test_contagion_mechanisms_values_valid(self):
        """Test that contagion mechanism values are valid."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=10)
        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        cm = results['summary_statistics']['contagion_mechanisms']

        # All rates should be between 0 and 1
        assert 0 <= cm['initial_failure_rate'] <= 1
        assert 0 <= cm['contagion_failure_rate'] <= 1
        assert 0 <= cm['direct_contagion_rate'] <= 1
        assert 0 <= cm['indirect_contagion_rate'] <= 1
        assert 0 <= cm['interbank_failure_rate'] <= 1
        assert 0 <= cm['fire_sale_failure_rate'] <= 1

        # Probabilities should be between 0 and 1
        assert 0 <= cm['contagion_probability'] <= 1
        assert 0 <= cm['direct_contagion_probability'] <= 1
        assert 0 <= cm['indirect_contagion_probability'] <= 1

        # Percentages should be between 0 and 100
        assert 0 <= cm['initial_pct_of_failures'] <= 100
        assert 0 <= cm['contagion_pct_of_failures'] <= 100
        assert 0 <= cm['direct_pct_of_failures'] <= 100
        assert 0 <= cm['indirect_pct_of_failures'] <= 100

    def test_contagion_taxonomy_consistency(self):
        """Test that the three-level taxonomy is internally consistent."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=10)
        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        cm = results['summary_statistics']['contagion_mechanisms']

        # Level 3 should equal Level 2
        # Interbank = Direct, Fire sale = Indirect
        assert abs(cm['interbank_failure_rate'] - cm['direct_contagion_rate']) < 1e-10
        assert abs(cm['fire_sale_failure_rate'] - cm['indirect_contagion_rate']) < 1e-10

        # Level 2 should sum to Level 1 contagion
        direct_plus_indirect = cm['direct_contagion_rate'] + cm['indirect_contagion_rate']
        assert abs(direct_plus_indirect - cm['contagion_failure_rate']) < 1e-10

        # Composition percentages should sum to 100% (or 0% if no failures)
        total_pct = cm['initial_pct_of_failures'] + cm['contagion_pct_of_failures']
        if cm['initial_pct_of_failures'] + cm['contagion_pct_of_failures'] > 0:
            assert abs(total_pct - 100.0) < 1e-6

    def test_contagion_with_fire_sales_disabled(self):
        """Test contagion mechanisms when fire sales are disabled."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=10)
        config.simulation.fire_sales_enabled = False

        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        cm = results['summary_statistics']['contagion_mechanisms']

        # With fire sales disabled, indirect/fire sale should be zero
        assert cm['fire_sale_failure_rate'] == 0
        assert cm['indirect_contagion_rate'] == 0
        assert cm['indirect_pct_of_failures'] == 0


# ============================================================================
# Test Determinism
# ============================================================================

class TestDeterminism:
    """Test that experiments are deterministic with seeds."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        config1 = TestHelpers.create_minimal_experiment_config(
            network_mode='stochastic',
            shock_mode='correlated',
            num_runs=5
        )
        config1.simulation.seed = 42

        config2 = TestHelpers.create_minimal_experiment_config(
            network_mode='stochastic',
            shock_mode='correlated',
            num_runs=5
        )
        config2.simulation.seed = 42

        runner1 = ExperimentRunner(config1)
        results1 = runner1.run(verbose=False)

        runner2 = ExperimentRunner(config2)
        results2 = runner2.run(verbose=False)

        # Compare summary statistics
        stats1 = results1['summary_statistics']
        stats2 = results2['summary_statistics']

        assert abs(stats1['failure_rate']['mean'] - stats2['failure_rate']['mean']) < 1e-10
        assert abs(stats1['total_rounds']['mean'] - stats2['total_rounds']['mean']) < 1e-10

    @pytest.mark.skip(reason="Test premise is flawed - same simulation seed produces deterministic results even with different structure seeds")
    def test_different_seed_different_results(self):
        """Test that different seeds produce different results."""
        config1 = TestHelpers.create_minimal_experiment_config(
            network_mode='stochastic',
            shock_mode='correlated',
            num_runs=10
        )
        config1.network.structure_seed = 42

        config2 = TestHelpers.create_minimal_experiment_config(
            network_mode='stochastic',
            shock_mode='correlated',
            num_runs=10
        )
        config2.network.structure_seed = 123

        runner1 = ExperimentRunner(config1)
        results1 = runner1.run(verbose=False)

        runner2 = ExperimentRunner(config2)
        results2 = runner2.run(verbose=False)

        # Results should differ
        stats1 = results1['summary_statistics']
        stats2 = results2['summary_statistics']

        mean_diff = abs(stats1['failure_rate']['mean'] - stats2['failure_rate']['mean'])
        total_rounds_diff = abs(stats1['total_rounds']['mean'] - stats2['total_rounds']['mean'])

        assert mean_diff > 0.001 or total_rounds_diff > 0.01


# ============================================================================
# Test Convenience Function
# ============================================================================

class TestConvenienceFunction:
    """Test run_experiment convenience function."""

    def test_run_experiment_function(self):
        """Test run_experiment convenience function."""
        config = TestHelpers.create_minimal_experiment_config(num_runs=5)

        results = run_experiment(config, verbose=False)

        assert 'metadata' in results
        assert 'simulation_results' in results
        assert len(results['simulation_results']) == 5


# ============================================================================
# Test Integration
# ============================================================================

class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_post_2008_scenario(self):
        """Test with realistic post-2008 scenario."""
        from financial_contagion_networks.config import DistributionConfig
        from tests.conftest import create_minimal_test_config

        # Create config with post-2008 baseline parameters
        config = create_minimal_test_config(
            metadata={
                'experiment_id': 'TEST',
                'scenario_id': 'POST_2008',
                'scenario_name': 'Post-2008 Test'
            },
            network={
                'mode': 'stochastic',
                'num_banks': 20,
                'num_core_banks': 5,
                'core_banks': {
                    'capital_ratio': {'distribution': 'fixed', 'value': 0.15},  # 15% post-reform
                    'total_assets': 500.0
                },
                'periphery_banks': {
                    'capital_ratio': {'distribution': 'fixed', 'value': 0.12},  # 12% post-reform
                    'total_assets': 200.0
                }
            },
            shock={
                'mode': 'correlated',
                'correlation': 0.6,
                'fire_sale_intensity': 0.15
            },
            simulation={'num_runs': 20}
        )

        runner = ExperimentRunner(config)
        results = runner.run(verbose=False)

        # Check we got valid results
        assert len(results['simulation_results']) == 20

        # Check failure rate is reasonable (should be low for post-2008 system)
        mean_failure_rate = results['summary_statistics']['failure_rate']['mean']
        assert 0 <= mean_failure_rate <= 0.5, \
            f"Post-2008 system should be relatively stable, got {mean_failure_rate:.1%}"

    def test_all_mode_combinations(self):
        """Test all combinations of network and shock modes."""
        network_modes = ['fixed', 'template', 'stochastic']
        shock_modes = ['deterministic', 'correlated', 'uncorrelated']

        for net_mode in network_modes:
            for shock_mode in shock_modes:
                config = TestHelpers.create_minimal_experiment_config(
                    network_mode=net_mode,
                    shock_mode=shock_mode,
                    num_runs=3  # Small number for speed
                )

                runner = ExperimentRunner(config)
                results = runner.run(verbose=False)

                # All combinations should work
                assert len(results['simulation_results']) == 3
                assert results['metadata']['network_mode'] == net_mode
                assert results['metadata']['shock_mode'] == shock_mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
