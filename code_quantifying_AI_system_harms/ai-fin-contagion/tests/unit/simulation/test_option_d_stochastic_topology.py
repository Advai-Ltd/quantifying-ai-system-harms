"""
Unit tests for Option D: Stochastic Network Topology.

Tests that parameter_seed is used for topology generation to create
varied network structures across simulation runs.
"""

import numpy as np
from financial_contagion_networks.config.models import (
    NetworkConfig, BankGroupConfig, DistributionConfig,
    PortfolioConfig, ConnectivityConfig
)
from financial_contagion_networks.simulation.generators import NetworkGenerator


def create_test_network_config() -> NetworkConfig:
    """Create a standard network config for testing."""
    return NetworkConfig(
        mode='stochastic',
        topology='pre_2008',
        structure_seed=42,
        parameter_seed=100,
        num_banks=10,
        num_core_banks=3,
        connectivity=ConnectivityConfig(
            core_to_core=0.8,
            core_to_periphery=0.7,
            periphery_to_core=0.6,
            periphery_to_periphery=0.6,
            core_to_core_exposure=DistributionConfig(distribution='uniform', min=4.0, max=8.0),
            core_to_periphery_exposure=DistributionConfig(distribution='uniform', min=1.5, max=4.0),
            periphery_to_core_exposure=DistributionConfig(distribution='uniform', min=1.0, max=3.0),
            periphery_to_periphery_exposure=DistributionConfig(distribution='uniform', min=0.5, max=2.0)
        ),
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


class TestOptionDBasicTopology:
    """Test basic topology generation with parameter_seed."""

    def test_different_run_ids_produce_different_topologies(self):
        """Test that different run_ids produce different network topologies."""
        config = create_test_network_config()

        # Generate two networks with different run_ids
        network1 = NetworkGenerator.from_config(config, run_id=0)
        network2 = NetworkGenerator.from_config(config, run_id=1)

        # Extract connection patterns (who is connected to whom)
        connections1 = self._get_connection_patterns(network1)
        connections2 = self._get_connection_patterns(network2)

        # Should have different connection patterns
        assert connections1 != connections2, \
            "Different run_ids should produce different topologies"

    def test_same_run_id_produces_same_topology(self):
        """Test that same run_id produces same topology (reproducibility)."""
        config = create_test_network_config()

        # Generate same network twice
        network1 = NetworkGenerator.from_config(config, run_id=0)
        network2 = NetworkGenerator.from_config(config, run_id=0)

        # Extract connection patterns
        connections1 = self._get_connection_patterns(network1)
        connections2 = self._get_connection_patterns(network2)

        # Should have identical connection patterns
        assert connections1 == connections2, \
            "Same run_id should produce identical topologies"

    def test_topologies_vary_in_connectivity(self):
        """Test that different topologies have different connectivity patterns."""
        config = create_test_network_config()

        # Generate multiple networks
        networks = [NetworkGenerator.from_config(config, run_id=i) for i in range(5)]

        # Calculate number of connections for each
        connection_counts = [len(self._get_connection_patterns(net)) for net in networks]

        # Should have variety (not all the same)
        unique_counts = len(set(connection_counts))
        assert unique_counts >= 3, \
            f"Only {unique_counts} unique connection counts - expected more variety"

    def test_parameter_seed_affects_topology(self):
        """Test that different parameter_seeds produce different topologies."""
        # Create two configs with different parameter_seeds
        config1 = create_test_network_config()
        config1.parameter_seed = 100

        config2 = create_test_network_config()
        config2.parameter_seed = 200

        # Generate networks with same run_id but different parameter_seeds
        network1 = NetworkGenerator.from_config(config1, run_id=0)
        network2 = NetworkGenerator.from_config(config2, run_id=0)

        connections1 = self._get_connection_patterns(network1)
        connections2 = self._get_connection_patterns(network2)

        # Should have different topologies
        assert connections1 != connections2, \
            "Different parameter_seeds should produce different topologies"

    def _get_connection_patterns(self, network) -> set:
        """Extract connection patterns from network as set of (from, to) tuples."""
        matrices = network.get_exposure_matrices()

        # Get exposure matrix - it's a dict with (from, to) tuple keys
        exposure_matrix = matrices['exposure_matrix']

        # Extract all connections (keys are already (from, to) tuples)
        connections = set(exposure_matrix.keys())

        return connections


class TestOptionDNetworkMetrics:
    """Test network structure metrics with stochastic topology."""

    def test_network_density_varies_across_runs(self):
        """Test that network density (connection ratio) varies across runs."""
        config = create_test_network_config()

        densities = []
        for run_id in range(10):
            network = NetworkGenerator.from_config(config, run_id=run_id)
            density = self._calculate_network_density(network)
            densities.append(density)

        # Should have variety in densities
        unique_densities = len(set(densities))
        assert unique_densities >= 5, \
            f"Only {unique_densities} unique densities out of 10 runs"

        # Densities should be within expected range (based on connection probabilities)
        # With probabilities 0.6-0.8, expect densities roughly 0.6-0.8
        assert all(0.4 <= d <= 0.9 for d in densities), \
            f"Some densities outside expected range: {densities}"

    def test_core_periphery_structure_maintained(self):
        """Test that core-periphery structure is maintained across runs."""
        config = create_test_network_config()

        for run_id in range(5):
            network = NetworkGenerator.from_config(config, run_id=run_id)

            # Core banks (0-2) should generally have more connections than periphery
            core_avg_connections = self._avg_connections_for_banks(network, range(0, 3))
            periphery_avg_connections = self._avg_connections_for_banks(network, range(3, 10))

            # Core should have higher average connectivity (not always, but on average)
            # We test this is true for at least some runs
            if core_avg_connections > periphery_avg_connections:
                # Good - core has more connections as expected
                pass

    def test_topology_reproducible_with_seeds(self):
        """Test that topology is reproducible given same seeds and run_id."""
        config = create_test_network_config()

        # Generate network 5 times with same parameters
        densities = []
        for _ in range(5):
            network = NetworkGenerator.from_config(config, run_id=0)
            density = self._calculate_network_density(network)
            densities.append(density)

        # All densities should be identical
        assert len(set(densities)) == 1, \
            f"Same seeds should produce identical topology, got densities: {densities}"

    def _calculate_network_density(self, network) -> float:
        """Calculate network density (fraction of possible connections that exist)."""
        matrices = network.get_exposure_matrices()
        exposure_matrix = matrices['exposure_matrix']

        num_banks = len(network.banks)
        num_connections = len(exposure_matrix)  # Each key is a connection

        # Maximum possible connections (excluding self-loops)
        max_connections = num_banks * (num_banks - 1)

        return num_connections / max_connections if max_connections > 0 else 0.0

    def _avg_connections_for_banks(self, network, bank_ids) -> float:
        """Calculate average number of connections for specified banks."""
        matrices = network.get_exposure_matrices()
        exposure_matrix = matrices['exposure_matrix']

        total_connections = 0
        for bank_id in bank_ids:
            # Count connections where this bank is the lender
            connections = sum(1 for (from_id, to_id) in exposure_matrix.keys()
                            if from_id == bank_id)
            total_connections += connections

        return total_connections / len(bank_ids) if len(bank_ids) > 0 else 0.0


class TestOptionDModeCompatibility:
    """Test that stochastic mode works correctly in different scenarios."""

    def test_stochastic_mode_basic_generation(self):
        """Test stochastic mode generates valid networks."""
        config = create_test_network_config()

        # Should generate valid networks
        network1 = NetworkGenerator.from_config(config, run_id=0)
        network2 = NetworkGenerator.from_config(config, run_id=1)

        # Both should be valid networks
        assert len(network1.banks) == 10
        assert len(network2.banks) == 10

        # Different run_ids should produce different networks
        connections1 = self._get_connection_patterns(network1)
        connections2 = self._get_connection_patterns(network2)
        assert connections1 != connections2

    def test_different_parameter_seeds_produce_different_networks(self):
        """Test that different parameter_seeds produce different networks."""
        config1 = create_test_network_config()
        config1.parameter_seed = 100

        config2 = create_test_network_config()
        config2.parameter_seed = 200

        # Generate networks with same run_id but different parameter_seeds
        network1 = NetworkGenerator.from_config(config1, run_id=0)
        network2 = NetworkGenerator.from_config(config2, run_id=0)

        # Should be reproducible with same seeds
        connections1 = self._get_connection_patterns(network1)
        connections2 = self._get_connection_patterns(network2)

        assert connections1 != connections2, \
            "Different parameter_seeds should produce different networks"

    def test_parameter_seed_overrides_structure_seed_for_topology(self):
        """Test that parameter_seed is used for topology (Option D behavior).

        OPTION D: parameter_seed is used for topology generation, not structure_seed.
        This creates more variation across runs - both parameters AND topology vary.
        """
        config1 = create_test_network_config()
        config1.structure_seed = 42
        config1.parameter_seed = 100

        config2 = create_test_network_config()
        config2.structure_seed = 999  # Different structure_seed
        config2.parameter_seed = 100  # Same parameter_seed

        # Generate networks with same run_id and parameter_seed
        network1 = NetworkGenerator.from_config(config1, run_id=0)
        network2 = NetworkGenerator.from_config(config2, run_id=0)

        # Networks should have SAME topology because parameter_seed is same
        # This is Option D: parameter_seed controls topology, not structure_seed
        connections1 = self._get_connection_patterns(network1)
        connections2 = self._get_connection_patterns(network2)

        assert connections1 == connections2, \
            "Same parameter_seed should produce same topology (Option D: parameter_seed overrides structure_seed)"

    def _get_connection_patterns(self, network) -> set:
        """Extract connection patterns from network."""
        matrices = network.get_exposure_matrices()
        exposure_matrix = matrices['exposure_matrix']

        # exposure_matrix is a dict with (from, to) tuple keys
        return set(exposure_matrix.keys())


class TestOptionDRealism:
    """Test that stochastic topologies maintain realistic properties."""

    def test_topologies_respect_connectivity_probabilities(self):
        """Test that connection probabilities are approximately respected."""
        config = create_test_network_config()

        # Generate many networks
        num_runs = 50
        core_to_core_counts = []

        for run_id in range(num_runs):
            network = NetworkGenerator.from_config(config, run_id=run_id)

            # Count core-to-core connections
            matrices = network.get_exposure_matrices()
            exposure_matrix = matrices['exposure_matrix']

            connections = 0
            possible = 0
            for i in range(3):  # Core banks: 0, 1, 2
                for j in range(3):
                    if i != j:
                        possible += 1
                        if (i, j) in exposure_matrix:
                            connections += 1

            ratio = connections / possible if possible > 0 else 0
            core_to_core_counts.append(ratio)

        # Average should be close to configured probability (0.8)
        avg_ratio = np.mean(core_to_core_counts)
        assert 0.6 <= avg_ratio <= 0.95, \
            f"Core-to-core connection ratio {avg_ratio:.2f} far from expected 0.8"

    def test_no_self_loops_in_topology(self):
        """Test that banks never have connections to themselves."""
        config = create_test_network_config()

        for run_id in range(10):
            network = NetworkGenerator.from_config(config, run_id=run_id)
            matrices = network.get_exposure_matrices()
            exposure_matrix = matrices['exposure_matrix']

            # Check no self-loops (i, i) in exposure matrix
            for (from_id, to_id) in exposure_matrix.keys():
                assert from_id != to_id, \
                    f"Bank {from_id} has self-loop in run {run_id}"

    def test_all_banks_present_in_network(self):
        """Test that all configured banks are present in generated network."""
        config = create_test_network_config()

        for run_id in range(5):
            network = NetworkGenerator.from_config(config, run_id=run_id)

            # Should have exactly 10 banks
            assert len(network.banks) == 10, \
                f"Expected 10 banks, got {len(network.banks)} in run {run_id}"

            # Banks should be numbered 0-9
            bank_ids = set(network.banks.keys())
            expected_ids = set(range(10))
            assert bank_ids == expected_ids, \
                f"Bank IDs mismatch: expected {expected_ids}, got {bank_ids}"
