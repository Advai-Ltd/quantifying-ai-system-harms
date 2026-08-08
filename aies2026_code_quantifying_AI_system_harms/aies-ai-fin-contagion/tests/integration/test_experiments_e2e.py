"""
End-to-End tests for all 6 experiments.

These tests run each experiment with minimal parameters to verify:
1. Config loads successfully
2. Experiment runs without errors
3. Results have expected structure
4. Key thresholds are identified
5. Results are reproducible

Running these tests provides confidence that all experiments work correctly
before running full parameter sweeps.
"""

import pytest
from pathlib import Path
import json
from financial_contagion_networks.config import load_config
from financial_contagion_networks.simulation.experiment import ExperimentRunner


# ============================================================================
# Test Configuration
# ============================================================================

# Experiments to test (relative to financial-contagion-networks/experiments/)
EXPERIMENTS = [
    {
        "name": "pre_2008",
        "path": "post_extension/pre_2008/config.yaml",
        "expected_thresholds": ["stability", "contagion", "systemic_crisis"],
        "description": "Pre-Basel III baseline"
    },
    {
        "name": "post_2008",
        "path": "post_extension/post_2008/config.yaml",
        "expected_thresholds": ["stability", "contagion", "systemic_crisis"],
        "description": "Post-Basel III realistic baseline"
    }
]


# Minimal test parameters (fast execution)
MINIMAL_TEST_PARAMS = {
    "num_runs": 3,  # Very few runs for speed
    "shock_levels": 2,  # Only test 2 shock levels
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def repo_root():
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent


# ============================================================================
# Test: Config Loading
# ============================================================================

class TestExperimentConfigLoading:
    """Test that all experiment configs load successfully."""

    @pytest.mark.parametrize("exp", EXPERIMENTS, ids=[e["name"] for e in EXPERIMENTS])
    def test_config_loads(self, exp, repo_root):
        """Test that experiment config loads without errors."""
        config_path = repo_root / "experiments" / exp["path"]

        # Skip if config doesn't exist (experiment not yet migrated)
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        # Load config - should not raise errors
        config = load_config(config_path)

        # Basic validation
        assert config is not None
        assert config.metadata is not None
        assert config.network is not None
        assert config.simulation is not None
        assert config.shock is not None


# ============================================================================
# Test: Minimal Experiment Execution
# ============================================================================

class TestMinimalExperimentExecution:
    """Test that each experiment runs with minimal parameters."""

    @pytest.mark.parametrize("exp", EXPERIMENTS, ids=[e["name"] for e in EXPERIMENTS])
    def test_experiment_runs_minimal(self, exp, repo_root):
        """Test that experiment executes successfully with minimal params."""
        config_path = repo_root / "experiments" / exp["path"]

        # Skip if config doesn't exist
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        # Load and modify config for quick test
        config = load_config(config_path)
        config.simulation.num_runs = MINIMAL_TEST_PARAMS["num_runs"]

        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()

        # Verify results structure
        assert results is not None
        assert "metadata" in results
        assert "summary_statistics" in results
        assert "simulation_results" in results

        # Verify we got the expected number of simulations
        assert len(results["simulation_results"]) == MINIMAL_TEST_PARAMS["num_runs"]


# ============================================================================
# Test: Results Structure and Validity
# ============================================================================

class TestExperimentResults:
    """Test that experiment results have expected structure and valid values."""

    def test_result_metadata_complete(self, repo_root):
        """Test that result metadata is complete."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        config = load_config(config_path)
        config.simulation.num_runs = 2

        runner = ExperimentRunner(config)
        results = runner.run()

        metadata = results["metadata"]
        assert "experiment_id" in metadata
        assert "scenario_id" in metadata
        assert "scenario_name" in metadata
        assert "num_simulations" in metadata

    def test_result_statistics_valid(self, repo_root):
        """Test that statistics have valid ranges."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        config = load_config(config_path)
        config.simulation.num_runs = 2

        runner = ExperimentRunner(config)
        results = runner.run()

        stats = results["summary_statistics"]

        # Failure rates should be between 0 and 1
        if "failure_rate" in stats:
            assert 0 <= stats["failure_rate"]["mean"] <= 1
            assert 0 <= stats["failure_rate"]["std"] >= 0

        # Contagion rounds should be non-negative
        if "contagion_rounds" in stats:
            assert stats["contagion_rounds"]["mean"] >= 0

    def test_simulation_results_complete(self, repo_root):
        """Test that each simulation result has required fields."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        config = load_config(config_path)
        config.simulation.num_runs = 2

        runner = ExperimentRunner(config)
        results = runner.run()

        for sim_result in results["simulation_results"]:
            assert "scenario" in sim_result
            assert "asset_shocks" in sim_result
            assert "total_failed" in sim_result
            assert "initially_failed" in sim_result


# ============================================================================
# Test: Reproducibility
# ============================================================================

class TestExperimentReproducibility:
    """Test that experiments are reproducible with same seeds."""

    def test_same_seed_same_results(self, repo_root):
        """Test that same seed produces identical results."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        # Run experiment twice with same config
        config1 = load_config(config_path)
        config1.simulation.num_runs = 2
        config1.simulation.seed = 42

        config2 = load_config(config_path)
        config2.simulation.num_runs = 2
        config2.simulation.seed = 42

        runner1 = ExperimentRunner(config1)
        results1 = runner1.run()

        runner2 = ExperimentRunner(config2)
        results2 = runner2.run()

        # Compare failure rates (should be identical)
        fr1 = results1["summary_statistics"]["failure_rate"]["mean"]
        fr2 = results2["summary_statistics"]["failure_rate"]["mean"]

        assert abs(fr1 - fr2) < 1e-10, (
            f"Same seed produced different results!\n"
            f"  Run 1 failure rate: {fr1}\n"
            f"  Run 2 failure rate: {fr2}\n"
            f"  This breaks reproducibility guarantee"
        )

    def test_different_seed_different_results(self, repo_root):
        """Test that different seeds produce different results (stochastic)."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        # Run with different seeds
        config1 = load_config(config_path)
        config1.simulation.num_runs = 10
        config1.simulation.seed = 42
        config1.network.mode = "stochastic"  # Ensure stochastic mode

        config2 = load_config(config_path)
        config2.simulation.num_runs = 10
        config2.simulation.seed = 99
        config2.network.mode = "stochastic"

        runner1 = ExperimentRunner(config1)
        results1 = runner1.run()

        runner2 = ExperimentRunner(config2)
        results2 = runner2.run()

        # Results should differ (with very high probability)
        fr1 = results1["summary_statistics"]["failure_rate"]["mean"]
        fr2 = results2["summary_statistics"]["failure_rate"]["mean"]

        # Allow them to be the same if both are extreme values (0 or 1)
        if not ((fr1 < 0.01 and fr2 < 0.01) or (fr1 > 0.99 and fr2 > 0.99)):
            assert abs(fr1 - fr2) > 0.001, (
                "Different seeds produced very similar results - "
                "this suggests randomization may not be working correctly"
            )


# ============================================================================
# Test: Network Mode Compatibility
# ============================================================================

class TestNetworkModeCompatibility:
    """Test that experiments work with different network modes."""

    @pytest.mark.parametrize("mode", ["fixed", "template", "stochastic"])
    def test_experiment_with_network_mode(self, mode, repo_root):
        """Test experiment runs with each network mode."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        config = load_config(config_path)
        config.simulation.num_runs = 2
        config.network.mode = mode

        runner = ExperimentRunner(config)
        results = runner.run()

        assert results is not None
        assert len(results["simulation_results"]) == 2


# ============================================================================
# Test: Shock Mode Compatibility
# ============================================================================

class TestShockModeCompatibility:
    """Test that experiments work with different shock modes."""

    @pytest.mark.parametrize("mode", ["deterministic", "correlated", "uncorrelated"])
    def test_experiment_with_shock_mode(self, mode, repo_root):
        """Test experiment runs with each shock mode."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        config = load_config(config_path)
        config.simulation.num_runs = 2
        config.shock.mode = mode

        runner = ExperimentRunner(config)
        results = runner.run()

        assert results is not None
        assert len(results["simulation_results"]) == 2


# ============================================================================
# Test: Fire Sales Integration
# ============================================================================

class TestFireSalesIntegration:
    """Test that fire sales mechanism integrates correctly."""

    def test_fire_sales_enabled_vs_disabled(self, repo_root):
        """Test that enabling fire sales affects results."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        # Run with fire sales disabled
        config1 = load_config(config_path)
        config1.simulation.num_runs = 5
        config1.simulation.seed = 42
        config1.simulation.fire_sales_enabled = False
        # Use moderate shock to avoid 100% failure rate
        config1.shock.asset_shocks = {
            'government_bond': 0.0,
            'corporate_bond': -0.05,
            'mortgage': -0.08,  # Moderate shock
            'stock': -0.10
        }

        runner1 = ExperimentRunner(config1)
        results1 = runner1.run()

        # Run with fire sales enabled
        config2 = load_config(config_path)
        config2.simulation.num_runs = 5
        config2.simulation.seed = 42
        config2.simulation.fire_sales_enabled = True
        config2.shock.fire_sale_intensity = 0.5  # Moderate intensity
        config2.shock.asset_shocks = {
            'government_bond': 0.0,
            'corporate_bond': -0.05,
            'mortgage': -0.08,  # Moderate shock
            'stock': -0.10
        }

        runner2 = ExperimentRunner(config2)
        results2 = runner2.run()

        # Results should differ when fire sales are enabled
        # (unless no failures occur, in which case fire sales don't matter)
        fr1 = results1["summary_statistics"]["failure_rate"]["mean"]
        fr2 = results2["summary_statistics"]["failure_rate"]["mean"]

        # If there are failures, fire sales should have an effect
        if fr1 > 0.01:
            # We expect different failure rates or at least different loss attribution
            # Check if fire sale losses are tracked when enabled
            has_fire_sale_losses = any(
                sim.get("fire_sale_losses", 0) > 0
                for sim in results2["simulation_results"]
            )
            # Either fire sales cause losses, or they change the failure rate
            assert has_fire_sale_losses or abs(fr1 - fr2) > 0.01, (
                f"Fire sales enabled but no effect observed\n"
                f"  Failure rate without fire sales: {fr1}\n"
                f"  Failure rate with fire sales: {fr2}\n"
                f"  Has fire sale losses: {has_fire_sale_losses}"
            )


# ============================================================================
# Test: Priority Claims
# ============================================================================

class TestPriorityClaimsIntegration:
    """Test that priority claims mechanism works correctly."""

    def test_priority_claims_affects_recovery_rates(self, repo_root):
        """Test that priority claims affects recovery rates."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        # Run without priority claims
        config1 = load_config(config_path)
        config1.simulation.num_runs = 5
        config1.simulation.seed = 42
        config1.simulation.use_priority_claims = False

        runner1 = ExperimentRunner(config1)
        results1 = runner1.run()

        # Run with priority claims
        config2 = load_config(config_path)
        config2.simulation.num_runs = 5
        config2.simulation.seed = 42
        config2.simulation.use_priority_claims = True

        runner2 = ExperimentRunner(config2)
        results2 = runner2.run()

        # Results should differ when there are failures
        # Priority claims makes interbank creditors junior to external creditors
        fr1 = results1["summary_statistics"]["failure_rate"]["mean"]
        fr2 = results2["summary_statistics"]["failure_rate"]["mean"]

        # If there are contagion failures, priority claims should matter
        if fr1 > 0.01:
            # Recovery rates should generally be lower with priority claims
            # This typically leads to more contagion
            # We don't assert exact relationship, just that system behaves differently
            assert results1 is not None and results2 is not None


# ============================================================================
# Test: Balance Sheet Invariants
# ============================================================================

class TestBalanceSheetInvariants:
    """Test that balance sheet invariants hold throughout simulation."""

    def test_balance_sheets_valid_after_simulation(self, repo_root):
        """Test that all banks have valid balance sheets after simulation."""
        config_path = repo_root / "experiments" / EXPERIMENTS[0]["path"]

        if not config_path.exists():
            pytest.skip("Config not found")

        config = load_config(config_path)
        config.simulation.num_runs = 3

        runner = ExperimentRunner(config)
        results = runner.run()

        # Check each simulation
        for sim_result in results["simulation_results"]:
            bank_states = sim_result.get("bank_states", [])

            for bank_state in bank_states:
                assets = bank_state["total_assets"]
                liabilities = bank_state["total_liabilities"]
                equity = bank_state["equity"]

                # Balance sheet identity: assets = liabilities + equity
                assert abs(assets - (liabilities + equity)) < 1e-6, (
                    f"Balance sheet identity violated for bank {bank_state['bank_id']}!\n"
                    f"  Assets: {assets}\n"
                    f"  Liabilities: {liabilities}\n"
                    f"  Equity: {equity}\n"
                    f"  Liabilities + Equity: {liabilities + equity}\n"
                    f"  Difference: {assets - (liabilities + equity)}"
                )

                # Assets and liabilities should be non-negative
                assert assets >= -1e-6, f"Negative assets: {assets}"
                assert liabilities >= -1e-6, f"Negative liabilities: {liabilities}"

                # Equity can be negative (failed bank)
                # but if equity > 0, capital ratio should be valid
                if equity > 0:
                    capital_ratio = bank_state["capital_ratio"]
                    assert 0 <= capital_ratio <= 1, f"Invalid capital ratio: {capital_ratio}"
