"""
Experiment runner for config-driven simulations.

Provides ExperimentRunner for executing Monte Carlo simulations
based on configuration files, supporting all network/shock modes.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import json
from enum import Enum
import hashlib

from financial_contagion_networks.config import ExperimentConfig
from financial_contagion_networks.simulation.generators import NetworkGenerator, NetworkTemplate
from financial_contagion_networks.core.network import ContagionNetwork
from financial_contagion_networks.core.shocks import ShockScenario, ShockGenerator
from financial_contagion_networks.core.assets import AssetType
from financial_contagion_networks.simulation.runner import monte_carlo_portfolio_simulation


def hash_experiment_config(config: ExperimentConfig, base_seed: int = 42) -> int:
    """
    Generate deterministic seed based on experiment configuration.

    Ensures different parameter combinations get different seeds,
    avoiding identical seeds across parameter sweeps.

    Args:
        config: Experiment configuration
        base_seed: Base random seed (default: 42)

    Returns:
        Unique integer seed based on config parameters

    Usage:
        config = load_config(...)
        config.shock.fire_sale_intensity = 0.5
        config.simulation.seed = hash_experiment_config(config)
    """
    # Collect key parameters that define this experiment variant
    params = {
        'fire_sale_intensity': config.shock.fire_sale_intensity,
        'mortgage_shock': config.shock.asset_shocks.get('mortgage', 0),
        'corporate_shock': config.shock.asset_shocks.get('corporate_bond', 0),
        'stock_shock': config.shock.asset_shocks.get('stock', 0),
        'correlation': config.shock.correlation,
        'shock_volatility': config.shock.shock_volatility,
        'num_banks': config.network.num_banks,
        'num_core_banks': config.network.num_core_banks,
    }

    # Create deterministic hash from parameters
    param_str = json.dumps(params, sort_keys=True)
    hash_obj = hashlib.md5(param_str.encode())
    hash_int = int(hash_obj.hexdigest()[:8], 16)  # Use first 8 hex digits

    # Combine with base seed to allow global seed control
    return (base_seed + hash_int) % (2**31 - 1)  # Keep within valid range


def convert_for_json(obj):
    """
    Recursively convert objects to JSON-serializable format.
    Handles enums (including as dict keys), numpy types, tuples, nested structures.
    """
    if isinstance(obj, dict):
        # Convert dict keys to strings (handles enums, tuples, etc.)
        result = {}
        for k, v in obj.items():
            if isinstance(k, Enum):
                key = k.value
            elif isinstance(k, (tuple, list)):
                key = str(k)  # Convert tuples/lists to strings
            elif isinstance(k, (str, int, float, bool)) or k is None:
                key = k
            else:
                key = str(k)  # Fallback: convert to string
            result[key] = convert_for_json(v)
        return result
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ExperimentRunner:
    """
    Runs Monte Carlo experiments from configuration.

    Supports:
    - All three network modes (fixed, template, stochastic)
    - All three shock modes (deterministic, correlated, uncorrelated)
    - Integration with monte_carlo_portfolio_simulation()
    - Reproducible results with seed management

    Usage:
        config = load_config('experiment.yaml')
        runner = ExperimentRunner(config)
        results = runner.run()
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: Complete experiment configuration
        """
        self.config = config
        config.validate()  # Ensure config is valid

        # Pre-generate network template if in template mode
        self.network_template: Optional[NetworkTemplate] = None
        if config.network.mode == 'template':
            self.network_template = NetworkGenerator.create_template(config.network)

    def run(self, verbose: bool = None) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Args:
            verbose: Override config verbosity setting

        Returns:
            Dictionary with experiment results and metadata
        """
        if verbose is None:
            verbose = self.config.output.verbose

        if verbose:
            self._print_experiment_header()

        # Create shock scenario from config
        shock_scenario = self._create_shock_scenario()

        # Create network generator function
        network_generator_func = self._create_network_generator()

        # Run Monte Carlo simulations
        if verbose:
            print(f"\nRunning {self.config.simulation.num_runs} Monte Carlo simulations...")

        # Prepare network metadata
        network_metadata = {
            'mode': self.config.network.mode,  # Mode is string, not enum
            'topology': self.config.network.topology,
            'structure_seed': self.config.network.structure_seed,
            'parameter_seed': self.config.network.parameter_seed,
            'num_banks': self.config.network.num_banks,
            'num_core_banks': self.config.network.num_core_banks
        }

        simulation_results = monte_carlo_portfolio_simulation(
            network_generator_func=network_generator_func,
            scenario=shock_scenario,
            num_simulations=self.config.simulation.num_runs,
            fire_sales_enabled=self.config.simulation.fire_sales_enabled,
            seed=self.config.simulation.seed,
            use_correlated_shocks=True,  # We handle shock modes in scenario
            shock_volatility=self.config.shock.shock_volatility,
            use_priority_claims=getattr(self.config.simulation, 'use_priority_claims', False),
            num_banks=self.config.network.num_banks,
            num_core_banks=self.config.network.num_core_banks,
            network_metadata=network_metadata
        )

        # Package results with metadata
        results = {
            'metadata': self._create_metadata(),
            'config': self._config_to_dict(),
            'simulation_results': simulation_results,
            'summary_statistics': self._compute_summary_statistics(simulation_results)
        }

        if verbose:
            self._print_results_summary(results)

        # Auto-save results according to config settings
        self.save(results)

        return results

    def save(self, results: Dict[str, Any]) -> None:
        """
        Save experiment results according to config settings.

        Args:
            results: Results dictionary from run()
        """
        output_dir = Path(self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary statistics
        if self.config.output.save_summary:
            summary_file = output_dir / "summary_statistics.json"
            data = convert_for_json({
                'metadata': results['metadata'],
                'summary_statistics': results['summary_statistics']
            })
            with open(summary_file, 'w') as f:
                json.dump(data, f, indent=2)

        # Save detailed simulation results
        if self.config.output.save_detailed_results:
            detailed_file = output_dir / "detailed_results.json"
            data = convert_for_json({
                'simulations': results['simulation_results'],
                'summary_statistics': results['summary_statistics'],
                'metadata': results['metadata']
            })
            with open(detailed_file, 'w') as f:
                json.dump(data, f, indent=2)

        # Save config copy
        if self.config.output.save_config_copy:
            config_file = output_dir / "config.yaml"
            data = convert_for_json(results['config'])
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)

    def _create_shock_scenario(self) -> ShockScenario:
        """
        Create ShockScenario from config.

        Returns:
            ShockScenario instance
        """
        shock_config = self.config.shock

        # Convert asset shock keys from strings to AssetType enums
        asset_shocks = {}
        for asset_name, shock_value in shock_config.asset_shocks.items():
            # Map string names to AssetType enum
            asset_type_map = {
                'government_bond': AssetType.GOVERNMENT_BOND,
                'corporate_bond': AssetType.CORPORATE_BOND,
                'mortgage': AssetType.MORTGAGE,
                'stock': AssetType.STOCK
            }
            asset_type = asset_type_map.get(asset_name)
            if asset_type:
                asset_shocks[asset_type] = shock_value

        return ShockScenario(
            name=self.config.metadata.scenario_name,
            asset_shocks=asset_shocks,
            fire_sale_intensity=shock_config.fire_sale_intensity,
            correlation=shock_config.correlation,
            mode=shock_config.mode,
            shock_volatility=shock_config.shock_volatility,
            description=self.config.metadata.description
        )

    def _create_network_generator(self):
        """
        Create network generator function compatible with monte_carlo_portfolio_simulation.

        The generator function signature must be:
            func(seed=None, **kwargs) -> ContagionNetwork

        Returns:
            Network generator function
        """
        network_config = self.config.network

        # Mode is stored as string (e.g., 'fixed', 'template', 'stochastic')
        if network_config.mode == 'fixed':
            # Fixed mode: Generate once, return same network every time
            fixed_network = NetworkGenerator.from_config(network_config, run_id=0)

            def generator(seed=None, **kwargs):
                return fixed_network

            return generator

        elif network_config.mode == 'template':
            # Template mode: Fixed structure, vary parameters
            template = self.network_template

            def generator(seed=None, **kwargs):
                # Use seed as run_id to get different parameters
                run_id = seed if seed is not None else 0
                return NetworkGenerator.from_config(
                    network_config,
                    run_id=run_id,
                    template=template
                )

            return generator

        elif network_config.mode == 'stochastic':
            # Stochastic mode: New network each time
            def generator(seed=None, **kwargs):
                run_id = seed if seed is not None else 0
                return NetworkGenerator.from_config(
                    network_config,
                    run_id=run_id
                )

            return generator

        else:
            raise ValueError(f"Unknown network mode: {network_config.mode}")

    def _create_metadata(self) -> Dict[str, Any]:
        """
        Create metadata dictionary for results.

        Returns:
            Metadata dictionary
        """
        return {
            'experiment_id': self.config.metadata.experiment_id,
            'scenario_id': self.config.metadata.scenario_id,
            'scenario_name': self.config.metadata.scenario_name,
            'hypothesis': self.config.metadata.hypothesis,
            'description': self.config.metadata.description,
            'tags': self.config.metadata.tags,
            'network_mode': self.config.network.mode,  # Already a string
            'shock_mode': self.config.shock.mode,  # Already a string
            'num_simulations': self.config.simulation.num_runs,
            'seed': self.config.simulation.seed,  # Include seed for reproducibility
            'fire_sale_intensity': self.config.shock.fire_sale_intensity,
            'correlation': self.config.shock.correlation,
            'shock_volatility': self.config.shock.shock_volatility
        }

    def _config_to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary for serialization.

        Returns:
            Config as dictionary
        """
        # Convert config Pydantic models to dicts
        config_dict = {
            'metadata': self.config.metadata.model_dump(),
            'simulation': self.config.simulation.model_dump(),
            'output': self.config.output.model_dump(),
        }

        # Convert network config (more complex due to nested structures)
        network_dict = {
            'mode': self.config.network.mode,  # Mode is string, not enum
            'topology': self.config.network.topology,
            'structure_seed': self.config.network.structure_seed,
            'parameter_seed': self.config.network.parameter_seed,
            'num_banks': self.config.network.num_banks,
            'num_core_banks': self.config.network.num_core_banks
        }
        config_dict['network'] = network_dict

        # Convert shock config
        shock_dict = {
            'mode': self.config.shock.mode,
            'asset_shocks': self.config.shock.asset_shocks,
            'correlation': self.config.shock.correlation,
            'shock_volatility': self.config.shock.shock_volatility,
            'fire_sale_intensity': self.config.shock.fire_sale_intensity,
            'scenario_name': self.config.metadata.scenario_name,
            'description': self.config.metadata.description
        }
        config_dict['shock'] = shock_dict

        return config_dict

    def _analyze_contagion_mechanisms(self, simulation_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze failure mechanisms using three-level contagion hierarchy.

        Level 1 (Top): Initial vs Contagion
        Level 2 (Middle): Direct vs Indirect Contagion
        Level 3 (Implementation): Interbank vs Fire Sale

        Args:
            simulation_results: List of individual simulation results

        Returns:
            Dictionary with contagion mechanism analysis
        """
        total_sims = len(simulation_results)
        num_banks = self.config.network.num_banks
        total_banks = total_sims * num_banks

        # Counters for each failure type
        initial_failures = 0
        interbank_failures = 0  # Direct contagion
        fire_sale_failures = 0  # Indirect contagion

        # Count simulations with each contagion type
        sims_with_any_contagion = 0
        sims_with_direct_contagion = 0
        sims_with_indirect_contagion = 0

        # Analyze each simulation
        for sim in simulation_results:
            # Check if simulation has detailed bank data
            if 'banks' in sim and isinstance(sim['banks'], dict):
                sim_initial = 0
                sim_interbank = 0
                sim_fire_sale = 0

                for bank_data in sim['banks'].values():
                    if 'failure_metadata' in bank_data:
                        cause = bank_data['failure_metadata'].get('failure_cause')

                        if cause == 'initial_shock':
                            sim_initial += 1
                            initial_failures += 1
                        elif cause == 'contagion':
                            sim_interbank += 1
                            interbank_failures += 1
                        elif cause == 'fire_sale':
                            sim_fire_sale += 1
                            fire_sale_failures += 1

                # Track which simulations had contagion
                if sim_interbank > 0 or sim_fire_sale > 0:
                    sims_with_any_contagion += 1
                if sim_interbank > 0:
                    sims_with_direct_contagion += 1
                if sim_fire_sale > 0:
                    sims_with_indirect_contagion += 1

            # Fallback: use total_rounds if detailed data not available
            elif 'total_rounds' in sim and sim['total_rounds'] > 0:
                sims_with_any_contagion += 1

        # Calculate contagion failures
        contagion_failures = interbank_failures + fire_sale_failures
        total_failures = initial_failures + contagion_failures

        return {
            # Level 1: Top-level classification
            'initial_failure_rate': initial_failures / total_banks if total_banks > 0 else 0,
            'contagion_failure_rate': contagion_failures / total_banks if total_banks > 0 else 0,

            # Level 2: Direct vs Indirect contagion
            'direct_contagion_rate': interbank_failures / total_banks if total_banks > 0 else 0,
            'indirect_contagion_rate': fire_sale_failures / total_banks if total_banks > 0 else 0,

            # Level 3: Implementation detail
            'interbank_failure_rate': interbank_failures / total_banks if total_banks > 0 else 0,
            'fire_sale_failure_rate': fire_sale_failures / total_banks if total_banks > 0 else 0,

            # Contagion probability metrics
            'contagion_probability': sims_with_any_contagion / total_sims if total_sims > 0 else 0,
            'direct_contagion_probability': sims_with_direct_contagion / total_sims if total_sims > 0 else 0,
            'indirect_contagion_probability': sims_with_indirect_contagion / total_sims if total_sims > 0 else 0,

            # Absolute counts
            'initial_failures_total': initial_failures,
            'interbank_failures_total': interbank_failures,
            'fire_sale_failures_total': fire_sale_failures,
            'contagion_failures_total': contagion_failures,

            # Composition percentages
            'initial_pct_of_failures': (initial_failures / total_failures * 100) if total_failures > 0 else 0,
            'contagion_pct_of_failures': (contagion_failures / total_failures * 100) if total_failures > 0 else 0,
            'direct_pct_of_failures': (interbank_failures / total_failures * 100) if total_failures > 0 else 0,
            'indirect_pct_of_failures': (fire_sale_failures / total_failures * 100) if total_failures > 0 else 0,
        }

    def _compute_summary_statistics(self, simulation_results: List[Dict]) -> Dict[str, Any]:
        """
        Compute summary statistics from simulation results.

        Args:
            simulation_results: List of individual simulation results

        Returns:
            Dictionary of summary statistics
        """
        # Extract key metrics
        failure_rates = [r['failure_rate'] for r in simulation_results]
        total_rounds = [r['total_rounds'] for r in simulation_results]
        asset_losses = [r['total_asset_losses'] for r in simulation_results]
        fire_sale_losses = [r['fire_sale_losses'] for r in simulation_results]

        # Analyze contagion mechanisms (three-level hierarchy)
        contagion_analysis = self._analyze_contagion_mechanisms(simulation_results)

        return {
            'num_simulations': len(simulation_results),
            'failure_rate': {
                'mean': float(np.mean(failure_rates)),
                'std': float(np.std(failure_rates)),
                'min': float(np.min(failure_rates)),
                'max': float(np.max(failure_rates)),
                'median': float(np.median(failure_rates)),
                'percentile_5': float(np.percentile(failure_rates, 5)),
                'percentile_25': float(np.percentile(failure_rates, 25)),
                'percentile_75': float(np.percentile(failure_rates, 75)),
                'percentile_95': float(np.percentile(failure_rates, 95))
            },
            'total_rounds': {
                'mean': float(np.mean(total_rounds)),
                'std': float(np.std(total_rounds)),
                'min': int(np.min(total_rounds)),
                'max': int(np.max(total_rounds)),
                'median': float(np.median(total_rounds))
            },
            'asset_losses': {
                'mean': float(np.mean(asset_losses)),
                'std': float(np.std(asset_losses)),
                'total': float(np.sum(asset_losses))
            },
            'fire_sale_losses': {
                'mean': float(np.mean(fire_sale_losses)),
                'std': float(np.std(fire_sale_losses)),
                'total': float(np.sum(fire_sale_losses))
            },
            'systemic_crisis_probability': float(np.mean(np.array(failure_rates) > 0.30)),
            # Three-level contagion mechanism analysis
            'contagion_mechanisms': contagion_analysis
        }

    def _print_experiment_header(self):
        """Print experiment information header."""
        print("=" * 80)
        print(f"EXPERIMENT: {self.config.metadata.experiment_id}")
        print(f"SCENARIO: {self.config.metadata.scenario_id} - {self.config.metadata.scenario_name}")
        print("=" * 80)
        if self.config.metadata.hypothesis:
            print(f"\nHypothesis: {self.config.metadata.hypothesis}")
        if self.config.metadata.description:
            print(f"Description: {self.config.metadata.description}")
        print(f"\nNetwork Mode: {self.config.network.mode}")
        print(f"Shock Mode: {self.config.shock.mode}")
        print(f"Number of banks: {self.config.network.num_banks}")
        print(f"Simulations: {self.config.simulation.num_runs}")
        print()

    def _print_results_summary(self, results: Dict[str, Any]):
        """Print results summary."""
        stats = results['summary_statistics']

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"\nFailure Rate:")
        print(f"  Mean: {stats['failure_rate']['mean']:.1%}")
        print(f"  Std:  {stats['failure_rate']['std']:.1%}")
        print(f"  Median: {stats['failure_rate']['median']:.1%}")
        print(f"  95th percentile: {stats['failure_rate']['percentile_95']:.1%}")

        print(f"\nContagion Rounds:")
        print(f"  Mean: {stats['total_rounds']['mean']:.1f}")
        print(f"  Range: {stats['total_rounds']['min']} - {stats['total_rounds']['max']}")

        print(f"\nSystemic Crisis Probability (>30% failures): {stats['systemic_crisis_probability']:.1%}")

        # Three-level contagion mechanism analysis
        if 'contagion_mechanisms' in stats:
            cm = stats['contagion_mechanisms']

            print(f"\nContagion Mechanism Analysis:")
            print(f"  Level 1 - Initial vs Contagion:")
            print(f"    Initial failures:   {cm['initial_failure_rate']:.1%} "
                  f"({cm['initial_pct_of_failures']:.1f}% of all failures)")
            print(f"    Contagion failures: {cm['contagion_failure_rate']:.1%} "
                  f"({cm['contagion_pct_of_failures']:.1f}% of all failures)")

            print(f"\n  Level 2 - Direct vs Indirect Contagion:")
            print(f"    Direct contagion:   {cm['direct_contagion_rate']:.1%} "
                  f"({cm['direct_pct_of_failures']:.1f}% of all failures)")
            print(f"    Indirect contagion: {cm['indirect_contagion_rate']:.1%} "
                  f"({cm['indirect_pct_of_failures']:.1f}% of all failures)")

            print(f"\n  Level 3 - Implementation Detail:")
            print(f"    Interbank failures: {cm['interbank_failure_rate']:.1%}")
            print(f"    Fire sale failures: {cm['fire_sale_failure_rate']:.1%}")

            print(f"\n  Contagion Probability (% of simulations):")
            print(f"    Any contagion:      {cm['contagion_probability']:.1%}")
            print(f"    Direct contagion:   {cm['direct_contagion_probability']:.1%}")
            print(f"    Indirect contagion: {cm['indirect_contagion_probability']:.1%}")

        print("=" * 80 + "\n")


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run an experiment from config.

    Args:
        config: Experiment configuration
        verbose: Print progress and results

    Returns:
        Results dictionary
    """
    runner = ExperimentRunner(config)
    return runner.run(verbose=verbose)
