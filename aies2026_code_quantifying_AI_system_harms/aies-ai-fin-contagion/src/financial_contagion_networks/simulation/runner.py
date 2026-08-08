"""
Advanced simulation with portfolio shocks and fire sales.

Extends basic contagion with:
- Portfolio composition (different asset types)
- Correlated asset shocks
- Fire sale contagion
- Monte Carlo simulations with parameter sweeps
"""

from typing import Dict, List, Optional
import numpy as np

from financial_contagion_networks.core.bank import Bank, BankStatus
from financial_contagion_networks.core.network import ContagionNetwork
from financial_contagion_networks.core.assets import AssetType
from financial_contagion_networks.core.shocks import ShockScenario, ShockGenerator


def simulate_with_portfolio_shocks(
    network: ContagionNetwork,
    scenario: ShockScenario,
    fire_sales_enabled: bool = True,
    verbose: bool = False,
    actual_shocks: Optional[Dict[AssetType, float]] = None,
    num_core_banks: Optional[int] = None,
    network_metadata: Optional[Dict] = None,
    use_priority_claims: bool = False
) -> Dict:
    """
    Simulate contagion with portfolio shocks and fire sales.

    Process:
    1. Apply initial asset shocks to all banks
    2. Mark insolvent banks as failed
    3. Propagate interbank contagion
    4. Apply fire sale effects (if enabled)
    5. Check for additional failures

    Args:
        network: ContagionNetwork with banks that have portfolios
        scenario: ShockScenario defining asset shocks
        fire_sales_enabled: Whether to model fire sale contagion
        verbose: Print detailed output
        actual_shocks: Optional dict of actual shocks to apply (if None, uses scenario.asset_shocks)
        num_core_banks: Optional number of core banks (for labeling core vs periphery)
        network_metadata: Optional dict with network generation parameters (topology, mode, etc.)
        use_priority_claims: If True, interbank creditors are junior to external creditors

    Returns:
        Dict with simulation results
    """
    # Use provided shocks or fall back to scenario's fixed shocks
    shocks_to_apply = actual_shocks if actual_shocks is not None else scenario.asset_shocks

    if verbose:
        print(f"Simulating scenario: {scenario.name}")
        print(f"Asset shocks: {shocks_to_apply}")
        print(f"Fire sales enabled: {fire_sales_enabled}")
        print()

    # CRITICAL DATA: Capture initial state before applying shocks
    # This allows analysis of shock impact (initial → final state)
    initial_state = {}
    for bank_id, bank in network.banks.items():
        bank_data = bank.to_dict()
        # Add bank type (core vs periphery) if num_core_banks is provided
        if num_core_banks is not None:
            bank_data['bank_type'] = 'core' if bank_id < num_core_banks else 'periphery'
        initial_state[bank_id] = bank_data

    # CRITICAL DATA: Capture exposure matrices at initial state
    # Essential for understanding network structure and contagion pathways
    exposure_matrices = network.get_exposure_matrices()

    # CRITICAL DATA: Capture shock generation details
    # This helps understand shock correlation and reproducibility
    shock_metadata = {
        'mode': scenario.mode.value if hasattr(scenario.mode, 'value') else str(scenario.mode),
        'correlation': scenario.correlation,
        'fire_sale_intensity': scenario.fire_sale_intensity,
        'shock_volatility': scenario.shock_volatility
    }

    # Phase 1: Apply asset shocks to all banks
    if verbose:
        print("Phase 1: Applying asset shocks...")

    # CRITICAL DATA: Track detailed asset shock application
    asset_shock_details = []
    total_asset_losses = 0.0
    banks_with_portfolios = [b for b in network.banks.values() if b.has_portfolio()]

    for bank in banks_with_portfolios:
        # Capture state before shock
        capital_before = bank.get_capital_ratio()
        portfolio_composition = bank.get_portfolio_composition()

        # Apply shock
        loss = bank.apply_portfolio_shocks(shocks_to_apply, failure_round=0)
        total_asset_losses += loss

        # Capture state after shock
        capital_after = bank.get_capital_ratio()
        caused_failure = (bank.get_status() == BankStatus.FAILED)

        # Record detailed information
        shock_detail = {
            'bank_id': bank.bank_id,
            'portfolio_composition': {
                asset_type.value: weight
                for asset_type, weight in portfolio_composition.items()
            } if portfolio_composition else {},
            'shocks_applied': {
                asset_type.value: shock
                for asset_type, shock in shocks_to_apply.items()
            },
            'losses_by_asset_type': dict(bank.loss_tracking['asset_shocks']),
            'total_loss': loss,
            'capital_before': capital_before,
            'capital_after': capital_after,
            'caused_failure': caused_failure
        }
        asset_shock_details.append(shock_detail)

        if verbose and loss > 0:
            print(f"  Bank {bank.bank_id}: Lost ${loss:.2f} from asset shocks")

    # Identify initially failed banks (from asset shocks)
    # Banks auto-update their status during shock application, so we just check final state
    initially_failed = []
    for bank in network.banks.values():
        if bank.get_status() == BankStatus.FAILED:
            initially_failed.append(bank.bank_id)

    if verbose:
        print(f"\nInitially failed banks: {initially_failed}")
        print(f"Total asset losses: ${total_asset_losses:.2f}")
        print()

    # Phase 2: Interbank contagion with integrated fire sales
    if verbose:
        print("Phase 2: Propagating interbank contagion with fire sales...")

    # Fire sales integrated into each contagion round
    # This is economically correct - fire sales happen DURING rounds, not after
    if fire_sales_enabled:
        contagion_result = network.propagate_contagion_with_fire_sales(
            failed_bank_ids=initially_failed,
            fire_sale_intensity=scenario.fire_sale_intensity,
            use_priority_claims=use_priority_claims,
            verbose=verbose
        )

        # Extract fire sale data from integrated results
        fire_sale_losses = contagion_result['total_fire_sale_losses']
        fire_sale_failures = contagion_result['fire_sale_failed_bank_ids']

        # Construct fire_sale_details from round-by-round data
        fire_sale_details = []
        for round_data in contagion_result['fire_sales_by_round']:
            fire_sale_details.extend(round_data['details'])
    else:
        # No fire sales - use old method
        contagion_result = network.propagate_contagion_from_failed(
            failed_bank_ids=initially_failed,
            use_priority_claims=use_priority_claims
        )

        fire_sale_losses = 0.0
        fire_sale_failures = []
        fire_sale_details = []

    # Calculate final statistics
    final_stats = network.get_network_statistics()

    # CRITICAL FIX: Get actual failed banks from final network state
    # The contagion tracking was broken - this ensures we capture reality
    actual_failed_bank_ids = [
        bank_id for bank_id, bank in network.banks.items()
        if bank.get_status() == BankStatus.FAILED
    ]

    # CRITICAL DATA: Capture network structure (who lends to whom)
    # This is essential for network visualization and contagion pathway analysis
    network_structure = {
        'edges': [
            {
                'from_bank': int(from_id),
                'to_bank': int(to_id),
                'exposure': float(network.graph[from_id][to_id]['weight'])
            }
            for from_id, to_id in network.graph.edges()
        ],
        'num_banks': len(network.banks),
        'num_edges': network.graph.number_of_edges()
    }

    # CRITICAL DATA: Capture core/periphery labels
    # Banks 0 to num_core_banks-1 are core, rest are periphery
    bank_labels = None
    if num_core_banks is not None:
        bank_labels = {
            'core_banks': list(range(num_core_banks)),
            'periphery_banks': list(range(num_core_banks, len(network.banks))),
            'num_core': num_core_banks,
            'num_periphery': len(network.banks) - num_core_banks
        }

    # Extract bank-level data including failure_metadata for contagion analysis
    banks_data = {}
    for bank_id, bank in network.banks.items():
        banks_data[bank_id] = {
            'failed': bank.get_status() == BankStatus.FAILED,
            'failure_metadata': bank.failure_metadata
        }

    result = {
        'scenario': scenario.name,
        'asset_shocks': shocks_to_apply,  # Use actual applied shocks, not scenario base
        'asset_shock_details': asset_shock_details,  # Per-bank shock application details
        'total_asset_losses': total_asset_losses,
        'initially_failed': initially_failed,
        'contagion_failures': len(actual_failed_bank_ids) - len(initially_failed),
        'fire_sale_losses': fire_sale_losses,
        'fire_sale_details': fire_sale_details,  # Per-bank fire sale details
        'fire_sale_failures': fire_sale_failures,
        'total_failed': len(actual_failed_bank_ids),
        'failed_bank_ids': actual_failed_bank_ids,
        'failure_rate': final_stats['failure_rate'],
        'total_rounds': contagion_result['total_rounds'],
        'contagion_history': contagion_result['contagion_history'],
        'network_structure': network_structure,  # Network topology
        'initial_state': initial_state,  # Pre-shock state
        'exposure_matrices': exposure_matrices,  # Exposure/obligation matrices
        'shock_metadata': shock_metadata,  # Shock generation details
        'banks': banks_data  # Bank-level data with failure_metadata
    }

    # Add bank labels if available
    if bank_labels is not None:
        result['bank_labels'] = bank_labels

    # Add network metadata if available
    if network_metadata is not None:
        result['network_metadata'] = network_metadata

    if verbose:
        print(f"\n{'='*60}")
        print(f"Final Results:")
        print(f"  Total failures: {result['total_failed']} / {len(network.banks)}")
        print(f"  Failure rate: {result['failure_rate']:.1%}")
        print(f"  Asset losses: ${result['total_asset_losses']:.2f}")
        print(f"  Fire sale losses: ${result['fire_sale_losses']:.2f}")
        print(f"{'='*60}\n")

    return result


def monte_carlo_portfolio_simulation(
    network_generator_func,
    scenario: ShockScenario,
    num_simulations: int,
    fire_sales_enabled: bool = True,
    seed: Optional[int] = None,
    use_correlated_shocks: bool = True,
    shock_volatility: float = 0.03,
    use_priority_claims: bool = False,
    **network_kwargs
) -> List[Dict]:
    """
    Run Monte Carlo simulations with portfolio shocks.

    Args:
        network_generator_func: Function to generate networks
        scenario: ShockScenario to apply
        num_simulations: Number of simulations
        fire_sales_enabled: Enable fire sale contagion
        seed: Random seed
        use_correlated_shocks: If True, generate random correlated shocks around scenario means
                               If False, use fixed shocks from scenario
        shock_volatility: Standard deviation of shock randomness (default 3%)
        use_priority_claims: If True, interbank creditors are junior to external creditors
        **network_kwargs: Arguments for network generator

    Returns:
        List of simulation results
    """
    if seed is not None:
        np.random.seed(seed)

    results = []

    # Extract num_core_banks from network_kwargs if available
    num_core_banks = network_kwargs.get('num_core_banks', None)

    # Extract network_metadata from network_kwargs if available
    network_metadata = network_kwargs.get('network_metadata', None)

    # Create shock generator for correlated random shocks
    shock_generator = ShockGenerator(seed=seed)

    for sim_id in range(num_simulations):
        # Generate network
        network = network_generator_func(seed=seed + sim_id if seed else None, **network_kwargs)

        # Generate correlated random shocks for this run (if enabled)
        if use_correlated_shocks:
            actual_shocks = shock_generator.generate_correlated_shocks_from_scenario(
                scenario,
                volatility=shock_volatility
            )
        else:
            actual_shocks = None  # Will use scenario's fixed shocks

        # Run simulation
        result = simulate_with_portfolio_shocks(
            network,
            scenario,
            fire_sales_enabled=fire_sales_enabled,
            verbose=False,
            actual_shocks=actual_shocks,
            num_core_banks=num_core_banks,
            network_metadata=network_metadata,
            use_priority_claims=use_priority_claims
        )

        result['simulation_id'] = sim_id
        results.append(result)

    return results


def compare_scenarios(
    network_generator_func,
    scenarios: List[ShockScenario],
    num_simulations: int = 100,
    fire_sales_enabled: bool = True,
    seed: Optional[int] = None,
    **network_kwargs
) -> Dict[str, List[Dict]]:
    """
    Compare multiple shock scenarios using Monte Carlo.

    Args:
        network_generator_func: Function to generate networks
        scenarios: List of scenarios to compare
        num_simulations: Number of simulations per scenario
        fire_sales_enabled: Enable fire sale contagion
        seed: Random seed
        **network_kwargs: Arguments for network generator

    Returns:
        Dict mapping scenario names to results
    """
    comparison = {}

    for scenario in scenarios:
        print(f"Running {num_simulations} simulations for: {scenario.name}")
        results = monte_carlo_portfolio_simulation(
            network_generator_func,
            scenario,
            num_simulations,
            fire_sales_enabled=fire_sales_enabled,
            seed=seed,
            **network_kwargs
        )
        comparison[scenario.name] = results

        # Print summary
        failure_rates = [r['failure_rate'] for r in results]
        print(f"  Mean failure rate: {np.mean(failure_rates):.2%}")
        print(f"  Std failure rate: {np.std(failure_rates):.2%}")
        print(f"  Max failure rate: {np.max(failure_rates):.2%}")
        print()

    return comparison
