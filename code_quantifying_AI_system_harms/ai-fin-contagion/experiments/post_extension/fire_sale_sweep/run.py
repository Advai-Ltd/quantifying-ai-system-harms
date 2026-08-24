#!/usr/bin/env python3
"""
Post-Extension Fire Sale Parameter Sweep

Tests how fire sale intensity affects Basel III reform effectiveness with
FIXED fire sale mechanics (no compounding, asset-specific markdowns).

RESEARCH QUESTION:
With realistic fire sale implementation, how does fire sale intensity
affect the systemic crisis threshold in the post-2008 reformed system?

BASELINE (from post_extension/post_2008):
- Fire sale intensity: 0.05 (5%)
- Systemic crisis threshold: ~-14% shock
- Reform benefit vs pre-2008: +9pp improvement

EXPERIMENTAL APPROACH:
- Use post-2008 reformed system (Basel III parameters)
- Sweep fire_sale_intensity: 0%, 1%, 3%, 5%, 7%, 10%, 12%, 15%
- For each intensity, sweep shocks from -1% to -30%
- Measure how systemic crisis threshold shifts

INTENSITY INTERPRETATION:
- 0%: No fire sales (counterfactual)
- 1%: Minimal market impact
- 3%: Low market impact
- 5%: Moderate market impact (current baseline)
- 7%: Elevated market stress
- 10%: High market stress
- 12%: Severe market stress
- 15%: Extreme market stress (pre-extension baseline)

USAGE:
    # Full experiment (8 intensities × 29 shocks × 200 runs = 46,400 simulations)
    uv run python experiments/post_extension/fire_sale_sweep/run.py --yes

    # Quick test (2 intensities × 10 shocks × 50 runs)
    uv run python experiments/post_extension/fire_sale_sweep/run.py \\
      --intensities 0.00 0.05 --min-shock -0.05 --max-shock -0.15 --step 0.01 --num-runs 50 --yes
"""

import re
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from numpy import arange

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from financial_contagion_networks.config import load_config
from financial_contagion_networks.simulation.experiment import ExperimentRunner
from experiments.post_extension.thresholds_analysis import analyse_thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Post-Extension Fire Sale Parameter Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')
    
    parser.add_argument('--scenario-table', type=str, default=None,
                        help='Path to CSV with a firesale_term column; one scenario per row, intensities rounded to 3 sig figs (overrides --min-firesale/--max-firesale/--firesale-step)')
    parser.add_argument("--skip-market-structure", type=str, required=False, default=None,
                        help="Skip rows with this market_structure value (e.g. 'Skewed oligopoly')")
    parser.add_argument("--skip-adversarial", default=False, action='store_true',
                        help="Pass argument to skip adversarial firesale values")
    parser.add_argument("--min-firesale", type=float, default=0.05,
                        help="Minimum firesale intensity (ignored if --scenario-table is given)")
    parser.add_argument("--max-firesale", type=float, default=0.25,
                        help="Maximum firesale intensity (ignored if --scenario-table is given)")
    parser.add_argument("--firesale-step", type=float, default=0.025,
                        help="Firesale intensity step size (ignored if --scenario-table is given)")
    parser.add_argument('--min-shock', type=float, default=-0.01,
                        help='Minimum shock level (default: -0.01 = -1%%)')
    parser.add_argument('--max-shock', type=float, default=-0.30,
                        help='Maximum shock level (default: -0.30 = -30%%)')
    parser.add_argument('--step', type=float, default=0.01,
                        help='Shock step size (default: 0.01 = 1%%)')
    parser.add_argument('--num-runs', type=int, default=200,
                        help='Number of Monte Carlo runs per (intensity, shock) pair (default: 200)')
    parser.add_argument('--output-subdir', type=str, default=None,
                        help='Optional subdirectory under output/ (default: timestamp)')
    parser.add_argument('--no-run-output', action='store_true',
                        help='Skip saving per-shock run data (no subdirectories written); only the sweep summary JSON is saved')
    args = parser.parse_args()

    # Validate arguments
    if args.min_shock <= args.max_shock:
        print("ERROR: min-shock must be greater than max-shock")
        return 1
    if args.step <= 0:
        print("ERROR: step must be positive")
        return 1

    if args.scenario_table:
        # Extract scaler and fH from the input filename
        _stem = Path(args.scenario_table).stem
        _m_scaler = re.search(r'scaler_([\.\d]+)', _stem, re.IGNORECASE)
        _m_fh = re.search(r'fh_([\.\d]+)', _stem, re.IGNORECASE)
        scaler_val = _m_scaler.group(1) if _m_scaler else None
        fH_val = _m_fh.group(1) if _m_fh else None

        with open(args.scenario_table, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        scenarios = []
        for row in rows:
            is_adversarial = row.get('adversarial', '').strip().lower() == 'true'
            if args.skip_adversarial and is_adversarial:
                continue
            if args.skip_market_structure and row.get('market_structure', '').lower() == args.skip_market_structure.lower():
                continue
            
            scenario = {
                'scenario_id': int(row['scenario_id']),
                'intensity': float(row['firesale_term']),
                'label': row.get('label', ''),
                'market_structure': row.get('market_structure', ''),
                'ai_adoption_pct': round(100.0 - float(row['Human Share %']), 4),
                'openai_pct': float(row['OpenAI Share %']),
                'google_pct': float(row['Google Share %']),
                'anthropic_pct': float(row['Anthropic Share %']),
                'adversarial': row.get('adversarial', '').strip().lower() == 'true',
            }
            scenarios.append(scenario)
    else:
        scaler_val = None
        fH_val = None
        scenarios = [
            {'intensity': round(x, 4), 'label': None}
            for x in arange(args.min_firesale, args.max_firesale, args.firesale_step)
        ]
    intensities = [s['intensity'] for s in scenarios]

    for intensity in intensities:
        if intensity < 0:
            print(f"ERROR: fire_sale_intensity must be >= 0, got {intensity}")
            return 1

    # Generate shock levels
    shock_levels = []
    shock = args.min_shock
    while shock >= args.max_shock:
        shock_levels.append(shock)
        shock -= args.step

    # Setup paths
    experiment_dir = Path(__file__).parent
    base_config_path = experiment_dir / "config.yaml"

    output_dir = experiment_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scenario_table:
        # Write JSON directly into output/ without creating a subdirectory
        output_base = output_dir
    else:
        output_subdir = args.output_subdir or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_base = output_dir / output_subdir
        output_base.mkdir(parents=True, exist_ok=True)

    # Calculate total simulations
    total_simulations = len(intensities) * len(shock_levels) * args.num_runs
    estimated_time_minutes = total_simulations * 0.02 / 60

    # Print experiment summary
    print("=" * 80)
    print("POST-EXTENSION FIRE SALE PARAMETER SWEEP")
    print("=" * 80)
    print(f"\nRESEARCH QUESTION:")
    print("  How does fire sale intensity affect Basel III reform effectiveness")
    print("  with fixed fire sale mechanics (no compounding)?")
    print(f"\nBASELINE (intensity = 0.05, from post_extension/post_2008):")
    print("  • Systemic crisis threshold: ~-14%% shock")
    print("  • Reform benefit vs pre-2008: +9pp improvement")
    print(f"\nFIRE SALE INTENSITY SWEEP:")
    print(f"  Testing {len(scenarios)} intensity levels")
    if args.scenario_table:
        print(f"  Source: {args.scenario_table}")
        for s in scenarios:
            lbl = f"  ({s['label']})" if s['label'] else ""
            print(f"    {s['intensity']:.3g}{lbl}")
    else:
        print(f"  Intensities: {', '.join(f'{x:.2%}' for x in intensities)}")
        print(f"\n  Interpretation:")
        print(f"    0%%:  No fire sales (counterfactual)")
        print(f"    1%%:  Minimal market impact")
        print(f"    3%%:  Low market impact")
        print(f"    5%%:  Moderate impact (current baseline)")
        print(f"    7%%:  Elevated market stress")
        print(f"    10%%: High market stress")
        print(f"    12%%: Severe market stress")
        print(f"    15%%: Extreme stress (pre-extension baseline)")
    print(f"\nSHOCK SWEEP (per intensity):")
    print(f"  Testing {len(shock_levels)} shock levels")
    print(f"  Range: {args.min_shock:.0%} to {args.max_shock:.0%}")
    print(f"  Step size: {args.step:.1%}")
    print(f"  Monte Carlo runs per (intensity, shock) pair: {args.num_runs}")
    print(f"\nTOTAL COMPUTATIONAL LOAD:")
    print(f"  {len(scenarios)} intensities × {len(shock_levels)} shocks × {args.num_runs} runs")
    print(f"  = {total_simulations:,} simulations")
    print(f"  Estimated time: ~{estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.1f} hours)")

    if not args.yes:
        response = input("\nProceed? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    print("\n" + "=" * 80)
    print("RUNNING FIRE SALE INTENSITY SWEEP")
    print("=" * 80)

    experiment_start = datetime.now()
    all_scenario_results = []

    for scenario_idx, scenario in enumerate(scenarios, 1):
        intensity = scenario['intensity']
        label_str = f" ({scenario['label']})" if scenario['label'] else ""
        print(f"\n{'=' * 80}")
        print(f"SCENARIO {scenario_idx}/{len(scenarios)}: {intensity:.3g}{label_str}".center(80))
        print(f"{'=' * 80}\n")

        intensity_results = []
        intensity_start = datetime.now()

        # Create subdirectory for this intensity level (skipped with --no-run-output or --scenario-table)
        intensity_label = f"scenario_{scenario_idx:03d}_{intensity:.3g}".replace('.', 'p')
        if not args.no_run_output and not args.scenario_table:
            intensity_output = output_base / intensity_label
            intensity_output.mkdir(exist_ok=True)

        for shock_idx, shock in enumerate(shock_levels, 1):
            print(f"[{scenario_idx}/{len(scenarios)}] " +
                  f"[{shock_idx}/{len(shock_levels)}] " +
                  f"Shock: {shock:.1%} | ", end='', flush=True)

            # Load and modify config
            config = load_config(str(base_config_path))

            # Modify fire sale intensity
            config.shock.fire_sale_intensity = intensity

            # Modify shock level (maintain proportional relationships)
            config.shock.asset_shocks['mortgage'] = shock
            config.shock.asset_shocks['corporate_bond'] = shock / 2
            config.shock.asset_shocks['stock'] = shock * 0.75

            # Set number of runs
            config.simulation.num_runs = args.num_runs

            # Set output directory for this (intensity, shock) pair
            shock_label = f"shock_{abs(shock):.0%}".replace('.', 'p')
            if args.no_run_output or args.scenario_table:
                config.output.save_summary = False
                config.output.save_detailed_results = False
                config.output.save_config_copy = False
                config.output.output_dir = str(output_base)  # already exists; mkdir is a no-op
            else:
                config.output.output_dir = str(intensity_output / shock_label)

            # Run experiment
            try:
                runner = ExperimentRunner(config)
                exp_results = runner.run(verbose=False)

                s = exp_results['summary_statistics']

                # Extract key metrics
                result = {
                    'intensity': intensity,
                    'shock': shock,
                    'shock_label': shock_label,
                    'failure_rate_mean': s['failure_rate']['mean'],
                    'failure_rate_std': s['failure_rate']['std'],
                    'failure_rate_median': s['failure_rate']['median'],
                    'contagion_rounds_mean': s['total_rounds']['mean'],
                    'contagion_rounds_std': s['total_rounds']['std'],
                    'contagion_rounds_max': s['total_rounds']['max'],
                    'asset_losses_mean': s['asset_losses']['mean'],
                    'asset_losses_std': s['asset_losses']['std'],
                    'fire_sale_losses_mean': s['fire_sale_losses']['mean'],
                    'fire_sale_losses_std': s['fire_sale_losses']['std'],
                    'systemic_crisis_prob': s['systemic_crisis_probability'],
                    'output_dir': str(config.output.output_dir)
                }
                intensity_results.append(result)

                # Print compact summary
                print(f"FR: {result['failure_rate_mean']:5.1%} ± {result['failure_rate_std']:5.1%} | " +
                      f"Systemic: {result['systemic_crisis_prob']:5.1%}")

            except Exception as e:
                print(f"ERROR: {str(e)}")
                result = {
                    'intensity': intensity,
                    'shock': shock,
                    'shock_label': shock_label,
                    'error': str(e)
                }
                intensity_results.append(result)

        intensity_end = datetime.now()
        intensity_elapsed = (intensity_end - intensity_start).total_seconds()

        # Analyse thresholds for this intensity
        thresholds = analyse_thresholds(intensity_results)

        # Store results for this scenario
        all_scenario_results.append({
            'scenario': scenario,
            'results': intensity_results,
            'thresholds': thresholds,
            'elapsed_seconds': intensity_elapsed
        })

        # Print threshold summary for this scenario
        print(f"\n{'-' * 80}")
        print(f"THRESHOLDS AT INTENSITY {intensity:.3g}:")
        print(f"{'-' * 80}")

        if thresholds:
            if thresholds['systemic_crisis']:
                print(f"  Systemic Crisis (>30%%): {thresholds['systemic_crisis']:.1%} shock")

                # Calculate change relative to first (baseline) scenario
                if scenario_idx == 1:
                    print(f"  (This is the baseline)")
                elif all_scenario_results:
                    _baseline_t = all_scenario_results[0]['thresholds'].get('systemic_crisis')
                    if _baseline_t:
                        change = (thresholds['systemic_crisis'] - _baseline_t) * 100
                        if change > 0:
                            print(f"  Change from baseline: +{change:.1f}pp (more resilient)")
                        else:
                            print(f"  Change from baseline: {change:.1f}pp (less resilient)")

            if thresholds['collapse']:
                print(f"  Collapse (>50%%):        {thresholds['collapse']:.1%} shock")
            if thresholds['stability']:
                print(f"  Stability (>1%%):        {thresholds['stability']:.1%} shock")

        print(f"\n  Completed in {intensity_elapsed / 60:.1f} minutes")

    experiment_end = datetime.now()
    total_elapsed = (experiment_end - experiment_start).total_seconds()

    # Final summary and comparison
    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)
    print("\nSystemic Crisis Threshold (>30%% failure rate) by Fire Sale Intensity:")
    print(f"{'Intensity':<10} {'Threshold':<15} {'Change from baseline':<22} Label")
    print("-" * 80)

    baseline_threshold = None
    if all_scenario_results:
        _bt = all_scenario_results[0]['thresholds']
        if _bt and _bt.get('systemic_crisis'):
            baseline_threshold = _bt['systemic_crisis']

    for entry in all_scenario_results:
        intensity = entry['scenario']['intensity']
        label = entry['scenario']['label'] or f"{intensity:.3g}"
        thresholds = entry['thresholds']
        is_baseline = entry is all_scenario_results[0]

        if thresholds and thresholds.get('systemic_crisis'):
            threshold = thresholds['systemic_crisis']

            if baseline_threshold and not is_baseline:
                change = (threshold - baseline_threshold) * 100
                if change > 0:
                    print(f"{intensity:<10.3g} {threshold:<15.1%} +{change:>6.1f}pp (more resilient)  {label}")
                else:
                    print(f"{intensity:<10.3g} {threshold:<15.1%} {change:>6.1f}pp (less resilient)  {label}")
            else:
                print(f"{intensity:<10.3g} {threshold:<15.1%} {'baseline':<22}  {label}")
        else:
            print(f"{intensity:<10.3g} {'N/A':<15} {'-':<22}  {label}")

    # Save comprehensive summary
    _ts = experiment_start.strftime('%Y%m%d_%H%M%S')
    if args.scenario_table and (scaler_val is not None or fH_val is not None):
        _scaler_part = f"_scaler_{scaler_val}" if scaler_val is not None else ""
        _fH_part = f"_fH_{fH_val}" if fH_val is not None else ""
        summary_filename = f"fire_sale_sweep_summary{_scaler_part}{_fH_part}_{_ts}.json"
    else:
        summary_filename = f"fire_sale_sweep_summary_{_ts}.json"
    summary_file = output_base / summary_filename

    summary = {
        'experiment': 'post_extension_fire_sale_parameter_sweep',
        'description': 'Fire sale intensity sweep on post-2008 system with fixed fire sale mechanics',
        'timestamp': experiment_start.isoformat(),
        'elapsed_seconds': total_elapsed,
        'config': {
            'base_config': str(base_config_path),
            'scenario_table': args.scenario_table,
            'scaler': scaler_val,
            'fH': fH_val,
            'intensities': intensities,
            'shock_range': [args.min_shock, args.max_shock],
            'step_size': args.step,
            'num_runs_per_pair': args.num_runs,
            'total_simulations': total_simulations,
            'fire_sale_compounding': False,
            'heterogeneous_portfolios': True,
            'basel_iii': True
        },
        'baseline_results': {
            'fire_sale_intensity': 0.05,
            'systemic_crisis_threshold': baseline_threshold,
            'pre_2008_systemic_crisis': -0.05,
            'reform_gain': 0.09
        },
        'intensity_results': {}
    }

    seen_keys: set = set()
    for idx, entry in enumerate(all_scenario_results):
        intensity = entry['scenario']['intensity']
        key = str(intensity)
        if key in seen_keys:
            key = f"{key}_{idx}"
        seen_keys.add(key)
        s = entry['scenario']
        summary['intensity_results'][key] = {
            'scenario_id': s.get('scenario_id',''),
            'market_structure': s.get('market_structure', ''),
            'ai_adoption_pct': s.get('ai_adoption_pct'),
            'anthropic_pct': s.get('anthropic_pct'),
            'google_pct': s.get('google_pct'),
            'openai_pct': s.get('openai_pct'),
            'adversarial': s.get('adversarial'),
            'thresholds': entry['thresholds'],
            'elapsed_seconds': entry['elapsed_seconds'],
            'results': entry['results']
        }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_elapsed / 60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"Total simulations: {total_simulations:,}")
    print(f"Results saved to: {summary_file}")
    print(f"Individual results in: {output_base}/")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
