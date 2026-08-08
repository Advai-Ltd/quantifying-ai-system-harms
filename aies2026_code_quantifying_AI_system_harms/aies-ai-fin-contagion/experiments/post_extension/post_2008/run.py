#!/usr/bin/env python3
"""
Post-Extension Post-2008 Shock Sweep

Sweeps mortgage shocks from -1% to -30% to identify critical thresholds
with FIXED fire sale settings AND Basel III reforms.

IMPROVEMENTS VS PRE-EXTENSION:
- Fire sale intensity: 0.05 (vs 0.15 baseline)
- Fire sale compounding: DISABLED
- Heterogeneous portfolios (Option F)
- Variable interbank exposure (Option A)
- Stochastic topology (Option D)
- Basel III reforms (higher capital, lower connectivity)

EXPECTED: Requires larger shocks than pre-2008 to cause failures

USAGE:
    uv run python experiments/post_extension/post_2008/run.py --yes
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from financial_contagion_networks.config import load_config
from financial_contagion_networks.simulation.experiment import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Post-Extension Post-2008 Shock Sweep")
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    parser.add_argument('--min-shock', type=float, default=-0.01, help='Min shock (default: -0.01)')
    parser.add_argument('--max-shock', type=float, default=-0.30, help='Max shock (default: -0.30)')
    parser.add_argument('--step', type=float, default=0.01, help='Step (default: 0.01)')
    parser.add_argument('--num-runs', type=int, default=200, help='Runs per shock (default: 200)')
    parser.add_argument('--output-subdir', type=str, default=None, help='Output subdir (default: timestamp)')
    args = parser.parse_args()

    if args.min_shock <= args.max_shock:
        print("ERROR: min-shock must be greater than max-shock")
        return 1
    if args.step <= 0:
        print("ERROR: step must be positive")
        return 1

    shock_levels = []
    shock = args.min_shock
    while shock >= args.max_shock:
        shock_levels.append(shock)
        shock -= args.step

    experiment_dir = Path(__file__).parent
    base_config_path = experiment_dir / "config.yaml"

    output_subdir = args.output_subdir or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base = experiment_dir / "output" / output_subdir
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("POST-EXTENSION POST-2008: SHOCK SWEEP")
    print("=" * 80)
    print(f"\nShock levels: {len(shock_levels)} ({args.min_shock:.0%} to {args.max_shock:.0%})")
    print(f"Runs per shock: {args.num_runs}")
    print(f"Expected: More resilient than pre-2008 (Basel III reforms)")
    print(f"Estimated time: ~{len(shock_levels) * args.num_runs * 0.02 / 60:.1f} minutes")

    if not args.yes:
        if input("\nProceed? (y/n): ").lower() != 'y':
            return 0

    results = []
    start_time = datetime.now()

    for i, shock in enumerate(shock_levels, 1):
        print(f"\n[{i}/{len(shock_levels)}] Shock: {shock:.1%}")
        print("-" * 80)

        config = load_config(str(base_config_path))
        config.shock.asset_shocks['mortgage'] = shock
        config.shock.asset_shocks['corporate_bond'] = shock / 2
        config.shock.asset_shocks['stock'] = shock * 0.75
        config.simulation.num_runs = args.num_runs

        shock_label = f"shock_{abs(shock):.0%}".replace('.', 'p')
        config.output.output_dir = str(output_base / shock_label)

        try:
            runner = ExperimentRunner(config)
            exp_results = runner.run(verbose=False)
            s = exp_results['summary_statistics']

            result = {
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
            results.append(result)

            print(f"  ✓ FR: {result['failure_rate_mean']:.1%} ± {result['failure_rate_std']:.1%}")
            print(f"  ✓ Systemic: {result['systemic_crisis_prob']:.1%}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({'shock': shock, 'shock_label': shock_label, 'error': str(e)})

    elapsed = (datetime.now() - start_time).total_seconds()

    # Threshold analysis
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        print("\nERROR: No valid results")
        return 1

    first_failure = next((r for r in valid_results if r['failure_rate_mean'] > 0.01), None)
    contagion_start = next((r for r in valid_results if r['contagion_rounds_mean'] > 0.5), None)
    systemic_crisis = next((r for r in valid_results if r['failure_rate_mean'] > 0.30), None)
    collapse = next((r for r in valid_results if r['failure_rate_mean'] > 0.50), None)

    critical_point = None
    if len(valid_results) > 2:
        max_derivative = 0
        for i in range(1, len(valid_results)):
            derivative = abs(valid_results[i]['failure_rate_mean'] - valid_results[i-1]['failure_rate_mean'])
            if derivative > max_derivative:
                max_derivative = derivative
                critical_point = valid_results[i]

    if first_failure:
        print(f"\n1. STABILITY: ~{first_failure['shock']:.1%} shock")
        print(f"   FR: {first_failure['failure_rate_mean']:.1%}")
    if contagion_start:
        print(f"\n2. CONTAGION: ~{contagion_start['shock']:.1%} shock")
        print(f"   Rounds: {contagion_start['contagion_rounds_mean']:.2f}")
    if critical_point:
        print(f"\n3. CRITICAL: ~{critical_point['shock']:.1%} shock")
        print(f"   FR: {critical_point['failure_rate_mean']:.1%}")
    if systemic_crisis:
        print(f"\n4. SYSTEMIC CRISIS: ~{systemic_crisis['shock']:.1%} shock")
        print(f"   FR: {systemic_crisis['failure_rate_mean']:.1%}")
    if collapse:
        print(f"\n5. COLLAPSE: ~{collapse['shock']:.1%} shock")
        print(f"   FR: {collapse['failure_rate_mean']:.1%}")

    # Save summary
    summary = {
        'experiment': 'post_extension_post_2008_shock_sweep',
        'description': 'Post-extension post-2008 with fixed fire sales and Basel III',
        'timestamp': start_time.isoformat(),
        'elapsed_seconds': elapsed,
        'config': {
            'base_config': str(base_config_path),
            'shock_range': [args.min_shock, args.max_shock],
            'step_size': args.step,
            'num_runs_per_shock': args.num_runs,
            'fire_sale_intensity': 0.05,
            'fire_sale_compounding': False,
            'heterogeneous_portfolios': True,
            'basel_iii': True
        },
        'thresholds': {
            'stability': first_failure['shock'] if first_failure else None,
            'contagion': contagion_start['shock'] if contagion_start else None,
            'critical': critical_point['shock'] if critical_point else None,
            'systemic_crisis': systemic_crisis['shock'] if systemic_crisis else None,
            'collapse': collapse['shock'] if collapse else None
        },
        'results': results
    }

    summary_file = output_base / "shock_sweep_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print("=" * 80)
    print(f"Time: {elapsed / 60:.1f} minutes")
    print(f"Summary: {summary_file}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
