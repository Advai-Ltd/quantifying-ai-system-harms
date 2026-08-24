#!/usr/bin/env python3
"""
Compare Post-Extension Pre-2008 vs Post-2008 Reform Effectiveness

Generates comparison graphs showing Basel III reform impact WITH FIXED FIRE SALES.

This comparison uses:
- Fire sale intensity: 0.05 (fixed, asset-specific)
- Fire sale compounding: DISABLED
- Heterogeneous portfolios (Option F)
- Asset-specific fire sales

Expected: Post-2008 shows similar improvement pattern but with realistic partial failures.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_results(experiment_dir):
    """Load latest experiment results."""
    output_dir = Path(experiment_dir) / "output"

    # Find most recent run
    run_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                     key=lambda x: x.name, reverse=True)

    if not run_dirs:
        raise FileNotFoundError(f"No results found in {output_dir}")

    latest_run = run_dirs[0]
    summary_file = latest_run / "shock_sweep_summary.json"

    with open(summary_file, 'r') as f:
        data = json.load(f)

    return data, latest_run


def plot_failure_curves(pre_data, post_data, output_dir):
    """Plot failure rate curves for both systems."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract data
    pre_shocks = [r['shock'] * 100 for r in pre_data['results'] if 'error' not in r]
    pre_failures = [r['failure_rate_mean'] * 100 for r in pre_data['results'] if 'error' not in r]
    pre_std = [r['failure_rate_std'] * 100 for r in pre_data['results'] if 'error' not in r]

    post_shocks = [r['shock'] * 100 for r in post_data['results'] if 'error' not in r]
    post_failures = [r['failure_rate_mean'] * 100 for r in post_data['results'] if 'error' not in r]
    post_std = [r['failure_rate_std'] * 100 for r in post_data['results'] if 'error' not in r]

    # Plot with confidence bands
    ax.plot(pre_shocks, pre_failures, 'o-', linewidth=2.5, markersize=6,
            color='#d62728', label='Pre-2008 (5-9% capital)', zorder=3)
    ax.fill_between(pre_shocks,
                    np.array(pre_failures) - np.array(pre_std),
                    np.array(pre_failures) + np.array(pre_std),
                    alpha=0.2, color='#d62728')

    ax.plot(post_shocks, post_failures, 's-', linewidth=2.5, markersize=6,
            color='#2ca02c', label='Post-2008 (10-13% capital)', zorder=3)
    ax.fill_between(post_shocks,
                    np.array(post_failures) - np.array(post_std),
                    np.array(post_failures) + np.array(post_std),
                    alpha=0.2, color='#2ca02c')

    # Mark critical thresholds
    pre_crisis = pre_data['thresholds'].get('systemic_crisis')
    post_crisis = post_data['thresholds'].get('systemic_crisis')

    ax.axhline(y=30, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Systemic Crisis (30%)')

    if pre_crisis:
        ax.axvline(x=pre_crisis * 100, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.7)
    if post_crisis:
        ax.axvline(x=post_crisis * 100, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.7)

    # Add improvement annotation
    if pre_crisis and post_crisis:
        improvement = abs(post_crisis - pre_crisis) * 100
        ax.annotate(f'+{improvement:.0f}pp\nimprovement',
                    xy=((pre_crisis + post_crisis) * 50, 35),
                    fontsize=12, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Mortgage Shock (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Bank Failure Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Basel III Reform Effectiveness (WITH FIXED FIRE SALES)\n' +
                 'Post-Extension: 0.05 fire sales, no compounding, heterogeneous portfolios',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(pre_shocks) - 1, max(pre_shocks) + 1)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'failure_rate_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'failure_rate_comparison.png'}")
    plt.close()


def main():
    print("=" * 80)
    print("POST-EXTENSION BASEL III REFORM COMPARISON")
    print("=" * 80)

    # Load results
    print("\nLoading experiment results...")
    pre_data, pre_run = load_results("experiments/post_extension/pre_2008")
    post_data, post_run = load_results("experiments/post_extension/post_2008")

    print(f"  ✓ Pre-2008 results: {pre_run.name}")
    print(f"  ✓ Post-2008 results: {post_run.name}")

    # Create output directory
    output_dir = Path("experiments/post_extension/comparison_graphs")
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating comparison graphs...")

    # Generate graphs
    plot_failure_curves(pre_data, post_data, output_dir)

    # Print threshold comparison
    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)

    thresholds = ['stability', 'contagion', 'systemic_crisis', 'collapse']
    for thresh in thresholds:
        pre_val = pre_data['thresholds'].get(thresh)
        post_val = post_data['thresholds'].get(thresh)

        if pre_val and post_val:
            improvement = abs(post_val - pre_val) * 100
            print(f"\n{thresh.upper()}:")
            print(f"  Pre-2008:  {abs(pre_val)*100:.0f}% shock")
            print(f"  Post-2008: {abs(post_val)*100:.0f}% shock")
            print(f"  Improvement: +{improvement:.0f}pp")

    print(f"\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nGraphs saved to: {output_dir.absolute()}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
