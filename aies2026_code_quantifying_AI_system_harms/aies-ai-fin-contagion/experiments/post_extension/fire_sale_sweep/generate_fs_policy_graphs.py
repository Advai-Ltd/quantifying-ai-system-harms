#!/usr/bin/env python3
"""
Generate policy-focused visualizations for fire sale parameter sweep.

This script creates comprehensive graphs demonstrating the multi-dimensional
danger of fire sales beyond simple threshold shifts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Optional, List
import pandas as pd
 

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

def load_data(summary_file):
    """Load the fire sale sweep summary data."""
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # Clean up floating-point precision in intensity_results keys
    if 'intensity_results' in data:
        cleaned_results = {}
        for key, value in data['intensity_results'].items():
            # Round to 4 decimal places and format as string
            rounded_intensity = round(float(key), 4)
            cleaned_key = f"{rounded_intensity:.4f}".rstrip('0').rstrip('.')
            cleaned_results[cleaned_key] = value
        data['intensity_results'] = cleaned_results
    
    # Also clean the intensities list in config if present
    if 'config' in data and 'intensities' in data['config']:
        data['config']['intensities'] = [round(x, 4) for x in data['config']['intensities']]
    
    return data

def illustrative_sri_shock_thresholds(data, output_dir, threshold: int = 30,
                                      output_filename: Optional[str] = None) -> None:
    """
    Create illustrative figure with red/amber/green operating zones for a single
    bank failure rate threshold, showing how fire sale intensity erodes resilience.

    Green  = intensities where the shock threshold hasn't moved from the baseline.
    Amber  = intensities where partial but not maximum erosion has occurred.
    Red    = intensities where the shock threshold has reached its worst value.
    """
    intensities = sorted(data['config']['intensities'])

    # Compute shock threshold (most lenient shock that still triggers ≥ threshold% failures)
    shock_values = []
    for intensity in intensities:
        intensity_data = data['intensity_results'][str(intensity)]
        results = intensity_data['results']

        min_shock = None
        for result in sorted(results, key=lambda x: x.get('shock', float('inf')), reverse=True):
            if 'error' not in result and 'failure_rate_mean' in result:
                if result['failure_rate_mean'] * 100 >= threshold:
                    min_shock = abs(result['shock']) * 100
                    break
        shock_values.append(min_shock)

    valid_pairs = [(i * 100, s) for i, s in zip(intensities, shock_values) if s is not None]
    if not valid_pairs:
        print("⚠ No valid data for illustrative SRI shock thresholds plot")
        return

    valid_intensities, valid_shocks = map(list, zip(*valid_pairs))

    baseline_shock = max(valid_shocks)   # most resilient = green baseline
    worst_shock    = min(valid_shocks)   # most fragile = red ceiling

    # Green → Amber boundary: last intensity still at baseline
    green_end = valid_intensities[0]
    for intensity, shock in zip(valid_intensities, valid_shocks):
        if shock >= baseline_shock:
            green_end = intensity
        else:
            break

    # Amber → Red boundary: first intensity that has reached worst degradation
    red_start = valid_intensities[-1]
    for intensity, shock in zip(valid_intensities, valid_shocks):
        if shock <= worst_shock:
            red_start = intensity
            break

    x_min = valid_intensities[0]
    x_max = valid_intensities[-1]

    fig, ax = plt.subplots(figsize=(11, 7))

    # Operating zone fills
    if green_end > x_min:
        ax.axvspan(x_min, green_end, alpha=0.18, color='green', label='Safe Zone (no erosion)')
    if green_end < red_start:
        ax.axvspan(green_end, red_start, alpha=0.18, color='orange', label='Caution Zone')
    if red_start < x_max:
        ax.axvspan(red_start, x_max, alpha=0.18, color='red', label='Danger Zone (maximum erosion)')

    # Zone boundary lines
    if green_end > x_min:
        ax.axvline(x=green_end, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    if red_start < x_max and red_start != green_end:
        ax.axvline(x=red_start, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Main threshold line
    ax.plot(valid_intensities, valid_shocks, '-o', color='black',
            linewidth=2.5, markersize=6, zorder=5,
            label=f'{threshold}% Bank Failure Rate Threshold')

    ax.set_xlabel('Fire Sale Intensity (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Minimum Mortgage Shock Required (%)', fontweight='bold', fontsize=12)
    ax.set_title(
        f'Systemic Risk Indicator: Shock Resistance at {threshold}% Bank Failure Rate\n'
        '(Green = Safe, Amber = Caution, Red = Danger)',
        fontweight='bold', fontsize=13, pad=15)

    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.98, 'More Resilient ↑', transform=ax.transAxes,
            fontsize=10, va='top', ha='left')
    ax.text(0.02, 0.05, 'More Fragile ↓', transform=ax.transAxes,
            fontsize=10, va='bottom', ha='left')

    plt.tight_layout()

    if output_filename is None:
        output_filename = f'illustrative_sri_shock_thresholds_{threshold}pct.png'

    output_path = Path(output_dir) / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_threshold_shocks_vs_firesale_plot(data, output_dir, thresholds=[10, 30, 50],
                                            output_filename: Optional[str] = None,
                                            ai_scenario_file: Optional[str] = None,
                                            scenarios_to_plot: Optional[List[str]] = None) -> None:

    intensities = sorted(data['config']['intensities'])
    
    n_thresholds = len(thresholds)
    cmap = plt.cm.plasma
    colors = [cmap(i / (n_thresholds - 1)) for i in range(n_thresholds)]
    
    plt.figure(figsize=(10, 7))
    
    for i, threshold in enumerate(thresholds):
        shock_values = []
        
        for intensity in intensities:
            intensity_data = data['intensity_results'][str(intensity)]
            results = intensity_data['results']
            
            # Find minimum shock where failure_rate_mean >= threshold
            min_shock = None
            for result in sorted(results, key=lambda x: x.get('shock', float('inf')),reverse=True):
                if 'error' not in result and 'failure_rate_mean' in result:
                    if result['failure_rate_mean'] * 100 >= threshold:
                        min_shock = abs(result['shock']) * 100  # Convert to positive percentage
                        break
            
            shock_values.append(min_shock)
        
        # print(threshold, shock_values)
        # Plot line with markers
        valid_intensities = []
        valid_shocks = []
        for j, shock in enumerate(shock_values):
            if shock is not None:
                valid_intensities.append(intensities[j] * 100)  # Convert to percentage
                valid_shocks.append(shock)
        
        if valid_shocks:  # Only plot if we have valid data
            plt.plot(valid_intensities, valid_shocks, '-o', 
                    color=colors[i % len(colors)], 
                    linewidth=2.5, 
                    markersize=6,
                    label=f'{threshold}% Bank Failure Rate')
        
    if ai_scenario_file is not None:   
        scenarios_file = Path(ai_scenario_file)
        if scenarios_file.exists():
            df = pd.read_csv(scenarios_file)
            labeled_scenarios = df[df['label'].notna() & (df['label'].str.strip() != '')]
            if scenarios_to_plot is not None:
                labeled_scenarios = labeled_scenarios[labeled_scenarios['label'].str.strip("'").isin(scenarios_to_plot)]
            
            for _, row in labeled_scenarios.iterrows():
                fire_sale_intensity = row['firesale_term'] * 100  # Convert to percentage
                label = row['label'].strip("'")  # Remove quotes if present
                
                # Add vertical dashed line
                plt.axvline(x=fire_sale_intensity, color='gray', linestyle='--', 
                        alpha=0.7, linewidth=1.5)
                
                y_pos = plt.gca().get_ylim()[0] + (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) * 0.95
                plt.text(fire_sale_intensity, y_pos, label, 
                        rotation=90, verticalalignment='top', horizontalalignment='right',
                        fontsize=9, color='darkgray', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            print(f"✓ Added {len(labeled_scenarios)} AI scenario markers")
        else:
            print(f"⚠ AI scenarios file not found: {scenarios_file}")
        
    plt.xlabel('Fire Sale Intensity (%)', fontweight='bold', fontsize=12)
    plt.ylabel('Minimum Mortgage Shock Required (%)', fontweight='bold', fontsize=12)
    plt.title('Mortgage Shock Thresholds vs Fire Sale Intensity\n', 
              fontweight='bold', fontsize=13, pad=15)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=len(thresholds), framealpha=0.95, fontsize=11,
              borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.text(0.02, 0.98, 'More Resilient ↑', transform=plt.gca().transAxes, 
             fontsize=10, va='top', ha='left')
    plt.text(0.02, 0.05, 'More Fragile ↓', transform=plt.gca().transAxes, 
             fontsize=10, va='bottom', ha='left')
    
    plt.tight_layout()
    
    if output_filename is None:
        output_filename = "shock_thresholds_vs_firesales_2.png"
    
    output_path = Path(output_dir) / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")

def create_ai_scenario_comparison_figure(data, output_dir, 
                                         ai_scenario_file: str,
                                         scenarios_to_plot: Optional[List[str]] = None,
                                         output_filename: Optional[str] = None,
                                         pre_2008_file: Optional[str] = None,
                                         post_2008_file: Optional[str] = None) -> None:
    """
    Create 2-panel figure comparing AI scenarios to baseline.
    
    Panel 1: Failure rate curves with AI scenario annotations
    Panel 2: Increased failure rate compared to 5% baseline for AI scenarios
    """
    
    # Load AI scenarios
    scenarios_file = Path(ai_scenario_file)
    if not scenarios_file.exists():
        print(f"⚠ AI scenarios file not found: {scenarios_file}")
        return
    
    df = pd.read_csv(scenarios_file)
    labeled_scenarios = df[df['label'].notna() & (df['label'].str.strip() != '')].copy()
    labeled_scenarios['label'] = labeled_scenarios['label'].str.strip("'")

    if scenarios_to_plot is not None:
        labeled_scenarios = labeled_scenarios[labeled_scenarios["label"].isin(scenarios_to_plot)]

    labeled_scenarios = labeled_scenarios.sort_values('firesale_term').reset_index(drop=True)

    # Get available intensities from data
    available_intensities = sorted([float(k) for k in data['intensity_results'].keys()])
    
    def find_closest_intensity(target):
        return min(available_intensities, key=lambda x: abs(x - target))
    
    labeled_scenarios['closest_intensity'] = labeled_scenarios['firesale_term'].apply(find_closest_intensity)
    
    # Create a mapping of intensity to AI scenario label
    intensity_to_label = {}
    for _, row in labeled_scenarios.iterrows():
        intensity_to_label[row['closest_intensity']] = row['label']

    # Load pre/post-2008 configuration data
    config_data_2008 = {
        'Pre-2008': {'file': pre_2008_file, 'data': None, 'color': 'darkblue', 'linestyle': '--', 'marker': 's'},
        'Post-2008': {'file': post_2008_file, 'data': None, 'color': 'darkgreen', 'linestyle': '-.', 'marker': '^'}
    }
    
    for config_name, config_info in config_data_2008.items():
        if config_info['file'] is not None:
            path = Path(config_info['file'])
            if path.exists():
                with open(path, 'r') as f:
                    config_info['data'] = json.load(f)
                print(f"✓ Loaded {config_name} configuration data")
            else:
                print(f"⚠ {config_name} file not found: {path}")
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color setup
    norm = plt.Normalize(min(available_intensities), max(available_intensities))
    cmap = plt.cm.plasma
    
    # ============================================================================
    # Panel 1: Failure Rate Curves with AI Scenario Labels in Legend
    # ============================================================================
    
    # Plot all intensity curves from fire sale sweep
    for intensity in available_intensities:
        intensity_data = data['intensity_results'][str(intensity)]
        results = intensity_data['results']
        
        shocks = [r['shock'] * 100 for r in results if 'error' not in r]
        failures = [r['failure_rate_mean'] * 100 for r in results if 'error' not in r]
        
        color = cmap(norm(intensity))
        is_ai_scenario = intensity in intensity_to_label
        
        # Build legend label and styling
        if intensity == 0.05:
            legend_label = f'{intensity:.1%} (Baseline - No AI)'
            linewidth, alpha = 3.0, 1.0
        elif is_ai_scenario:
            legend_label = f'{intensity:.1%} ({intensity_to_label[intensity]})'
            linewidth, alpha = 2.5, 0.9
        else:
            continue
        
        ax1.plot(shocks, failures, '-', color=color, linewidth=linewidth,
                alpha=alpha, label=legend_label, marker='o', markersize=4)
    
    # Plot pre/post-2008 configurations
    for config_name, config_info in config_data_2008.items():
        if config_info['data'] is not None:
            results = config_info['data']['results']
            shocks = [r['shock'] * 100 for r in results if 'error' not in r]
            failures = [r['failure_rate_mean'] * 100 for r in results if 'error' not in r]
            
            fs_intensity = config_info['data']['config'].get('fire_sale_intensity', 0)
            
            ax1.plot(shocks, failures, config_info['linestyle'], 
                    color=config_info['color'], linewidth=2.5, alpha=0.8,
                    label=f'{config_name} Config (FS={fs_intensity:.1%})', 
                    marker=config_info['marker'], markersize=5)
    
    ax1.set_xlabel('Mortgage Shock (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Bank Failure Rate (%)', fontweight='bold', fontsize=12)
    ax1.set_title('A. Bank Failure Rate vs Mortgage Shock\n(AI Scenarios Highlighted)', 
                  fontweight='bold', fontsize=13, pad=15)
    ax1.legend(loc='lower left', framealpha=0.95, fontsize=8, ncol=1,
               title='Fire Sale Intensity Value', title_fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-42, -3])
    ax1.set_ylim([-5, 105])
    
    # ============================================================================
    # Panel 2: Increased Failure Rate vs Baseline (5%)
    # ============================================================================
    
    # Get baseline (5%) data
    baseline_intensity = 0.05
    if str(baseline_intensity) not in data['intensity_results']:
        baseline_intensity = find_closest_intensity(0.05)
    
    baseline_data = data['intensity_results'][str(baseline_intensity)]
    baseline_results = {r['shock']: r['failure_rate_mean'] 
                       for r in baseline_data['results'] if 'error' not in r}
    
    # Use the same colormap and normalization as Panel 1 for consistency
    # (norm and cmap are already defined above for Panel 1)
    
    # Plot increased failure rate for each AI scenario
    for idx, (_, row) in enumerate(labeled_scenarios.iterrows()):
        closest_intensity = row['closest_intensity']
        if closest_intensity == baseline_intensity:
            continue
        
        intensity_data = data['intensity_results'][str(closest_intensity)]
        results = intensity_data['results']
        
        differences, valid_shocks = [], []
        for result in sorted(results, key=lambda r: r.get('shock', 0)):
            if 'error' not in result and result['shock'] in baseline_results:
                diff = (result['failure_rate_mean'] - baseline_results[result['shock']]) * 100
                differences.append(diff)
                valid_shocks.append(result['shock'] * 100)
        
        if valid_shocks:
            # Use the same color mapping as Panel 1 based on intensity value
            color = cmap(norm(closest_intensity))
            ax2.plot(valid_shocks, differences, '-o', color=color,
                    linewidth=2.5, markersize=6,
                    label=f'{row["label"]} ({closest_intensity:.1%})')
    
    # Plot pre/post-2008 vs baseline
    for config_name, config_info in config_data_2008.items():
        if config_info['data'] is not None:
            results = config_info['data']['results']
            differences, valid_shocks = [], []
            
            for result in sorted(results, key=lambda r: r.get('shock', 0)):
                if 'error' not in result and result['shock'] in baseline_results:
                    diff = (result['failure_rate_mean'] - baseline_results[result['shock']]) * 100
                    differences.append(diff)
                    valid_shocks.append(result['shock'] * 100)
            
            if valid_shocks:
                fs_intensity = config_info['data']['config'].get('fire_sale_intensity', 0)
                ax2.plot(valid_shocks, differences, config_info['linestyle'],
                        color=config_info['color'], linewidth=2.5,
                        marker=config_info['marker'], markersize=6,
                        label=f'{config_name} Config (FS={fs_intensity:.1%})')
    
    # Add reference line at zero
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Mortgage Shock (%)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Increase in Failure Rate vs Baseline (pp)', fontweight='bold', fontsize=12)
    ax2.set_title('B. Additional Bank Failures Due to AI-Driven Fire Sales\n(Absolute Difference vs 5% Baseline)', 
                  fontweight='bold', fontsize=13, pad=15)
    ax2.legend(loc='upper left', framealpha=0.95, fontsize=10,
               title='Fire Sale Intensity Value', title_fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-42, -3])
    
    plt.tight_layout()
    
    if output_filename is None:
        output_filename = "ai_scenario_comparison.png"
    
    output_path = Path(output_dir) / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Fire Sale Parameter Sweep Plotter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--summary-file', required=False, default=None,
                        help='Path to fire_sale_sweep_summary.json file; if not provided, defaults to latest')
    parser.add_argument('--ai-scenario-file', required=False, help='Path to AI component scenario file')
    parser.add_argument('--pre-2008-file', required=False, help='Path to pre-2008 configuration results JSON')
    parser.add_argument('--post-2008-file', required=False, help='Path to post-2008 configuration results JSON')
    parser.add_argument('--figure-name', action='store_true', default=False,
                        help='If set, incorporate the AI scenario CSV stem into output filenames')
    
    args = parser.parse_args()
    summary_file_arg = args.summary_file
    pre_2008_file = args.pre_2008_file
    post_2008_file = args.post_2008_file
    ai_scenario_file = args.ai_scenario_file

    figure_suffix = None
    if args.figure_name and ai_scenario_file is not None:
        figure_suffix = Path(ai_scenario_file).stem
    
    base_dir = Path(__file__).parent
     
    if summary_file_arg is None:
        # Find the latest run
        output_base = base_dir / "output"
        run_dirs = sorted([d for d in output_base.iterdir() if d.is_dir() and d.name.startswith('run_')],
                        key=lambda x: x.name, reverse=True)

        if not run_dirs:
            print("ERROR: No run directories found")
            return 1

        summary_file = run_dirs[0] / "fire_sale_sweep_summary.json"
    else:
        summary_file = Path(summary_file_arg)
    
    if not summary_file.exists():
        print(f"ERROR: Summary file not found: {summary_file}")
        return 1

    print(f"Loading data from: {summary_file}")
    data = load_data(summary_file)

    # Create output directory for graphs next to the summary file
    graph_dir = summary_file.parent / "policy_graphs"
    graph_dir.mkdir(exist_ok=True)

    print("\nGenerating policy visualizations...")

    scenarios_to_plot = ['25% AI adoption - Equal diversity',
                         '50% AI adoption - Equal diversity',
                         '100% AI adoption - Equal diversity',
                         '100% AI adoption - Equal diversity (Adversarial)',
                         '50% AI adoption - Equal diversity (Adversarial)',
                         "100% AI adoption - Monopoly (OpenAI)",
                         "100% AI adoption - Monopoly (OpenAI) (Adversarial)",]

    sri_filename = (f"illustrative_sri_shock_thresholds_{figure_suffix}.png"
                    if figure_suffix else None)
    illustrative_sri_shock_thresholds(data, graph_dir, threshold=30,
                                      output_filename=sri_filename)

    threshold_filename = (f"shock_thresholds_vs_firesales_2_{figure_suffix}.png"
                          if figure_suffix else None)
    create_threshold_shocks_vs_firesale_plot(data, graph_dir, thresholds=[10, 15, 30, 50],
                                             ai_scenario_file=ai_scenario_file,
                                             scenarios_to_plot=scenarios_to_plot,
                                             output_filename=threshold_filename)

    ai_comparison_filename = (f"ai_scenario_comparison_{figure_suffix}.png"
                               if figure_suffix else None)
    create_ai_scenario_comparison_figure(data, graph_dir,
                                         ai_scenario_file=ai_scenario_file,
                                         scenarios_to_plot=scenarios_to_plot,
                                         pre_2008_file=pre_2008_file,
                                         post_2008_file=post_2008_file,
                                         output_filename=ai_comparison_filename)

    # print(f"\n✓ All graphs saved to: {graph_dir.absolute()}")
    # print("\nNext: Use these graphs in the policy brief")

    return 0

if __name__ == "__main__":
    exit(main())
