import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D
from experiments.simulation_types import AttackType

MODEL_MARKERS = {
    'openai/gpt-5-mini': 'o',
    'anthropic/claude-3-5-haiku-20241022': 's',
    'gemini/gemini-flash-lite-latest': '^',
}

MODEL_LABELS = {
    'openai/gpt-5-mini': 'GPT-5 mini',
    'anthropic/claude-3-5-haiku-20241022': 'Claude 3.5 Haiku',
    'gemini/gemini-flash-lite-latest': 'Gemini Flash',
}

ASSET_COLS = {
    'equities': 'equities_weighting',
    'corporate_bonds': 'corporate_bonds_weighting',
    'government_bonds': 'government_bonds_weighting',
    'mortgages_backed_securities': 'mortgages_backed_securities_weighting',
}

ASSET_NAMES = {
    'equities': 'Equities',
    'corporate_bonds': 'Corporate Bonds',
    'government_bonds': 'Government Bonds',
    'mortgages_backed_securities': 'Mortgages Backed Securities',
}

def plot_by_attack_type(baseline_path: str, attack_path: str, output_dir: str = None):
    """Create separate 4-panel plots for each attack type and an average."""
    
    df_baseline = pd.read_csv(baseline_path)
    df_attack = pd.read_csv(attack_path)
    _display_to_key = {v: k for k, v in ASSET_NAMES.items()}
    for df in [df_baseline, df_attack]:
        df['has_attack'] = df['has_attack'].map(lambda x: str(x).strip().upper() == 'TRUE')
        df['fixed_asset_type'] = df['fixed_asset_type'].map(
            lambda x: _display_to_key.get(str(x).strip(), str(x).strip())
        )
    
    attack_types = [attack.value for attack in AttackType]
    
    # Fixed colors for consistent experiment types
    CONDITION_COLORS = {
        'AllNeutralRun': '#1f77b4',  # blue
        'FixedBearishNeutralRun': '#ff7f0e',  # orange
        'FixedBearishNeutralRun+Adversarial': '#d62728',  # red
    }
    
    # Plot each attack type + average
    for attack_type in attack_types + ['average']:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        all_labels_with_data = set()  # Track labels that actually have data points
        
        for idx, (asset_key, asset_col) in enumerate(ASSET_COLS.items()):
            ax = axes[idx]
            asset_name = ASSET_NAMES[asset_key]
            
            # Baseline data
            df_base = df_baseline[
                ((df_baseline['experiment_type'] == 'AllNeutralRun') & (df_baseline['has_attack'] == False)) |
                ((df_baseline['experiment_type'] == 'FixedBearishNeutralRun') & 
                (df_baseline['fixed_asset_type'] == asset_key) & 
                (df_baseline['has_attack'] == False))
            ].copy()
            df_base['has_attack'] = False
            
            # Attack data
            df_atk = df_attack[df_attack['attack_target_asset'] == asset_name].copy()
            if attack_type != 'average':
                df_atk = df_atk[df_atk['attack_type'] == attack_type]
            df_atk['has_attack'] = True
            
            df = pd.concat([df_base, df_atk], ignore_index=True)
            
            # Calculate stats
            stats = []
            panel_labels = set()  # Track labels for this specific panel
            for (exp_type, has_attack), group in df.groupby(['experiment_type', 'has_attack']):
                if len(group) == 0:  # Skip if no data
                    continue
                    
                suffix = "+Adversarial" if has_attack else ""
                label = exp_type + suffix
                panel_labels.add(label)
                all_labels_with_data.add(label)
                
                for model, model_group in group.groupby('inference_model'):
                    weights = model_group[asset_col].astype(float)
                    stats.append({
                        'Condition': label,
                        'Model': model,
                        'Mean': weights.mean(),
                        'Std': weights.std(),
                        'N': len(weights)
                    })
            
            stats_df = pd.DataFrame(stats)
            
            # Plot only if we have data
            for _, row in stats_df.iterrows():
                ax.scatter(row['Mean'], row['Std'],
                          color=CONDITION_COLORS.get(row['Condition'], 'gray'),
                          marker=MODEL_MARKERS.get(row['Model'], 'o'),
                          s=120)
                ax.annotate(f"n={row['N']}", (row['Mean'], row['Std']), 
                          fontsize=7, xytext=(3, 3), textcoords='offset points')
            
            ax.set_xlabel("Mean Weighting", fontsize=10)
            ax.set_ylabel("Std Dev", fontsize=10)
            ax.set_title(asset_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Legends - only show conditions that actually have data
        condition_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=CONDITION_COLORS.get(label, 'gray'),
                   markersize=10, label=label, markeredgecolor='black', markeredgewidth=1)
            for label in sorted(all_labels_with_data)
        ]
        
        # Show all defined models consistently across plots
        model_handles = [
            Line2D([0], [0], marker=MODEL_MARKERS[model], color='w', markerfacecolor='gray',
                   markersize=10, label=MODEL_LABELS.get(model, model), 
                   markeredgecolor='black', markeredgewidth=1)
            for model in MODEL_MARKERS.keys()
        ]
        
        fig.legend(handles=condition_handles, loc='lower left', 
                  title='Condition', fontsize=11, title_fontsize=12,
                  bbox_to_anchor=(0.02, 0.01), ncol=len(all_labels_with_data),
                  bbox_transform=fig.transFigure)
        fig.legend(handles=model_handles, loc='lower right',
                  title='Model', fontsize=11, title_fontsize=12,
                  bbox_to_anchor=(0.98, 0.01), ncol=len(MODEL_MARKERS),
                  bbox_transform=fig.transFigure)
        
        title = f"Attack Type: {attack_type.replace('_', ' ').title()}" if attack_type != 'average' else "Average Across All Attack Types"
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.10, 1, 0.96])
        
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            filename = f"attack_{attack_type}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.show()
        
        plt.close()

def plot_attack_shift(baseline_path: str, attack_path: str, output_dir: str = None):
    """
    Plot relative shift in allocation for different attacks using box plots.
    
    Panels 1-4: Box plots for each asset showing all attack types (only when that asset is targeted)
    Panel 5: Average shift across all assets
    """
    df_baseline = pd.read_csv(baseline_path)
    df_attack = pd.read_csv(attack_path)
    _display_to_key = {v: k for k, v in ASSET_NAMES.items()}
    for df in [df_baseline, df_attack]:
        df['has_attack'] = df['has_attack'].map(lambda x: str(x).strip().upper() == 'TRUE')
        df['fixed_asset_type'] = df['fixed_asset_type'].map(
            lambda x: _display_to_key.get(str(x).strip(), str(x).strip())
        )
    
    attack_types = [attack.value for attack in AttackType]
    
    # Calculate baseline (FixedBearishNeutral, no adversarial) means per asset per model
    baseline_means = {}
    for asset_key, asset_col in ASSET_COLS.items():
        asset_name = ASSET_NAMES[asset_key]
        df_base = df_baseline[
            (df_baseline['experiment_type'] == 'FixedBearishNeutralRun') & 
            (df_baseline['fixed_asset_type'] == asset_key) &
            (df_baseline['has_attack'] == False)
        ]
        # Store mean per model
        baseline_means[asset_key] = df_base.groupby('inference_model')[asset_col].mean().to_dict()
    
    # Calculate shifts for each attack type and asset (only when asset is targeted)
    shifts_data = {asset: {attack: [] for attack in attack_types} for asset in ASSET_COLS.keys()}
    
    for attack_type in attack_types:
        for asset_key, asset_col in ASSET_COLS.items():
            asset_name = ASSET_NAMES[asset_key]
            # Only get data where this specific asset was targeted
            df_atk = df_attack[
                (df_attack['attack_target_asset'] == asset_name) &
                (df_attack['attack_type'] == attack_type)
            ]
            
            # Calculate shift for each episode where this asset was targeted
            for _, row in df_atk.iterrows():
                model = row['inference_model']
                baseline = baseline_means[asset_key].get(model, 0)
                shift = row[asset_col] - baseline
                shifts_data[asset_key][attack_type].append(shift)
    
    # Create plot
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.4, wspace=0.35)
    
    # Panels 1-4: One asset per panel
    asset_keys = list(ASSET_COLS.keys())
    attack_labels = [a.replace('_', ' ').title() for a in attack_types]
    colors = plt.cm.Set2(range(len(attack_types)))
    
    for idx, asset_key in enumerate(asset_keys):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Prepare data for box plot
        box_data = [shifts_data[asset_key][attack] for attack in attack_types]
        
        bp = ax.boxplot(box_data, labels=attack_labels, patch_artist=True,
                       whis=[0, 100], showfliers=False)  # 0-100 percentile, no outliers
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticklabels(attack_labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel("Allocation Shift", fontsize=10)
        ax.set_title(ASSET_NAMES[asset_key], fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Average across all assets
    ax_avg = fig.add_subplot(gs[2, :])
    
    # Calculate average shifts across all targeted assets
    avg_shifts_data = {attack: [] for attack in attack_types}
    
    # For each attack type, collect all shifts across all assets where they were targeted
    for attack in attack_types:
        all_shifts = []
        for asset in asset_keys:
            all_shifts.extend(shifts_data[asset][attack])
        avg_shifts_data[attack] = all_shifts
    
    box_data_avg = [avg_shifts_data[attack] for attack in attack_types]
    
    bp_avg = ax_avg.boxplot(box_data_avg, labels=attack_labels, patch_artist=True,
                            whis=[0, 100], showfliers=False)  # 0-100 percentile, no outliers
    
    # Color boxes
    for patch, color in zip(bp_avg['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax_avg.set_xticklabels(attack_labels, rotation=30, ha='right', fontsize=11)
    ax_avg.set_ylabel("Average Allocation Shift", fontsize=12)
    ax_avg.set_title("Average Shift Across All Targeted Assets", fontsize=13, fontweight='bold')
    ax_avg.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax_avg.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Allocation Shift by Attack Type (Relative to FixedBearishNeutral Baseline)", 
                fontsize=14, fontweight='bold', y=0.995)
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "attack_shift_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: attack_shift_comparison.png")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate attack type comparison plots")
    parser.add_argument('baseline', help='Path to baseline CSV')
    parser.add_argument('attack', help='Path to attack CSV')
    parser.add_argument('--output-dir', help='Directory to save plots')
    args = parser.parse_args()

    plot_by_attack_type(args.baseline, args.attack, args.output_dir)
    plot_attack_shift(args.baseline, args.attack, args.output_dir)

if __name__ == "__main__":
    main()