#!/usr/bin/env python3
"""
Fire Sale Sweep Summary Analyser

Compares key metrics across scenario groups from a fire_sale_sweep_summary.json:

  - Adversarial vs non-adversarial scenarios
  - Market structure groups (Monopoly, Equal diversity, No AI) — non-adversarial only

Metrics compared:
  - Mean asset loss   (averaged across all shock levels, then across scenarios in group)
  - Mean failure rate (same averaging)
  - Systemic crisis threshold

USAGE:
    # Single file:
    uv run python experiments/post_extension/fire_sale_sweep/analyse_summary.py \\
        experiments/post_extension/fire_sale_sweep/output/<run>/fire_sale_sweep_summary.json

    # Directory (iterates all matching files and builds a DataFrame):
    uv run python experiments/post_extension/fire_sale_sweep/analyse_summary.py \\
        experiments/post_extension/fire_sale_sweep/output/<run>/
"""

import sys
import json
import re
import argparse
from pathlib import Path
from statistics import mean

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def scenario_means(entry: dict) -> dict:
    """
    Average per-shock metrics across all shock levels for a single scenario entry.
    Returns None if no valid results.
    """
    results = [r for r in entry.get('results', []) if 'error' not in r]
    if not results:
        return None
    return {
        'asset_losses_mean': mean(r['asset_losses_mean'] for r in results),
        'failure_rate_mean': mean(r['failure_rate_mean'] for r in results),
        'fire_sale_losses_mean': mean(r['fire_sale_losses_mean'] for r in results),
    }


def group_means(entries: list[dict]) -> dict | None:
    """Average scenario_means across a group of entries."""
    per_scenario = [scenario_means(e) for e in entries]
    per_scenario = [s for s in per_scenario if s is not None]
    if not per_scenario:
        return None

    return {
        'n_scenarios': len(per_scenario),
        'asset_losses_mean': mean(s['asset_losses_mean'] for s in per_scenario),
        'failure_rate_mean': mean(s['failure_rate_mean'] for s in per_scenario),
        'fire_sale_losses_mean': mean(s['fire_sale_losses_mean'] for s in per_scenario),
    }


def adversarial_amplification(entries: list[dict], print_details: bool = True) -> dict | None:
    """
    For each adoption scenario (matched by scenario_id), compute per-shock
    differences between adversarial and non-adversarial runs for:
      - failure_rate_mean (absolute pp difference)
      - fire_sale_losses_mean (% increase relative to non-adversarial baseline)
    Take the top-3 by magnitude and average → one value per scenario.
    Report the mean of these values across all scenarios.
    0% AI adoption scenarios are excluded (no adversarial effect there).
    """
    from collections import defaultdict

    # Exclude 0% AI adoption: no difference between adversarial/non-adversarial
    filtered = [e for e in entries if (e.get('ai_adoption_pct') or 0) != 0]

    by_id: dict = defaultdict(dict)
    for e in filtered:
        sid = e.get('scenario_id')
        if sid is None:
            continue
        if e.get('adversarial'):
            by_id[sid]['adv'] = e
        else:
            by_id[sid]['nonadv'] = e

    fr_amplifications = []
    fs_amplifications = []
    al_amplifications = []
    pair_rows = []
    for sid in sorted(by_id):
        pair = by_id[sid]
        if 'adv' not in pair or 'nonadv' not in pair:
            continue

        adv_results    = {r['shock']: r for r in pair['adv'].get('results', [])    if 'error' not in r}
        nonadv_results = {r['shock']: r for r in pair['nonadv'].get('results', []) if 'error' not in r}
        common = sorted(set(adv_results) & set(nonadv_results))
        if not common:
            continue

        # Failure rate: absolute difference (pp)
        fr_diffs = [adv_results[s]['failure_rate_mean'] - nonadv_results[s]['failure_rate_mean']
                    for s in common]
        fr_top3   = sorted(fr_diffs, key=abs, reverse=True)[:3]
        avg_fr    = mean(fr_top3)

        def _pct_amplification(key: str) -> float:
            diffs = []
            for s in common:
                base = nonadv_results[s][key]
                adv  = adv_results[s][key]
                if base and base != 0:
                    diffs.append((adv - base) / abs(base))
            if not diffs:
                return float('nan')
            return mean(sorted(diffs, key=abs, reverse=True)[:3])

        avg_fs = _pct_amplification('fire_sale_losses_mean')
        avg_al = _pct_amplification('asset_losses_mean')

        fr_amplifications.append(avg_fr)
        fs_amplifications.append(avg_fs)
        al_amplifications.append(avg_al)
        pair_rows.append((
            sid,
            pair['nonadv'].get('market_structure', '?'),
            pair['nonadv'].get('ai_adoption_pct', float('nan')),
            avg_fr,
            avg_fs,
            avg_al,
        ))

    valid_fs = [x for x in fs_amplifications if x == x]  # drop NaN
    valid_al = [x for x in al_amplifications if x == x]

    if print_details:
        width = 95
        print(f"\n{'=' * width}")
        print("ADVERSARIAL AMPLIFICATION (avg of top-3 shock differences per adoption scenario)".center(width))
        print(f"{'=' * width}")

        if not pair_rows:
            print("  No complete adversarial/non-adversarial pairs found."
                  "  (check that scenario_id is present in the JSON)")
        else:
            print(f"  {'ScenID':>7}  {'Market Structure':<22}  {'AI Adopt%':>9}  "
                  f"{'Avg Top-3 Δfailure rate':>23}  {'Avg Top-3 % Δfiresale loss':>26}  {'Avg Top-3 % Δasset loss':>24}")
            print(f"  {'-' * 100}")
            for sid, ms, ai_pct, avg_fr, avg_fs, avg_al in pair_rows:
                ai_str = f"{ai_pct:.1f}%" if isinstance(ai_pct, (int, float)) else str(ai_pct)
                fs_str = f"{avg_fs:>26.2%}" if avg_fs == avg_fs else f"{'N/A':>26}"
                al_str = f"{avg_al:>24.2%}" if avg_al == avg_al else f"{'N/A':>24}"
                print(f"  {sid:>7}  {ms:<22}  {ai_str:>9}  {avg_fr:>23.4%}  {fs_str}  {al_str}")
            print(f"  {'-' * 100}")

            fs_mean_str = f"{mean(valid_fs):.2%}" if valid_fs else "N/A"
            al_mean_str = f"{mean(valid_al):.2%}" if valid_al else "N/A"
            print(f"  Mean max amplification across {len(fr_amplifications)} adoption scenario(s):")
            print(f"    Δfailure rate:      {mean(fr_amplifications):.4%}")
            print(f"    % Δfiresale loss:   {fs_mean_str}")
            print(f"    % Δasset loss:      {al_mean_str}")

    if not pair_rows:
        return None

    return {
        'mean_delta_failure_rate':      mean(fr_amplifications),
        'mean_pct_delta_firesale_loss': mean(valid_fs) if valid_fs else float('nan'),
        'mean_pct_delta_asset_loss':    mean(valid_al) if valid_al else float('nan'),
        'n_pairs':                      len(fr_amplifications),
    }


_FILENAME_RE = re.compile(r'scaler_([\d.]+)_fH_([\d.]+)_')


def process_directory(dir_path: Path) -> int:
    """
    Iterate all fire_sale_sweep_summary*.json files in *dir_path*, extract
    scaler and fH from each filename, compute adversarial amplification
    (0% AI adoption excluded), and print a sorted pandas DataFrame.
    """
    files = sorted(dir_path.glob('fire_sale_sweep_summary*.json'))
    if not files:
        print(f"ERROR: no fire_sale_sweep_summary*.json files found in {dir_path}")
        return 1

    print(f"\nFound {len(files)} file(s) in {dir_path}")
    rows = []
    for fpath in files:
        m = _FILENAME_RE.search(fpath.name)
        if m is None:
            print(f"  Warning: could not parse scaler/fH from {fpath.name}, skipping")
            continue
        scaler = float(m.group(1))
        fh     = float(m.group(2))

        with open(fpath) as f:
            summary = json.load(f)

        entries = list(summary.get('intensity_results', {}).values())
        if not entries:
            print(f"  Warning: no intensity_results in {fpath.name}, skipping")
            continue

        metrics = adversarial_amplification(entries, print_details=False)
        if metrics is None:
            print(f"  Warning: no valid adversarial pairs in {fpath.name}, skipping")
            continue

        rows.append({
            'scaler':            scaler,
            'fH':                fh,
            'mean_Δfailure_rate':    metrics['mean_delta_failure_rate'],
            'mean_%Δfiresale_loss':  metrics['mean_pct_delta_firesale_loss'],
            'mean_%Δasset_loss':     metrics['mean_pct_delta_asset_loss'],
            'n_pairs':               metrics['n_pairs'],
        })

    if not rows:
        print("ERROR: no valid data collected from directory")
        return 1

    df = pd.DataFrame(rows).sort_values(['scaler', 'fH']).reset_index(drop=True)

    df_display = df.copy()
    df_display['mean_Δfailure_rate']   = df_display['mean_Δfailure_rate'].map('{:.4%}'.format)
    df_display['mean_%Δfiresale_loss'] = df_display['mean_%Δfiresale_loss'].map(
        lambda x: f'{x:.2%}' if x == x else 'N/A'
    )
    df_display['mean_%Δasset_loss']    = df_display['mean_%Δasset_loss'].map(
        lambda x: f'{x:.2%}' if x == x else 'N/A'
    )

    width = 90
    print(f"\n{'=' * width}")
    print(f"ADVERSARIAL AMPLIFICATION SWEEP ({len(rows)} parameter combinations)".center(width))
    print(f"{'=' * width}")
    print(df_display.to_string(index=False))
    print(f"{'=' * width}\n")

    _plot_heatmap(
        df, dir_path,
        col='mean_Δfailure_rate',
        scale=100,
        fmt='{:.2f} pp',
        cbar_label='Mean Δ failure rate (pp)',
        cbar_fmt='%.1f pp',
        title='Adversarial amplification: mean Δ failure rate (pp)\n(0% AI adoption excluded)',
        filename='failure_rate_amplification_heatmap.png',
    )
    _plot_heatmap(
        df, dir_path,
        col='mean_%Δfiresale_loss',
        scale=100,
        fmt='{:.2f}%',
        cbar_label='Mean % Δ fire-sale loss',
        cbar_fmt='%.1f%%',
        title='Adversarial amplification: mean % Δ fire-sale loss\n(0% AI adoption excluded)',
        filename='firesale_loss_amplification_heatmap.png',
    )
    _plot_heatmap(
        df, dir_path,
        col='mean_%Δasset_loss',
        scale=100,
        fmt='{:.2f}%',
        cbar_label='Mean % Δ asset loss',
        cbar_fmt='%.1f%%',
        title='Adversarial amplification: mean % Δ asset loss\n(0% AI adoption excluded)',
        filename='asset_loss_amplification_heatmap.png',
    )

    return 0


def _plot_heatmap(
    df: pd.DataFrame,
    dir_path: Path,
    col: str,
    scale: float,
    fmt: str,
    cbar_label: str,
    cbar_fmt: str,
    title: str,
    filename: str,
) -> None:
    """
    Generic heatmap: y-axis → scaler (K), x-axis → fH.
    Values are multiplied by *scale* before display (e.g. 100 for %).
    """
    pivot = df.pivot(index='scaler', columns='fH', values=col)
    pivot = pivot.sort_index(ascending=False)   # y: high K at top
    pivot = pivot.sort_index(axis=1)            # x: low fH at left

    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.1),
                                    max(4, pivot.shape[0] * 0.9)))

    data_scaled = pivot.values * scale

    im = ax.imshow(data_scaled, aspect='auto', cmap='RdYlGn_r',
                   vmin=np.nanmin(data_scaled), vmax=np.nanmax(data_scaled))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(cbar_fmt))

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f'{v:.4g}' for v in pivot.columns], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f'{v:.4g}' for v in pivot.index], fontsize=9)

    ax.set_xlabel('$f_H$ (Human Firesale Term)', fontsize=12)
    ax.set_ylabel('Scaler $K$', fontsize=12)
    ax.set_title(title, fontsize=13)

    max_abs = np.nanmax(np.abs(data_scaled))
    for row_i in range(pivot.shape[0]):
        for col_i in range(pivot.shape[1]):
            val = data_scaled[row_i, col_i]
            if not np.isnan(val):
                text_color = 'white' if max_abs and abs(val) > 0.6 * max_abs else 'black'
                ax.text(col_i, row_i, fmt.format(val), ha='center', va='center',
                        fontsize=8, color=text_color)

    fig.tight_layout()
    out_path = dir_path / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to: {out_path}")



def print_comparison(title: str, groups: dict[str, list[dict]]):
    header_width = 80
    print(f"\n{'=' * header_width}")
    print(title.center(header_width))
    print(f"{'=' * header_width}")
    print(f"{'Group':<30} {'N':>4}  {'Avg Asset Loss':>16}  {'Avg Failure Rate':>16}  {'Avg Firesale Loss':>18}")
    print(f"{'-' * header_width}")

    for label, entries in groups.items():
        gm = group_means(entries)
        if gm is None:
            print(f"  {label:<28}  (no data)")
            continue
        print(
            f"  {label:<28} {gm['n_scenarios']:>4}  "
            f"{gm['asset_losses_mean']:>16.2f}  "
            f"{gm['failure_rate_mean']:>16.2%}  "
            f"{gm['fire_sale_losses_mean']:>18.2f}  "
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyse fire_sale_sweep_summary.json — compare groups of scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('summary_file', type=str,
                        help='Path to fire_sale_sweep_summary.json')
    
    args = parser.parse_args()

    summary_path = Path(args.summary_file)
    if not summary_path.exists():
        print(f"ERROR: path not found: {summary_path}")
        return 1

    if summary_path.is_dir():
        return process_directory(summary_path)

    with open(summary_path) as f:
        summary = json.load(f)

    intensity_results = summary.get('intensity_results', {})
    if not intensity_results:
        print("ERROR: no intensity_results found in summary")
        return 1

    entries = list(intensity_results.values())

    print(f"\nLoaded {len(entries)} scenarios from {summary_path.name}")
    print(f"Timestamp: {summary.get('timestamp', 'unknown')}")
    print(f"Shock range: {summary['config']['shock_range']}, step: {summary['config']['step_size']:.2%}")
    print(f"Runs per (scenario, shock): {summary['config']['num_runs_per_pair']}")

    # ── 0. Adversarial amplification (per-pair, top-3 shock diffs) ─────────
    adversarial_amplification(entries)

    # ── 1. Adversarial vs non-adversarial ────────────────────────────────────
    adversarial_entries     = [e for e in entries if e.get('adversarial')]
    non_adversarial_entries = [e for e in entries if not e.get('adversarial')]

    print_comparison(
        "ADVERSARIAL vs NON-ADVERSARIAL",
        {
            'Non-adversarial': non_adversarial_entries,
            'Adversarial':     adversarial_entries,
        }
    )

    # # ── 2. Market structure (non-adversarial only) ────────────────────────────
    # structures = {}
    # for e in non_adversarial_entries:
    #     ms = e.get('market_structure') or 'Unknown'
    #     structures.setdefault(ms, []).append(e)

    # print_comparison(
    #     "MARKET STRUCTURE (non-adversarial only)",
    #     structures,
    # )

    # # ── 3. AI adoption quartiles (non-adversarial only) ───────────────────────
    # brackets = {'0% (no AI)': [], '1–49%': [], '50–74%': [], '75–100%': []}
    # for e in non_adversarial_entries:
    #     pct = e.get('ai_adoption_pct') or 0.0
    #     if pct == 0:
    #         brackets['0% (no AI)'].append(e)
    #     elif pct < 50:
    #         brackets['1–49%'].append(e)
    #     elif pct < 75:
    #         brackets['50–74%'].append(e)
    #     else:
    #         brackets['75–100%'].append(e)

    # print_comparison(
    #     "AI ADOPTION LEVEL (non-adversarial only)",
    #     {k: v for k, v in brackets.items() if v},
    # )

    # return 0


if __name__ == '__main__':
    sys.exit(main())
