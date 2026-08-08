# Post-Extension Experiments

Main results using the extended model: fixed fire sale mechanics (no compounding, asset-specific markdowns), heterogeneous portfolios (Option F), variable interbank exposure (Option A), and stochastic topology (Option D).

## Experiments

| Directory | Description |
|---|---|
| `pre_2008/` | Pre-Basel III shock sweep with extended model |
| `post_2008/` | Post-Basel III shock sweep with extended model |
| `fire_sale_sweep/` | Sweeps fire sale intensity (0–25%) across AI market-structure scenarios |
| `correlation_sweep/` | Sweeps shock correlation coefficient across scenarios |

## Key scripts

```bash
# Baseline shock sweeps
uv run python experiments/post_extension/pre_2008/run.py --yes
uv run python experiments/post_extension/post_2008/run.py --yes

# Fire sale parameter sweep (batch via scenario CSV)
./experiments/post_extension/fire_sale_sweep/run_all_scenarios.sh

# Correlation sweep
uv run python experiments/post_extension/correlation_sweep/run.py --yes

# Compare pre-2008 vs post-2008 results
uv run python experiments/post_extension/compare_reforms.py

# Generate policy graphs from fire sale sweep output
uv run python experiments/post_extension/fire_sale_sweep/generate_fs_policy_graphs.py \
    experiments/post_extension/fire_sale_sweep/output/<run>/fire_sale_sweep_summary.json
```

## AI market-structure scenarios

The fire sale sweep is parameterised by AI adoption level and market structure (monopoly, equal diversity, no AI) sourced from CSV files in `fire_sale_sweep/firesale_ai_component_table/`. Adversarial and non-adversarial variants are included.
