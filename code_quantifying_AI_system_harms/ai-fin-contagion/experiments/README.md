# Experiments

Reproducible experiment scripts and results for the financial contagion model. Each subdirectory is a self-contained experiment group with its own `run.py`, `config.yaml`, and output directory.

## Groups

| Directory | Purpose |
|---|---|
| `post_extension/` | Main results: shock, fire sale, and correlation sweeps using the extended model (fixed fire sales, heterogeneous portfolios) |
| `heterogeneity_investigation/` | Investigates sensitivity to portfolio heterogeneity |
| `post_extension_experiments/` | Development validation experiments used to verify model fixes |

## Running experiments

All scripts use `uv`:

```bash
uv run python experiments/<group>/<experiment>/run.py --help
```

Pass `--yes` to skip the confirmation prompt for long runs. Results are written to `output/` subdirectories as JSON summaries.

## Comparing pre-2008 vs post-2008

```bash
uv run python experiments/post_extension/compare_reforms.py
```
