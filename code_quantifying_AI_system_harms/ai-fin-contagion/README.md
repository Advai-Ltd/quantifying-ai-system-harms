# financial-contagion-networks

Code repository for AIES 2026 Paper: *Quantifying System-Level Harms from AI Adoption in Complex Sociotechnical Systems*.

**Authors:** Paul Vautravers, Oliver Chalkley, Gabriel Downer, Kate S, Damian Ruck  
**Conference:** AAAI Conference on AI, Ethics, and Society (AIES 2026)

simulation of financial contagion in interbank networks, extended to model AI-driven fire sale dynamics. Banks hold diversified asset portfolios; correlated mortgage shocks propagate through bilateral exposures. The model supports parameter sweeps over fire sale intensity and shock correlation to identify systemic crisis thresholds under different AI market-structure and adversarial-behaviour assumptions.

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (dependency and environment manager)

Install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick start

```bash
# Install all dependencies into an isolated virtual environment
uv sync

# Run the main fire sale sweep (see --help for options)
uv run python experiments/post_extension/fire_sale_sweep/run.py --help

# Run the test suite
uv run pytest tests/
```

All `uv run` commands automatically use the project's managed environment — no manual `source .venv/bin/activate` needed.

## Repository structure

```
src/financial_contagion_networks/
    config/        # Pydantic config models and YAML loader
    core/          # Bank, Portfolio, ContagionNetwork, ShockGenerator
    simulation/    # NetworkGenerator, Monte Carlo runner, ExperimentRunner

experiments/
    post_extension/
        pre_2008/          # Pre-Basel III shock sweep
        post_2008/         # Post-Basel III shock sweep
        fire_sale_sweep/   # Main sweep: fire sale intensity × AI market structure
        correlation_sweep/ # Sweep over shock correlation coefficient
        compare_reforms.py # Pre- vs post-2008 comparison graphs
    heterogeneity_investigation/  # Sensitivity to portfolio heterogeneity

tests/
    unit/         # Unit tests for core components
    integration/  # End-to-end experiment tests
```

See `experiments/README.md` for a full description of each experiment group.
