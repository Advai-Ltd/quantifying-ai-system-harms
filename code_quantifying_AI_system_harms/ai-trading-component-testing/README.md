# aies-ai-trading-component-testing

Supporting code for AIES 2026 Paper: *Quantifying System-Level Harms from AI Adoption in Complex Sociotechnical Systems*.

**Authors:** Paul Vautravers, Oliver Chalkley, Gabriel Downer, Kate S, Damian Ruck  
**Conference:** AAAI Conference on AI, Ethics, and Society (AIES 2026)

This repository implements a single-turn interaction with a stylised LLM-based tool (repeated across many independent calls) that decides asset allocations across four asset classes in response to synthetic market news articles. Experiments explore how sentiment and prompt-injection style manipulations affect the tool's outputs. The measured allocation shifts can then be used to explore how widespread AI adoption for this use case might affect system-level market dynamics — in particular, the propensity for fire-sale contagion (see the companion `aies-ai-fin-contagion` repository).

---

## Repository structure

```
src/
  main.py                      # CLI entry point for experiments
  record_results.py            # CSV / JSON results persistence
  utils.py                     # Filesystem and HTML utilities
  experiments/                 # Experiment types, episode logic, adversarial insertion
  model/                       # LLM wrappers, prompt templates, tool definitions
  analysis/                    # Aggregation, plotting, and firesale scenario mapping
  news_generator/              # Synthetic news article generation

simulation_inputs/
  adversarial_attack_configs/  # JSON configs for each (asset, attack-type) pair
  input_article_data/          # News article HTML files used as simulation inputs

simulation_shell_scripts/
  comprehensive_experiment_run.sh   # Run all attack × asset experiments
  firesale_parameter_sweep.sh       # Grid-search firesale parameters (κ, f_H)
  security_tests.sh                 # Prompt-injection robustness checks

misc_code/
  generate_jsons_for_attacks.py     # Regenerate adversarial config JSONs
  latex_tables.py                   # Convert result CSVs to LaTeX tables
  clean_csvs.py                     # One-off CSV reconciliation utility

tests/                         # Unit tests (pytest)
```

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install the project and all dependencies (requires Python ≥ 3.12):
   ```bash
   uv sync
   ```

   This creates a virtual environment at `.venv/` and installs everything pinned in `pyproject.toml`. No need to manually activate it — prefix commands with `uv run` as shown below.

---

## API key setup

The simulation calls OpenAI, Anthropic, and Google APIs via [LiteLLM](https://github.com/BerriAI/litellm). Set the relevant keys as environment variables before running:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

---

## Running experiments

### All-neutral baseline
```bash
uv run python src/main.py all-neutral simulation_inputs/input_article_data/<run_dir> \
    --inference-models openai/gpt-5-mini \
    --num-episodes 30 --batch-mode --batch-size 10 \
    --output-dir simulation_outputs/my_run
```

### Fixed-bearish-neutral (per asset)
```bash
uv run python src/main.py fixed-bearish-neutral simulation_inputs/input_article_data/<run_dir> \
    --inference-models openai/gpt-5-mini \
    --fixed-assets "Equities" \
    --num-episodes 30 --batch-mode --batch-size 10 \
    --output-dir simulation_outputs/my_run
```

### Adversarial attack experiment
```bash
uv run python src/main.py fixed-bearish-neutral simulation_inputs/input_article_data/<run_dir> \
    --inference-models openai/gpt-5-mini \
    --fixed-assets "Equities" \
    --attack-config simulation_inputs/adversarial_attack_configs/asset_Equities--attack_escalation--insertion_by_asset.json \
    --num-episodes 30 --batch-mode --batch-size 10 \
    --output-dir simulation_outputs/my_run
```

See `simulation_shell_scripts/` for full multi-asset, multi-attack-type experiment orchestration.

---

## Firesale analysis

After generating experiment results, run the firesale scenario mapping to map measured component level qualities to scenario based system level firesale values:

```bash
uv run python src/analysis/stats_mapping_analysis.py <responses.csv> \
    --adversarial-responses-csv <adversarial_responses.csv> \
    --firesale-mapping \
    --adversarial-attack-type escalation \
    --firesale-scaler 1.0 \
    --human-firesale-term 0.05 \
    --output-dir simulation_outputs/firesale_results
```

To repeat the above, sweep over κ and f_H values:
```bash
bash simulation_shell_scripts/firesale_parameter_sweep.sh
```

---

## Citation

Citation will be added once the AIES 2026 proceedings are available. arXiv and proceedings links will be added shortly.
