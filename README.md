# Quantifying System-Level Harms from AI Adoption in Complex Sociotechnical Systems

Code and supplementary materials for **AIES 2026 Paper 883**.

**Authors:** Paul Vautravers, Oliver Chalkley, Gabriel Downer, Kate S, Damian Ruck  
**Affiliations:** Advai Ltd; UK National Cyber Security Centre

> **Links:** arXiv and AIES 2026 proceedings links will be added shortly.

---

## Repository contents

| Path | Description |
|------|-------------|
| `aies2026_code_quantifying_AI_system_harms/aies-ai-fin-contagion/` | Financial contagion model — simulates AI-driven fire sale dynamics across an interbank network |
| `aies2026_code_quantifying_AI_system_harms/aies-ai-trading-component-testing/` | LLM trading component — measures asset allocation shifts under adversarial prompt injection |
| `aies2026_stpa_tables.xlsx` | Full STPA artefacts: losses, hazards, system constraints, unsafe control actions, and AI-driven loss scenarios |

---

## Quick start

This repository contains two independent code components, each managed with its own [uv](https://docs.astral.sh/uv/) environment. **They must be set up and run separately** — do not run `uv sync` from the repository root.

The two components also target different Python versions:
- `aies-ai-fin-contagion` — Python ≥ 3.11
- `aies-ai-trading-component-testing` — Python ≥ 3.12

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up and run each component from its own directory:

```bash
# Financial contagion model
cd aies2026_code_quantifying_AI_system_harms/aies-ai-fin-contagion
uv sync                          # creates .venv with Python 3.11
uv run python experiments/post_extension/fire_sale_sweep/run.py --help

# LLM trading component testing (requires LLM API keys — see subdirectory README)
cd aies2026_code_quantifying_AI_system_harms/aies-ai-trading-component-testing
uv sync                          # creates .venv with Python 3.12
uv run python src/main.py --help
```

See each subdirectory's `README.md` for full setup and experiment instructions.

---

## Versioning

This repository is versioned as a whole in the top-level `VERSION` file. A version increment here reflects any change to the code, the extended appendix, or the STPA tables. `1.0.0` is the version at the point of public release accompanying the AIES 2026 submission.

Each code subdirectory also carries its own `VERSION` file tracking changes to that component independently.

---

## Citation

Citation will be added once the AIES 2026 proceedings are available.