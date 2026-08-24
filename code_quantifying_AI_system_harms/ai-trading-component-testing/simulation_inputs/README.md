# Simulation Inputs

## adversarial_attack_configs/

JSON configuration files defining adversarial prompt injection attacks. One file per (target asset, attack type) combination. Each config specifies the `attack_type`, `target_asset`, `insertion_logic`, and `custom_text` fields used to inject adversarial content into the news articles presented to trading agents.

To regenerate these configs (e.g. after adding new asset or attack types):

```bash
python misc_code/generate_jsons_for_attacks.py
```

## input_article_data/

Synthetic financial news articles used as inputs to the trading agent simulation. **All articles are fully LLM-generated — no real news content from any outlet is used.** See [`src/news_generator/README.md`](../src/news_generator/README.md) for details on how they were produced and a note on the FT-style HTML formatting.

Each run directory contains:
- `articles/` — HTML-formatted synthetic news articles
- `metadata/` — JSON metadata per article (model, asset class, sentiment, timestamp)
- `config_<timestamp>.json` — the generation config used for that run
