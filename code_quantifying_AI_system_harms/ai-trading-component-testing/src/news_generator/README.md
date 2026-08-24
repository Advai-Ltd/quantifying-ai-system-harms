# News Generator

Generates fully synthetic financial news articles for use as simulation inputs.

## How articles are generated

Articles are produced by prompting LLMs (via [LiteLLM](https://github.com/BerriAI/litellm)) to write structured financial news content for a given asset class and sentiment (bullish / neutral / bearish). The structured JSON response is then rendered into HTML using a Jinja2 template (`src/model/templates/ft.j2`).

## Note on article styling

The HTML template styles articles to resemble a financial news webpage. The generated articles carry the "Financial Times" name and FT visual styling solely because this outlet was chosen as a realistic format for the simulation context — **no real Financial Times content is used**. All article text, author names, company names, and data are entirely LLM-generated fiction. Each article includes a footer disclaimer to this effect:

> *"This is a mock webpage created for demonstration purposes only."*

The choice of outlet style has no bearing on the simulation results; it exists only to present a realistic-looking news stimulus to the trading agents.

## Usage

```bash
python src/news_generator/generate.py \
    --config_file src/news_generator/config.json \
    --output_dir simulation_inputs/input_article_data/my_run
```

See `config.json` for the full schema (models, asset classes, sentiments, repeats per combination).
