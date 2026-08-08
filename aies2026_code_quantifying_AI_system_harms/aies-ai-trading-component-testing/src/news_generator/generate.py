"""Synthetic financial news article generator.

Uses LLMs to produce HTML-formatted news articles across combinations of asset class
and sentiment, structured to resemble Financial Times articles. Output is used as
simulation input for trading agent experiments.

**How it works**:
1. A JSON config file specifies which models, asset classes, sentiments, and how many
   repeats per combination to generate.
2. For each (model, asset_class, sentiment, repeat) combination, the LLM is prompted
   with a Jinja2 template and returns a structured ``NewsTemplate`` JSON payload.
3. The payload is rendered into an HTML article via a provider template (``ft.j2``).
4. Articles and their metadata are saved under timestamped subdirectories.

**Config file schema** (see ``config.json`` for a full example)::

    {
        "repeats_per_combination": <int>,
        "models": ["openai/gpt-5-mini", ...],
        "asset_classes": ["Equities", "Corporate Bonds", ...],
        "sentiment": ["bullish", "neutral", "bearish"]
    }

**Usage**::

    python src/news_generator/generate.py \\
        --config_file src/news_generator/config.json \\
        --output_dir simulation_inputs/input_article_data/my_run

Articles are written to ``<output_dir>/<timestamp>/articles/`` and metadata to
``<output_dir>/<timestamp>/metadata/``.
"""
import itertools
import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Tuple
import os
from pydantic import BaseModel

from model.llm import LLM
from model.prompt_loader import PromptLoader
from utils import make_dir_if_not_exists
import argparse
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()


class NewsTemplate(BaseModel):
    headline: str
    summary: str
    author: str
    location: str
    date_time: str
    category: str
    paragraphs: List[str]
    key_developments: Optional[List[str]] = None
    quote: Optional[str] = None
    quote_attribution: Optional[str] = None


class NewsMetadata(BaseModel):
    version: int
    model: str
    asset_class: str
    sentiment: str
    timestamp: str  # formatted as DDMMYYYY_HHMMSS


class PromptVariables(BaseModel):
    asset_class: str
    sentiment: str


class Generator:

    def __init__(self, config_path: Optional[str] = "src/news_generator/config.json", output_dir: str = "./data/dummy"):

        self.config = self._load_config(config_path)
        self.output_dir = output_dir
        self.llm = LLM(system_prompt_path="gen_news_system_template.j2")
        self.prompt_loader = PromptLoader()
        logger.info("Generator initialized successfully")

    def _load_config(self, news_gen_config_path: str):
        """Load json config for news generation."""

        logger.info(f"Loading configuration from {news_gen_config_path}")
        try:
            with open(news_gen_config_path, "r") as f:
                config = json.load(f)

            logger.info(
                f"Configuration loaded successfully: {len(config.get('models', []))} models, "
                f"{len(config.get('asset_classes', []))} asset classes, "
                f"{len(config.get('sentiment', []))} sentiments"
            )
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {news_gen_config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise

    def load_provider_template(self, model: str, prompt_vars: PromptVariables) -> str:
        """Generate content using model and render with provider template."""
        logger.info(
            f"Loading provider template for {prompt_vars.asset_class}, model: {model}, sentiment: {prompt_vars.sentiment}"

        )
        template = self.prompt_loader.load_template("ft.j2")

        # Generate content using the model
        self.llm._initialize_conversation()
        self.llm.messages.append(
            {
                "role": "user",
                "content": self.prompt_loader.render_template(
                    "gen_news_template.j2",
                    prompt_vars.model_dump(),
                ),
            }
        )

        # Call model to get content
        response = self.llm.call_model_with_output_format(response_format=NewsTemplate)
        content_response = response["choices"][0]["message"]["content"]
        try:
            content_data = json.loads(content_response)
            logger.debug(f"Successfully parsed JSON response from {model}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response from {model}: {e}")
            raise

        # Render template with generated content
        html = template.render(**content_data)
        logger.info(
            f"Successfully generated HTML content for {prompt_vars.asset_class}, sentiment: {prompt_vars.sentiment} (length: {len(html)} chars)"
        )

        return html

    def generate_news(self) -> tuple[list[str], list[NewsMetadata]]:
        """Generate news article based on the config.

        Every combination of model, provider and sector should be generated.
        """
        logger.info("Starting news generation process")
        news_articles = []
        metadata: list[NewsMetadata] = []
        n_repeats = self.config.get("repeats_per_combination", 0)
        models = self.config.get("models", [])
        assets = self.config.get("asset_classes", [])
        sentiments = self.config.get("sentiment", [])

        # Generate all combinations
        combinations: List[Tuple[str, str, str, int]] = list(
            itertools.product(models, assets, sentiments, range(n_repeats))
        )
        total_combinations = len(combinations)

        logger.info(
            f"Planning to generate {total_combinations} articles across {len(models)} models, "
            f"{len(assets)} asset classes, and {len(sentiments)} sentiments"
        )

        for article_count, (model, asset, sentiment, i) in enumerate(combinations, 1):
            self.llm.set_model(model)
            logger.info(
                f"Generating article {article_count}/{total_combinations}: "
                f"{model}/{asset}/{sentiment}/{i}"
            )

            try:
                # Build prompt variables
                prompt_vars = PromptVariables(asset_class=asset, sentiment=sentiment)
                # Use the template-based generation for FT
                html: str = self.load_provider_template(model, prompt_vars)
                news_articles.append(html)
                metadata.append(
                    NewsMetadata(
                        version=i,
                        model=model,
                        asset_class=asset,
                        sentiment=sentiment,
                        timestamp=datetime.now().strftime("%d%m%Y_%H%M%S"),
                    )
                )
                logger.info(
                    f"Successfully generated article content for {model}/{asset}/{sentiment}/{i}"
                )
            except Exception as e:
                logger.error(
                    f"Error generating article for {model}/{asset}/{sentiment}/{i}: {e}"
                )
                continue

        logger.info(
            f"News generation completed. Generated {len(news_articles)} articles."
        )
        return news_articles, metadata

    def save_news_to_file(
        self, news_articles: list[str], metadata: list[NewsMetadata]

    ) -> None:
        """Save generated news articles as a HTML file."""
        logger.info(f"Saving {len(news_articles)} articles to {self.output_dir}")

        try:
            for i, (article, meta) in enumerate(zip(news_articles, metadata)):
                # generate UUID for file name
                uuid_str = uuid.uuid4()

                file_basename = f"{meta.timestamp}_{uuid_str}"

                articles_dir_path = os.path.join(self.output_dir, "articles")
                metadata_dir_path = os.path.join(self.output_dir, "metadata")

                make_dir_if_not_exists(articles_dir_path)
                make_dir_if_not_exists(metadata_dir_path)
                                       
                article_path = os.path.join(articles_dir_path, f"{file_basename}.html")
                metadata_path = os.path.join(metadata_dir_path, f"{file_basename}.json")
                                       
                with open(article_path, "w", encoding="utf-8") as f:
                    f.write(article)

                with open(metadata_path, "w", encoding="utf-8") as f:
                    f.write(meta.model_dump_json())

                logger.info(
                    f"Article, Metadata {i}/{len(news_articles)} saved file basename: {file_basename} "
                    f"(Article length: {len(article)} characters)"
                )
            
            print(f"News articles saved to {articles_dir_path}")

            logger.info(
                f"Successfully saved all {len(news_articles)} articles to {self.output_dir}"
            )
        except Exception as e:
            logger.error(f"Error saving news articles to file: {e}")
            print(f"Error saving news articles to file: {e}")
            raise


def main(news_gen_config: str, output_dir: str):

    time_stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_subdir = os.path.join(output_dir, time_stamp)
    make_dir_if_not_exists(output_subdir)

    config_copy_path = os.path.join(output_subdir, f"config_{time_stamp}.json")
    with open(news_gen_config, 'r') as src, open(config_copy_path, 'w') as dst:
        dst.write(src.read())
    logger.info(f"Config file copied to: {config_copy_path}")
        

    logger.info("=== Starting news generation application ===")

    try:
        generator = Generator(news_gen_config, output_subdir)
        news, metadata = generator.generate_news()
        generator.save_news_to_file(
            news_articles=news, metadata=metadata,
        )
        logger.info(f"Total Cost: ${generator.llm.cost:.4f}")
    except Exception as e:
        logger.error(f"=== Application failed with error: {e} ===")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",default="src/news_generator/config.json",
                        help="The path to the config file for generating the data")
    parser.add_argument("--output_dir", default="./data/dummy",help="Output directory to generate news content into")
    args = parser.parse_args()
    main(args.config_file, args.output_dir)

