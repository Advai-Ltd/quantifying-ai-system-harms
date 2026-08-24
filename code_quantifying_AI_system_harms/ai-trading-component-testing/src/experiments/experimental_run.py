from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
import uuid
import random
import json
import os
from pathlib import Path
from experiments.simulation_types import Article, Sentiment, AssetType, InsertionRule, AttackType
from experiments.episode import Episode
from experiments.adversarial_inputs import TextInsertion
import logging

DEFAULT_BULLISH_ASSET = AssetType.EQUITIES
DEFAULT_BEARISH_ASSET = AssetType.MORTGAGES_BACKED_SECURITIES

class ExperimentalRun(BaseModel):
    """Base class for experimental runs."""
    # experiment_type: Optional[str] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    episodes: List[Episode] = Field(default_factory=list)
    seed: Optional[int] = 42
    source_news_dir: str = Field(description="Directory where articles were loaded from")
    allowed_article_models: Optional[List[str]] = Field(
        default=None, 
        description="Models used to filter articles (None = all models)"
    )
    available_articles: Dict[AssetType, List[Article]] = Field(
        default_factory=dict, 
        description="All articles available for this experiment"
    )
    fixed_assets: Optional[List[AssetType]] = None,

    def __getitem__(self, idx: int):
        return self.episodes[idx]

    def save_definition(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            f.write(self.model_dump_json(indent=2))
        print(f"Experimental run definition saved to: {filepath}")

    @property
    def experiment_type(self) -> str:
        return self.__class__.__name__ 
    
    @classmethod
    def load_definition(cls, filepath: str) -> "ExperimentalRun":
        with open(filepath, 'r') as f:
            return cls.model_validate_json(f.read())
    
    @staticmethod
    def load_articles_from_directory(news_dir: str, allowed_models: Optional[List[str]]) -> Dict[AssetType, List[Article]]:
        """Load articles from generated news directory.
        Args:
            news_dir: Directory containing articles and metadata
            allowed_models: Optional list of model names to filter by. 
                        If None, load all articles.
        Returns:
            Dictionary mapping AssetType to list of Articles
        """
        logging.info(f"Loading articles from dir {news_dir} - filtering for allowed models: {allowed_models}")
        articles_by_asset = {asset_type: [] for asset_type in AssetType}
        metadata_dir = Path(os.path.join(news_dir, "metadata"))

        if not metadata_dir.exists():
            raise ValueError(f"Metadata directory not found: {metadata_dir}")
        
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if allowed_models is not None and metadata['model'] not in allowed_models:
                continue
            
            article = Article(
                file_path=str(os.path.join(news_dir,"articles", metadata_file.stem + ".html")),
                asset_type=AssetType(metadata['asset_class']),
                sentiment=Sentiment(metadata['sentiment']),
                model=metadata['model'],
                repeat_id=metadata['version']
            )
            
            articles_by_asset[article.asset_type].append(article)

        for asset_type, articles in articles_by_asset.items():
            if not articles:
                filter_msg = f" with allowed models {allowed_models}" if allowed_models else ""
                raise ValueError(f"No articles found for {asset_type.value}{filter_msg}")
        
        return articles_by_asset
    
    @staticmethod
    def create_episode_with_adversarial_insertion(
        articles: List[Article],
        insertion_config: Dict[str, Any]
    ) -> Episode:
        """Helper to create an episode with optional insertion."""
        text_insertion = None
        if insertion_config:
            text_insertion = TextInsertion.resolve_for_episode(
                articles=articles,
                attack_type=AttackType(insertion_config["attack_type"]),
                target_asset=AssetType(insertion_config["target_asset"]),
                insertion_logic=InsertionRule(insertion_config["insertion_logic"]),
                fixed_position_idx=insertion_config.get("fixed_position_idx", -1),
                custom_text=insertion_config.get("custom_text")
            )
                
        return Episode.from_articles_and_insertion(articles, text_insertion=text_insertion)

class RandomSentimentRun(ExperimentalRun):
    """Random sentiment combinations."""
    
    @classmethod
    def generate_run_from_dir(
        cls, 
        name: str,
        source_news_dir: str, 
        num_episodes: int = 10,
        seed: Optional[int] = None,
        allowed_models: Optional[List[str]] = None,
        fixed_assets: Optional[List[AssetType]] = None,
        insertion_config: Optional[Dict[str, Any]] = None,
    ) -> "RandomSentimentRun":
        
        if not seed:
            seed = cls.seed
        random.seed(seed)

        available_articles = cls.load_articles_from_directory(source_news_dir, allowed_models)
        
        episodes = []
        for _ in range(num_episodes):
            selected_articles = []
            for asset_type in AssetType:
                article = random.choice(available_articles[asset_type])
                selected_articles.append(article)
            
            random.shuffle(selected_articles)
            episode = cls.create_episode_with_adversarial_insertion(selected_articles, insertion_config)
            episodes.append(episode)
        
        return cls(name=name, 
                   episodes=episodes,
                   seed=seed,
                   fixed_assets=fixed_assets,
                   source_news_dir=source_news_dir,
                   allowed_article_models=allowed_models,
                   available_articles=available_articles)
    
class RandomBullishBearishRun(ExperimentalRun):
    """1 bullish, 3 bearish articles."""
    
    @classmethod
    def generate_run_from_dir(
        cls, 
        name: str,
        source_news_dir: str,
        num_episodes: int = 10,
        seed: Optional[int] = None,
        allowed_models: Optional[List[str]] = None,
        fixed_assets: Optional[List[AssetType]] = None,
        insertion_config: Optional[Dict[str, Any]] = None,
    ) -> "RandomBullishBearishRun":
        
        if not seed:
            seed = cls.seed
        random.seed(seed)

        available_articles = cls.load_articles_from_directory(source_news_dir, allowed_models)
        
        episodes = []
        for _ in range(num_episodes):
            bullish_asset = random.choice(list(AssetType))
            
            selected_articles = []
            for asset_type in AssetType:
                if asset_type == bullish_asset:
                    # Find bullish article for this asset
                    bullish_articles = [a for a in available_articles[asset_type] if a.sentiment == Sentiment.BULLISH]
                    article = random.choice(bullish_articles)
                else:
                    bearish_articles = [a for a in available_articles[asset_type] if a.sentiment == Sentiment.BEARISH]
                    article = random.choice(bearish_articles)
                
                selected_articles.append(article)
            
            random.shuffle(selected_articles)
            episode = cls.create_episode_with_adversarial_insertion(selected_articles, insertion_config)
            episodes.append(episode)
        
        return cls(
            name=name,
            episodes=episodes,
            seed=seed,
            fixed_assets=fixed_assets,
            source_news_dir=source_news_dir,
            allowed_article_models=allowed_models,
            available_articles=available_articles
        )
    
class FixedBullishBearishRun(ExperimentalRun):
    """Bullish Assets fixed throughout experiment run; remaining bearish """
    
    @classmethod
    def generate_run_from_dir(
        cls, 
        name: str,
        source_news_dir: str,
        num_episodes: int = 10,
        seed: Optional[int] = None,
        allowed_models: Optional[List[str]] = None,
        fixed_assets: Optional[List[AssetType]] = None,
        insertion_config: Optional[Dict[str, Any]] = None,
    ) -> "FixedBullishBearishRun":
        
        if not seed:
            seed = cls.seed
        random.seed(seed)

        available_articles = cls.load_articles_from_directory(source_news_dir, allowed_models)
        
        if fixed_assets is None: 
            fixed_assets = [DEFAULT_BULLISH_ASSET]
        logging.info(f"Generating FixedBullishBearish experiment with bullish assets: {fixed_assets}")
        
        episodes = []
        for _ in range(num_episodes):
            
            selected_articles = []
            for asset_type in AssetType:
                if asset_type in fixed_assets:
                    bullish_articles = [a for a in available_articles[asset_type] if a.sentiment == Sentiment.BULLISH]
                    article = random.choice(bullish_articles)
                else:
                    bearish_articles = [a for a in available_articles[asset_type] if a.sentiment == Sentiment.BEARISH]
                    article = random.choice(bearish_articles)
                
                selected_articles.append(article)
            
            random.shuffle(selected_articles)
            episode = cls.create_episode_with_adversarial_insertion(selected_articles, insertion_config)
            episodes.append(episode)
        
        return cls(
            name=name,
            episodes=episodes,
            seed=seed,
            fixed_assets=fixed_assets,
            source_news_dir=source_news_dir,
            allowed_article_models=allowed_models,
            available_articles=available_articles
        )
    
class FixedBearishNeutralRun(ExperimentalRun):

    @classmethod
    def generate_run_from_dir(
        cls, 
        name: str,
        source_news_dir: str,
        num_episodes: int = 10,
        seed: Optional[int] = None,
        allowed_models: Optional[List[str]] = None,
        fixed_assets: Optional[List[AssetType]] = None,
        insertion_config: Optional[Dict[str, Any]] = None,
    ) -> "FixedBearishNeutralRun":
        
        if not seed:
            seed = cls.seed
        random.seed(seed)

        available_articles = cls.load_articles_from_directory(source_news_dir, allowed_models)
        
        print(f"fixed_assets at experimentalRun level -fBearishNeutral - {fixed_assets}")
        if fixed_assets is None: 
            fixed_assets = [DEFAULT_BEARISH_ASSET]
        logging.info(f"Generating FixedBearishNeutralRun experiment with bearish assets: {fixed_assets}")
        
        episodes = []
        for _ in range(num_episodes):
            
            selected_articles = []
            for asset_type in AssetType:
                if asset_type in fixed_assets:
                    bearish_articles = [a for a in available_articles[asset_type] if a.sentiment == Sentiment.BEARISH]
                    article = random.choice(bearish_articles)
                else:
                    neutral_articles = [a for a in available_articles[asset_type] if a.sentiment == Sentiment.NEUTRAL]
                    article = random.choice(neutral_articles)
                
                selected_articles.append(article)
            
            random.shuffle(selected_articles)
            episode = cls.create_episode_with_adversarial_insertion(selected_articles, insertion_config)
            episodes.append(episode)
        
        return cls(
            name=name,
            episodes=episodes,
            seed=seed,
            fixed_assets=fixed_assets,
            source_news_dir=source_news_dir,
            allowed_article_models=allowed_models,
            available_articles=available_articles
        )
    
class AllNeutralRun(ExperimentalRun):
    """All articles have neutral sentiment"""
    
    @classmethod
    def generate_run_from_dir(
        cls, 
        name: str,
        source_news_dir: str,
        num_episodes: int = 10,
        seed: Optional[int] = None,
        allowed_models: Optional[List[str]] = None,
        fixed_assets: Optional[List[AssetType]] = None,
        insertion_config: Optional[Dict[str, Any]] = None,
    ) -> "AllNeutralRun":
        
        if not seed:
            seed = cls.seed
        random.seed(seed)

        available_articles = cls.load_articles_from_directory(source_news_dir, allowed_models)
        
        episodes = []
        for _ in range(num_episodes):
                        
            selected_articles = []
            for asset_type in AssetType:
                neutral_articles = [a for a in available_articles[asset_type] if a.sentiment == Sentiment.NEUTRAL]
                article = random.choice(neutral_articles)

                selected_articles.append(article)
            
            random.shuffle(selected_articles)
            episode = cls.create_episode_with_adversarial_insertion(selected_articles, insertion_config)
            episodes.append(episode)
        
        return cls(
            name=name,
            episodes=episodes,
            seed=seed,
            fixed_assets=fixed_assets,
            source_news_dir=source_news_dir,
            allowed_article_models=allowed_models,
            available_articles=available_articles
        )
    

if __name__ == "__main__":
    eq_art = Article(file_path="./",asset_type=AssetType.EQUITIES, 
                     sentiment=Sentiment.BEARISH, model="chatgpt", repeat_id=1)
    mort_art = Article(file_path="./",asset_type=AssetType.MORTGAGES_BACKED_SECURITIES, 
                     sentiment=Sentiment.BEARISH, model="chatgpt", repeat_id=1)
    corp_art = Article(file_path="./",asset_type=AssetType.CORPORATE_BONDS, 
                     sentiment=Sentiment.BEARISH, model="chatgpt", repeat_id=1)
    gov_art = Article(file_path="./",asset_type=AssetType.GOVERNMENT_BONDS, 
                     sentiment=Sentiment.BEARISH, model="chatgpt", repeat_id=1)

    available_articles = {
        AssetType.EQUITIES: [eq_art],
        AssetType.CORPORATE_BONDS: [corp_art], 
        AssetType.GOVERNMENT_BONDS: [gov_art],
        AssetType.MORTGAGES_BACKED_SECURITIES: [mort_art]
    }
    
    # Generate experimental runs
    random_run = RandomSentimentRun.generate_run(
        name="Random Test",
        available_articles=available_articles,
        num_episodes=3,
        seed=42
    )

    print(random_run)