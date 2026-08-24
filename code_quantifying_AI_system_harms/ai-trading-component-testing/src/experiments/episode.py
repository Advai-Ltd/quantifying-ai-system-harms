from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from enum import Enum
import random
import numpy as np
from experiments.simulation_types import Article, AssetType, Sentiment
from experiments.adversarial_inputs import TextInsertion

class Episode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    articles: List[Article] = Field(min_length=4, max_length=4)
    text_insertion: Optional[TextInsertion] = Field(default=None)
    model_origin: str = Field(default=None, description="Model used to generate articles in episode (None if mixed)")

    def __getitem__(self, idx: int):
        return self.articles[idx]

    @classmethod
    def from_articles_and_insertion(cls, articles: List[Article], 
                                    text_insertion: Optional[TextInsertion] = None):
        
        if not articles:
            raise ValueError("Articles list cannot be empty")
        
        models = {article.model for article in articles}
        model_origin = articles[0].model if len(models) == 1 else None

        return cls(
            articles=articles,
            model_origin=model_origin,
            text_insertion=text_insertion
        )

    @field_validator('articles')
    @classmethod
    def validate_one_per_asset(cls, v):
        """Ensure exactly one article per asset type."""
        asset_types = [article.asset_type for article in v]
        if set(asset_types) != set(AssetType):
            raise ValueError("Must include all 4 asset types")
        return v
    
    def randomize_order(self, seed: Optional[int] = None) -> "Episode":
        """Return new episode with same articles but randomized order."""
        if seed:
            random.seed(seed)
        
        indices = list(range(len(self.articles)))
        original_indices = indices.copy()
        
        new_indices = np.random.permutation(indices).tolist()
        while new_indices == original_indices:
            new_indices = np.random.permutation(indices).tolist()
        
        shuffled_articles = [self.articles[i] for i in new_indices]
        
        return Episode(
            articles=shuffled_articles,
            model_origin=self.model_origin
        )   
    
        
    def print_info(self) -> str:
        """Return basic episode information."""
        insertion_info = ""
        if self.text_insertion:
            insertion_info = f"\n  Insertion at idx {self.text_insertion.insertion_idx}: {self.text_insertion.attack_type.value}"
        else: 
            insertion_info = self.text_insertion
        
        return (
            f"Episode {self.id[:8]}:\n"
            f"  Model: {self.model_origin}\n"
            f"  Asset Type Order: {[article.asset_type.value for article in self.articles]}\n"
            f"  Sentiment Type Order: {[article.sentiment.value for article in self.articles]}\n"
            f"  Timestamp: {self.timestamp}\n"
            f"  Adversarial Insertion: {insertion_info}\n"
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
    episode = Episode.from_articles_and_insertion([eq_art,mort_art, corp_art, gov_art], 
                                                  text_insertion=None)

    print(episode.print_info())

    random_episode = episode.randomize_order(seed=42)
    print("\nRandomized:")
    print(random_episode.print_info())
