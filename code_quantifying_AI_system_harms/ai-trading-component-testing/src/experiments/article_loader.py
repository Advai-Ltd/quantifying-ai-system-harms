import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from experiments.episode import Article, AssetType, Sentiment
from experiments.experimental_run import RandomSentimentRun

def load_articles_from_generated_news(news_dir: str, allowed_models: Optional[List[str]]) -> Dict[AssetType, List[Article]]:
    """Load articles from generated news directory.
    Args:
        news_dir: Directory containing articles and metadata
        allowed_models: Optional list of model names to filter by. 
                       If None, load all articles.
    
    Returns:
        Dictionary mapping AssetType to list of Articles
    """

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
    
    return articles_by_asset


if __name__=="__main__":

    available_articles = load_articles_from_generated_news("./data/dummy/2025_11_28-17_59_47")

    random_run = RandomSentimentRun.generate_run(
        name='Test Loader', 
        available_articles=available_articles,
        num_episodes=10,
        seed=100
    )

    print("Oth element (episode) of experimental run")
    print(random_run[0]) 

    print("Oth element (article) of episode above")
    print(random_run[0][0]) 