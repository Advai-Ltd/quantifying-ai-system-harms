import pytest
from src.experiments.episode import Episode, Article, AssetType, Sentiment

class TestEpisode:
    
    def setup_method(self):
        self.eq_art = Article(file_path="./test1.html", asset_type=AssetType.EQUITIES, 
                             sentiment=Sentiment.BEARISH, model="test-model", repeat_id=1)
        self.corp_art = Article(file_path="./test2.html", asset_type=AssetType.CORPORATE_BONDS, 
                               sentiment=Sentiment.BULLISH, model="test-model", repeat_id=1)
        self.gov_art = Article(file_path="./test3.html", asset_type=AssetType.GOVERNMENT_BONDS, 
                              sentiment=Sentiment.NEUTRAL, model="test-model", repeat_id=1)
        self.mbs_art = Article(file_path="./test4.html", asset_type=AssetType.MORTGAGES_BACKED_SECURITIES, 
                              sentiment=Sentiment.BEARISH, model="test-model", repeat_id=1)
    
    def test_episode_creation_from_articles(self):
        episode = Episode.from_articles([self.eq_art, self.corp_art, self.gov_art, self.mbs_art])
        
        assert len(episode.articles) == 4
        assert episode.model_origin == "test-model"
        assert episode.id is not None
        assert len({art.asset_type for art in episode.articles}) == 4
    
    def test_episode_rejects_mixed_models(self):
        bad_article = Article(file_path="./bad.html", asset_type=AssetType.EQUITIES, 
                             sentiment=Sentiment.BEARISH, model="different-model", repeat_id=1)
        
        with pytest.raises(ValueError) as excinfo:
            Episode.from_articles([bad_article, self.corp_art, self.gov_art, self.mbs_art])
        
        assert "same model" in str(excinfo.value)
    
    def test_randomize_order_creates_new_episode(self):
        episode = Episode.from_articles([self.eq_art, self.corp_art, self.gov_art, self.mbs_art])
        random_episode = episode.randomize_order(seed=42)
        
        assert episode.id != random_episode.id
        assert episode.model_origin == random_episode.model_origin
        
        # Same articles, different order
        assert episode.articles != random_episode.articles
        episode_asset_types = {a.asset_type for a in episode.articles}
        random_asset_types = {a.asset_type for a in random_episode.articles}
        assert episode_asset_types == random_asset_types
