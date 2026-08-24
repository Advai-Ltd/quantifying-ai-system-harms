import unittest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from src.news_generator.generate import Generator, NewsMetadata, PromptVariables, NewsTemplate
import itertools


class TestGenerator(unittest.TestCase):
    
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_config = {
            "repeats_per_combination": 1,
            "models": ["test-model"],
            "asset_classes": ["Equities"],
            "sentiment": ["bullish"]
        }

        self.sample_articles = [
            "<html><h1>Test Article</h1><p>Content</p></html>"
        ]
        
        self.sample_metadata = [
            NewsMetadata(
                version=0,
                model="test-model", 
                asset_class="Equities",
                sentiment="bullish",
                timestamp="01012025_120000"
            )
        ]

    def test_generator_initialization_with_valid_config(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f)
            config_path = f.name
        
        try:
            with patch('src.news_generator.generate.LLM'), \
                    patch('src.news_generator.generate.PromptLoader'):
                generator = Generator(config_path=config_path, output_dir="./test_output")
                self.assertIsNotNone(generator.config)
                self.assertEqual(generator.output_dir, "./test_output")
        finally:
            os.unlink(config_path)

    def test_load_config_with_missing_file(self):
        """Test that missing config file raises FileNotFoundError."""
        with patch('src.news_generator.generate.LLM'), \
             patch('src.news_generator.generate.PromptLoader'):
            with self.assertRaises(FileNotFoundError):
                Generator(config_path="nonexistent.json")


    def test_prompt_variables_creation(self):
        """Test PromptVariables model creation."""
        prompt_vars = PromptVariables(asset_class="Equities", sentiment="bullish")
        self.assertEqual(prompt_vars.asset_class, "Equities")
        self.assertEqual(prompt_vars.sentiment, "bullish")


    @patch.object(Generator, 'load_provider_template')
    def test_generate_news_returns_correct_count(self, mock_load_template):
        """Test that generate_news returns expected number of articles."""
        # Mock just the HTML output
        mock_load_template.return_value = "<html>Mock Article</html>"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f)
            config_path = f.name
        
        try:
            generator = Generator(config_path=config_path)
            news, metadata = generator.generate_news()
            
            # Basic checks
            self.assertEqual(len(news), 1)
            self.assertEqual(len(metadata), 1)
            self.assertEqual(news[0], "<html>Mock Article</html>")
            
            mock_load_template.assert_called_once()
            
        finally:
            os.unlink(config_path)

    @patch('src.news_generator.generate.LLM')
    @patch('src.news_generator.generate.PromptLoader')
    def test_save_news_creates_files(self, mock_prompt_loader, mock_llm):
        """Test that save_news_to_file creates the correct files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.test_config, f)
            
            generator = Generator(config_path=config_path, output_dir=temp_dir)
            generator.save_news_to_file(self.sample_articles, self.sample_metadata)
            
            # Check files were created
            articles_dir = os.path.join(temp_dir, "articles")
            metadata_dir = os.path.join(temp_dir, "metadata")
            
            self.assertTrue(os.path.exists(articles_dir))
            self.assertTrue(os.path.exists(metadata_dir))
            self.assertEqual(len(os.listdir(articles_dir)), 1)
            self.assertEqual(len(os.listdir(metadata_dir)), 1)