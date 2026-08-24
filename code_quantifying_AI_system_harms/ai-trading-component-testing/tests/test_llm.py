import pytest
from typing import Optional
from src.model.llm import JSONResponseModel, LLM
from unittest.mock import Mock, patch
import json

class TestResponseValidation:

    def test_weightings_throws_value_error(self):

        with pytest.raises(ValueError):

            bad_json_instance = JSONResponseModel(equity_weighting=0.2, corporate_bonds_weighting=0.6,
                                                   mortgage_backed_securities_weighting=0.2, government_bonds_weighting=0.1)


class TestLLM:

    def test_call_tool_returns_articles_with_mock(self):
        """Test call_tool returns article data using Mock."""
        
        # Create a mock Tools instance
        mock_tools = Mock()
        mock_tools.get_news_articles.return_value = [
            {
                "title": "Test Article",
                "file_path": "/test.txt", 
                "date": "Today",
                "content": "Test content"
            }
        ]
        
        # Create LLM and inject the mock
        llm = LLM(model="test-model")
        llm.tools = mock_tools  # Replace the real tools with our mock
        
        # Test the method
        _, articles = llm.call_tool("get_news_articles", {})
        
        assert len(articles) == 1
        assert articles[0][0] == "Test Article"
        assert articles[0][1] == "/test.txt"
        
        # Verify the mock was called
        mock_tools.get_news_articles.assert_called_once()

    def test_unknown_tool_raises_error(self):
        """Test calling unknown tool fails."""
        with patch('src.model.tools.Tools.get_news_articles'), patch('src.model.prompt_loader.PromptLoader'):
            llm = LLM(model="test-model")
            
            with pytest.raises(Exception):
                llm.call_tool("fake_tool", {})

    def test_run_with_article_filepaths(self, tmp_path):
        """Test run_with_article_filepaths loads articles and calls model."""
    
        # Create temporary HTML article files
        article_paths = []
        for i in range(4):
            article_file = tmp_path / f"test_article_{i}.html"
            article_file.write_text(f"<html><body><h1>Article {i}</h1><p>Content for article {i}</p></body></html>")
            article_paths.append(str(article_file))
        
        # Mock the completion response
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "equity_weighting": 0.4,
                        "corporate_bonds_weighting": 0.3,
                        "government_bonds_weighting": 0.2,
                        "mortgage_backed_securities_weighting": 0.1,
                        "rationale": "Test rationale"
                    })
                }
            }]
        }
        
        with patch('src.model.llm.completion', return_value=mock_response), \
            patch('src.model.llm.completion_cost', return_value=0.001), \
            patch('src.model.prompt_loader.PromptLoader'):
            
            llm = LLM(model="test-model")
            response = llm.run_with_article_filepaths(article_paths)
            
            # Verify response structure
            assert response["equity_weighting"] == 0.4
            assert response["corporate_bonds_weighting"] == 0.3
            assert response["government_bonds_weighting"] == 0.2
            assert response["mortgage_backed_securities_weighting"] == 0.1
            assert response["rationale"] == "Test rationale"
            
            assert len(llm.messages) == 2
            assert llm.messages[1]["role"] == "user"

    def test_run_with_article_filepaths_maintains_article_order(self, tmp_path):
        """Test that articles are processed in the order provided."""
        
        # Create articles with numbered content
        article_paths = []
        for i in range(4):
            article_file = tmp_path / f"article_{i}.html"
            article_file.write_text(f"<html><body><h1>Article Number {i}</h1><p>Position {i}</p></body></html>")
            article_paths.append(str(article_file))
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "equity_weighting": 0.1,
                        "corporate_bonds_weighting": 0.2,
                        "government_bonds_weighting": 0.3,
                        "mortgage_backed_securities_weighting": 0.4,
                        "rationale": "Weighted by position"
                    })
                }
            }]
        }
        
        with patch('src.model.llm.completion', return_value=mock_response) as mock_completion, \
            patch('src.model.llm.completion_cost', return_value=0.001), \
            patch('src.model.prompt_loader.PromptLoader'):
            
            llm = LLM(model="test-model")
            response = llm.run_with_article_filepaths(article_paths)
            
            # Check the order in the prompt sent to the model
            call_args = mock_completion.call_args
            messages = call_args.kwargs['messages']
            user_message = messages[1]['content']
            
            # Find positions of each article in the message
            pos_0 = user_message.find("Article Number 0")
            pos_1 = user_message.find("Article Number 1")
            pos_2 = user_message.find("Article Number 2")
            pos_3 = user_message.find("Article Number 3")
            
            # Verify they appear in order
            assert pos_0 < pos_1 < pos_2 < pos_3
            assert response["rationale"] == "Weighted by position"
