import json
import logging

import litellm
from litellm import completion, completion_cost, batch_completion
from litellm.utils import ModelResponse
from pydantic import BaseModel, Field, model_validator
from typing import Literal, List, Dict, Tuple, Optional
import os 
from model.prompt_loader import PromptLoader
from model.tools import TOOLS, Tools
from experiments.episode import Episode
from utils import make_dir_if_not_exists, extract_text_from_html


DATA_DIR = "./data"

litellm.enable_json_schema_validation = True
logger = logging.getLogger(__name__)


class JSONResponseModel(BaseModel):
    equities_weighting: float = Field(..., ge=0, le=1)
    corporate_bonds_weighting: float = Field(..., ge=0, le=1)
    government_bonds_weighting: float = Field(..., ge=0, le=1)
    mortgages_backed_securities_weighting: float = Field(..., ge=0, le=1)
    rationale: str

    @model_validator(mode="after")
    def validate_weightings_sum_to_one(self):
        total = (self.equities_weighting + self.corporate_bonds_weighting 
        + self.mortgages_backed_securities_weighting + self.government_bonds_weighting)

        if not total == 1.0:
            raise ValueError(f"Exposures must sum to 1; instead got: {total}")
        return self


class LLM:
    def __init__(
        self, 
        model: str = "", 
        mitigations: Optional[List[str]] = None
    ) -> None:
        self.prompt_loader = PromptLoader()
        self.tools = Tools()
        self.mitigations = mitigations or []

        prompt_dict = self._select_llm_prompts_by_mitigation()

        self.system_prompt = self.prompt_loader.render_template(prompt_dict['system_template'], {})
        self.articles_template_path = prompt_dict['articles_template']
        self.model = model
        self.cost = 0.0

    def _select_llm_prompts_by_mitigation(self) -> dict[str,str]:

        prompt_templates = {"system_template": None, "articles_template": None}
        for k,v in prompt_templates.items():
            path_value = k
            if "prompt-hardening" in self.mitigations:
                path_value += "_hardened"
            
            path_value += ".j2"
            prompt_templates[k] = path_value
        return prompt_templates

    def call_model_with_tools(self,  messages: List[str]):
        """Call the actual language model API."""

        response = completion(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            num_retries=3,
        )
        self.cost += completion_cost(completion_response=response)
        return response

    def call_model_with_output_format(self, response_format, messages: List[str]):
        """Call the actual language model API."""

        response = completion(
            model=self.model,
            messages=messages,
            response_format=response_format,
            num_retries=6,
        )
        logger.info(f"Run Cost: ${completion_cost(completion_response=response):.4f}")
        self.cost += completion_cost(completion_response=response)

        return response

    def call_tool(self, tool_name: str, tool_input: dict) -> tuple[str, list[str]]:
        """Call a tool and return its output."""
        if tool_name == "get_news_articles":
            articles = self.tools.get_news_articles()
            logging.info(f"Articles: {articles}")
            article_names_filepaths = [(article["title"],article['file_path']) for article in articles]
            prompt = self.prompt_loader.render_template(
                self.articles_template_path, {"articles": articles}
            )
            return prompt, article_names_filepaths
        raise ValueError("Failed to call tool.")

    def _process_tool_calls(self, completion):
        """Process tool calls from the model's completion."""
        if not completion.choices[0].message.tool_calls:
            return []

        for tool_call in completion.choices[0].message.tool_calls:
            logging.info(
                f"Calling tool: {tool_call.function.name} with args {tool_call.function.arguments}"
            )
            args = json.loads(tool_call.function.arguments)
            result, article_names_filepaths = self.call_tool(tool_call.function.name, args)

            self.messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
            )
            return article_names_filepaths

    def get_total_cost(self) -> float:
        """Get the total cost incurred so far."""
        return self.cost

    def set_model(self, model: str) -> None:
        """Set the model to be used for LLM calls."""
        self.model = model

    def run_on_random_articles(self):
        user_message = self.prompt_loader.render_template("tool_call_template.j2", {})

        self.messages.append({"role": "user", "content": user_message})

        # Call model to get tool calls
        completion = self.call_model_with_tools()
        self.messages.append(completion["choices"][0]["message"])

        article_names_filepaths = self._process_tool_calls(completion)

        # Call model to get final response
        final_response = self.call_model_with_output_format(
            response_format=JSONResponseModel
        )
        logging.info("Final response from model:")
        logging.info(final_response["choices"][0]["message"]["content"])

        return json.loads(
            final_response["choices"][0]["message"]["content"]
        ), article_names_filepaths
    
    @staticmethod
    def collect_article_text(article_paths: List[str]) -> List[str]:

        articles = []
        for path in article_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                extracted_text = extract_text_from_html(html_content)
                article_name = os.path.basename(path).replace(".html", "").replace("_", " ").title()
                
                article = {
                    "title": article_name,
                    "content": extracted_text,
                }
                articles.append(article)
                
            except Exception as e:
                logger.error(f"Error loading {path}: {str(e)}")
                continue
        
        return articles
    
    @staticmethod
    def collect_article_text_with_insertions(episode: Episode) -> List[Dict[str, str]]:

        articles = episode.articles
        article_paths = [article.file_path for article in articles]
        article_texts = LLM.collect_article_text(article_paths=article_paths)

        if episode.text_insertion:
            idx = episode.text_insertion.insertion_idx
            article_texts[idx]['content'] += f"\n\n{episode.text_insertion.text}"
        
        return article_texts
    
    def run_with_episode(self, episode: Episode) -> Tuple[Dict, float]:
        articles = self.collect_article_text_with_insertions(episode)

        prompt = self.prompt_loader.render_template(
            self.articles_template_path, {"articles": articles}
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        logging.info(f"Submitting article content to model {self.model}")
        raw_response = completion(
            model=self.model,
            messages=messages,
            response_format=JSONResponseModel,
            num_retries=6,
        )
        raw_response_cost = completion_cost(raw_response)
        logger.info(f"Run Cost: ${raw_response_cost:.4f}")
        self.cost += raw_response_cost

        response = json.loads(raw_response["choices"][0]["message"]["content"])
        
        logging.info(f"Final response from model {self.model}:")
        logging.info(response)

        return response, raw_response_cost

    def run_batch_with_episodes(
            self, 
            batch_episodes: List[Episode],
    ) -> List[dict[str]]:
        
        batch_messages = []

        for episode in batch_episodes:
            
            # article_paths = [article.file_path for article in episode.articles]
            articles = self.collect_article_text_with_insertions(episode)

            prompt = self.prompt_loader.render_template(
                self.articles_template_path, {"articles": articles}
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            batch_messages.append(messages)

        logger.info(f"Submitting batch of {len(batch_messages)} episodes to {self.model}")
        
        raw_responses = batch_completion(
            model=self.model,
            messages=batch_messages,
            response_format=JSONResponseModel,
            num_retries=6
        )
        
        responses, raw_response_costs = [],[]
        batch_cost = 0.0

        for idx, raw_response in enumerate(raw_responses):
            try:
                if not isinstance(raw_response, ModelResponse) or "choices" not in raw_response:
                    print(raw_response)
                    raise ValueError(f"Response is not ModelResponse: {type(raw_response).__name__}")
        
                parsed = json.loads(raw_response["choices"][0]["message"]["content"])
                responses.append(parsed)
                
                raw_response_cost = completion_cost(completion_response=raw_response)
                raw_response_costs.append(raw_response_cost)
                self.cost += raw_response_cost
                batch_cost += raw_response_cost
                
                logger.info(f"Model {self.model} - Episode {idx+1}/{len(batch_episodes)} - Cost: ${raw_response_cost:.4f}")
            except Exception as e:
                logger.error(f"Failed to parse model {self.model} response for episode {idx+1}/{len(batch_episodes)}: {type(e).__name__}: {e}")
                responses.append(None)
                raw_response_costs.append(0.0)
        
        logger.info(f"Model {self.model} Batch completed. Total batch cost: ${batch_cost:.4f}")
        
        return responses, raw_response_costs



    