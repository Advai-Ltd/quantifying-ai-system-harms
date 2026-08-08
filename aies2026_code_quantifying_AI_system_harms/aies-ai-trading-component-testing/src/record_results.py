import csv
import logging
import os
import uuid
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from experiments.simulation_types import AssetType
from experiments.episode import Episode
from experiments.experimental_run import ExperimentalRun
from pathlib import Path 

RESPONSE_DIR = "./responses/"


class ResultsRecorder:
    """Class to handle recording LLM results to CSV file."""

    def __init__(self, 
                experimental_run: 'ExperimentalRun',
                results_output_dir: str = "./responses", 
                csv_basename: str = "results.csv"):
        """Initialize the ResultsRecorder.
        :param csv_file_path: Path to the CSV file where results will be stored
        """
        self.experimental_run = experimental_run
        self.results_dir = results_output_dir
        self.csv_file_path = os.path.join(self.results_dir, csv_basename)

        self.fieldnames = [
            "timestamp",
            "experiment_id",
            "experiment_type",
            "fixed_asset_type", 
            "experiment_name",
            "episode_id", 
            "episode_article_dir",
            "has_attack",
            "attack_type",
            "attack_target_asset",
            "insertion_idx",
            "attack_text",
            "mitigations",
            "article_1_filename",
            "article_1_fields", #JSON dict, e.g. : {"asset": "Equities", "sentiment": "bullish", "repeat": 1, "model": ".."}
            "article_2_filename",
            "article_2_fields",
            "article_3_filename",
            "article_3_fields",
            "article_4_filename",  
            "article_4_fields",
            "inference_model",
            "equities_weighting",
            "corporate_bonds_weighting",
            "government_bonds_weighting",
            "mortgages_backed_securities_weighting",
            "response_file",
            "cost"]
        
        self.logger = logging.getLogger(__name__)
        self._ensure_csv_exists()
        self._ensure_response_dir_exists()

    def _ensure_response_dir_exists(self) -> None:
        """Ensure the response directory exists."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            self.logger.info(f"Created response directory: {self.results_dir}")
        else:
            self.logger.debug(f"Response directory already exists: {self.results_dir}")

    def _ensure_csv_exists(self) -> None:
        """Check if CSV file exists, create it with headers if it doesn't."""
        if not os.path.exists(self.csv_file_path):
            self._create_csv_with_headers()
        else:
            self.logger.debug(f"CSV file already exists: {self.csv_file_path}")

    def _create_csv_with_headers(self) -> None:
        """Create a new CSV file with the required headers."""
        with open(self.csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            self.logger.info(f"Created new CSV file: {self.csv_file_path}")

    def save_response(self, response: str) -> str:
        """Save the response to a text file for backup.

        :param response: The LLM's response/output
        """
        response_file_path: str = os.path.join(self.results_dir, f"{uuid.uuid4()}.json")

        try:
            with open(response_file_path, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Response saved to {response_file_path}")
            return response_file_path
        except Exception as e:
            self.logger.error(f"Error saving response to file: {e}")
            raise

    def record_result(
        self, episode: 'Episode', model: str, response: dict[str, Any], episode_cost: float,
        mitigations: Optional[List[str]] = None
    ) -> None:
        """Record a single LLM result to the CSV file.

        :param data: The input data for the new tool
        :param model: The model used to generate the response
        :param topic_files: The topics or input data processed
        :param response: The LLM's response/output
        """

        experiment_type = self.experimental_run.experiment_type
        experiment_fixed_assets = self.experimental_run.fixed_assets
        if isinstance(experiment_fixed_assets, list) and len(experiment_fixed_assets) == 1:
            experiment_fixed_assets = experiment_fixed_assets[0]

        episode_cost = 0.0 if episode_cost is None else episode_cost

        response_file_path = self.save_response(response)
        article_paths = [article.file_path for article in episode.articles]

        article_parentdir_set = {Path(article).parent for article in article_paths}
        if len(article_parentdir_set) != 1: 
            raise ValueError("Number of unique parent directories of articles in episode was not 1")
        
        article_filenames = [os.path.basename(article_fp) for article_fp in article_paths]
        article_fields = [article.model_dump_json(exclude={"file_path"}) for article in episode.articles]

        has_attack = episode.text_insertion is not None
        attack_type = episode.text_insertion.attack_type.value if has_attack else None
        attack_target_asset = episode.text_insertion.target_asset.value if has_attack else None
        insertion_idx = episode.text_insertion.insertion_idx if has_attack else None
        attack_text = episode.text_insertion.text if has_attack else None
        
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experimental_run.id,
            "experiment_type": experiment_type, 
            "fixed_asset_type": experiment_fixed_assets, 
            "experiment_name": self.experimental_run.name,
            "episode_id": episode.id,
            "episode_article_dir": list(article_parentdir_set)[0],
            "has_attack": has_attack, 
            "attack_type": attack_type,
            "attack_target_asset": attack_target_asset,
            "insertion_idx": insertion_idx,
            "attack_text": attack_text,
            "mitigations": json.dumps(mitigations) if mitigations else None,
            "article_1_filename":article_filenames[0],
            "article_1_fields": article_fields[0],
            "article_2_filename":article_filenames[1],
            "article_2_fields": article_fields[1],
            "article_3_filename": article_filenames[2],
            "article_3_fields": article_fields[2],
            "article_4_filename": article_filenames[3],  
            "article_4_fields": article_fields[3],
            "inference_model": model,
            "equities_weighting": response["equities_weighting"],
            "corporate_bonds_weighting": response["corporate_bonds_weighting"],
            "government_bonds_weighting": response["government_bonds_weighting"],
            "mortgages_backed_securities_weighting": response["mortgages_backed_securities_weighting"],
            "response_file": response_file_path,
            "cost": episode_cost
        }
        
        try:
            with open(self.csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(result_data)
                self.logger.info(
                    f"Successfully recorded result to {self.csv_file_path}"
                )
        except Exception as e:
            self.logger.error(f"Error recording result: {e}")
            raise

    def record_multiple_results(self, results: list[Dict[str, Any]]) -> None:
        """
        Record multiple LLM results to the CSV file.

        :param results: List of dictionaries containing result data
        """
        try:
            with open(self.csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                for result in results:
                    writer.writerow(result)
                self.logger.info(
                    f"Successfully recorded {len(results)} results to {self.csv_file_path}"
                )
        except Exception as e:
            self.logger.error(f"Error recording multiple results: {e}")
            raise

    def get_existing_results_count(self) -> int:
        """
        Get the number of existing results in the CSV file.

        :return: Number of rows in the CSV (excluding header)
        """
        if not os.path.exists(self.csv_file_path):
            self.logger.debug(f"CSV file does not exist: {self.csv_file_path}")
            return 0

        try:
            with open(self.csv_file_path, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                # Skip header and count remaining rows
                next(reader, None)  # Skip header
                count = sum(1 for _ in reader)
                self.logger.debug(
                    f"Found {count} existing results in {self.csv_file_path}"
                )
                return count
        except Exception as e:
            self.logger.error(f"Error counting existing results: {e}")
            return 0
