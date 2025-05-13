"""
This script is used to score the chatbot responses using other LLM.

The score is a list of values based on the following criteria:
- correctness (from the given ground_truth value)
- style (this is a chatbot response and engaging and friendly)
- response time (from the given response_time value)

Create a LLMScorer class that takes a filename and a model name and provider (which
is a LLM judge) and returns a list of scores.

"""

import click
import json
import csv
import logging
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Protocol

from components import LLMManager, check_api_keys, LLMType, setup_logging


setup_logging()


class ScoringStrategy(ABC):
    """Abstract base class for scoring strategies"""

    @abstractmethod
    def score(self, data: Dict[str, Any]) -> float:
        pass


class CorrectnessScoringStrategy(ScoringStrategy):
    def __init__(self, llm):
        self.llm = llm

    def score(self, data: Dict[str, Any]) -> float:
        prompt = f"""Rate the correctness of the response compared to the ground truth on a scale of 0-1.
        Consider if the response contains the key information from the ground truth.
        0 means the response is incorrect or does not contain the key information from the ground truth.
        1 means the response is correct and contains the key information from the ground truth.
        the values between 0 and 1 are for the response that is close to the ground truth or missing some information.
        provide only score, no other text.
        
        Ground Truth: {data["ground_truth"]}
        Response: {data["answer"]}
        
        Score (0-1):"""

        result = self.llm.invoke(prompt).content
        try:
            return float(result.strip())
        except ValueError:
            logging.warning("Failed to parse correctness score, returning 0.0")
            return 0.0


class StyleScoringStrategy(ScoringStrategy):
    def __init__(self, llm):
        self.llm = llm

    def score(self, data: Dict[str, Any]) -> float:
        prompt = f"""Rate the style of this chatbot response on a scale of 0-1.
        Consider if the response is engaging, friendly, and appropriate for a chatbot.
        0 means the response has only a few words and is not engaging.
        1 means the response is engaging, friendly, and appropriate for a chatbot.
        provide only score, no other text.
        
        Response: {data["answer"]}
        
        Score (0-1):"""

        result = self.llm.invoke(prompt).content
        try:
            return float(result.strip())
        except ValueError:
            logging.warning("Failed to parse style score, returning 0.0")
            return 0.0


class ResponseTimeScoringStrategy(ScoringStrategy):
    def score(self, data: Dict[str, Any]) -> float:
        response_time = data["response_time"]
        if response_time <= 1.0:
            return 1.0
        elif response_time >= 15.0:
            return 0.0
        return 1.0 - ((response_time - 1.0) / 14.0)


class LLMFactory:
    """Factory for creating LLM instances"""

    @staticmethod
    def create_llm(provider: str, model_name: str) -> Any:
        llm_manager = LLMManager(check_api_keys())

        if (
            provider.lower() == "google"
            and model_name == "gemini-2.5-flash-preview-04-17"
        ):
            llm_type = LLMType.GEMINI_25_FLASH
        elif (
            provider.lower() == "anthropic"
            and model_name == "claude-3-5-sonnet-20240620"
        ):
            llm_type = LLMType.CLAUDE_35_SONNET
        elif provider.lower() == "openai" and model_name == "gpt-4o-mini":
            llm_type = LLMType.GPT4O_MINI
        else:
            raise ValueError(
                f"Unsupported model combination: {provider} - {model_name}"
            )

        llm_manager.select_llm(llm_type)
        return llm_manager.get_current_llm()


class ProgressObserver(Protocol):
    def update(self, completed: int, total: int) -> None:
        pass


class LoggingProgressObserver:
    def update(self, completed: int, total: int) -> None:
        logging.info(f"Progress: {completed}/{total} responses scored")


class LLMScorer:
    def __init__(self, filename: str, model_name: str, provider: str):
        """
        Initialize the LLMScorer with the given parameters.

        Args:
            filename (str): Path to the file containing responses to score
            model_name (str): Name of the LLM model to use for scoring
            provider (str): Provider of the LLM model
        """

        self.filename = filename
        logging.info(f"Initializing LLMScorer with file: {filename}")

        self.llm = LLMFactory.create_llm(provider, model_name)
        self.scoring_strategies = {
            "correctness": CorrectnessScoringStrategy(self.llm),
            "style": StyleScoringStrategy(self.llm),
            "response_time": ResponseTimeScoringStrategy(),
        }
        self.observers: List[ProgressObserver] = [LoggingProgressObserver()]

    def add_observer(self, observer: ProgressObserver) -> None:
        self.observers.append(observer)

    def _notify_observers(self, completed: int, total: int) -> None:
        for observer in self.observers:
            observer.update(completed, total)

    def _load_responses(self) -> list:
        """Load responses from the specified file."""
        logging.info(f"Loading responses from {self.filename}")
        with open(self.filename, "r") as f:
            responses = json.load(f)
        logging.info(f"Loaded {len(responses)} responses")
        return responses

    def _score_single_response(self, response_data: Dict[str, Any]) -> Dict[str, float]:
        """Score a single response using all criteria."""

        logging.info("Scoring single response")
        scores = {
            name: strategy.score(response_data)
            for name, strategy in self.scoring_strategies.items()
        }
        scores.update(
            {
                "id": response_data["id"],
                "model_name": response_data["model_name"],
                "temperature": response_data["temperature"],
                "difficulty": response_data["difficulty"],
            }
        )
        logging.debug(f"Response scored: {scores}")
        return scores

    def score_responses(
        self, max_workers: int = 4, use_multithreading: bool = True
    ) -> list:
        """
        Score all responses in the file and return a list of scores.

        Args:
            max_workers (int): Maximum number of worker threads to use
            use_multithreading (bool): Whether to use multithreading for scoring

        Returns:
            list: List of dictionaries containing scores for each response
        """

        responses = self._load_responses()
        scores = []

        if use_multithreading:
            logging.info(f"Starting parallel scoring with {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all scoring tasks
                future_to_response = {
                    executor.submit(
                        self._score_single_response, response_data
                    ): response_data
                    for response_data in responses
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_response):
                    completed += 1
                    self._notify_observers(completed, len(responses))
                    try:
                        scores.append(future.result())
                    except Exception as e:
                        response_data = future_to_response[future]
                        logging.error(f"Error processing response: {e}")
                        scores.append(self._create_default_score(response_data))
        else:
            logging.info("Starting sequential scoring")
            response_data = None
            for i, response_data in enumerate(responses, 1):
                self._notify_observers(i, len(responses))
                try:
                    scores.append(self._score_single_response(response_data))
                except Exception as e:
                    logging.error(f"Error processing response: {e}")
                    scores.append(self._create_default_score(response_data))

        logging.info(f"Completed scoring {len(scores)} responses")
        return scores

    def _create_default_score(self, response_data: Dict[str, Any]) -> Dict[str, float]:
        return {
            "correctness": 0.0,
            "style": 0.0,
            "response_time": 0.0,
            "model_name": response_data["model_name"],
            "temperature": response_data["temperature"],
        }


class ScoringCommand:
    """Command pattern implementation for scoring operation"""

    def __init__(self, scorer: LLMScorer, output_file: str):
        self.scorer = scorer
        self.output_file = output_file

    def execute(self, max_workers: int, use_multithreading: bool) -> None:
        scores = self.scorer.score_responses(max_workers, use_multithreading)
        self._save_scores(scores)

    def _save_scores(self, scores: List[Dict[str, float]]) -> None:
        logging.info(f"Saving scores to {self.output_file}")
        # Ensure the output directory exists
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

        # Save scores to CSV
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "correctness",
                    "style",
                    "response_time",
                    "model_name",
                    "temperature",
                    "difficulty",
                ],
            )
            writer.writeheader()
            writer.writerows(scores)

        logging.info(f"Successfully saved {len(scores)} scores to {self.output_file}")


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="evaluation/configs/batch_config.yaml",
    help="Path to the configuration file",
)
def main(config: str):
    # Load configuration
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)

    # Use command line arguments if provided, otherwise use config values
    filename = config_data.get("chatbot_results_path")
    model_name = config_data.get("scoring_model_name")
    provider = config_data.get("scoring_model_provider")
    output_file = config_data.get("scoring_output_file")
    max_workers = config_data.get("scoring_max_workers", 8)
    use_multithreading = config_data.get("scoring_use_multithreading", True)

    logging.info(
        f"Starting LLM scoring process with model: {model_name} from {provider}"
    )
    logging.info(f"Multithreading: {'enabled' if use_multithreading else 'disabled'}")

    scorer = LLMScorer(filename, model_name, provider)
    command = ScoringCommand(scorer, output_file)
    command.execute(max_workers, use_multithreading)


if __name__ == "__main__":
    main()
