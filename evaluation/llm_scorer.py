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
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

from src import LLMManager, check_api_keys, LLMType, setup_logging


setup_logging()


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
        
        self.llm_manager = LLMManager(check_api_keys())
        logging.info("LLM Manager initialized with available API keys")
        
        # Map provider and model_name to LLMType
        if provider.lower() == 'google' and model_name == 'gemini-2.5-flash-preview-04-17':
            llm_type = LLMType.GEMINI_25_FLASH
        elif provider.lower() == 'anthropic' and model_name == 'claude-3-5-sonnet-20240620':
            llm_type = LLMType.CLAUDE_35_SONNET
        elif provider.lower() == 'openai' and model_name == 'gpt-4o-mini':
            llm_type = LLMType.GPT4O_MINI
        else:
            raise ValueError(f"Unsupported model combination: {provider} - {model_name}")
            
        self.llm_manager.select_llm(llm_type)
        self.llm = self.llm_manager.get_current_llm()
        
    def _load_responses(self) -> list:
        """Load responses from the specified file."""
        logging.info(f"Loading responses from {self.filename}")
        with open(self.filename, 'r') as f:
            responses = json.load(f)
        logging.info(f"Loaded {len(responses)} responses")
        return responses
            
    def _score_correctness(self, response: str, ground_truth: str) -> float:
        """Score the correctness of the response compared to ground truth."""
        prompt = f"""Rate the correctness of the response compared to the ground truth on a scale of 0-1.
        Consider if the response contains the key information from the ground truth.
        provide only score, no other text.
        
        Ground Truth: {ground_truth}
        Response: {response}
        
        Score (0-1):"""
        
        result = self.llm.invoke(prompt).content

        try:
            score = float(result.strip())
            logging.info(f"Correctness score: {score}")
            return score
        except ValueError:
            logging.warning("Failed to parse correctness score, returning 0.0")
            return 0.0
            
    def _score_style(self, response: str) -> float:
        """Score the style of the response for engagement and friendliness."""
        prompt = f"""Rate the style of this chatbot response on a scale of 0-1.
        Consider if it is engaging, friendly, and appropriate for a chatbot.
        provide only score, no other text.
        
        Response: {response}
        
        Score (0-1):"""
        
        result = self.llm.invoke(prompt).content
        print(result)
        try:
            score = float(result.strip())
            logging.info(f"Style score: {score}")
            return score
        except ValueError:
            logging.warning("Failed to parse style score, returning 0.0")
            return 0.0
            
    def _score_response_time(self, response_time: float) -> float:
        """Score the response time on a scale of 0-1."""
        logging.debug(f"Scoring response time: {response_time}s")
        # Assuming response time is in seconds
        # Score decreases as response time increases
        # Perfect score (1.0) for responses under 1 second
        # Minimum score (0.0) for responses over 10 seconds
        if response_time <= 1.0:
            score = 1.0
        elif response_time >= 10.0:
            score = 0.0
        else:
            score = 1.0 - ((response_time - 1.0) / 9.0)
        logging.debug(f"Response time score: {score}")
        return score
    
    def _score_single_response(self, response_data: Dict[str, Any]) -> Dict[str, float]:
        """Score a single response using all criteria."""
        logging.info("Scoring single response")
        response = response_data['answer']
        ground_truth = response_data['ground_truth']
        response_time = response_data['response_time']
        
        score = {
            'correctness': self._score_correctness(response, ground_truth),
            'style': self._score_style(response),
            'response_time': self._score_response_time(response_time),
            'model_name': response_data['model_name'],
            'temperature': response_data['temperature']
        }
        logging.info(f"Response scored: {score}")
        return score
            
    def score_responses(self, max_workers: int = 4, use_multithreading: bool = True) -> list:
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
                    executor.submit(self._score_single_response, response_data): response_data 
                    for response_data in responses
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_response):
                    completed += 1
                    logging.info(f"Progress: {completed}/{len(responses)} responses scored")
                    try:
                        score = future.result()
                        scores.append(score)
                    except Exception as e:
                        logging.error(f"Error processing response: {e}")
                        scores.append({
                            'correctness': 0.0,
                            'style': 0.0,
                            'response_time': 0.0,
                            'model_name': response_data['model_name'],
                            'temperature': response_data['temperature']
                        })
        else:
            logging.info("Starting sequential scoring")
            for i, response_data in enumerate(responses, 1):
                logging.info(f"Progress: {i}/{len(responses)} responses scored")
                try:
                    score = self._score_single_response(response_data)
                    scores.append(score)
                except Exception as e:
                    logging.error(f"Error processing response: {e}")
                    scores.append({
                        'correctness': 0.0,
                        'style': 0.0,
                        'response_time': 0.0,
                        'model_name': response_data['model_name'],
                        'temperature': response_data['temperature']
                    })
        
        logging.info(f"Completed scoring {len(scores)} responses")
        return scores


@click.command()
@click.option('--config', type=click.Path(exists=True), default='evaluation/configs/batch_config.yaml', help='Path to the configuration file')
@click.option('--filename', type=click.Path(exists=True), help='Path to the file containing responses to score (overrides config)')
@click.option('--model_name', type=str, help='Name of the LLM model to use for scoring (overrides config)')
@click.option('--provider', type=str, help='Provider of the LLM model (overrides config)')
@click.option('--output_file', type=click.Path(), help='Path to save the output CSV file (overrides config)')
@click.option('--max_workers', type=int, help='Maximum number of worker threads to use (overrides config)')
@click.option('--use-multithreading/--no-multithreading', help='Whether to use multithreading for scoring (overrides config)')
def main(config: str, filename: str, model_name: str, provider: str, output_file: str, max_workers: int, use_multithreading: bool):
    # Load configuration
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Use command line arguments if provided, otherwise use config values
    filename = filename or config_data.get('chatbot_results_path')
    model_name = model_name or config_data.get('scoring_model_name')
    provider = provider or config_data.get('scoring_model_provider')
    output_file = output_file or config_data.get('scoring_output_file')
    max_workers = max_workers or config_data.get('scoring_max_workers', 8)
    use_multithreading = use_multithreading if use_multithreading is not None else config_data.get('scoring_use_multithreading', True)
    
    logging.info(f"Starting LLM scoring process with model: {model_name} from {provider}")
    logging.info(f"Multithreading: {'enabled' if use_multithreading else 'disabled'}")
    
    scorer = LLMScorer(filename, model_name, provider)
    scores = scorer.score_responses(max_workers=max_workers, use_multithreading=use_multithreading)
    
    # Generate default output filename if not provided
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        output_file = f'evaluation/chatbot_results/llm_score_{timestamp}.csv'
    
    logging.info(f"Saving scores to {output_file}")
    
    # Ensure the output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save scores to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['correctness', 'style', 'response_time', 'model_name', 'temperature'])
        writer.writeheader()
        writer.writerows(scores)
    
    logging.info(f"Successfully saved {len(scores)} scores to {output_file}")

if __name__ == '__main__':
    main()