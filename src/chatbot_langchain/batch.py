"""
Batch Chatbot Processing Utility

This script processes a batch of questions using a chatbot engine, generating answers and saving results to a JSON file. It supports configurable settings and can repeat answer generation multiple times per question for robustness testing or sampling.

Key Functions:
- load_config: Load configuration from a YAML file.
- load_dataset: Load questions from a JSON dataset.
- save_results: Save results to a JSON file.
- remove_think_tags: Remove <think>...</think> tags from generated answers.
- main: Command-line entry point for batch processing with options for config and repeat count.
"""

import os
import json
import logging
import click
import yaml
from datetime import datetime
from dotenv import load_dotenv
from components.chat_engine import ChatEngine
from components.utils import setup_logging
import re

setup_logging()

# Load environment variables
load_dotenv()


def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path (str): Path to the configuration YAML file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            if not config:
                raise ValueError("Configuration file is empty")
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")


def load_dataset(dataset_path):
    """Load questions from JSON dataset."""
    with open(dataset_path, "r") as f:
        return json.load(f)


def save_results(results, output_path):
    """Save results to JSON file."""
    if not results:
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results with pretty formatting
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def remove_think_tags(text):
    """Remove content between <think> and </think> tags from text and clean up newlines.

    Args:
        text (str): Input text that may contain think tags

    Returns:
        str: Text with think tags and their content removed, and double newlines cleaned up
    """
    if "<think>" not in text or "</think>" not in text:
        return text

    # Remove think tags and their content
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove double newlines
    result = result.replace("\n\n", "\n")

    return result


@click.command()
@click.option(
    "--config",
    default="evaluation/configs/batch_config.yaml",
    help="Path to the configuration YAML file",
)
def main(config):
    """Process questions from dataset and generate answers."""
    try:
        # Load configuration
        config_data = load_config(config)

        # Set default output path if not provided
        if config_data.get("chatbot_results_path") is None:
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            config_data["chatbot_results_path"] = (
                f"evaluation/chatbot_results/results_{timestamp}.json"
            )

        # Load dataset
        questions = load_dataset(config_data["dataset_path"])

        # Initialize ChatEngine
        chat_engine = ChatEngine(
            model_name=config_data["chat_model_name"],
            model_provider=config_data["chat_model_provider"],
            temperature=config_data["chat_temperature"],
            file_path=config_data["knowledge_path"],
            text_splitter_type=config_data["chat_text_splitter_type"],
            top_k=config_data["chat_top_k"],
        )

        # Process questions and generate answers
        repeat = config_data["repeat"]
        results = []
        for idx, q in enumerate(questions, start=1):
            for repeat_id in range(1, repeat + 1):
                try:
                    # Generate answer using ChatEngine
                    answer, response_time = chat_engine.generate_response(q["question"])

                    # Create result entry
                    result = {
                        "id": idx,
                        "repeat_id": repeat_id,
                        "question": q["question"],
                        "answer": remove_think_tags(answer),
                        "ground_truth": q["answer"],
                        "source": q["source"],
                        "difficulty": q["difficulty"],
                        "confidence": q["confidence"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_name": config_data["chat_model_name"],
                        "model_provider": config_data["chat_model_provider"],
                        "temperature": config_data["chat_temperature"],
                        "response_time": response_time,
                    }
                    results.append(result)

                except Exception as e:
                    logging.error(f"Error processing question: {e}")
                    continue

        # Save results
        save_results(results, config_data["chatbot_results_path"])
        logging.info(f"... Results saved to {config_data['chatbot_results_path']}")

    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        logging.error(f"Configuration error: {e}")
        raise click.Abort()


if __name__ == "__main__":
    main()
