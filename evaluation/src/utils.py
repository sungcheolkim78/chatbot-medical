"""
Utility functions for the dataset generator.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv
import re


def setup_logging() -> None:
    """Set up logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"dataset_generator_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )


def check_api_keys() -> Dict[str, bool]:
    """Check if required API keys are set."""
    load_dotenv()  # Load environment variables from .env file

    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }

    available_models = {}
    for key_name, key_value in api_keys.items():
        if key_value:
            logging.info(f"{key_name} is set")
            available_models[key_name] = True
        else:
            logging.warning(f"{key_name} is not set")
            available_models[key_name] = False

    return available_models


def preprocess_text(text: str) -> str:
    """Preprocess text to fix spacing and formatting issues."""
    # Fix common PDF text extraction issues
    processed = text

    # Fix missing spaces after periods
    processed = processed.replace(".", ". ")

    # Fix missing spaces after commas
    processed = processed.replace(",", ", ")

    # Fix missing spaces after semicolons
    processed = processed.replace(";", "; ")

    # Fix missing spaces after colons
    processed = processed.replace(":", ": ")

    # Fix missing spaces between parentheses
    processed = processed.replace(")(", ") (")

    # Fix missing spaces between word and parenthesis
    processed = processed.replace("(", " (")
    processed = processed.replace(")", ") ")

    # Fix missing spaces between word and reference numbers
    processed = re.sub(r"([a-zA-Z])(\d+)", r"\1 \2", processed)
    processed = re.sub(r"(\d+)([a-zA-Z])", r"\1 \2", processed)

    # Fix extra spaces
    processed = re.sub(r"\s+", " ", processed)

    # Fix spaces around hyphens in compound words
    processed = re.sub(r"\s*-\s*", "-", processed)

    return processed.strip() 