"""
LLM management functionality for the dataset generator.
"""

from typing import Dict
import logging

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from .models import LLMType


class LLMManager:
    def __init__(self, available_models: Dict[str, bool]):
        self.available_models = available_models
        self.current_llm_type = None
        self.llm = None
        self.temperature = 0.1

    def select_llm(self, llm_type: LLMType) -> None:
        """Select and initialize the LLM based on type."""

        logging.info(f"... Selecting LLM: {llm_type} - {llm_type.value}")
        try:
            if llm_type in [LLMType.GPT4O_MINI]:
                if not self.available_models.get("OPENAI_API_KEY"):
                    raise ValueError("OpenAI API key is not set")
                self.llm = ChatOpenAI(
                    model=llm_type.value, temperature=self.temperature
                )
                logging.info(
                    f"... Successfully initialized {llm_type} - {llm_type.value}"
                )
            elif llm_type in [LLMType.CLAUDE_35_SONNET]:
                if not self.available_models.get("ANTHROPIC_API_KEY"):
                    raise ValueError("Anthropic API key is not set")
                self.llm = ChatAnthropic(
                    model=llm_type.value, temperature=self.temperature
                )
                logging.info(
                    f"... Successfully initialized {llm_type} - {llm_type.value}"
                )
            elif llm_type in [LLMType.GEMINI_25_FLASH]:
                if not self.available_models.get("GOOGLE_API_KEY"):
                    raise ValueError("Google API key is not set")
                self.llm = ChatGoogleGenerativeAI(
                    model=llm_type.value, temperature=self.temperature
                )
                logging.info(
                    f"... Successfully initialized {llm_type} - {llm_type.value}"
                )
            else:
                raise ValueError(f"Invalid LLM type: {llm_type}")

            self.current_llm_type = llm_type
        except Exception as e:
            logging.error(f"Error selecting LLM {llm_type}: {str(e)}")
            raise

    def get_current_llm(self):
        """Get the currently selected LLM."""
        if not self.llm:
            raise ValueError("No LLM has been selected")
        return self.llm
