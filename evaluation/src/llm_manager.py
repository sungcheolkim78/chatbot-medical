"""
LLM management functionality for the dataset generator.
"""

from typing import Dict, Optional
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

    def select_llm(self, llm_type: LLMType) -> None:
        """Select and initialize the LLM based on type."""
        logging.info(f"Selecting LLM: {llm_type}")
        try:
            if llm_type == LLMType.GPT4_MINI:
                if not self.available_models.get("OPENAI_API_KEY"):
                    raise ValueError("OpenAI API key is not set")
                self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            elif llm_type in [LLMType.CLAUDE_35, LLMType.CLAUDE_37]:
                if not self.available_models.get("ANTHROPIC_API_KEY"):
                    raise ValueError("Anthropic API key is not set")
                model = (
                    "claude-3-opus-20240229"
                    if llm_type == LLMType.CLAUDE_35
                    else "claude-3-sonnet-20240229"
                )
                self.llm = ChatAnthropic(model=model)
            elif llm_type == LLMType.GEMINI_15_PRO:
                if not self.available_models.get("GOOGLE_API_KEY"):
                    raise ValueError("Google API key is not set")
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

            self.current_llm_type = llm_type
            logging.info(f"Successfully initialized {llm_type}")
        except Exception as e:
            logging.error(f"Error selecting LLM {llm_type}: {str(e)}")
            raise

    def get_current_llm(self):
        """Get the currently selected LLM."""
        if not self.llm:
            raise ValueError("No LLM has been selected")
        return self.llm 