"""
Question and Answer generation functionality for the dataset generator.
"""

import json
import logging
from typing import Dict

from .models import Difficulty, Confidence
from .llm_manager import LLMManager
from .document_loader import DocumentLoader


class QAGenerator:
    def __init__(self, document_loader: DocumentLoader, llm_manager: LLMManager):
        self.document_loader = document_loader
        self.llm_manager = llm_manager
        self.difficulty = None
        self.num_chunks = 1

    def set_difficulty(self, difficulty: Difficulty) -> None:
        """Set the difficulty level for question generation."""

        self.difficulty = difficulty
        logging.info(f"Set difficulty level to: {difficulty}")

    def generate_qa_pair(self) -> Dict[str, str]:
        """Generate a question-answer pair based on random portions of the paper content."""

        try:
            # Get random context from the paper
            context = self.document_loader.get_random_context(num_chunks=self.num_chunks)

            # Generate question and answer using the LLM
            qa_prompt = f"""You are a medical doctor specializing in breast cancer and oncogene research. 
Your task is to create a question and answer pair based on the given excerpt.

Paper excerpt:
{context}

Instructions:
1. Generate a {self.difficulty.value} level question and answer pair. Difficulty level is based on the target audience with different levels of knowledge.
2. The question must be answerable from the given excerpt
3. Include specific quotes or data from the excerpt in the answer
4. Format your response EXACTLY as shown in the example below

Example formats:
{{
    "question": "What is the specific finding about X described in the excerpt?",
    "answer": "According to the excerpt, the study found that X has Y effect. Specifically, the paper states: '[direct quote from text]'. This demonstrates that...",
    "source": "[exact quote from the excerpt that contains the answer]"
}}
{{
    "question": "What is the definition of X described in the excerpt?",
    "answer": "X is defined as Y in the excerpt. Specifically, the paper states: '[direct quote from text]'. This demonstrates that...",
    "source": "[exact quote from the excerpt that contains the answer]"
}}

IMPORTANT: 
- Respond ONLY with the JSON object
- Use proper JSON formatting with double quotes
- Do not include any other text before or after the JSON
- Make sure all quotes in the text are properly escaped
"""

            llm = self.llm_manager.get_current_llm()
            response = llm.invoke(qa_prompt)
            try:
                # Extract content from AIMessage
                response_content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Clean up the response to ensure valid JSON
                # Remove any markdown code block indicators
                response_content = response_content.strip()
                response_content = response_content.replace("```json", "").replace(
                    "```", ""
                )
                response_content = response_content.strip()

                # Try to parse the JSON
                try:
                    qa_pair = json.loads(response_content)
                except json.JSONDecodeError as e:
                    logging.warning(f"Initial JSON parsing failed: {str(e)}")
                    # Try to fix common JSON issues
                    response_content = response_content.replace(
                        "'", '"'
                    )  # Replace single quotes with double quotes
                    response_content = response_content.replace(
                        "\n", " "
                    )  # Remove newlines
                    qa_pair = json.loads(response_content)

                # Validate required fields
                required_fields = ["question", "answer", "source"]
                if not all(field in qa_pair for field in required_fields):
                    raise ValueError(
                        f"Missing required fields. Response: {response_content}"
                    )

                # Add source context metadata
                qa_pair["full_context"] = context

                logging.debug("Successfully generated QA pair")
                return qa_pair

            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(
                    f"Failed to parse LLM response: {str(e)}\nResponse content: {response_content}"
                )
                # Create a fallback response
                return {
                    "question": "Error: Failed to generate valid question",
                    "answer": "Error: Failed to generate valid answer",
                    "source": "Error: Failed to parse response",
                    "full_context": context,
                    "error": str(e),
                    "original_response": response_content,
                }
        except Exception as e:
            logging.error(f"Error generating QA pair: {str(e)}")
            raise

    def check_answer_confidence(self, answer: str, source: str) -> Confidence:
        """Check the confidence of the generated answer against the source."""

        try:
            confidence_prompt = f"""
            Compare the following answer with the source text and rate the confidence level (low, medium, high):
            
            Answer: {answer}
            Source: {source}
            
            Consider:
            1. Accuracy of information
            2. Completeness of the answer
            3. Alignment with source text
            
            Return only one word: low, medium, or high
            """

            llm = self.llm_manager.get_current_llm()
            response = llm.invoke(confidence_prompt)
            # Extract content from AIMessage
            confidence = (
                response.content if hasattr(response, "content") else str(response)
            )
            confidence = confidence.strip().lower()

            return Confidence(confidence)
        except Exception as e:
            logging.error(f"Error checking confidence: {str(e)}")
            raise
