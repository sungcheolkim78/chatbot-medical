"""
Dataset generator for chatbot evaluation
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import click
import random

from components import (
    Difficulty,
    LLMType,
    setup_logging,
    check_api_keys,
    DocumentLoader,
    LLMManager,
    QAGenerator,
)


class DatasetGenerator:
    def __init__(self, paper_path: str):
        # Initialize components
        self.document_loader = DocumentLoader(
            paper_path,
            pdf_loader_type=None,
            text_splitter_type="sentence_transformers_token",
        )
        self.llm_manager = LLMManager(check_api_keys())
        self.qa_generator = QAGenerator(self.document_loader, self.llm_manager)

        # Set up output directory and file
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        self.output_dir = Path("evaluation/datasets")
        self.output_dir.mkdir(exist_ok=True)
        self.output_file = self.output_dir / f"dataset_{timestamp}.json"
        self.output_file.write_text("[]")  # Initialize with empty array

        logging.info(f"... Initialized DatasetGenerator with paper: {paper_path}")
        logging.info(f"... Dataset will be saved to: {self.output_file}")

    def save_qa_pairs(
        self, qa_pairs: list, llm_type: LLMType, difficulty: Difficulty
    ) -> None:
        """Save QA pairs to the output file."""

        try:
            current_data = json.loads(self.output_file.read_text())

            # Add metadata to each entry
            for entry in qa_pairs:
                entry["llm_type"] = llm_type.value
                entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            current_data.extend(qa_pairs)
            self.output_file.write_text(json.dumps(current_data, indent=2))

            logging.info(
                f"... Saved {len(qa_pairs)} QA pairs for {llm_type.value} - {difficulty.value}"
            )
        except Exception as e:
            logging.error(f"Error saving QA pairs: {str(e)}")
            raise

    def process_combination(
        self, llm_type: LLMType, difficulty: Difficulty, num_questions: int
    ) -> None:
        """Process a single LLM-difficulty combination."""

        try:
            # Set a unique random seed for this thread based on LLM type and difficulty
            thread_seed = hash(f"{llm_type.value}_{difficulty.value}_{time.time()}")
            random.seed(thread_seed)

            logging.info(f"... Processing: {llm_type.value} - {difficulty.value}")

            # Set up LLM and difficulty
            self.llm_manager.select_llm(llm_type)
            self.qa_generator.set_difficulty(difficulty)

            # Generate QA pairs
            qa_pairs = []
            for i in range(num_questions):
                logging.info(f"Generating question {i + 1}/{num_questions}")

                qa_pair = self.qa_generator.generate_qa_pair()
                confidence = self.qa_generator.check_answer_confidence(
                    qa_pair["answer"], qa_pair["source"]
                )

                qa_pairs.append(
                    {
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "source": qa_pair["source"],
                        "full_context": qa_pair["full_context"],
                        "difficulty": difficulty.value,
                        "confidence": confidence.value,
                    }
                )

            # Save the generated QA pairs
            self.save_qa_pairs(qa_pairs, llm_type, difficulty)
            logging.info(f"... Completed: {llm_type.value} - {difficulty.value}")

        except Exception as e:
            logging.error(
                f"Failed to process {llm_type.value} - {difficulty.value}: {str(e)}"
            )
            raise


@click.command()
@click.option(
    "--paper-path",
    type=click.Path(exists=True),
    default="knowledge/slamon1987_claude.md",
    help="Path to the paper PDF file",
)
@click.option(
    "--num-questions",
    type=int,
    default=2,
    help="Number of questions to generate per combination",
)
def main(paper_path: str, num_questions: int):
    """Generate datasets for chatbot evaluation using different LLMs and difficulty levels."""

    setup_logging()
    logging.info(f"... Starting dataset generation with {paper_path}")

    try:
        generator = DatasetGenerator(paper_path)
        generator.document_loader.load_paper()

        # Create all combinations
        combinations = [
            (llm_type, difficulty) for llm_type in LLMType for difficulty in Difficulty
        ]
        logging.info(f"... Processing {len(combinations)} combinations")

        # Process combinations in parallel
        with ThreadPoolExecutor(max_workers=9) as executor:
            futures = [
                executor.submit(
                    generator.process_combination, llm_type, difficulty, num_questions
                )
                for llm_type, difficulty in combinations
            ]

            # Wait for all combinations to complete
            completed = 0
            for future in as_completed(futures):
                completed += 1
                logging.info(
                    f"... Progress: {completed}/{len(combinations)} combinations completed"
                )
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Combination failed: {str(e)}")

        logging.info(
            f"... Dataset generation completed. Results saved in: {generator.output_file}"
        )

    except Exception as e:
        logging.error(f"Dataset generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
