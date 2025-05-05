"""
Data models and enums for the dataset generator.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Set, Dict, Tuple
import time


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LLMType(str, Enum):
    GPT4_MINI = "gpt-4o-mini"
    CLAUDE_35 = "claude-3-opus-20240229"
    CLAUDE_37 = "claude-3-sonnet-20240229"
    GEMINI_15_PRO = "gemini-1.5-pro"


@dataclass
class ProgressStats:
    total_combinations: int
    completed: Set[Tuple[LLMType, Difficulty]]
    failed: Dict[Tuple[LLMType, Difficulty], int]  # Combination -> retry count
    start_time: float

    def get_progress_str(self) -> str:
        completed = len(self.completed)
        elapsed_time = time.time() - self.start_time
        if completed > 0:
            avg_time_per_item = elapsed_time / completed
            estimated_remaining = avg_time_per_item * (
                self.total_combinations - completed
            )
            return (
                f"Progress: {completed}/{self.total_combinations} combinations "
                f"({(completed / self.total_combinations) * 100:.1f}%) | "
                f"Elapsed: {elapsed_time:.1f}s | "
                f"Estimated remaining: {estimated_remaining:.1f}s | "
                f"Failed attempts: {sum(self.failed.values())}"
            )
        return f"Progress: 0/{self.total_combinations} combinations" 