from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from datetime import datetime
import time

# Note: if you want to use pdf file directly, use PDFKnowledgeSource
# from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from crewai.knowledge.knowledge_config import KnowledgeConfig

from typing import List


@CrewBase
class ChatbotEngine:
    """Chatbot Engine with CrewAI"""

    agents: List[BaseAgent]
    tasks: List[Task]
    knowledge_source = CrewDoclingSource(
        file_paths=["slamon1987_claude.md"],
        chunk_size=4000,
        chunk_overlap=200,
    )
    knowledge_config = KnowledgeConfig(results_limit=3, score_threshold=0.35)

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            verbose=False,
        )

    @agent
    def assistant(self) -> Agent:
        return Agent(
            config=self.agents_config["assistant"],
            verbose=False,
        )

    @task
    def research_task(self) -> Task:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_time = int(time.time() * 1000)  # Current time in milliseconds
        return Task(
            config=self.tasks_config["research_task"],
            output_file=f"logs/research_{timestamp}_{response_time}.md",
        )

    @task
    def chat_task(self) -> Task:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_time = int(time.time() * 1000)  # Current time in milliseconds
        return Task(
            config=self.tasks_config["chat_task"],
            output_file=f"logs/chat_{timestamp}_{response_time}.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the crew for the chatbot"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=False,
            output_log_file=True,
            knowledge_sources=[self.knowledge_source],
            knowledge_config=self.knowledge_config,
            memory=True,
            long_term_memory=LongTermMemory(
                storage=LTMSQLiteStorage(db_path="./memory.db")
            ),
        )
