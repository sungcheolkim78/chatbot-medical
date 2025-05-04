from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# pdf_source = PDFKnowledgeSource(file_paths=["slamon1987.pdf"])
pdf_source = PDFKnowledgeSource(file_paths=["2024.12.03.24318322v2.full.pdf"])

memory_config = {
    "provider": "mem0",
    "config": {"user_id": "User"},
}


@CrewBase
class CrewaiKnowledgeChatbot:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def assistant(self) -> Agent:
        return Agent(
            config=self.agents_config["assistant"],
            memory=True,
            memory_config=memory_config,
            verbose=True,
        )

    @task
    def chat_task(self) -> Task:
        return Task(
            config=self.tasks_config["chat_task"],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=[pdf_source],
            verbose=True,
        )
