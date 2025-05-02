#!/usr/bin/env python
import warnings
import logging
import os

from dotenv import load_dotenv

from crewai_knowledge_chatbot.crew import CrewaiKnowledgeChatbot
from mem0 import MemoryClient

# warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
client = MemoryClient()


def run():
    history = []

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! It was nice talking to you.")
            break

        chat_history = "\\n".join(history)

        inputs = {
            "user_message": f"{user_input}",
            "history": f"{chat_history}",
        }

        response = CrewaiKnowledgeChatbot().crew().kickoff(inputs=inputs)

        history.append(f"User: {user_input}")
        history.append(f"Assistant: {response}")
        client.add(user_input, user_id="User")

        print(f"Assistant: {response}")
