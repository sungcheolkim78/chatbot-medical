#!/usr/bin/env python
import sys
import warnings
import streamlit as st
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
from chatbot_crewai.crew import ChatbotEngine

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Constants
DEFAULT_KNOWLEDGE_FILE = "slamon1987_claude.md"
KNOWLEDGE_DIR = "knowledge"
EXAMPLE_QUESTIONS = [
    "What was observed regarding the HER-2/neu loci in the study?",
    "What impact does HER-2/neu gene amplification have on patient outcomes according to the study?",
    "What statistical method was used to evaluate the predictive power of prognostic factors in the study?",
]

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

def setup_page_config() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(page_title="Medical Chatbot", page_icon="💬", layout="wide")

def create_sidebar() -> None:
    """Create and handle the sidebar functionality."""
    with st.sidebar:
        st.title("Settings")
        if st.button("Clear History", type="primary"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

def display_knowledge_base(file_name: str) -> None:
    """Display the knowledge base content in the right column."""
    st.markdown("### Knowledge Base")
    
    if not file_name:
        st.warning("No knowledge file selected")
        return

    file_path = os.path.join(KNOWLEDGE_DIR, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            with st.container():
                st.markdown(
                    f"""
                    <div style="height: 700px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(f"Error reading knowledge file: {str(e)}")

def display_chat_interface() -> None:
    """Display the main chat interface."""
    st.title("Medical Chatbot")
    st.markdown(
        "A chatbot powered by CrewAI and Streamlit. You can ask questions about the knowledge base."
    )
    with st.expander("Example questions"):
        st.markdown("\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS))

def display_chat_history() -> None:
    """Display the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def process_user_input(prompt: str) -> None:
    """Process user input and generate a response."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            inputs = {
                "topic": prompt,
                "current_year": str(datetime.now().year),
                "user_message": prompt,
                "history": "\n".join(st.session_state.history),
            }
            response = ChatbotEngine().crew().kickoff(inputs=inputs)
            end_time = time.time()
            response_time = end_time - start_time
            
            st.markdown(response)
            st.caption(f"Response time: {response_time:.2f} seconds")
            st.session_state.history.append(f"User: {prompt}")
            st.session_state.history.append(f"Assistant: {response}")

    st.session_state.messages.append({"role": "assistant", "content": response})

def chat() -> None:
    """Main chat interface function."""
    setup_page_config()
    initialize_session_state()
    create_sidebar()

    chat_column, right_column = st.columns([2, 1], border=False, vertical_alignment="top")

    with chat_column:
        display_chat_interface()
        display_chat_history()

        if prompt := st.chat_input("Ask something..."):
            process_user_input(prompt)

    with right_column:
        display_knowledge_base(DEFAULT_KNOWLEDGE_FILE)

def run() -> None:
    """Run the crew with default inputs."""
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        ChatbotEngine().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def test() -> None:
    """Test the crew execution with specified iterations and evaluation LLM."""
    if len(sys.argv) < 3:
        raise ValueError("Please provide number of iterations and evaluation LLM as arguments")

    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        ChatbotEngine().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    chat()
