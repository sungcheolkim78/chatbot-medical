"""
This is a simple chatbot powered by LangChain and Streamlit.

It uses the following LLM providers:
- OpenAI
- Anthropic
- Google Generative AI
- Ollama

The knowledge base is the `knowledge` folder in the root of the project. And the chatbot
will answer questions based on the content of the files in the folder.

The API keys are loaded from load_dotenv() and the .env file is in the root of the project.
"""

import streamlit as st
from dotenv import load_dotenv
from components.llm_config import get_llm_config
from components.chat_engine import ChatEngine
from components.gui_handler import GUIHandler

# Load environment variables
load_dotenv()


def main():
    # Set up the Streamlit page configuration (must be the first Streamlit command)
    st.set_page_config(page_title="Medical Chatbot", page_icon="💬", layout="wide")

    # Create columns for the layout
    left_column, chat_column, right_column = st.columns([1, 4, 2])

    # Initialize components
    gui = GUIHandler(left_column, chat_column, right_column)

    # Get LLM configuration
    with gui.left_column:
        st.markdown("### Settings")
        (
            model_provider,
            model_name,
            temperature,
            selected_file,
            text_splitter_type,
            top_k,
        ) = get_llm_config()

    # Initialize LLM and conversation if not already done
    chat_engine = ChatEngine(
        model_name=model_name,
        model_provider=model_provider,
        temperature=temperature,
        file_path=f"knowledge/{selected_file}",
        text_splitter_type=text_splitter_type,
        top_k=top_k,
    )

    # Render UI components
    gui.render_header()
    gui.render_knowledge_base(selected_file)
    gui.render_chat_messages(chat_engine)
    gui.render_chat_input(chat_engine)
    gui.render_instructions()


if __name__ == "__main__":
    main()
