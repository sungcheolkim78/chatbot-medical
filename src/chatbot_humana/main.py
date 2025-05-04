#!/usr/bin/env python
import sys
import warnings
import streamlit as st

from datetime import datetime

from chatbot_humana.crew import ChatbotHumana

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def chat():
    """
    Chat with the human chatbot. The chatbot will be able to answer questions about the
    clinical publications in knowlege folder.
    """
    st.title("Chat with the Humana Chatbot")
    st.write("Ask me anything about the clinical publications in the knowledge folder.")
    st.write("For example: 'What is the main idea of the paper?' or 'What is the best-known oncogene in breast cancer?'")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assisntant"):
            with st.spinner("Thinking..."):
                inputs = {
                    'topic': prompt, 
                    'current_year': str(datetime.now().year),
                    'user_message': prompt,
                    'history': "\n".join(st.session_state.history)
                }
                response = ChatbotHumana().crew().kickoff(inputs=inputs)
                st.markdown(response)
                st.session_state.history.append(f"User: {prompt}")
                st.session_state.history.append(f"Assistant: {response}")

        st.session_state.messages.append({"role": "assistant", "content": response})
            

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }
    
    try:
        ChatbotHumana().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        ChatbotHumana().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        ChatbotHumana().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        ChatbotHumana().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    chat()