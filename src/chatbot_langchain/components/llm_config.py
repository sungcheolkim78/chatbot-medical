"""
LLM configuration and model selection module.
"""

import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm_config():
    """Get LLM configuration from the UI."""
    model_provider = st.selectbox(
        "Select a model provider:", ["Ollama", "OpenAI", "Anthropic", "Google Gemini"]
    )

    # Model selection based on provider
    if model_provider == "Ollama":
        model_name = st.selectbox(
            "Select Ollama model:", 
            [
                "llama3.1:latest", 
                "llama3.2:latest",
                "granite3.2:2b",
                "granite3.1-dense:2b",
                "qwen3:1.7b",
                "qwen3:8b",
                "qwen3:14b",
            ]
        )
    elif model_provider == "OpenAI":
        model_name = st.selectbox("Select OpenAI model:", ["gpt-3.5-turbo", "gpt-4"])
    elif model_provider == "Anthropic":
        model_name = st.selectbox(
            "Select Anthropic model:",
            [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
        )
    else:  # Google Gemini
        model_name = st.selectbox(
            "Select Gemini model:", ["gemini-pro", "gemini-pro-vision"]
        )

    temperature = st.slider(
        "Temperature:", min_value=0.0, max_value=1.0, value=0.3, step=0.1
    )

    # Text splitter configuration
    text_splitter_type = st.selectbox(
        "Select text splitter type:",
        ["sentence_transformers_token", "recursive_character", "character"],
        help="Choose how to split the document into chunks"
    )

    # Top-k configuration
    top_k = st.slider(
        "Number of chunks to retrieve (top-k):",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Number of most relevant document chunks to retrieve for answering questions"
    )

    # Get list of markdown files from knowledge folder
    knowledge_files = [
        f for f in os.listdir("knowledge") if f.endswith((".md", ".txt"))
    ]

    # File selector
    selected_file = st.selectbox(
        "Select a document:",
        knowledge_files,
        index=knowledge_files.index("slamon1987_claude.md")
        if "slamon1987_claude.md" in knowledge_files
        else 0,
    )

    return model_provider, model_name, temperature, selected_file, text_splitter_type, top_k


def get_llm(model_provider: str, model_name: str, temperature: float):
    """Initialize and return the appropriate LLM based on configuration."""
    if model_provider == "OpenAI":
        return ChatOpenAI(temperature=temperature, model_name=model_name)
    elif model_provider == "Anthropic":
        return ChatAnthropic(temperature=temperature, model_name=model_name)
    else:  # Google Gemini
        return ChatGoogleGenerativeAI(temperature=temperature, model_name=model_name)
