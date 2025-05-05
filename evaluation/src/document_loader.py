"""
Document loading and processing functionality for the dataset generator.
"""

import logging
from pathlib import Path
from typing import List
import random

from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from .utils import preprocess_text


class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.paper_content = None
        self.vectorstore = None

    def load_paper(self) -> None:
        """Load and process the paper content."""
        logging.info("Loading paper content...")
        try:
            loader = DoclingLoader(file_path=self.file_path)
            pages = loader.load()
            logging.info(f"Successfully loaded {len(pages)} pages from the paper")

            # Preprocess text content of each page
            for page in pages:
                page.page_content = preprocess_text(page.page_content)
                logging.debug(f"Preprocessed text sample: {page.page_content[:200]}...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            self.paper_content = text_splitter.split_documents(pages)
            logging.info(f"Split paper into {len(self.paper_content)} chunks")

            # Create vector store for similarity search
            logging.info("Creating vector store...")
            embeddings = HuggingFaceEmbeddings()
            self.vectorstore = FAISS.from_documents(self.paper_content, embeddings)
            logging.info("Vector store creation completed")

        except Exception as e:
            logging.error(f"Error loading paper: {str(e)}")
            raise

    def get_random_context(self, num_chunks: int = 2) -> str:
        """Get random chunks of text from the paper for context."""
        if not self.paper_content:
            raise ValueError("Paper content not loaded")

        # Select random chunks
        selected_chunks = random.sample(
            self.paper_content, min(num_chunks, len(self.paper_content))
        )

        # Combine the chunks and their surrounding context
        context_texts = []
        for chunk in selected_chunks:
            # Get the chunk's page number if available
            page_num = chunk.metadata.get("page", "Unknown")
            # Add page number to help track source
            context_texts.append(f"[Page {page_num}] {chunk.page_content}")

        context = "\n\n".join(context_texts)
        logging.debug(f"Selected random context from {len(selected_chunks)} chunks")
        return context 