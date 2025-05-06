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
        """Get random chunks of text from the paper for context, including adjacent chunks for continuity."""
        if not self.paper_content:
            raise ValueError("Paper content not loaded")

        # Select a random starting chunk
        start_idx = random.randint(0, len(self.paper_content) - 1)
        
        # Calculate the range of chunks to include
        # For num_chunks=2, we'll get the selected chunk and one adjacent chunk
        # For num_chunks=3, we'll get the selected chunk and both adjacent chunks
        chunks_to_get = []
        
        if num_chunks == 2:
            # Get the selected chunk and either previous or next chunk
            chunks_to_get.append(start_idx)
            if start_idx > 0 and random.random() < 0.5:
                chunks_to_get.append(start_idx - 1)  # Previous chunk
            elif start_idx < len(self.paper_content) - 1:
                chunks_to_get.append(start_idx + 1)  # Next chunk
        else:
            # Get the selected chunk and both adjacent chunks if available
            chunks_to_get = [start_idx]
            if start_idx > 0:
                chunks_to_get.append(start_idx - 1)  # Previous chunk
            if start_idx < len(self.paper_content) - 1:
                chunks_to_get.append(start_idx + 1)  # Next chunk

        # Sort chunks to maintain original order
        chunks_to_get.sort()
        
        # Get the selected chunks
        selected_chunks = [self.paper_content[i] for i in chunks_to_get]

        # Combine the chunks and their surrounding context
        context_texts = []
        for chunk in selected_chunks:
            # Get the chunk's page number if available
            page_num = chunk.metadata.get("page", "Unknown")
            # Add page number to help track source
            context_texts.append(f"[Page {page_num}] {chunk.page_content}")

        context = "\n\n".join(context_texts)
        logging.debug(f"Selected context from {len(selected_chunks)} chunks")
        return context 