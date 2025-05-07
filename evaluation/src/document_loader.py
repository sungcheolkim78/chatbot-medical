"""
Document loading and processing functionality for the dataset generator.
"""

import logging
import random
import os
import pickle
from pathlib import Path

from langchain_docling import DoclingLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .utils import preprocess_text


class DocumentLoader:
    # Class-level cache for paper content
    _paper_content_cache = {}

    def __init__(
        self,
        file_path: str,
        pdf_loader_type: str = "unstructured",
        text_splitter_type: str = "sentence_transformers_token",
    ):
        self.file_path = file_path
        self.paper_content = None
        self.vectorstore = None
        self.pdf_loader_type = pdf_loader_type
        self.text_splitter_type = text_splitter_type
        self.need_vectorstore = False  # This is for the chatbot
        self.cache_file = self._get_cache_file_path()

    def _get_cache_file_path(self) -> str:
        """Get the path for the cache file."""

        file_path = Path(self.file_path)
        return str(
            file_path.with_stem(
                f"{file_path.stem}_{self.text_splitter_type}_{self.pdf_loader_type}"
            ).with_suffix(".cache")
        )

    def _load_from_cache_file(self) -> bool:
        """Load content from cache file if it exists."""

        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.paper_content = pickle.load(f)
                logging.info(f"Loaded paper content from cache file: {self.cache_file}")
                return True
            except Exception as e:
                logging.warning(f"Failed to load cache file: {str(e)}")
        return False

    def _save_to_cache_file(self) -> None:
        """Save content to cache file."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.paper_content, f)
            logging.info(f"Saved paper content to cache file: {self.cache_file}")
        except Exception as e:
            logging.error(f"Failed to save cache file: {str(e)}")

    def load_paper(self) -> None:
        """Load and process the paper content."""

        logging.info("Loading paper content...")

        try:
            # First try to load from cache file
            if self._load_from_cache_file():
                pass
            # Then check memory cache
            elif self.file_path in self._paper_content_cache:
                logging.info("Using cached paper content from memory")
                self.paper_content = self._paper_content_cache[self.file_path]
            else:
                # TODO: Try different loaders
                # - PyPDFLoader
                # - Use multimodal models
                if self.pdf_loader_type == "docling":
                    loader = DoclingLoader(file_path=self.file_path)
                elif self.pdf_loader_type == "unstructured":
                    loader = UnstructuredLoader(
                        file_path=self.file_path,
                        strategy="hi_res",
                        partition_via_api=True,
                        coordinates=True,
                    )
                elif self.pdf_loader_type == "pdfminer":
                    loader = PDFMinerLoader(file_path=self.file_path)
                else:
                    raise ValueError(f"Invalid PDF loader type: {self.pdf_loader_type}")

                pages = loader.load()
                logging.info(f"Successfully loaded {len(pages)} pages from the paper")

                # Preprocess text content of each page
                for page in pages:
                    page.page_content = preprocess_text(page.page_content)
                    logging.debug(
                        f"Preprocessed text sample: {page.page_content[:200]}..."
                    )

                # TODO: Try more different splitters
                # - SemanticChunker
                # - SentenceTransformersTokenTextSplitter
                # - CharacterTextSplitter.from_huggingface_tokenizer
                if self.text_splitter_type == "recursive_character":
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=10
                    )
                elif self.text_splitter_type == "sentence_transformers_token":
                    text_splitter = SentenceTransformersTokenTextSplitter(
                        tokens_per_chunk=350, chunk_overlap=10
                    )
                self.paper_content = text_splitter.split_documents(pages)

                # Store in memory cache
                self._paper_content_cache[self.file_path] = self.paper_content
                # Save to cache file
                self._save_to_cache_file()

                logging.info(
                    f"... Split paper into {len(self.paper_content)} chunks and cached"
                )

            # Create vector store for similarity search
            # TODO: This is not needed for the dataset generator
            if self.need_vectorstore:
                logging.info("... Creating vector store")
                embeddings = HuggingFaceEmbeddings()
                self.vectorstore = FAISS.from_documents(self.paper_content, embeddings)
                logging.info("... Vector store creation completed")

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

        # Combine the chunks and their surrounding context
        context_texts = []
        for chunk_id in chunks_to_get:
            chunk = self.paper_content[chunk_id]

            # Get the chunk's page number if available
            page_num = chunk.metadata.get("page", "Unknown")

            if page_num != "Unknown":
                # Add page number to help track source
                context_texts.append(f"[Page {page_num}] {chunk.page_content}")
            else:
                # Add chunk number to help track source
                context_texts.append(f"[Chunk {chunk_id}] {chunk.page_content}")

        context = "\n\n".join(context_texts)
        logging.info(
            f"... Selected context from {len(chunks_to_get)} chunks - {chunks_to_get}"
        )

        return context
