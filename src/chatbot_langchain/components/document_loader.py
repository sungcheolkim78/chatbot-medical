"""
Document loading and processing functionality for the dataset generator.
"""

import logging
import random
import os
import pickle
from pathlib import Path
from typing import List, Optional, Protocol

from langchain_docling import DoclingLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import (
    PDFMinerLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from .utils import preprocess_text


class DocumentLoaderStrategy(Protocol):
    """Strategy interface for document loading."""

    def load(self) -> List[Document]:
        """Load documents from a source."""
        pass


class DoclingLoaderStrategy(DocumentLoaderStrategy):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = DoclingLoader(file_path=file_path)

    def load(self) -> List[Document]:
        return self.loader.load()


class UnstructuredLoaderStrategy(DocumentLoaderStrategy):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = UnstructuredLoader(
            file_path=file_path,
            strategy="hi_res",
            partition_via_api=True,
            coordinates=True,
        )

    def load(self) -> List[Document]:
        return self.loader.load()


class PDFMinerLoaderStrategy(DocumentLoaderStrategy):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = PDFMinerLoader(file_path=file_path)

    def load(self) -> List[Document]:
        return self.loader.load()


class UnstructuredMarkdownLoaderStrategy(DocumentLoaderStrategy):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = UnstructuredMarkdownLoader(file_path=file_path)

    def load(self) -> List[Document]:
        return self.loader.load()


class TextSplitterStrategy(Protocol):
    """Strategy interface for text splitting."""

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        pass


class RecursiveCharacterSplitterStrategy(TextSplitterStrategy):
    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        return splitter.split_documents(documents)


class SentenceTransformersSplitterStrategy(TextSplitterStrategy):
    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=150, chunk_overlap=3
        )
        return splitter.split_documents(documents)


class DocumentLoaderFactory:
    """Factory for creating document loaders and text splitters."""

    # Mapping of file extensions to loader types
    EXTENSION_TO_LOADER = {
        ".md": "markdown",
        ".pdf": "pdfminer",  # Default PDF loader
    }

    # Mapping of loader types to strategy classes
    LOADER_STRATEGIES = {
        "docling": DoclingLoaderStrategy,
        "unstructured": UnstructuredLoaderStrategy,
        "pdfminer": PDFMinerLoaderStrategy,
        "markdown": UnstructuredMarkdownLoaderStrategy,
    }

    @classmethod
    def create_document_loader(
        cls, loader_type: str, file_path: str
    ) -> DocumentLoaderStrategy:
        """Create a document loader based on type and file path."""
        # If loader_type is not specified, try to determine from file extension
        if not loader_type:
            file_extension = Path(file_path).suffix.lower()
            loader_type = cls.EXTENSION_TO_LOADER.get(file_extension, "unstructured")
            logging.info(f"Auto-detected loader type: {loader_type}")

        if loader_type not in cls.LOADER_STRATEGIES:
            raise ValueError(f"Invalid loader type: {loader_type}")

        return cls.LOADER_STRATEGIES[loader_type](file_path)

    @staticmethod
    def create_text_splitter(splitter_type: str) -> TextSplitterStrategy:
        splitters = {
            "recursive_character": RecursiveCharacterSplitterStrategy,
            "sentence_transformers_token": SentenceTransformersSplitterStrategy,
        }
        if splitter_type not in splitters:
            raise ValueError(f"Invalid text splitter type: {splitter_type}")
        return splitters[splitter_type]()


class CacheStrategy(Protocol):
    """Strategy interface for caching."""

    def save(self, content: List[Document]) -> None:
        """Save content to cache."""
        pass

    def load(self) -> Optional[List[Document]]:
        """Load content from cache."""
        pass


class FileCacheStrategy(CacheStrategy):
    def __init__(self, cache_file: str):
        self.cache_file = cache_file

    def save(self, content: List[Document]) -> None:
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(content, f)
            logging.info(f"Saved paper content to cache file: {self.cache_file}")
        except Exception as e:
            logging.error(f"Failed to save cache file: {str(e)}")

    def load(self) -> Optional[List[Document]]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache file: {str(e)}")
        return None


class MemoryCacheStrategy(CacheStrategy):
    def __init__(self, cache: dict):
        self.cache = cache
        self.key = None

    def set_key(self, key: str):
        self.key = key

    def save(self, content: List[Document]) -> None:
        if self.key:
            self.cache[self.key] = content

    def load(self) -> Optional[List[Document]]:
        if self.key and self.key in self.cache:
            return self.cache[self.key]
        return None


class DocumentLoader:
    """Template class for document loading and processing."""

    # Class-level cache for paper content
    _paper_content_cache = {}

    def __init__(
        self,
        file_path: str,
        pdf_loader_type: str = None,  # Made optional to allow auto-detection
        text_splitter_type: str = "sentence_transformers_token",
        need_vectorstore: bool = True,
    ):
        self.file_path = file_path
        self.paper_content = None
        self.vectorstore = None
        self.pdf_loader_type = pdf_loader_type
        self.text_splitter_type = text_splitter_type
        self.need_vectorstore = need_vectorstore

        # Initialize strategies
        self.document_loader = DocumentLoaderFactory.create_document_loader(
            pdf_loader_type, file_path
        )
        self.text_splitter = DocumentLoaderFactory.create_text_splitter(
            text_splitter_type
        )

        # Initialize cache strategies
        cache_file = self._get_cache_file_path()
        self.file_cache = FileCacheStrategy(cache_file)
        self.memory_cache = MemoryCacheStrategy(self._paper_content_cache)
        self.memory_cache.set_key(file_path)

    def _get_cache_file_path(self) -> str:
        """Get the path for the cache file."""
        file_path = Path(self.file_path)
        return str(
            file_path.with_stem(
                f"{file_path.stem}_{self.text_splitter_type}_{self.pdf_loader_type}"
            ).with_suffix(".cache")
        )

    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Preprocess the text content of each document."""
        for doc in documents:
            if self.pdf_loader_type != "markdown":
                doc.page_content = preprocess_text(doc.page_content)
            logging.debug(f"Preprocessed text sample: {doc.page_content[:200]}...")
        return documents

    def _create_vector_store(self) -> None:
        """Create vector store for similarity search if needed."""
        if not self.need_vectorstore:
            return

        logging.info("... Creating vector store")
        embeddings = HuggingFaceEmbeddings()
        self.vectorstore = FAISS.from_documents(self.paper_content, embeddings)
        logging.info("... Vector store creation completed")

    def load_paper(self) -> None:
        """Template method for loading and processing paper content."""
        logging.info("Loading paper content...")

        try:
            # Try loading from cache first
            if cached_content := self.file_cache.load():
                self.paper_content = cached_content
                self._create_vector_store()
                return

            # Check memory cache
            if cached_content := self.memory_cache.load():
                logging.info("Using cached paper content from memory")
                self.paper_content = cached_content
                self._create_vector_store()
                return

            # Load and process documents
            self._load_and_process_documents()
            self._create_vector_store()

        except Exception as e:
            logging.error(f"Error loading paper: {str(e)}")
            raise

    def _load_and_process_documents(self) -> None:
        """Load and process documents using the selected strategies."""
        # Load documents
        pages = self.document_loader.load()
        logging.info(f"Successfully loaded {len(pages)} pages from the paper")

        # Preprocess documents
        pages = self._preprocess_documents(pages)

        # Split documents
        self.paper_content = self.text_splitter.split_documents(pages)
        logging.info(f"... Split paper into {len(self.paper_content)} chunks")

        # Cache the results
        self.memory_cache.save(self.paper_content)
        self.file_cache.save(self.paper_content)

    def get_random_context(self, num_chunks: int = 1) -> str:
        """Get random chunks of text from the paper for context, including adjacent chunks for continuity."""

        if not self.paper_content:
            raise ValueError("Paper content not loaded")

        # Select a random starting chunk
        start_idx = random.randint(0, len(self.paper_content) - 1)

        # Calculate the range of chunks to include
        # For num_chunks=1, we'll get the selected chunk
        # For num_chunks=2, we'll get the selected chunk and one adjacent chunk
        # For num_chunks=3, we'll get the selected chunk and both adjacent chunks
        chunks_to_get = [start_idx]

        if num_chunks == 2:
            # Get the selected chunk and either previous or next chunk
            if start_idx > 0 and random.random() < 0.5:
                chunks_to_get.append(start_idx - 1)  # Previous chunk
            elif start_idx < len(self.paper_content) - 1:
                chunks_to_get.append(start_idx + 1)  # Next chunk
        elif num_chunks == 3:
            # Get the selected chunk and both adjacent chunks if available
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
