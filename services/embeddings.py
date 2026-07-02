"""
Embedding service using BAAI/bge-m3.

Provides a single embedding model instance used both for document chunk
embeddings (during ingestion) and query embeddings (during retrieval),
ensuring both sides of the similarity search live in the same vector space.
"""

import logging
from typing import List
import os

CACHE_DIR = os.getenv("HF_CACHE_DIR", os.path.join(os.getcwd(), ".hf_cache"))

os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps the BAAI/bge-m3 embedding model for documents and queries."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        logger.info("Loading embedding model '%s' on device '%s'.", model_name, device)
        self.model_name = model_name
        try:
            self._model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", model_name, exc)
            raise
        logger.info("Embedding model loaded successfully.")

    @property
    def langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Expose the underlying LangChain Embeddings object (required by SemanticChunker)."""
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunk texts for storage in the vector database."""
        try:
            return self._model.embed_documents(texts)
        except Exception as exc:
            logger.error("Failed to embed documents: %s", exc)
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query using the same model used for documents."""
        try:
            return self._model.embed_query(text)
        except Exception as exc:
            logger.error("Failed to embed query: %s", exc)
            raise

    def get_dimension(self) -> int:
        """Infer the embedding vector dimension by embedding a probe string."""
        vector = self.embed_query("dimension probe")
        return len(vector)
