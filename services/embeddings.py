"""
Embedding service using BAAI/bge-m3.

Provides a single embedding model instance used both for document chunk
embeddings (during ingestion) and query embeddings (during retrieval),
ensuring both sides of the similarity search live in the same vector space.
"""

import logging
from typing import List
import json
import os

import boto3

CACHE_DIR = os.getenv("HF_CACHE_DIR", os.path.join(os.getcwd(), ".hf_cache"))

os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps either a local HuggingFace embedding model or AWS Bedrock embeddings."""

    def __init__(self, model_name: str = None, device: str = "cpu"):
        provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
        self.provider = provider
        if provider == "bedrock":
            self.model_name = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        else:
            self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        self.device = device

        if provider == "bedrock":
            self._model = None
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            )
            logger.info("Using AWS Bedrock embeddings with model '%s'.", self.model_name)
            return

        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model '%s' on device '%s'.", self.model_name, device)
        try:
            self._model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", self.model_name, exc)
            raise
        logger.info("Embedding model loaded successfully.")

    @property
    def langchain_embeddings(self):
        """Expose the underlying embeddings object when a local model is used."""
        if self.provider == "bedrock":
            return None
        return self._model

    def _bedrock_embed(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            body = json.dumps({"inputText": text})
            response = self._bedrock_client.invoke_model(
                modelId=self.model_name,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read().decode("utf-8"))
            embedding = payload.get("embedding") or payload.get("embeddings") or []
            if not embedding:
                raise ValueError("Bedrock embedding response did not contain embedding data")
            embeddings.append(embedding)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunk texts for storage in the vector database."""
        if self.provider == "bedrock":
            return self._bedrock_embed(texts)
        try:
            return self._model.embed_documents(texts)
        except Exception as exc:
            logger.error("Failed to embed documents: %s", exc)
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query using the same model used for documents."""
        if self.provider == "bedrock":
            return self._bedrock_embed([text])[0]
        try:
            return self._model.embed_query(text)
        except Exception as exc:
            logger.error("Failed to embed query: %s", exc)
            raise

    def get_dimension(self) -> int:
        """Infer the embedding vector dimension by embedding a probe string."""
        vector = self.embed_query("dimension probe")
        return len(vector)
