"""
Configuration module for the RAG application.

Loads environment variables (via python-dotenv) and exposes a single
immutable `settings` object used across the application. Also configures
application-wide logging.
"""

import os
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logging for the entire application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass(frozen=True)
class Settings:
    # --- Paths ---
    PDF_FOLDER: str = os.getenv("PDF_FOLDER", "data/pdfs")

    # --- AWS S3 (optional: documents are mirrored to S3 on upload and
    # fetched back down from S3 into PDF_FOLDER as a local working cache.
    # If S3_BUCKET_NAME is unset, the app runs local-folder-only — no
    # boto3 calls are made and no AWS credentials are required.) ---
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
    S3_PREFIX: str = os.getenv("S3_PREFIX", "documents/")

    # --- Chunking (Semantic Chunker config) ---
    SEMANTIC_BUFFER_SIZE: int = int(os.getenv("SEMANTIC_BUFFER_SIZE", "1"))
    SEMANTIC_BREAKPOINT_TYPE: str = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile")
    SEMANTIC_BREAKPOINT_AMOUNT: float = float(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "95"))
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # --- Document Chunking (Stage 1: headings / sections / paragraphs) ---
    DOC_CHUNK_HEADING_MAX_LENGTH: int = int(os.getenv("DOC_CHUNK_HEADING_MAX_LENGTH", "80"))
    DOC_CHUNK_MIN_PARAGRAPH_LENGTH: int = int(os.getenv("DOC_CHUNK_MIN_PARAGRAPH_LENGTH", "20"))

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    INDEX_BATCH_SIZE: int = int(os.getenv("INDEX_BATCH_SIZE", "32"))

    # --- Qdrant ---
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "company_docs")

    # --- Retrieval ---
    TOP_K: int = int(os.getenv("TOP_K", "40"))
    TOP_K_SUMMARY: int = int(os.getenv("TOP_K_SUMMARY", "40"))
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.05"))

    # --- LLM provider switch: "openrouter" (default) or "bedrock" ---
    # Bedrock reuses the AWS_* credentials/region already configured above
    # for S3 — one set of AWS credentials covers both.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    BEDROCK_REGION: str = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))

    # --- OpenRouter (Qwen 2.5 72B) ---
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

    # --- Bedrock model IDs (used only when LLM_PROVIDER=bedrock) ---
    # Q&A + summary: Nova Micro — text-only, cheapest, lowest latency.
    BEDROCK_MODEL: str = os.getenv("BEDROCK_MODEL", "amazon.nova-micro-v1:0")
    # Presentation: Nova 2 Lite — needs a region-prefixed inference profile
    # ID for on-demand access from a US region (see bedrock_llm.py docstring).
    BEDROCK_PRESENTATION_MODEL: str = os.getenv("BEDROCK_PRESENTATION_MODEL", "us.amazon.nova-2-lite-v1:0")

    # Q&A answers: 1024 tokens gives 4-5 solid lines without cutting off
    OPENROUTER_MAX_TOKENS: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "1024"))

    OPENROUTER_TEMPERATURE: float = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
    OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "")
    OPENROUTER_SITE_NAME: str = os.getenv("OPENROUTER_SITE_NAME", "")

    # Summaries: 2048 tokens gives room for 8-10 detailed lines + key points
    SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", "2048"))

    # Training scripts: same model by default, but a much higher token cap
    # since a full multi-section narration script is long-form output.
    PRESENTATION_MODEL: str = os.getenv("PRESENTATION_MODEL", os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct"))
    PRESENTATION_MAX_TOKENS: int = int(os.getenv("PRESENTATION_MAX_TOKENS", "8192"))
    # Deliberately higher than OPENROUTER_TEMPERATURE (used for factual Q&A/summaries):
    # the training-script story needs to vary run to run, not converge on one answer.
    PRESENTATION_TEMPERATURE: float = float(os.getenv("PRESENTATION_TEMPERATURE", "0.9"))

    # Local folder every generated training script is saved to as a .txt file,
    # in addition to being shown/downloadable in the Streamlit UI.
    NARRATIVE_SCRIPTS_DIR: str = os.getenv("NARRATIVE_SCRIPTS_DIR", "Narrative_scripts")

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
setup_logging(settings.LOG_LEVEL)
