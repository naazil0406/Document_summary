"""
Interactive query CLI.

Flow:
Question -> BGE-M3 Query Embedding -> Qdrant Similarity Search ->
Filter by relevance score -> Intent detection (summary vs Q&A) ->
Qwen 2.5 72B (via OpenRouter) -> Final Answer

Routing:
  - Summary-style questions  → generate_summary()  with SUMMARY_MAX_TOKENS
  - All other questions       → generate_answer()   with OPENROUTER_MAX_TOKENS

Run with:
    python -m scripts.query
"""

import logging
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from services.embeddings import EmbeddingService
from services.qdrant_db import QdrantService
from services.retriever import Retriever, SUMMARY_KEYWORDS
from services.llm_service import OpenRouterLLMService, FALLBACK_ANSWER

logger = logging.getLogger(__name__)


def _normalize_question(question: str) -> str:
    normalized = question.lower().strip()
    normalized = re.sub(r"\bsumarize\b",  "summarize", normalized)
    normalized = re.sub(r"\bsummarise\b", "summarize", normalized)
    # Only expand "unit2" → "unit 2" when NOT followed by a file extension
    normalized = re.sub(r"\bunit\s*(\d+)\b(?![\.\w])", r"unit \1", normalized)
    return normalized


def _is_summary_question(question: str) -> bool:
    """Return True if the question is asking for a summary or overview."""
    q = _normalize_question(question)
    if any(keyword in q for keyword in SUMMARY_KEYWORDS):
        return True

    # Handle explicit unit summary requests and common misspellings.
    if re.search(r"\bunit\s*\d+\b", q) and re.search(
        r"\b(summary|summarize|overview|brief)\b",
        q,
    ):
        return True

    return False


def build_pipeline():
    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )
    qdrant_service = QdrantService(
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        api_key=settings.QDRANT_API_KEY,
    )
    retriever = Retriever(
        embedding_service,
        qdrant_service,
        top_k=settings.TOP_K,
        summary_top_k=settings.TOP_K_SUMMARY,
        min_relevance_score=settings.MIN_RELEVANCE_SCORE,
    )

    # Q&A service — standard token budget
    qa_llm = OpenRouterLLMService(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_MODEL,
        max_tokens=settings.OPENROUTER_MAX_TOKENS,
        temperature=settings.OPENROUTER_TEMPERATURE,
        site_url=settings.OPENROUTER_SITE_URL,
        site_name=settings.OPENROUTER_SITE_NAME,
    )

    # Summary service — larger token budget for 8-10 line output
    summary_llm = OpenRouterLLMService(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_MODEL,
        max_tokens=settings.SUMMARY_MAX_TOKENS,
        temperature=settings.OPENROUTER_TEMPERATURE,
        site_url=settings.OPENROUTER_SITE_URL,
        site_name=settings.OPENROUTER_SITE_NAME,
    )

    return retriever, qa_llm, summary_llm


def answer_question(
    retriever: Retriever,
    qa_llm: OpenRouterLLMService,
    summary_llm: OpenRouterLLMService,
    question: str,
) -> str:
    normalized_question = _normalize_question(question)
    try:
        chunks = retriever.retrieve(normalized_question)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return "An error occurred while retrieving relevant information."

    if not chunks:
        return FALLBACK_ANSWER

    # Route to the correct LLM method based on intent
    is_summary = _is_summary_question(normalized_question)
    logger.info("Query intent: %s", "SUMMARY" if is_summary else "Q&A")

    try:
        if is_summary:
            return summary_llm.generate_summary(chunks)
        else:
            return qa_llm.generate_answer(chunks, question)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        return "An error occurred while generating the answer."


def main() -> None:
    retriever, qa_llm, summary_llm = build_pipeline()
    print("RAG Query CLI. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        answer = answer_question(retriever, qa_llm, summary_llm, question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Query CLI failed: %s", exc)
        sys.exit(1)