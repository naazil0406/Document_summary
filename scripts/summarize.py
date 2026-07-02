"""
Document summarization script.

Flow:
PDF -> PyMuPDF (+ OCR fallback) -> Document Chunking -> Semantic Chunking
-> Qwen 2.5 72B (via OpenRouter) -> Executive Summary

Unlike the query flow, summarization does NOT go through Qdrant or
embeddings/retrieval — it reads the target PDF directly from the local
PDF folder, runs it through the same two-stage chunking pipeline used
during ingestion, and sends the resulting chunks straight to Qwen 2.5
72B (via OpenRouter) to produce an executive summary.

Filename matching is fuzzy: passing "unit2", "unit 2", or "unit2.pdf"
all resolve to the correct file in the PDF folder.

Run with:
    python -m scripts.summarize <filename_or_stem>

Examples:
    python -m scripts.summarize unit2.pdf
    python -m scripts.summarize unit2
    python -m scripts.summarize "unit 2"
"""

import argparse
import logging
import os
import re
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from services.pdf_parser import PDFParser
from services.chunking import DocumentChunker, SemanticChunkingService
from services.embeddings import EmbeddingService
from services.openrouter_llm import OpenRouterLLMService

logger = logging.getLogger(__name__)


def _resolve_filename(name: str, pdf_folder: str) -> Optional[str]:
    """
    Resolve a user-supplied name (with or without .pdf, with or without
    spaces) to an actual filename in *pdf_folder*.

    Matching order:
      1. Exact filename match (case-insensitive).
      2. Stem match after stripping .pdf extension.
      3. Fuzzy: normalise spaces and compare stems.

    Returns the matched filename (basename only), or None if nothing matches.
    """
    try:
        candidates = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    except OSError:
        return None

    name_norm = name.strip().lower()
    # Strip extension if the user supplied it
    name_stem = re.sub(r"\.pdf$", "", name_norm)
    # Collapse spaces for fuzzy comparison
    name_nospace = re.sub(r"\s+", "", name_stem)

    for candidate in candidates:
        c_lower = candidate.lower()
        c_stem = re.sub(r"\.pdf$", "", c_lower)
        c_nospace = re.sub(r"\s+", "", c_stem)

        if c_lower == name_norm:                  # exact match
            return candidate
        if c_stem == name_stem:                   # stem match
            return candidate
        if c_nospace == name_nospace:             # no-space stem match
            return candidate
        # Partial: "unit 2" matches "unit2_notes.pdf"
        if name_nospace and name_nospace in c_nospace:
            return candidate

    return None


def summarize_document(name: str) -> str:
    # Resolve fuzzy name → actual filename
    resolved = _resolve_filename(name, settings.PDF_FOLDER)
    if resolved is None:
        available = ", ".join(
            f for f in os.listdir(settings.PDF_FOLDER) if f.lower().endswith(".pdf")
        ) if os.path.isdir(settings.PDF_FOLDER) else "(folder not found)"
        return (
            f"File not found for '{name}' in {settings.PDF_FOLDER}.\n"
            f"Available PDFs: {available or 'none'}"
        )

    file_path = os.path.join(settings.PDF_FOLDER, resolved)
    logger.info("Summarizing '%s' (resolved from '%s').", resolved, name)

    # --- Parse ---
    parser = PDFParser(pdf_folder=settings.PDF_FOLDER)
    pages = parser.extract_pages(file_path)
    if not pages:
        return f"No extractable text found in: {resolved}"

    # --- Stage 1: Document chunking (TOC-aware) ---
    document_chunker = DocumentChunker(
        heading_max_length=settings.DOC_CHUNK_HEADING_MAX_LENGTH,
        min_paragraph_length=settings.DOC_CHUNK_MIN_PARAGRAPH_LENGTH,
    )
    document_chunks = document_chunker.chunk_pages(pages)
    if not document_chunks:
        return f"Document chunking produced no content for: {resolved}"

    # --- Stage 2: Semantic chunking ---
    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )
    semantic_chunker = SemanticChunkingService(
        embeddings=embedding_service.langchain_embeddings,
        buffer_size=settings.SEMANTIC_BUFFER_SIZE,
        breakpoint_threshold_type=settings.SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=settings.SEMANTIC_BREAKPOINT_AMOUNT,
        max_chunk_size=settings.MAX_CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = semantic_chunker.chunk_documents(document_chunks)
    if not chunks:
        return f"Semantic chunking produced no content for: {resolved}"

    logger.info("Summarizing %d chunks from %s.", len(chunks), resolved)

    context_chunks = [
        {"text": c.text, "filename": c.filename, "page_label": c.page_label}
        for c in chunks
    ]

    llm_service = OpenRouterLLMService(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_MODEL,
        max_tokens=settings.SUMMARY_MAX_TOKENS,
        temperature=settings.OPENROUTER_TEMPERATURE,
        site_url=settings.OPENROUTER_SITE_URL,
        site_name=settings.OPENROUTER_SITE_NAME,
    )

    return llm_service.generate_summary(context_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a PDF document by name or stem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "filename",
        help=(
            "Name (or stem) of the PDF to summarize.  "
            "You can omit the .pdf extension and use spaces: "
            "\"unit2\", \"unit 2\", and \"unit2.pdf\" all work."
        ),
    )
    args = parser.parse_args()

    summary = summarize_document(args.filename)
    print(f"\nExecutive Summary of '{args.filename}':\n{summary}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Summarization failed: %s", exc)
        sys.exit(1)
