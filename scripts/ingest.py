"""
Ingestion pipeline entry point.

Flow:
PDF Folder -> Docling Lite -> Document Chunking -> Semantic Chunking ->
BGE-M3 Embeddings -> Qdrant

Run with:
    python -m scripts.ingest
"""

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from services.pdf_parser import PDFParser
from services.chunking import DocumentChunker, SemanticChunkingService
from services.embeddings import EmbeddingService
from services.qdrant_db import QdrantService

logger = logging.getLogger(__name__)


def run_ingestion() -> None:
    logger.info("Starting ingestion pipeline.")

    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )

    parser = PDFParser(pdf_folder=settings.PDF_FOLDER)

    pages = parser.extract_all()

    if not pages:
        logger.warning(
            "No pages extracted from %s. Aborting ingestion.",
            settings.PDF_FOLDER,
        )
        return

    logger.info(
        "Successfully extracted %d document(s).",
        len(pages),
    )

    document_chunker = DocumentChunker(
        heading_max_length=settings.DOC_CHUNK_HEADING_MAX_LENGTH,
        min_paragraph_length=settings.DOC_CHUNK_MIN_PARAGRAPH_LENGTH,
    )

    document_chunks = document_chunker.chunk_pages(pages)

    if not document_chunks:
        logger.warning("No document chunks generated. Aborting ingestion.")
        return

    logger.info(
        "Generated %d document chunks.",
        len(document_chunks),
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
        logger.warning("No chunks generated. Aborting ingestion.")
        return

    logger.info(
        "Generated %d semantic chunks.",
        len(chunks),
    )

    logger.info(
        "Embedding %d chunks with %s.",
        len(chunks),
        settings.EMBEDDING_MODEL_NAME,
    )

    chunk_texts = [c.text for c in chunks]

    chunk_embeddings = embedding_service.embed_documents(chunk_texts)

    qdrant_service = QdrantService(
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        api_key=settings.QDRANT_API_KEY,
    )

    qdrant_service.ensure_collection(
        vector_size=len(chunk_embeddings[0])
    )

    qdrant_service.upsert_chunks(
        chunks,
        chunk_embeddings,
    )

    toc_records = [
        {
            "level": entry.level,
            "title": entry.title,
            "page_start": entry.page_start,
            "page_end": entry.page_end,
            "filename": filename,
        }
        for filename, entries in parser.toc_map.items()
        for entry in entries
    ]

    if toc_records:
        logger.info(
            "Embedding %d TOC entries with %s.",
            len(toc_records),
            settings.EMBEDDING_MODEL_NAME,
        )
        toc_embeddings = embedding_service.embed_documents(
            [entry["title"] for entry in toc_records]
        )
        qdrant_service.upsert_toc_entries(toc_records, toc_embeddings)
    else:
        logger.info("No PDF TOC entries found; only semantic chunks were stored.")

    logger.info(
        "Ingestion completed successfully. %d chunks and %d TOC entries "
        "stored in collection '%s'.",
        len(chunks),
        len(toc_records),
        settings.QDRANT_COLLECTION_NAME,
    )


if __name__ == "__main__":
    try:
        run_ingestion()
    except Exception as exc:
        logger.exception(
            "Ingestion pipeline failed: %s",
            exc,
        )
        sys.exit(1)
