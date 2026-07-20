"""
Ingestion pipeline entry point.

Flow:
S3 Bucket -> PDF Folder -> Docling Lite -> Document Chunking ->
Semantic Chunking -> BGE-M3 Embeddings -> Qdrant

Run with:
    python -m scripts.ingest
"""

import logging
import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from services.pdf_parser import PDFParser
from services.docx_parser import DocxParser
from services.excel_parser import ExcelParser
from services.pptx_parser import PptxParser
from services.image_parser import ImageParser, IMAGE_EXTENSIONS
from services.chunking import Chunk, DocumentChunker, SemanticChunkingService
from services.embeddings import EmbeddingService
from services.qdrant_db import QdrantService
from services.s3_storage import S3Storage
from services.canonical_naming import canonical_filename, is_canonical

logger = logging.getLogger(__name__)

SUPPORTED_INGESTION_EXTENSIONS = (
    ".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".pptx",
) + IMAGE_EXTENSIONS


def _list_ingestable_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if os.path.isfile(os.path.join(folder, name))
        and os.path.splitext(name)[1].lower() in SUPPORTED_INGESTION_EXTENSIONS
    ]


def _select_parser(file_path: str, folder: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".xlsx", ".xlsm", ".xls", ".csv"}:
        return ExcelParser(excel_folder=folder)
    if ext == ".docx":
        return DocxParser(docx_folder=folder)
    if ext == ".pptx":
        return PptxParser(pptx_folder=folder)
    if ext in IMAGE_EXTENSIONS:
        return ImageParser(image_folder=folder)
    return PDFParser(pdf_folder=folder)


def _parse_and_chunk(file_path: str, embedding_service: EmbeddingService, parser) -> list | None:
    pages = parser.extract_pages(file_path)
    if not pages:
        return None

    if isinstance(parser, ExcelParser):
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=page.text,
                filename=page.filename,
                page_number=page.page_number,
                page_label=page.page_label,
                page_start=page.metadata["row_start"],
                page_end=page.metadata["row_end"],
                metadata=dict(page.metadata),
                toc_section=page.metadata.get("toc_section", ""),
            )
            for page in pages
            if page.text.strip()
        ] or None

    document_chunker = DocumentChunker(
        heading_max_length=settings.DOC_CHUNK_HEADING_MAX_LENGTH,
        min_paragraph_length=settings.DOC_CHUNK_MIN_PARAGRAPH_LENGTH,
    )
    document_chunks = document_chunker.chunk_pages(pages)
    if not document_chunks:
        return None

    semantic_chunker = SemanticChunkingService(
        embeddings=embedding_service.langchain_embeddings,
        buffer_size=settings.SEMANTIC_BUFFER_SIZE,
        breakpoint_threshold_type=settings.SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=settings.SEMANTIC_BREAKPOINT_AMOUNT,
        max_chunk_size=settings.MAX_CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return semantic_chunker.chunk_documents(document_chunks) or None


def sync_from_s3() -> None:
    """Pull down anything in S3 that isn't already in the local PDF_FOLDER.
    No-op if S3_BUCKET_NAME isn't configured."""
    if not settings.S3_BUCKET_NAME:
        logger.info("S3_BUCKET_NAME not set; skipping S3 sync, using local PDF_FOLDER only.")
        return

    os.makedirs(settings.PDF_FOLDER, exist_ok=True)
    s3_storage = S3Storage(
        bucket_name=settings.S3_BUCKET_NAME,
        prefix=settings.S3_PREFIX,
    )
    logger.info(
        "Syncing from s3://%s/%s into %s ...",
        settings.S3_BUCKET_NAME,
        settings.S3_PREFIX,
        settings.PDF_FOLDER,
    )
    try:
        downloaded = s3_storage.sync_down(settings.PDF_FOLDER)
    except Exception as exc:
        logger.warning(
            "S3 sync failed; continuing with documents already available locally: %s",
            exc,
        )
        return
    if downloaded:
        logger.info("Downloaded %d file(s) from S3: %s", len(downloaded), downloaded)
    else:
        logger.info("Nothing new to download; local PDF_FOLDER already up to date with S3.")


def _rename_to_canonical(folder: str, qdrant_service: QdrantService) -> None:
    """Rename any non-canonical file in `folder` to "Unit N - 123456.ext"
    form -- locally, and in Qdrant if it was already indexed under the old
    name (Qdrant's `filename` payload is what the UI reads). Documents
    ingested before the canonical naming scheme was introduced get
    normalized the next time ingestion runs.

    The S3 object is intentionally never renamed here: canonical naming is
    a local + UI concern only, and S3 keys/filenames stay exactly as they
    were originally uploaded."""
    if not os.path.isdir(folder):
        return

    for old_name in sorted(os.listdir(folder)):
        old_path = os.path.join(folder, old_name)
        if not os.path.isfile(old_path):
            continue
        if os.path.splitext(old_name)[1].lower() not in SUPPORTED_INGESTION_EXTENSIONS:
            continue
        if is_canonical(old_name):
            continue

        new_name = canonical_filename(old_name)
        new_path = os.path.join(folder, new_name)
        try:
            os.rename(old_path, new_path)
            qdrant_service.rename_document(old_name, new_name)
            logger.info("Renamed '%s' -> '%s' (canonical form).", old_name, new_name)
        except Exception as exc:
            logger.warning("Could not rename '%s' to canonical form: %s", old_name, exc)


def run_ingestion() -> None:
    logger.info("Starting ingestion pipeline.")

    sync_from_s3()

    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )

    qdrant_service = QdrantService(
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        api_key=settings.QDRANT_API_KEY,
    )

    _rename_to_canonical(settings.PDF_FOLDER, qdrant_service)

    ingestable_files = _list_ingestable_files(settings.PDF_FOLDER)
    if not ingestable_files:
        logger.warning(
            "No supported files found in %s. Aborting ingestion.",
            settings.PDF_FOLDER,
        )
        return

    logger.info("Found %d supported file(s) to ingest.", len(ingestable_files))

    total_chunks = 0
    total_toc_entries = 0

    for file_path in ingestable_files:
        parser = _select_parser(file_path, settings.PDF_FOLDER)
        chunks = _parse_and_chunk(file_path, embedding_service, parser)
        if not chunks:
            logger.warning("No chunks generated for %s; skipping.", os.path.basename(file_path))
            continue

        batch_size = max(1, settings.INDEX_BATCH_SIZE)
        first_batch = chunks[:batch_size]
        first_embeddings = embedding_service.embed_documents(
            [chunk.text for chunk in first_batch]
        )
        if not first_embeddings:
            raise RuntimeError("Embedding service returned no vectors.")
        qdrant_service.ensure_collection(vector_size=len(first_embeddings[0]))
        qdrant_service.delete_document(first_batch[0].filename)
        qdrant_service.upsert_chunks(first_batch, first_embeddings)

        for offset in range(batch_size, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            embeddings = embedding_service.embed_documents([chunk.text for chunk in batch])
            if not embeddings:
                raise RuntimeError("Embedding service returned no vectors.")
            qdrant_service.upsert_chunks(batch, embeddings)
        total_chunks += len(chunks)

        toc_records = []
        if hasattr(parser, "toc_map"):
            toc_records = [
                {
                    "level": entry.level,
                    "title": entry.title,
                    "page_start": entry.page_start,
                    "page_end": entry.page_end,
                    "filename": os.path.basename(file_path),
                }
                for entry in parser.toc_map.get(os.path.basename(file_path), [])
            ]

        if toc_records:
            toc_embeddings = embedding_service.embed_documents(
                [entry["title"] for entry in toc_records]
            )
            qdrant_service.upsert_toc_entries(toc_records, toc_embeddings)
            total_toc_entries += len(toc_records)

        logger.info(
            "Ingested %s: %d chunks%s.",
            os.path.basename(file_path),
            len(chunks),
            f", {len(toc_records)} TOC entries" if toc_records else "",
        )

    logger.info(
        "Ingestion completed successfully. %d chunks and %d TOC entries "
        "stored in collection '%s'.",
        total_chunks,
        total_toc_entries,
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