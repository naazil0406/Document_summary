"""
FastAPI backend for the RAG bot — lives at the repo root, next to app.py,
config/ and services/.

This is a second front door onto the exact same orchestration logic that
app.py (Streamlit) implements: parsing, chunking, embedding, Qdrant
upsert/retrieval, S3 sync, LLM Q&A/summarization/transcript generation.
Nothing in services/ or config/ is touched — this file only adds a REST
API + static UI on top of them, mirroring app.py's functions.

Run with (from the repo root):

    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 — it serves static/index.html, which is
the UI wired to the endpoints below.
"""

import base64
import logging
import os
import re
import uuid
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import settings
from services.pdf_parser import PDFParser
from services.docx_parser import DocxParser
from services.excel_parser import ExcelParser
from services.pptx_parser import PptxParser
from services.image_parser import ImageParser, IMAGE_EXTENSIONS
from services.markdown_parser import MarkdownParser
from services.xml_parser import XMLParser
from services.json_parser import JSONParser
from services.transcript_parser import TranscriptParser
from services.reranker import ReRankerService
from services.s3_storage import S3Storage
from services.chunking import Chunk, chunk_extracted_pages, chunk_pages_legacy
from services.embeddings import EmbeddingService
from services.qdrant_db import QdrantService
from services.uuid7 import uuid7_str
from services.retriever import Retriever, SUMMARY_KEYWORDS
from services.llm_service import (
    FALLBACK_ANSWER,
    CONTENT_TYPES,
    OpenRouterLLMService,
    BedrockLLMService,
)
from services.document_resolver import (
    resolve_pdf_reference,
    resolve_summary_request,
    ambiguous_candidates,
)
from services.canonical_naming import canonical_display_name, current_month_folder, parse_canonical, unique_id_for
from services import name_mapping
from services.image_generation_service import HuggingFaceFluxService, PollinationsImageService, NovaCanvasService

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (
    ".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".pptx", ".md", ".xml", ".json", ".txt",
) + IMAGE_EXTENSIONS

# ===========================================================================
# Cached resources — heavy objects loaded once per process (replaces
# app.py's @st.cache_resource; functools.lru_cache gives the same
# "build once, reuse forever" behaviour outside of Streamlit).
# ===========================================================================
@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )


@lru_cache(maxsize=1)
def get_qdrant_service() -> QdrantService:
    return QdrantService(
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        api_key=settings.QDRANT_API_KEY,
    )


@lru_cache(maxsize=1)
def get_s3_storage() -> Optional[S3Storage]:
    if not settings.S3_BUCKET_NAME:
        return None
    return S3Storage(
        bucket_name=settings.S3_BUCKET_NAME,
        prefix=settings.S3_PREFIX,
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


@lru_cache(maxsize=1)
def get_reranker_service() -> Optional[ReRankerService]:
    if not settings.USE_RERANKER:
        return None
    return ReRankerService(
        model_name=settings.RERANKER_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(
        get_embedding_service(),
        get_qdrant_service(),
        top_k=settings.TOP_K,
        summary_top_k=settings.TOP_K_SUMMARY,
        min_relevance_score=settings.MIN_RELEVANCE_SCORE,
        reranker_service=get_reranker_service(),
    )


@lru_cache(maxsize=1)
def get_qa_llm() -> OpenRouterLLMService:
    if settings.LLM_PROVIDER == "bedrock":
        return BedrockLLMService(
            model=settings.BEDROCK_MODEL,
            max_tokens=settings.OPENROUTER_MAX_TOKENS,
            temperature=settings.OPENROUTER_TEMPERATURE,
            region_name=settings.BEDROCK_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return OpenRouterLLMService(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_MODEL,
        max_tokens=settings.OPENROUTER_MAX_TOKENS,
        temperature=settings.OPENROUTER_TEMPERATURE,
        site_url=settings.OPENROUTER_SITE_URL,
        site_name=settings.OPENROUTER_SITE_NAME,
    )


@lru_cache(maxsize=1)
def get_summary_llm() -> OpenRouterLLMService:
    if settings.LLM_PROVIDER == "bedrock":
        return BedrockLLMService(
            model=settings.BEDROCK_MODEL,
            max_tokens=settings.SUMMARY_MAX_TOKENS,
            temperature=settings.OPENROUTER_TEMPERATURE,
            region_name=settings.BEDROCK_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return OpenRouterLLMService(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_MODEL,
        max_tokens=settings.SUMMARY_MAX_TOKENS,
        temperature=settings.OPENROUTER_TEMPERATURE,
        site_url=settings.OPENROUTER_SITE_URL,
        site_name=settings.OPENROUTER_SITE_NAME,
    )


@lru_cache(maxsize=1)
def get_content_llm() -> OpenRouterLLMService:
    """Content Generation Agent — same provider switch as get_qa_llm()/
    get_summary_llm(), tuned with its own token cap + higher temperature
    (settings.CONTENT_MAX_TOKENS / CONTENT_TEMPERATURE) since this pipeline
    generates short (50-100 word), deliberately varied learning-feed text.
    """
    if settings.LLM_PROVIDER == "bedrock":
        return BedrockLLMService(
            model=settings.BEDROCK_MODEL,
            max_tokens=settings.CONTENT_MAX_TOKENS,
            temperature=settings.CONTENT_TEMPERATURE,
            region_name=settings.BEDROCK_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return OpenRouterLLMService(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_MODEL,
        max_tokens=settings.CONTENT_MAX_TOKENS,
        temperature=settings.CONTENT_TEMPERATURE,
        site_url=settings.OPENROUTER_SITE_URL,
        site_name=settings.OPENROUTER_SITE_NAME,
    )


@lru_cache(maxsize=1)
def get_image_prompt_llm() -> BedrockLLMService:
    """Nova Lite — always Bedrock, no OpenRouter fallback for this pipeline."""
    return BedrockLLMService(
        model=settings.BEDROCK_IMAGE_PROMPT_MODEL,
        max_tokens=settings.IMAGE_PROMPT_MAX_TOKENS,
        temperature=settings.IMAGE_PROMPT_TEMPERATURE,
        region_name=settings.AWS_IMAGE_REGION,
        aws_access_key_id=settings.AWS_IMAGE_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_IMAGE_SECRET_ACCESS_KEY,
    )


@lru_cache(maxsize=1)
def get_image_gen_service():
    """Returns the configured image-rendering backend (all three expose the
    same `.generate_image(prompt)` interface — see
    services/image_generation_service.py).

    Defaults to FLUX.1-dev via Hugging Face Inference Providers (requires
    HF_TOKEN). Set IMAGE_PROVIDER=pollinations for the no-signup free
    option, or IMAGE_PROVIDER=aws for Bedrock Nova Canvas.
    """
    if settings.IMAGE_PROVIDER == "aws":
        return NovaCanvasService(
            model=settings.BEDROCK_IMAGE_GEN_MODEL,
            region_name=settings.AWS_IMAGE_REGION,
            aws_access_key_id=settings.AWS_IMAGE_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_IMAGE_SECRET_ACCESS_KEY,
            width=settings.IMAGE_WIDTH,
            height=settings.IMAGE_HEIGHT,
            quality=settings.IMAGE_QUALITY,
            cfg_scale=settings.IMAGE_CFG_SCALE,
        )

    if settings.IMAGE_PROVIDER == "pollinations":
        return PollinationsImageService(
            model=settings.POLLINATIONS_MODEL,
            base_url=settings.POLLINATIONS_BASE_URL,
            width=settings.IMAGE_WIDTH,
            height=settings.IMAGE_HEIGHT,
        )

    return HuggingFaceFluxService(
        api_token=settings.HF_TOKEN,
        model=settings.HF_FLUX_MODEL,
        provider=settings.HF_INFERENCE_PROVIDER,
        width=settings.IMAGE_WIDTH,
        height=settings.IMAGE_HEIGHT,
        num_inference_steps=settings.FLUX_NUM_INFERENCE_STEPS,
        guidance_scale=settings.FLUX_GUIDANCE_SCALE,
    )


@lru_cache(maxsize=1)
def get_presentation_llm() -> OpenRouterLLMService:
    if settings.LLM_PROVIDER == "bedrock":
        return BedrockLLMService(
            model=settings.BEDROCK_PRESENTATION_MODEL,
            max_tokens=settings.PRESENTATION_MAX_TOKENS,
            temperature=settings.PRESENTATION_TEMPERATURE,
            region_name=settings.BEDROCK_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return OpenRouterLLMService(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.PRESENTATION_MODEL,
        max_tokens=settings.PRESENTATION_MAX_TOKENS,
        temperature=settings.PRESENTATION_TEMPERATURE,
        site_url=settings.OPENROUTER_SITE_URL,
        site_name=settings.OPENROUTER_SITE_NAME,
    )


# ===========================================================================
# Orchestration logic — ported 1:1 from app.py, with st.* calls removed.
# ===========================================================================
def _list_all_documents(pdf_folder: str) -> List[str]:
    """Basenames of every supported document under pdf_folder, at any
    nesting depth -- legacy flat files directly in pdf_folder, files inside
    a single month subfolder (e.g. pdf_folder/july/<file>), and files
    nested further still (e.g. pdf_folder/<synced-folder>/<subfolder>/<file>,
    as happens when an S3 prefix folder itself contains subfolders).
    Identifiers used throughout the app (Qdrant, cache, UI) are always the
    bare filename; how deep a file physically sits on disk is a storage
    detail only."""
    found: set[str] = set()
    for root, _dirs, files in os.walk(pdf_folder):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXTENSIONS):
                found.add(f)
    return sorted(found)


def local_read_path(filename: str) -> str:
    """Resolve `filename` to wherever it actually lives on disk: nested any
    number of levels deep (e.g. a synced S3 folder containing subfolders),
    inside a single month subfolder (current convention), or directly in
    PDF_FOLDER (legacy files saved before month folders existed). Falls
    back to the flat PDF_FOLDER path if the file isn't found anywhere, so
    callers still get a sensible path to report as missing."""
    root = settings.PDF_FOLDER
    flat = os.path.join(root, filename)
    if os.path.isfile(flat):
        return flat
    if os.path.isdir(root):
        for dirpath, _dirs, files in os.walk(root):
            if filename in files:
                return os.path.join(dirpath, filename)
    return flat


def local_write_path(filename: str) -> str:
    """Destination path for a newly saved file -- always under the current
    month's subfolder (e.g. data/pdfs/july/<filename>), matching the S3
    bucket's monthly-folder convention. Creates the subfolder if needed."""
    month_dir = os.path.join(settings.PDF_FOLDER, current_month_folder())
    os.makedirs(month_dir, exist_ok=True)
    return os.path.join(month_dir, filename)


def document_path_metadata(file_path: str, s3_key: str = "") -> dict:
    """Derive original filename + nested folder metadata from a local path.

    Relative path under PDF_FOLDER mirrors the S3 object hierarchy, e.g.
    ``Warehouse/Safety/Heat Stress Prevention Guide.pdf``.
    """
    from services.s3_storage import parse_s3_object_path

    abs_root = os.path.abspath(settings.PDF_FOLDER)
    abs_path = os.path.abspath(file_path)
    try:
        rel = os.path.relpath(abs_path, abs_root).replace("\\", "/")
    except ValueError:
        rel = os.path.basename(file_path)

    if rel.startswith(".."):
        rel = os.path.basename(file_path)

    key = (s3_key or rel).replace("\\", "/").lstrip("/")
    parsed = parse_s3_object_path(key)
    return {
        "filename": parsed["filename"] or os.path.basename(file_path),
        "folder": parsed["folder"],
        "subfolder": parsed["subfolder"],
        "folder_path": parsed["folder_path"],
        "s3_key": parsed["s3_key"] or key,
    }


def reconcile_local_names_with_s3(s3_storage: S3Storage, qdrant_service: QdrantService) -> dict:
    """Rename known legacy local aliases to their authoritative S3 names.

    The old Streamlit implementation recorded local-canonical -> S3-original
    mappings in ``.s3_name_map.json``.  Consume those mappings before a sync
    so the cache is repaired in place rather than downloading a duplicate.
    S3 objects and Qdrant vectors are never recreated or moved.
    """
    aliases = name_mapping._load(settings.PDF_FOLDER)
    renamed = {}
    for obj in s3_storage.list_objects():
        filename, key = obj["filename"], obj["key"]
        local_dir = os.path.join(settings.PDF_FOLDER, obj["folder_name"]) if obj["folder_name"] else settings.PDF_FOLDER
        target = os.path.join(local_dir, filename)
        if os.path.isfile(target):
            continue
        candidates = [
            local_name for local_name, s3_name in aliases.items()
            if s3_name == filename and os.path.isfile(local_read_path(local_name))
        ]
        if not candidates:
            # Earlier canonical migration runs embedded a deterministic id
            # in the local alias but did not always save .s3_name_map.json.
            # That id is derived from the original S3 filename, making this
            # a safe one-to-one fallback (also require the same extension).
            expected_id = unique_id_for(filename)
            candidates = [
                local_name for local_name in existing_documents()
                if (info := parse_canonical(local_name))
                and info.unique_id == expected_id
                and os.path.splitext(local_name)[1].lower() == os.path.splitext(filename)[1].lower()
            ]
        if len(candidates) != 1:
            continue  # Unknown aliases are intentionally not guessed.
        old_name = candidates[0]
        old_path = local_read_path(old_name)
        try:
            os.makedirs(local_dir, exist_ok=True)
            os.replace(old_path, target)
            qdrant_service.rename_document(old_name, filename)
            qdrant_service.enrich_document_metadata(
                filename, canonical_display_name(filename), key, obj["folder_name"], target, obj["upload_date"]
            )
            name_mapping.remove(settings.PDF_FOLDER, old_name)
            renamed[old_name] = filename
        except Exception as exc:
            logger.warning("Could not reconcile local '%s' with S3 '%s': %s", old_name, key, exc)
    return renamed


def resolve_filename(name: str, pdf_folder: str):
    candidates = _list_all_documents(pdf_folder)
    # During the one-time local/S3 migration, local aliases may still be
    # legacy ``Unit1 - 123456`` names. Include the persisted S3 originals in
    # matching, then return the local/Qdrant identity for retrieval.
    aliases = name_mapping._load(pdf_folder)
    s3_to_local = {s3_name: local_name for local_name, s3_name in aliases.items()}
    resolved = resolve_pdf_reference(name, candidates + list(s3_to_local))
    if resolved:
        return s3_to_local.get(resolved, resolved)
    normalized = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
    matches = [
        filename for filename in candidates
        if " ".join(re.findall(r"[a-z0-9]+", canonical_display_name(filename).lower())) in normalized
    ]
    return matches[0] if len(matches) == 1 else None


def _tag_folder(chunks: Optional[List[Chunk]], folder: str) -> Optional[List[Chunk]]:
    """Stamp every chunk's metadata with the S3 knowledge-repository folder
    it was ingested from (e.g. "video_scripts", "company_policies").
    QdrantService.upsert_chunks() promotes chunk.metadata["folder"] to a
    top-level, filterable payload field. A blank/None folder is a no-op,
    so ingestion without a folder behaves exactly as before."""
    if not chunks or not folder:
        return chunks
    for chunk in chunks:
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["folder"] = folder
    return chunks


def parse_and_chunk(
    file_path: str,
    embedding_service: EmbeddingService,
    parser=None,
    use_semantic_chunking: bool = True,
    folder: str = "",
):
    parser = parser or PDFParser(pdf_folder=settings.PDF_FOLDER)
    pages = parser.extract_pages(file_path)
    if not pages:
        return None

    chunk_kwargs = {
        "use_semantic_chunking": use_semantic_chunking,
        "folder": folder,
        "heading_max_length": settings.DOC_CHUNK_HEADING_MAX_LENGTH,
        "min_paragraph_length": settings.DOC_CHUNK_MIN_PARAGRAPH_LENGTH,
    }

    if parser and parser.__class__.__name__ == "ExcelParser":
        chunks = [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=p.text,
                filename=p.filename,
                page_number=p.page_number,
                page_label=p.page_label,
                page_start=p.page_number,
                page_end=p.page_number,
                metadata=dict(p.metadata or {}),
                toc_section=(p.metadata or {}).get("toc_section", ""),
            )
            for p in pages
        ]
    elif not use_semantic_chunking:
        chunks = chunk_pages_legacy(pages, embedding_service, **chunk_kwargs)
    elif settings.USE_SEMANTIC_BOUNDARY_DETECTION:
        chunks = chunk_extracted_pages(pages, embedding_service, **chunk_kwargs)
    else:
        chunks = chunk_pages_legacy(pages, embedding_service, **chunk_kwargs)

    return _tag_folder(chunks or None, folder)


def _get_or_create_document_id(filename: str, qdrant_service) -> str:
    """Return the permanent UUIDv7 document_id for `filename`.

    Re-indexing (the document was already indexed before, under this same
    filename) reuses its existing document_id unchanged. A genuinely new
    document -- including a filename that was previously deleted and is
    now being uploaded fresh -- gets a brand-new UUIDv7, since
    get_document_id() correctly returns None once delete_document() has
    removed every point that used to carry the old id.
    """
    existing = qdrant_service.get_document_id(filename)
    if existing:
        return existing
    return uuid7_str()


def _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta: Optional[dict] = None) -> None:
    filename = chunks[0].filename
    path_meta = path_meta or {}
    document_id = _get_or_create_document_id(filename, qdrant_service)
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata.update({
            "original_filename": path_meta.get("filename") or filename,
            # Display label only — never used as storage identity / filename.
            "canonical_name": canonical_display_name(filename),
            "s3_key": path_meta.get("s3_key") or metadata.get("s3_key") or filename,
            "folder": path_meta.get("folder") or metadata.get("folder") or "",
            "subfolder": path_meta.get("subfolder") or metadata.get("subfolder") or "",
            "local_path": local_read_path(filename),
            # Permanent internal identifier — never displayed to users.
            "document_id": document_id,
        })
        chunk.metadata = metadata
    batch_size = max(1, settings.INDEX_BATCH_SIZE)
    first_batch = chunks[:batch_size]
    if not first_batch:
        return

    embeddings = embedding_service.embed_documents([chunk.text for chunk in first_batch])
    if not embeddings:
        raise RuntimeError("Embedding service returned no vectors.")
    qdrant_service.ensure_collection(vector_size=len(embeddings[0]))
    qdrant_service.delete_document(first_batch[0].filename)
    qdrant_service.upsert_chunks(first_batch, embeddings)

    for offset in range(batch_size, len(chunks), batch_size):
        batch = chunks[offset:offset + batch_size]
        embeddings = embedding_service.embed_documents([chunk.text for chunk in batch])
        if not embeddings:
            raise RuntimeError("Embedding service returned no vectors.")
        qdrant_service.upsert_chunks(batch, embeddings)


def ingest_single_pdf(file_path: str, embedding_service, qdrant_service, folder: str = "") -> int:
    parser = PDFParser(pdf_folder=settings.PDF_FOLDER)
    path_meta = document_path_metadata(file_path)
    folder = folder or path_meta.get("folder", "")
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, folder=folder)
    if not chunks:
        return 0

    for chunk in chunks:
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata.setdefault("subfolder", path_meta.get("subfolder", ""))
        chunk.metadata.setdefault("s3_key", path_meta.get("s3_key", chunk.filename))

    path_meta = _apply_path_metadata(chunks, file_path, folder=folder)
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta=path_meta)

    filename = os.path.basename(file_path)
    document_id = qdrant_service.get_document_id(filename) or ""
    toc_records = [
        {
            "level": entry.level,
            "title": entry.title,
            "page_start": entry.page_start,
            "page_end": entry.page_end,
            "filename": filename,
            "document_id": document_id,
            "canonical_name": canonical_display_name(filename),
        }
        for entry in parser.toc_map.get(filename, [])
    ]
    if toc_records:
        toc_embeddings = embedding_service.embed_documents(
            [entry["title"] for entry in toc_records]
        )
        qdrant_service.upsert_toc_entries(toc_records, toc_embeddings)

    return len(chunks)


def _apply_path_metadata(chunks, file_path: str, folder: str = "") -> dict:
    """Stamp original S3 filename / folder / subfolder / s3_key onto chunks."""
    path_meta = document_path_metadata(file_path)
    if folder:
        path_meta["folder"] = folder
    for chunk in chunks:
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata.setdefault("folder", path_meta.get("folder", ""))
        chunk.metadata.setdefault("subfolder", path_meta.get("subfolder", ""))
        chunk.metadata.setdefault("s3_key", path_meta.get("s3_key", chunk.filename))
        chunk.metadata.setdefault("original_filename", path_meta.get("filename", chunk.filename))
    return path_meta


def ingest_single_docx(file_path: str, embedding_service, qdrant_service, folder: str = "") -> int:
    parser = DocxParser(docx_folder=settings.PDF_FOLDER)
    path_meta = document_path_metadata(file_path)
    folder = folder or path_meta.get("folder", "")
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, folder=folder)
    if not chunks:
        return 0
    path_meta = _apply_path_metadata(chunks, file_path, folder=folder)
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta=path_meta)
    return len(chunks)


def ingest_single_excel(file_path: str, embedding_service, qdrant_service, restructure_llm=None, folder: str = "") -> int:
    parser = ExcelParser(excel_folder=settings.PDF_FOLDER, llm_service=restructure_llm)
    path_meta = document_path_metadata(file_path)
    folder = folder or path_meta.get("folder", "")
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, use_semantic_chunking=False, folder=folder)
    if not chunks:
        return 0
    path_meta = _apply_path_metadata(chunks, file_path, folder=folder)
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta=path_meta)
    return len(chunks)


def ingest_single_pptx(file_path: str, embedding_service, qdrant_service, folder: str = "") -> int:
    parser = PptxParser(pptx_folder=settings.PDF_FOLDER)
    path_meta = document_path_metadata(file_path)
    folder = folder or path_meta.get("folder", "")
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, folder=folder)
    if not chunks:
        return 0
    path_meta = _apply_path_metadata(chunks, file_path, folder=folder)
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta=path_meta)
    return len(chunks)


def ingest_single_image(file_path: str, embedding_service, qdrant_service, folder: str = "") -> int:
    parser = ImageParser(image_folder=settings.PDF_FOLDER)
    path_meta = document_path_metadata(file_path)
    folder = folder or path_meta.get("folder", "")
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, folder=folder)
    if not chunks:
        return 0
    path_meta = _apply_path_metadata(chunks, file_path, folder=folder)
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta=path_meta)
    return len(chunks)


def ingest_single_markdown(file_path: str, embedding_service, qdrant_service, folder: str = "") -> int:
    parser = MarkdownParser(folder_path=settings.PDF_FOLDER)
    path_meta = document_path_metadata(file_path)
    folder = folder or path_meta.get("folder", "")
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, folder=folder)
    if not chunks:
        return 0
    path_meta = _apply_path_metadata(chunks, file_path, folder=folder)
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta=path_meta)
    return len(chunks)


def ingest_single_xml(file_path: str, embedding_service, qdrant_service, folder: str = "") -> int:
    parser = XMLParser(folder_path=settings.PDF_FOLDER)
    path_meta = document_path_metadata(file_path)
    folder = folder or path_meta.get("folder", "")
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, folder=folder)
    if not chunks:
        return 0
    path_meta = _apply_path_metadata(chunks, file_path, folder=folder)
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta=path_meta)
    return len(chunks)


def ingest_single_json(file_path: str, embedding_service, qdrant_service, folder: str = "") -> int:
    parser = JSONParser(folder_path=settings.PDF_FOLDER)
    path_meta = document_path_metadata(file_path)
    folder = folder or path_meta.get("folder", "")
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, folder=folder)
    if not chunks:
        return 0
    path_meta = _apply_path_metadata(chunks, file_path, folder=folder)
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta=path_meta)
    return len(chunks)


def ingest_single_transcript(file_path: str, embedding_service, qdrant_service, folder: str = "") -> int:
    parser = TranscriptParser(folder_path=settings.PDF_FOLDER)
    path_meta = document_path_metadata(file_path)
    folder = folder or path_meta.get("folder", "")
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, folder=folder)
    if not chunks:
        return 0
    path_meta = _apply_path_metadata(chunks, file_path, folder=folder)
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service, path_meta=path_meta)
    return len(chunks)


def ingest_document_by_extension(dest_path: str, embedding_service, qdrant_service, folder: str = "") -> int:
    file_ext = os.path.splitext(dest_path)[1].lower()
    if file_ext == ".docx":
        return ingest_single_docx(dest_path, embedding_service, qdrant_service, folder=folder)
    elif file_ext in (".xlsx", ".xlsm", ".xls", ".csv"):
        return ingest_single_excel(dest_path, embedding_service, qdrant_service, folder=folder)
    elif file_ext == ".pptx":
        return ingest_single_pptx(dest_path, embedding_service, qdrant_service, folder=folder)
    elif file_ext in IMAGE_EXTENSIONS:
        return ingest_single_image(dest_path, embedding_service, qdrant_service, folder=folder)
    elif file_ext == ".pdf":
        return ingest_single_pdf(dest_path, embedding_service, qdrant_service, folder=folder)
    elif file_ext == ".md":
        return ingest_single_markdown(dest_path, embedding_service, qdrant_service, folder=folder)
    elif file_ext == ".xml":
        return ingest_single_xml(dest_path, embedding_service, qdrant_service, folder=folder)
    elif file_ext == ".json":
        return ingest_single_json(dest_path, embedding_service, qdrant_service, folder=folder)
    elif file_ext == ".txt":
        return ingest_single_transcript(dest_path, embedding_service, qdrant_service, folder=folder)
    else:
        logger.warning("Skipping '%s': unsupported extension '%s'.", dest_path, file_ext)
        return 0


def files_needing_ingestion(filenames: List[str], qdrant_service) -> List[str]:
    pending = []
    for filename in filenames:
        try:
            already_indexed = bool(qdrant_service.retrieve_document(filename))
        except Exception:
            already_indexed = False
        if not already_indexed:
            pending.append(filename)
    return pending


def _get_all_chunks(name: str, qdrant_service):
    resolved = resolve_filename(name, settings.PDF_FOLDER)
    if resolved is None:
        return None
    raw = qdrant_service.retrieve_document(resolved)
    return [{"text": c["text"], "filename": c["filename"], "page_label": c["page_label"]} for c in raw]


# Matches requests like "give me an infographic of the documents I have",
# "an overview of all my documents", "everything I've uploaded", etc. — i.e.
# no single document is named, the request spans the whole corpus.
_ALL_DOCUMENTS_PATTERN = re.compile(
    r"\b("
    r"all( of)? (my|the) documents|"
    r"the documents i have|"
    r"all my (documents|files|uploads)|"
    r"all (the )?(documents|files)|"
    r"every document|"
    r"entire (document )?(library|collection)|"
    r"everything i(?:'ve| have) uploaded|"
    r"across (my|all)( of the)? documents"
    r")\b",
    re.IGNORECASE,
)


def _is_all_documents_request(query: str) -> bool:
    return bool(_ALL_DOCUMENTS_PATTERN.search(query))


def _collect_cross_document_chunks(
    qdrant_service,
    max_docs: int = 8,
    chunks_per_doc: int = 3,
) -> List[dict]:
    """Sample a few chunks from every ingested document.

    Used for "infographic of all my documents"-style requests, where the
    query text itself ("the documents I have") carries no topical signal
    for semantic search to latch onto. Rather than embedding that phrase
    and letting one document win arbitrarily, this pulls a spread of
    chunks (start / middle / end) from each file so the resulting
    infographic can actually reflect the whole corpus. Caps the number of
    documents sampled so the combined context stays a manageable size for
    the prompt-writing model.
    """
    filenames = existing_documents()
    if not filenames:
        return []

    collected: List[dict] = []
    for filename in filenames[:max_docs]:
        try:
            raw = qdrant_service.retrieve_document(filename)
        except Exception as exc:
            logger.warning(
                "Could not fetch chunks for '%s' during cross-document image request: %s",
                filename, exc,
            )
            continue
        if not raw:
            continue

        sample_indices = sorted(set(
            idx for idx in (0, len(raw) // 2, len(raw) - 1)
            if 0 <= idx < len(raw)
        ))[:chunks_per_doc]

        for idx in sample_indices:
            c = raw[idx]
            collected.append({
                "text": c["text"],
                "filename": c["filename"],
                "page_label": c.get("page_label"),
            })

    return collected


def save_narrative_script(script_text: str, label: str) -> str:
    os.makedirs(settings.NARRATIVE_SCRIPTS_DIR, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "script"
    filename = f"{safe_label}_script.txt"
    path = os.path.join(settings.NARRATIVE_SCRIPTS_DIR, filename)

    # Two generations with the same name shouldn't silently clobber each
    # other now that saved scripts are the permanent, browsable record.
    counter = 2
    while os.path.exists(path):
        filename = f"{safe_label}_script_{counter}.txt"
        path = os.path.join(settings.NARRATIVE_SCRIPTS_DIR, filename)
        counter += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(script_text)
    logger.info("Saved training script to '%s'.", path)
    return path


_LEGACY_SCRIPT_PATTERN = re.compile(r"^(?:training_script|video_script)_(.+)_\d{8}_\d{6}$")


def _migrate_legacy_script_filenames() -> None:
    """Rename scripts saved under the old 'training_script_{label}_{timestamp}.txt'
    / 'video_script_{label}_{timestamp}.txt' pattern to the current
    '{label}_script.txt' form, so every already-saved script shows up in the
    UI under the same naming convention as new ones."""
    folder = settings.NARRATIVE_SCRIPTS_DIR
    if not os.path.isdir(folder):
        return

    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(".txt"):
            continue
        stem = filename[: -len(".txt")]
        match = _LEGACY_SCRIPT_PATTERN.match(stem)
        if not match:
            continue

        label = match.group(1)
        new_name = f"{label}_script.txt"
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        if os.path.exists(new_path) and new_path != old_path:
            counter = 2
            while os.path.exists(new_path):
                new_name = f"{label}_script_{counter}.txt"
                new_path = os.path.join(folder, new_name)
                counter += 1

        try:
            os.rename(old_path, new_path)
            logger.info("Migrated legacy script '%s' -> '%s'.", filename, new_name)
        except OSError as exc:
            logger.warning("Could not migrate legacy script '%s': %s", filename, exc)


def _list_saved_scripts() -> List[dict]:
    """Every script currently saved on disk, oldest first -- this is the
    source of truth for what shows up in the Video Script rail, so scripts
    persist across restarts instead of only lasting one server session."""
    folder = settings.NARRATIVE_SCRIPTS_DIR
    if not os.path.isdir(folder):
        return []

    entries = []
    for filename in os.listdir(folder):
        if not filename.endswith("_script.txt"):
            continue
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue
        label = filename[: -len("_script.txt")].replace("_", " ")
        created_at = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
        entries.append(
            {
                "id": filename,
                "label": label,
                "saved_path": path,
                "created_at": created_at,
            }
        )

    entries.sort(key=lambda e: e["created_at"])
    return entries


def save_dual_script(video_script: str, story: str, label: str) -> Tuple[str, str]:
    """Save a Video Script + Story pair as two sibling .txt files under
    settings.DUAL_SCRIPTS_DIR, sharing one safe_label stem so they're easy
    to browse/pair up on disk. Returns (video_path, story_path)."""
    os.makedirs(settings.DUAL_SCRIPTS_DIR, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "video_story"

    video_filename = f"{safe_label}_video.txt"
    video_path = os.path.join(settings.DUAL_SCRIPTS_DIR, video_filename)
    counter = 2
    while os.path.exists(video_path):
        video_filename = f"{safe_label}_video_{counter}.txt"
        video_path = os.path.join(settings.DUAL_SCRIPTS_DIR, video_filename)
        counter += 1

    # Reuse whatever suffix counter the video file needed so the pair stays
    # aligned (e.g. "topic_video_2.txt" pairs with "topic_story_2.txt").
    story_filename = video_filename.replace("_video", "_story", 1) if "_video_" in video_filename or video_filename.endswith("_video.txt") else f"{safe_label}_story.txt"
    story_path = os.path.join(settings.DUAL_SCRIPTS_DIR, story_filename)

    with open(video_path, "w", encoding="utf-8") as f:
        f.write(video_script)
    with open(story_path, "w", encoding="utf-8") as f:
        f.write(story)
    logger.info("Saved video script + story pair to '%s' / '%s'.", video_path, story_path)
    return video_path, story_path


def _list_saved_dual_scripts() -> List[dict]:
    """Every Video Script + Story pair on disk, oldest first — the source
    of truth for the Video Script & Story rail."""
    folder = settings.DUAL_SCRIPTS_DIR
    if not os.path.isdir(folder):
        return []

    entries = []
    for filename in os.listdir(folder):
        if not filename.endswith("_video.txt"):
            continue
        video_path = os.path.join(folder, filename)
        story_path = os.path.join(folder, filename.replace("_video.txt", "_story.txt"))
        if not (os.path.isfile(video_path) and os.path.isfile(story_path)):
            continue
        label = filename[: -len("_video.txt")].replace("_", " ")
        created_at = datetime.fromtimestamp(os.path.getmtime(video_path)).isoformat()
        entries.append(
            {
                "id": filename,
                "label": label,
                "video_path": video_path,
                "story_path": story_path,
                "created_at": created_at,
            }
        )

    entries.sort(key=lambda e: e["created_at"])
    return entries


def save_generated_image(image_bytes: bytes, label: str) -> str:
    os.makedirs(settings.GENERATED_IMAGES_DIR, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "image"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_{safe_label}_{timestamp}.png"
    path = os.path.join(settings.GENERATED_IMAGES_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    logger.info("Saved generated image to '%s'.", path)
    return path


def generate_document_image(
    query: str,
    retriever,
    image_prompt_llm,
    image_gen_service,
) -> dict:
    """Full image pipeline: retrieve relevant chunks for the query -> Nova
    Lite turns (query + chunks) into one optimized prompt -> Nova Canvas
    renders the image. Returns the Nova Lite prompt alongside the saved
    image path and base64 payload so the caller/API can show both.

    Two retrieval modes, chosen by what the query is actually asking for:
      - A specific unit/chapter/section is named (e.g. "infographic for
        unit 2", "give me an image for unit1") -> normal retrieval via
        retriever.retrieve(), which scopes by that unit/chapter/section
        number (filenames are not used for matching in this deployment).
      - No unit/chapter/section is named and the request spans everything
        (e.g. "infographic of the documents I have") -> sample chunks
        across every ingested document instead of running one semantic
        search against a phrase with no topical content.
    """
    unit_hint = retriever.extract_unit_hint(query)

    if not unit_hint and _is_all_documents_request(query):
        logger.info(
            "Image request spans the whole corpus (no specific document named); "
            "sampling chunks across all ingested documents."
        )
        chunks = _collect_cross_document_chunks(retriever.qdrant_service)
        if not chunks:
            raise ValueError(
                "No indexed documents were found to build an infographic from. "
                "Upload and process documents first."
            )
    else:
        try:
            chunks = retriever.retrieve(query)
        except Exception as exc:
            logger.error("Retrieval failed during image generation: %s", exc)
            raise RuntimeError("An error occurred while retrieving relevant information.") from exc

        if not chunks:
            raise ValueError("No relevant context was found in the indexed documents for this request.")

    # This standalone endpoint has always been infographic-only (it predates
    # the Scenario/Spot-the-Mistake "scene" mode added for the Learning
    # Content Generation Engine below), so the mode is pinned explicitly
    # rather than left to a default.
    image_prompt = image_prompt_llm.generate_image_prompt(query, chunks, mode="infographic")
    negative_prompt = _negative_prompt_for_mode("infographic")
    image_bytes = image_gen_service.generate_image(image_prompt, negative_prompt=negative_prompt)
    saved_path = save_generated_image(image_bytes, label=query[:40])

    return {
        "image_prompt": image_prompt,
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "saved_path": saved_path,
    }


# Which image strategy each Content Type should use. Content types that
# describe one real, concrete situation get a photorealistic cinematic
# "scene" image (a still frame from a training video); conceptual/explainer
# content types keep the original flat-vector "infographic" look. See
# prompts/image_prompt_system.txt for what each mode actually produces.
_SCENE_MODE_CONTENT_TYPES = {"Scenario", "Spot the Mistake Challenge", "AI Image"}


def _image_mode_for_content_type(content_type: str) -> str:
    return "scene" if content_type in _SCENE_MODE_CONTENT_TYPES else "infographic"


# Negative prompts are mode-specific: a "scene" image needs photorealism
# and cinematic lighting, so the old blanket negative prompt (which banned
# exactly those things) actively fought against Scenario/Spot-the-Mistake
# images ever looking realistic. An "infographic" image still wants to
# avoid photorealism, so it keeps close to the original negative prompt.
_INFOGRAPHIC_NEGATIVE_PROMPT = (
    "photorealistic, realistic photo, photograph, cinematic lighting, "
    "realistic human faces, realistic skin texture, depth of field, "
    "camera bokeh, film grain, 3D render of real people, blurry, low detail, "
    "watermarks, logos"
)
_SCENE_NEGATIVE_PROMPT = (
    "infographic, icon, iconographic illustration, flat vector illustration, "
    "diagram, chart, flowchart, numbered steps, labeled boxes, comparison "
    "columns, collage, grid of panels, cartoon, clip art, text, captions, "
    "titles, labels, watermarks, logos, low quality, blurry, distorted "
    "anatomy, extra limbs"
)


def _negative_prompt_for_mode(mode: str) -> str:
    return _SCENE_NEGATIVE_PROMPT if mode == "scene" else _INFOGRAPHIC_NEGATIVE_PROMPT


def generate_learning_feed_item(
    content_type: str,
    topic: str,
    retriever,
    content_llm,
    image_prompt_llm,
    image_gen_service,
    monthly_topic_content: Optional[str] = None,
    common_data: Optional[str] = None,
    web_results: Optional[str] = None,
    folders: Optional[List[str]] = None,
) -> dict:
    """End-to-end Learning Content Generation Engine pipeline for one feed
    item:

      1. Knowledge Extraction — retrieve the Topic from the Knowledge Base
         (Qdrant, via retriever.retrieve()).
      2. Content Generation Agent — turn (Content Type + Topic + retrieved
         chunks, plus optional Monthly Topic Content / Common Knowledge /
         Internet Research) into one short, original piece of
         learner-facing text (services/llm_service.py
         generate_learning_content()).
      3. Image Prompt Generation Agent — turn (the generated text + Topic)
         into one optimized image-generation prompt (Nova Lite, the same
         generate_image_prompt() the standalone image pipeline uses).
      4. Image Generation Agent — render that prompt into an image via
         whichever backend settings.IMAGE_PROVIDER selects.

    monthly_topic_content / common_data / web_results are all optional —
    when omitted, this behaves exactly as before. web_results in
    particular is expected to be pre-fetched by the caller (e.g. a web
    search step upstream); this function does not perform web search
    itself.

    Every content type gets an image: the image prompt is always derived
    from the LLM-generated content_text (never the raw topic string alone),
    so the picture actually reflects what the card/scenario/quiz says.
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("topic must not be empty.")
    if content_type not in CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type '{content_type}'. "
            f"Supported types: {', '.join(CONTENT_TYPES)}"
        )

    try:
        chunks = retriever.retrieve(topic, folders=folders)
    except Exception as exc:
        logger.error("Retrieval failed during content generation: %s", exc)
        raise RuntimeError("An error occurred while retrieving relevant information.") from exc

    if not chunks:
        raise ValueError(
            f"No relevant context was found in the indexed documents for topic: {topic}. "
            "Upload and process the relevant document first."
        )

    content_text = content_llm.generate_learning_content(
        content_type,
        topic,
        chunks,
        monthly_topic_content=monthly_topic_content,
        common_data=common_data,
        web_results=web_results,
    )

    # The Image Prompt Generation Agent (Nova Lite) is driven by what the
    # content actually says, not just the bare topic, so the image matches
    # the specific fact/scenario/question generated above. Which visual
    # strategy it uses (photorealistic scene vs. flat-vector infographic)
    # is chosen explicitly by content_type, not inferred from wording.
    image_mode = _image_mode_for_content_type(content_type)
    image_query = f"{content_type} about \"{topic}\": {content_text}"
    image_prompt = image_prompt_llm.generate_image_prompt(image_query, chunks, mode=image_mode)

    negative_prompt = _negative_prompt_for_mode(image_mode)
    image_bytes = image_gen_service.generate_image(image_prompt, negative_prompt=negative_prompt)
    saved_path = save_generated_image(image_bytes, label=f"{content_type}_{topic[:30]}")

    return {
        "content_type": content_type,
        "topic": topic,
        "content_text": content_text,
        "image_prompt": image_prompt,
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "saved_path": saved_path,
    }


def summarize_indexed_document(name: str, qdrant_service, summary_llm) -> str:
    chunks = _get_all_chunks(name, qdrant_service)
    if chunks is None:
        available = ", ".join(existing_documents()) if os.path.isdir(settings.PDF_FOLDER) else "(folder not found)"
        raise ValueError(
            f"File not found for '{name}' in {settings.PDF_FOLDER}. "
            f"Available documents: {available or 'none'}"
        )
    if not chunks:
        raise ValueError(
            f"No indexed content was found for: {name}. Upload and process it first."
        )
    summary = summary_llm.generate_summary(chunks)
    if summary.strip() == FALLBACK_ANSWER:
        raise RuntimeError("The summary model returned no content. Please try again.")
    return summary


def normalize_question(question: str) -> str:
    normalized = question.lower().strip()
    normalized = re.sub(r"\bsumarize\b", "summarize", normalized)
    normalized = re.sub(r"\bsummarise\b", "summarize", normalized)
    normalized = re.sub(r"\bunit\s*(\d+)\b(?![\.\w])", r"unit \1", normalized)
    return normalized


def is_summary_question(question: str) -> bool:
    q = normalize_question(question)
    if any(keyword in q for keyword in SUMMARY_KEYWORDS):
        return True
    if re.search(r"\bunit\s*\d+\b", q) and re.search(r"\b(summary|summarize|overview|brief)\b", q):
        return True
    return False


def answer_question(retriever, qa_llm, summary_llm, question: str) -> str:
    normalized_question = normalize_question(question)
    try:
        chunks = retriever.retrieve(normalized_question)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return "An error occurred while retrieving relevant information."

    if not chunks:
        return FALLBACK_ANSWER

    is_summary = is_summary_question(normalized_question)
    logger.info("Query intent: %s", "SUMMARY" if is_summary else "Q&A")

    try:
        if is_summary:
            return summary_llm.generate_summary(chunks)
        return qa_llm.generate_answer(chunks, question)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        return "An error occurred while generating the answer."


def generate_suggested_questions(llm_service, summary_text: str, filename: str) -> list:
    prompt = (
        f"You produced this executive summary of the document \"{filename}\":\n\n"
        f"{summary_text}\n\n"
        "Suggest exactly 3 short, natural follow-up questions a reader might "
        "ask about this document, based only on the summary above. "
        "Return ONLY the 3 questions, one per line. "
        "Do not number them, do not use bullets or quotes, no extra text."
    )
    try:
        raw = llm_service._call_llm(prompt)
    except Exception:
        return []
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip().strip('"') for l in raw.splitlines()]
    lines = [l for l in lines if l]
    return lines[:3]


def existing_documents() -> List[str]:
    os.makedirs(settings.PDF_FOLDER, exist_ok=True)
    return _list_all_documents(settings.PDF_FOLDER)


# ===========================================================================
# FastAPI app
# ===========================================================================
app = FastAPI(title="Document Q&A Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-process cache: filename -> {"summary": str, "questions": list}
_summary_cache: dict = {}


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    type: str  # "summary" | "qa" | "info"
    filename: Optional[str] = None
    questions: List[str] = []


class SummarizeResponse(BaseModel):
    filename: str
    summary: str
    questions: List[str]


class TranscriptRequest(BaseModel):
    name: Optional[str] = None


class TranscriptResponse(BaseModel):
    id: str
    label: str
    script: str
    saved_path: str
    created_at: str


class TranscriptSummary(BaseModel):
    id: str
    label: str
    created_at: str


class VideoStoryRequest(BaseModel):
    learning_objectives: Optional[str] = ""
    name: Optional[str] = None
    hint: Optional[str] = None


class VideoStoryResponse(BaseModel):
    id: str
    label: str
    video_script: str
    story: str
    created_at: str


class VideoStorySummary(BaseModel):
    id: str
    label: str
    created_at: str


class ImageGenRequest(BaseModel):
    query: str


class ImageGenResponse(BaseModel):
    image_prompt: str
    image_base64: str
    saved_path: str


class ContentGenRequest(BaseModel):
    content_type: str
    topic: str
    # Optional extra sources described in prompts/content_generation_system.txt.
    # All default to None so existing callers keep working unchanged.
    monthly_topic_content: Optional[str] = None
    common_data: Optional[str] = None
    web_results: Optional[str] = None
    # Folder-based retrieval (see prompts.txt "FOLDER SELECTION LOGIC"):
    #   omitted/None -> auto-detect from `topic` text (e.g. "... from the
    #     video_scripts folder"), falling back to global search if no
    #     folder is named.
    #   [] (empty list) -> force global search across every folder, even
    #     if `topic` happens to mention a folder name.
    #   ["video_scripts"] -> folder-specific retrieval.
    #   ["video_scripts", "incident_reports"] -> multi-folder retrieval.
    folders: Optional[List[str]] = None


class ContentGenResponse(BaseModel):
    content_type: str
    topic: str
    content_text: str
    image_prompt: str
    image_base64: str
    saved_path: str


class DailyTipRequest(BaseModel):
    # Both optional: "give me today's daily tip" needs neither.
    topic: Optional[str] = None
    common_data: Optional[str] = None
    web_results: Optional[str] = None
    # How many distinct tips to generate (e.g. 10 for "give me 10 daily
    # tips"). Defaults to 1.
    count: int = 1
    # Folder scope (see ContentGenRequest.folders for the same semantics):
    #   None -> auto-detect from `topic` text (e.g. "... from the
    #     video_scripts folder"), falling back to global if none named.
    #   [] -> force global even if `topic` names a folder.
    #   ["video_scripts"] -> scope to just that folder.
    folders: Optional[List[str]] = None


class DailyTipResponse(BaseModel):
    topic: str
    tips: List[str]
    word_counts: List[int]


@app.on_event("startup")
def on_startup():
    os.makedirs(settings.PDF_FOLDER, exist_ok=True)
    _migrate_legacy_script_filenames()


@app.get("/api/status")
def status():
    s3 = get_s3_storage()
    return {
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": (
            settings.BEDROCK_MODEL if settings.LLM_PROVIDER == "bedrock" else settings.OPENROUTER_MODEL
        ),
        "s3_enabled": s3 is not None,
        "s3_bucket": settings.S3_BUCKET_NAME or None,
        "credentials_ok": (
            bool(settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY)
            if settings.LLM_PROVIDER == "bedrock"
            else bool(settings.OPENROUTER_API_KEY)
        ),
    }


@app.get("/api/documents")
def list_documents():
    return {"documents": [
        {"original_filename": filename, "canonical_name": canonical_display_name(filename)}
        for filename in existing_documents()
    ]}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), folder: str = Form("")):
    """folder: which S3 knowledge-repository category this document
    belongs to (e.g. "video_scripts", "company_policies",
    "incident_reports") -- whatever folders you actually have. Optional;
    an empty/omitted folder just means the document isn't scoped to any
    folder for retrieval (it still shows up in global searches)."""
    folder = folder.strip()
    original_filename = os.path.basename(file.filename)
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")

    # Uploads require S3: it is the only supported storage destination now,
    # not an optional backup. Fail fast rather than accepting a document
    # that would only ever live on local disk.
    s3_storage = get_s3_storage()
    if not s3_storage:
        raise HTTPException(
            503,
            "Document upload requires an S3 bucket, which isn't configured. "
            "Set S3_BUCKET_NAME (and credentials) and try again.",
        )

    # The original upload name is the storage identity in S3 (and in the
    # local working copy below). ``canonical_name`` is metadata/UI-only and
    # is never used as a path.
    filename = original_filename

    # The local file is only the transient working copy the parsers read
    # from during ingestion -- S3 is the actual store of record.
    dest_path = local_write_path(filename)
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    embedding_service = get_embedding_service()
    qdrant_service = get_qdrant_service()

    already_ingested = filename not in files_needing_ingestion([filename], qdrant_service)

    try:
        # A deliberate user upload is the one case that may update an
        # existing key. Migration/sync never upload or alter S3 objects.
        s3_uri = s3_storage.upload_file(dest_path, filename)
    except Exception as exc:
        raise HTTPException(502, f"S3 upload failed, so the document was not saved: {exc}")

    path_meta = document_path_metadata(dest_path, s3_key=s3_uri.split("/", 3)[-1] if s3_uri else "")
    folder = folder or path_meta.get("folder", "")

    if already_ingested:
        qdrant_service.enrich_document_metadata(
            filename, canonical_display_name(filename),
            path_meta.get("s3_key", filename), folder, dest_path,
            subfolder=path_meta.get("subfolder", ""),
        )
        return {
            "filename": filename,
            "original_filename": original_filename,
            "already_indexed": True,
            "chunks": 0,
        }

    try:
        n_chunks = ingest_document_by_extension(dest_path, embedding_service, qdrant_service, folder=folder)
    except Exception as exc:
        raise HTTPException(500, f"Something went wrong while processing the document: {exc}")

    if n_chunks == 0:
        raise HTTPException(422, "No extractable content found in this document.")

    qdrant_service.enrich_document_metadata(
        filename, canonical_display_name(filename),
        path_meta.get("s3_key", filename), folder, dest_path,
        subfolder=path_meta.get("subfolder", ""),
    )

    return {
        "filename": filename,
        "original_filename": original_filename,
        "already_indexed": False,
        "chunks": n_chunks,
    }


@app.post("/api/documents/{filename}/summarize", response_model=SummarizeResponse)
def summarize_document(filename: str):
    qdrant_service = get_qdrant_service()
    summary_llm = get_summary_llm()

    resolved = resolve_filename(filename, settings.PDF_FOLDER) or filename
    if resolved in _summary_cache:
        cached = _summary_cache[resolved]
        return SummarizeResponse(filename=resolved, summary=cached["summary"], questions=cached["questions"])

    try:
        summary = summarize_indexed_document(resolved, qdrant_service, summary_llm)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Something went wrong while summarizing: {exc}")

    questions = generate_suggested_questions(summary_llm, summary, resolved)
    _summary_cache[resolved] = {"summary": summary, "questions": questions}
    return SummarizeResponse(filename=resolved, summary=summary, questions=questions)


def migrate_document_ids(qdrant_service) -> dict:
    """One-time migration: assign a permanent UUIDv7 document_id to every
    already-indexed document that doesn't have one yet.

    Idempotent -- documents that already carry a document_id (from a
    previous run of this migration, or because they were ingested after
    the UUIDv7 feature existed) are left untouched and reported separately.
    Only the `document_id` payload field is written; text, vectors, and
    every other piece of metadata are left exactly as they were.
    """
    migrated = []
    already_had_id = []
    for filename in qdrant_service.list_all_filenames():
        existing = qdrant_service.get_document_id(filename)
        if existing:
            already_had_id.append({"filename": filename, "document_id": existing})
            continue
        document_id = uuid7_str()
        points_updated = qdrant_service.set_document_id_for_filename(filename, document_id)
        migrated.append({
            "filename": filename,
            "document_id": document_id,
            "points_updated": points_updated,
        })
    return {
        "migrated_count": len(migrated),
        "already_had_id_count": len(already_had_id),
        "migrated": migrated,
        "already_had_id": already_had_id,
    }


@app.post("/api/admin/migrate-document-ids")
def migrate_document_ids_endpoint():
    """Explicit, user-triggered one-time migration -- assigns a permanent
    UUIDv7 document_id to every already-indexed document that doesn't have
    one yet. Safe to call more than once; already-migrated documents are
    skipped and reported under `already_had_id` rather than re-processed.
    """
    qdrant_service = get_qdrant_service()
    try:
        return migrate_document_ids(qdrant_service)
    except Exception as exc:
        raise HTTPException(500, f"Document ID migration failed: {exc}")


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    qdrant_service = get_qdrant_service()
    local_path = local_read_path(filename)

    removed_local = False
    if os.path.isfile(local_path):
        os.remove(local_path)
        removed_local = True

    try:
        qdrant_service.delete_document(filename)
    except Exception as exc:
        logger.warning("Could not delete '%s' from Qdrant: %s", filename, exc)

    s3_storage = get_s3_storage()
    removed_s3 = False
    if s3_storage:
        s3_key = s3_storage._find_key(filename)
        if s3_key:
            try:
                s3_storage.client.delete_object(Bucket=s3_storage.bucket_name, Key=s3_key)
                removed_s3 = True
            except Exception as exc:
                logger.warning("Could not delete '%s' from S3: %s", filename, exc)

    _summary_cache.pop(filename, None)

    if not removed_local and not removed_s3:
        raise HTTPException(404, f"'{filename}' was not found locally or in S3.")

    return {"filename": filename, "removed_local": removed_local, "removed_s3": removed_s3}


@app.post("/api/sync")
def sync_from_s3():
    s3_storage = get_s3_storage()
    if not s3_storage:
        raise HTTPException(400, "S3 is not configured (set S3_BUCKET_NAME in .env).")

    qdrant_service = get_qdrant_service()
    try:
        renamed = reconcile_local_names_with_s3(s3_storage, qdrant_service)
        downloaded = s3_storage.sync_down(settings.PDF_FOLDER)
    except Exception as exc:
        raise HTTPException(502, f"Could not sync from S3: {exc}")

    local_files = existing_documents()
    embedding_service = get_embedding_service()
    pending = files_needing_ingestion(local_files, qdrant_service)

    indexed_count = 0
    error_count = 0
    for filename in pending:
        dest_path = local_read_path(filename)
        path_meta = document_path_metadata(dest_path)
        try:
            ingest_document_by_extension(
                dest_path,
                embedding_service,
                qdrant_service,
                folder=path_meta.get("folder", ""),
            )
            qdrant_service.enrich_document_metadata(
                filename,
                canonical_display_name(filename),
                path_meta.get("s3_key", filename),
                folder_name=path_meta.get("folder", ""),
                local_path=dest_path,
                subfolder=path_meta.get("subfolder", ""),
            )
            indexed_count += 1
        except Exception as exc:
            error_count += 1
            logger.warning("Failed to index '%s' during sync: %s", filename, exc)

    # Do not return document inventories to the client — only opaque status.
    return {
        "ok": True,
        "downloaded_count": len(downloaded),
        "renamed_count": len(renamed),
        "indexed_count": indexed_count,
        "error_count": error_count,
        "up_to_date": not downloaded and not pending and not renamed,
    }


_FIRST_PERSON_NAME_PATTERN = re.compile(
    r"\bmy name(?:'s| is)\s+([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2})",
    re.IGNORECASE,
)


def _extract_first_person_name(story_text: str) -> Optional[str]:
    """Pull the invented character's full name out of a first-person
    story's self-introduction (e.g. 'Hi, my name is Maria Rodriguez...'),
    per prompts/video_story_dual_system.txt's required story structure.
    Captures up to three consecutive capitalized words so multi-word names
    survive, not just the first token. Used to auto-name saved Video
    Script + Story pairs when the user doesn't type one."""
    match = _FIRST_PERSON_NAME_PATTERN.search(story_text)
    if not match:
        return None
    name = match.group(1).strip().rstrip(",.!?")
    return name or None





_STORY_HEADER_PATTERN = re.compile(r"^Story\s*-\s*(.+?)\.mp4\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_story_name(script_text: str) -> Optional[str]:
    """Pull the invented character's name out of the script's required first
    line ('Story - <name>.mp4', per prompts/presentation_prompt.txt), so
    generated scripts are auto-named after their story instead of a generic
    'Transcript N' when the user doesn't type a name."""
    match = _STORY_HEADER_PATTERN.search(script_text)
    if not match:
        return None
    name = match.group(1).strip()
    return name or None


@app.post("/api/transcript", response_model=TranscriptResponse)
def generate_transcript(req: TranscriptRequest = TranscriptRequest()):
    qdrant_service = get_qdrant_service()
    presentation_llm = get_presentation_llm()

    docs = existing_documents()
    all_chunks = []
    for pdf in docs:
        chunks = _get_all_chunks(pdf, qdrant_service)
        if chunks:
            all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(
            422, "No indexed content found. Please upload and process at least one document first."
        )

    try:
        script = presentation_llm.generate_presentation(all_chunks)
    except Exception as exc:
        raise HTTPException(500, f"Something went wrong while generating the training script: {exc}")

    custom_name = (req.name or "").strip()
    story_name = _extract_story_name(script)
    label = custom_name or story_name or f"Transcript {len(_list_saved_scripts()) + 1}"
    saved_path = save_narrative_script(script, label)

    entry = {
        "id": os.path.basename(saved_path),
        "label": os.path.basename(saved_path)[: -len("_script.txt")].replace("_", " "),
        "script": script,
        "saved_path": saved_path,
        "created_at": datetime.fromtimestamp(os.path.getmtime(saved_path)).isoformat(),
    }
    return TranscriptResponse(**entry)


@app.get("/api/transcripts")
def list_transcripts():
    return {
        "transcripts": [
            TranscriptSummary(id=t["id"], label=t["label"], created_at=t["created_at"])
            for t in _list_saved_scripts()
        ]
    }


@app.get("/api/transcripts/{transcript_id}", response_model=TranscriptResponse)
def get_transcript(transcript_id: str):
    for t in _list_saved_scripts():
        if t["id"] == transcript_id:
            try:
                with open(t["saved_path"], "r", encoding="utf-8") as f:
                    script_text = f.read()
            except OSError as exc:
                raise HTTPException(500, f"Could not read saved script: {exc}")
            return TranscriptResponse(
                id=t["id"],
                label=t["label"],
                script=script_text,
                saved_path=t["saved_path"],
                created_at=t["created_at"],
            )
    raise HTTPException(404, "Video script not found.")


@app.post("/api/video-story", response_model=VideoStoryResponse)
def generate_video_story(req: VideoStoryRequest):
    """Dual Video Script + Story generator (the 'DUAL-OUTPUT MODE' section
    of prompts/presentation_prompt.txt): no topic needed — it draws on
    everything indexed in the Knowledge Base and picks its own topic, then
    produces a scene-by-scene Video Script and a first-person incident
    Story together in one call, teaching the same lesson from two angles.
    """
    qdrant_service = get_qdrant_service()
    presentation_llm = get_presentation_llm()

    docs = existing_documents()
    all_chunks = []
    for pdf in docs:
        chunks = _get_all_chunks(pdf, qdrant_service)
        if chunks:
            all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(
            422, "No indexed content found. Please upload and process at least one document first."
        )

    try:
        video_script, story, seed_character_name = presentation_llm.generate_video_and_story(
            req.learning_objectives or "", all_chunks, scenario_hint=req.hint
        )
    except Exception as exc:
        raise HTTPException(500, f"Something went wrong while generating the video script and story: {exc}")

    custom_name = (req.name or "").strip()
    label = custom_name or seed_character_name or f"Video Story {len(_list_saved_dual_scripts()) + 1}"
    video_path, story_path = save_dual_script(video_script, story, label)

    entry = {
        "id": os.path.basename(video_path),
        "label": os.path.basename(video_path)[: -len("_video.txt")].replace("_", " "),
        "video_script": video_script,
        "story": story,
        "created_at": datetime.fromtimestamp(os.path.getmtime(video_path)).isoformat(),
    }
    return VideoStoryResponse(**entry)


@app.get("/api/video-stories")
def list_video_stories():
    return {
        "video_stories": [
            VideoStorySummary(id=t["id"], label=t["label"], created_at=t["created_at"])
            for t in _list_saved_dual_scripts()
        ]
    }


@app.get("/api/video-stories/{pair_id}", response_model=VideoStoryResponse)
def get_video_story(pair_id: str):
    for t in _list_saved_dual_scripts():
        if t["id"] == pair_id:
            try:
                with open(t["video_path"], "r", encoding="utf-8") as f:
                    video_script = f.read()
                with open(t["story_path"], "r", encoding="utf-8") as f:
                    story = f.read()
            except OSError as exc:
                raise HTTPException(500, f"Could not read saved video script / story: {exc}")
            return VideoStoryResponse(
                id=t["id"],
                label=t["label"],
                video_script=video_script,
                story=story,
                created_at=t["created_at"],
            )
    raise HTTPException(404, "Video script / story pair not found.")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty.")

    qdrant_service = get_qdrant_service()
    retriever = get_retriever()
    qa_llm = get_qa_llm()
    summary_llm = get_summary_llm()

    try:
        if is_summary_question(question):
            docs = existing_documents()
            target = resolve_filename(question, settings.PDF_FOLDER)
            if not target:
                target = resolve_summary_request(question, docs, None)

            if target:
                if target in _summary_cache:
                    cached = _summary_cache[target]
                    answer, questions = cached["summary"], cached["questions"]
                else:
                    try:
                        answer = summarize_indexed_document(target, qdrant_service, summary_llm)
                    except ValueError as exc:
                        return ChatResponse(answer=str(exc), type="info")
                    questions = generate_suggested_questions(summary_llm, answer, target)
                    _summary_cache[target] = {"summary": answer, "questions": questions}
                # Use the original S3 filename — never a generated display alias.
                return ChatResponse(answer=answer, type="summary", filename=target, questions=questions)

            q_lower = question.lower()
            all_keywords = [
                "all", "every", "each", "entire", "whole", "combined",
                "together", "documents", "all doc", "all pdf",
            ]
            wants_all = any(kw in q_lower for kw in all_keywords)
            if wants_all and docs:
                all_chunks = []
                for pdf in docs:
                    chunks = _get_all_chunks(pdf, qdrant_service)
                    if chunks:
                        all_chunks.extend(chunks)
                if all_chunks:
                    answer = summary_llm.generate_summary(all_chunks)
                    questions = generate_suggested_questions(summary_llm, answer, "Knowledge Base")
                    return ChatResponse(
                        answer=answer, type="summary", filename="", questions=questions
                    )
                return ChatResponse(
                    answer="I could not find this information in the indexed knowledge base.",
                    type="info",
                )

            tied = ambiguous_candidates(question, docs)
            if tied:
                display_names = [canonical_display_name(f) for f in tied]
                listed = "\n".join(f"- {name}" for name in sorted(display_names))
                return ChatResponse(
                    answer=(
                        "More than one document matches that request. "
                        "Which one did you mean?\n\n" + listed
                    ),
                    type="info",
                )

            return ChatResponse(
                answer=(
                    "I could not uniquely identify which document you mean. "
                    "Please ask about the topic directly, or name the original file "
                    "if you already know it."
                ),
                type="info",
            )

        # A document name in a normal Q&A prompt (not only "summarize")
        # scopes retrieval to that one document. This recognizes both the
        # S3 original name and the UI's canonical display name.
        target = resolve_filename(question, settings.PDF_FOLDER)
        if target:
            chunks = _get_all_chunks(target, qdrant_service)
            if not chunks:
                return ChatResponse(
                    answer="No indexed content was found for that document.", type="info"
                )
            answer = qa_llm.generate_answer(chunks, question)
        else:
            answer = answer_question(retriever, qa_llm, summary_llm, question)
        return ChatResponse(answer=answer, type="qa")
    except HTTPException:
        raise
    except Exception as exc:
        return ChatResponse(answer=f"An error occurred while answering: {exc}", type="info")


@app.post("/api/generate-image", response_model=ImageGenResponse)
def generate_image(req: ImageGenRequest):
    """Image generation pipeline: retrieve relevant chunks for req.query,
    have Nova Lite turn (query + chunks) into one optimized prompt, then
    have Nova Canvas render the image. Returns the Nova Lite prompt (for
    transparency/debugging) plus the image as base64 and the on-disk path
    it was also saved to.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "query must not be empty.")

    retriever = get_retriever()
    image_prompt_llm = get_image_prompt_llm()
    image_gen_service = get_image_gen_service()

    try:
        result = generate_document_image(query, retriever, image_prompt_llm, image_gen_service)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        logger.error("Image generation failed: %s", exc)
        raise HTTPException(500, "An error occurred while generating the image.")

    return ImageGenResponse(**result)


@app.get("/api/content-types")
def content_types():
    """List of Content Types the Learning Content Generation Engine
    supports, for the UI's dropdown."""
    return {"content_types": CONTENT_TYPES}


@app.post("/api/generate-content", response_model=ContentGenResponse)
def generate_content(req: ContentGenRequest):
    """Learning Content Generation Engine: Content Type + Topic in,
    original learner-facing text + a matching AI image out. See
    generate_learning_feed_item() for the full pipeline (Knowledge
    Extraction -> Content Generation Agent -> Image Prompt Generation
    Agent -> Image Generation Agent).
    """
    content_type = req.content_type.strip()
    topic = req.topic.strip()
    if not content_type:
        raise HTTPException(400, "content_type must not be empty.")
    if not topic:
        raise HTTPException(400, "topic must not be empty.")

    retriever = get_retriever()
    content_llm = get_content_llm()
    image_prompt_llm = get_image_prompt_llm()
    image_gen_service = get_image_gen_service()

    try:
        result = generate_learning_feed_item(
            content_type,
            topic,
            retriever,
            content_llm,
            image_prompt_llm,
            image_gen_service,
            monthly_topic_content=req.monthly_topic_content,
            common_data=req.common_data,
            web_results=req.web_results,
            folders=req.folders,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        logger.error("Content generation failed: %s", exc)
        raise HTTPException(500, "An error occurred while generating the content.")

    return ContentGenResponse(**result)


@app.get("/api/folders")
def list_folders():
    """Every folder currently indexed in Qdrant (e.g. "video_scripts",
    "company_policies") -- nothing hardcoded, this reflects whatever
    you've actually ingested, however many folders that is."""
    return {"folders": get_qdrant_service().list_folders()}


@app.post("/api/daily-tip", response_model=DailyTipResponse)
def daily_tip(req: DailyTipRequest):
    """Daily Tip: one or more 40-80 word practical tips.

    Folder scope: if req.folders is given, used as-is. Otherwise, folder
    names mentioned in req.topic (e.g. "10 daily tips from the
    video_scripts folder") are auto-detected against whatever's actually
    indexed; if none are found, retrieval is global across every folder
    (no dedicated daily_tips folder needed).

    req.count controls how many distinct tips come back (default 1).
    """
    retriever = get_retriever()
    content_llm = get_content_llm()
    topic = (req.topic or "").strip()
    count = max(1, min(req.count, 25))  # sane upper bound on one request

    folders = req.folders
    if folders is None and topic:
        known_folders = get_qdrant_service().list_folders()
        folders = retriever.extract_folder_hints(topic, known_folders) or None

    try:
        chunks = retriever.retrieve_for_daily_tip(topic, folders=folders)
    except Exception as exc:
        logger.error("Retrieval failed during daily tip generation: %s", exc)
        raise HTTPException(500, "An error occurred while retrieving relevant information.")

    if not chunks:
        scope_note = f" in folder(s) {', '.join(folders)}" if folders else ""
        raise HTTPException(
            404,
            f"No indexed content was found{scope_note} to generate a daily tip from. "
            "Upload and process documents first.",
        )

    try:
        tips = content_llm.generate_daily_tips(
            chunks,
            topic=topic,
            count=count,
            common_data=req.common_data,
            web_results=req.web_results,
        )
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        logger.error("Daily tip generation failed: %s", exc)
        raise HTTPException(500, "An error occurred while generating the daily tip(s).")

    return DailyTipResponse(
        topic=topic or "today's safety learning",
        tips=tips,
        word_counts=[len(t.split()) for t in tips],
    )


# ---------------------------------------------------------------------------
# Static frontend (static/index.html) — served last so /api/* takes priority.
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))