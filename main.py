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
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import settings
from services.pdf_parser import PDFParser
from services.docx_parser import DocxParser
from services.excel_parser import ExcelParser
from services.s3_storage import S3Storage
from services.chunking import Chunk, DocumentChunker, SemanticChunkingService
from services.embeddings import EmbeddingService
from services.qdrant_db import QdrantService
from services.retriever import Retriever, SUMMARY_KEYWORDS
from services.llm_service import (
    FALLBACK_ANSWER,
    OpenRouterLLMService,
    BedrockLLMService,
)
from services.document_resolver import (
    resolve_pdf_reference,
    resolve_summary_request,
)
from services.image_generation_service import NovaCanvasService

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv")

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
def get_retriever() -> Retriever:
    return Retriever(
        get_embedding_service(),
        get_qdrant_service(),
        top_k=settings.TOP_K,
        summary_top_k=settings.TOP_K_SUMMARY,
        min_relevance_score=settings.MIN_RELEVANCE_SCORE,
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
def get_image_prompt_llm() -> BedrockLLMService:
    """Nova Lite — always Bedrock, no OpenRouter fallback for this pipeline."""
    return BedrockLLMService(
        model=settings.BEDROCK_IMAGE_PROMPT_MODEL,
        max_tokens=settings.IMAGE_PROMPT_MAX_TOKENS,
        temperature=settings.IMAGE_PROMPT_TEMPERATURE,
        region_name=settings.BEDROCK_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


@lru_cache(maxsize=1)
def get_image_gen_service() -> NovaCanvasService:
    return NovaCanvasService(
        model=settings.BEDROCK_IMAGE_GEN_MODEL,
        region_name=settings.BEDROCK_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        width=settings.IMAGE_WIDTH,
        height=settings.IMAGE_HEIGHT,
        quality=settings.IMAGE_QUALITY,
        cfg_scale=settings.IMAGE_CFG_SCALE,
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
def resolve_filename(name: str, pdf_folder: str):
    try:
        candidates = [
            f for f in os.listdir(pdf_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
    except OSError:
        return None
    return resolve_pdf_reference(name, candidates)


def parse_and_chunk(
    file_path: str,
    embedding_service: EmbeddingService,
    parser=None,
    use_semantic_chunking: bool = True,
):
    parser = parser or PDFParser(pdf_folder=settings.PDF_FOLDER)
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
                page_start=(page.metadata or {}).get("row_start", page.page_number),
                page_end=(page.metadata or {}).get("row_end", page.page_number),
                metadata=dict(page.metadata or {}),
                toc_section=(page.metadata or {}).get("toc_section", ""),
            )
            for page in pages
            if page.text and page.text.strip()
        ] or None

    document_chunker = DocumentChunker(
        heading_max_length=settings.DOC_CHUNK_HEADING_MAX_LENGTH,
        min_paragraph_length=settings.DOC_CHUNK_MIN_PARAGRAPH_LENGTH,
    )
    document_chunks = document_chunker.chunk_pages(pages)
    if not document_chunks:
        return None

    if not use_semantic_chunking:
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=dc.text,
                filename=dc.filename,
                page_number=dc.page_number,
                page_label=dc.page_label,
                page_start=dc.page_start,
                page_end=dc.page_end,
                metadata=dict(dc.metadata or {}),
                toc_section=(dc.metadata or {}).get("toc_section", ""),
            )
            for dc in document_chunks
        ]

    semantic_chunker = SemanticChunkingService(
        embeddings=embedding_service.langchain_embeddings,
        buffer_size=settings.SEMANTIC_BUFFER_SIZE,
        breakpoint_threshold_type=settings.SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=settings.SEMANTIC_BREAKPOINT_AMOUNT,
        max_chunk_size=settings.MAX_CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = semantic_chunker.chunk_documents(document_chunks)
    return chunks or None


def _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service) -> None:
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


def ingest_single_pdf(file_path: str, embedding_service, qdrant_service) -> int:
    parser = PDFParser(pdf_folder=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0

    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)

    filename = os.path.basename(file_path)
    toc_records = [
        {
            "level": entry.level,
            "title": entry.title,
            "page_start": entry.page_start,
            "page_end": entry.page_end,
            "filename": filename,
        }
        for entry in parser.toc_map.get(filename, [])
    ]
    if toc_records:
        toc_embeddings = embedding_service.embed_documents(
            [entry["title"] for entry in toc_records]
        )
        qdrant_service.upsert_toc_entries(toc_records, toc_embeddings)

    return len(chunks)


def ingest_single_docx(file_path: str, embedding_service, qdrant_service) -> int:
    parser = DocxParser(docx_folder=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)
    return len(chunks)


def ingest_single_excel(file_path: str, embedding_service, qdrant_service, restructure_llm=None) -> int:
    parser = ExcelParser(excel_folder=settings.PDF_FOLDER, llm_service=restructure_llm)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, use_semantic_chunking=False)
    if not chunks:
        return 0
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)
    return len(chunks)


def ingest_document_by_extension(dest_path: str, embedding_service, qdrant_service) -> int:
    file_ext = os.path.splitext(dest_path)[1].lower()
    if file_ext == ".docx":
        return ingest_single_docx(dest_path, embedding_service, qdrant_service)
    elif file_ext in (".xlsx", ".xlsm", ".xls", ".csv"):
        return ingest_single_excel(dest_path, embedding_service, qdrant_service)
    elif file_ext == ".pdf":
        return ingest_single_pdf(dest_path, embedding_service, qdrant_service)
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


def save_narrative_script(script_text: str, label: str) -> str:
    os.makedirs(settings.NARRATIVE_SCRIPTS_DIR, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "script"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_script_{safe_label}_{timestamp}.txt"
    path = os.path.join(settings.NARRATIVE_SCRIPTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(script_text)
    logger.info("Saved training script to '%s'.", path)
    return path


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
    """
    try:
        chunks = retriever.retrieve(query)
    except Exception as exc:
        logger.error("Retrieval failed during image generation: %s", exc)
        raise RuntimeError("An error occurred while retrieving relevant information.") from exc

    if not chunks:
        raise ValueError("No relevant context was found in the indexed documents for this request.")

    image_prompt = image_prompt_llm.generate_image_prompt(query, chunks)
    image_bytes = image_gen_service.generate_image(image_prompt)
    saved_path = save_generated_image(image_bytes, label=query[:40])

    return {
        "image_prompt": image_prompt,
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "saved_path": saved_path,
    }


def summarize_indexed_document(name: str, qdrant_service, summary_llm) -> str:
    chunks = _get_all_chunks(name, qdrant_service)
    if chunks is None:
        available = ", ".join(
            f for f in os.listdir(settings.PDF_FOLDER) if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ) if os.path.isdir(settings.PDF_FOLDER) else "(folder not found)"
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
        return "The information is not available in the provided documents."

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
    return sorted(
        f for f in os.listdir(settings.PDF_FOLDER) if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )


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

# In-process store of every transcript generated this run, newest last.
# Each entry: {"id", "label", "script", "saved_path", "created_at"}
_transcripts: List[dict] = []


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


class ImageGenRequest(BaseModel):
    query: str


class ImageGenResponse(BaseModel):
    image_prompt: str
    image_base64: str
    saved_path: str


@app.on_event("startup")
def on_startup():
    os.makedirs(settings.PDF_FOLDER, exist_ok=True)


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
    return {"documents": existing_documents()}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")

    os.makedirs(settings.PDF_FOLDER, exist_ok=True)
    dest_path = os.path.join(settings.PDF_FOLDER, filename)
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    embedding_service = get_embedding_service()
    qdrant_service = get_qdrant_service()

    already_ingested = filename not in files_needing_ingestion([filename], qdrant_service)

    s3_uri = None
    s3_warning = None
    s3_storage = get_s3_storage()
    if s3_storage:
        try:
            s3_uri = s3_storage.upload_file(dest_path, filename)
        except Exception as exc:
            s3_warning = f"Saved locally, but S3 upload failed: {exc}"

    if already_ingested:
        return {
            "filename": filename,
            "already_indexed": True,
            "chunks": 0,
            "s3_uri": s3_uri,
            "s3_warning": s3_warning,
        }

    try:
        n_chunks = ingest_document_by_extension(dest_path, embedding_service, qdrant_service)
    except Exception as exc:
        raise HTTPException(500, f"Something went wrong while processing the document: {exc}")

    if n_chunks == 0:
        raise HTTPException(422, "No extractable content found in this document.")

    return {
        "filename": filename,
        "already_indexed": False,
        "chunks": n_chunks,
        "s3_uri": s3_uri,
        "s3_warning": s3_warning,
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
        summary = summarize_indexed_document(filename, qdrant_service, summary_llm)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Something went wrong while summarizing: {exc}")

    questions = generate_suggested_questions(summary_llm, summary, resolved)
    _summary_cache[resolved] = {"summary": summary, "questions": questions}
    return SummarizeResponse(filename=resolved, summary=summary, questions=questions)


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    qdrant_service = get_qdrant_service()
    local_path = os.path.join(settings.PDF_FOLDER, filename)

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
        try:
            s3_storage.client.delete_object(
                Bucket=s3_storage.bucket_name, Key=s3_storage._key_for(filename)
            )
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

    try:
        downloaded = s3_storage.sync_down(settings.PDF_FOLDER)
    except Exception as exc:
        raise HTTPException(502, f"Could not sync from S3: {exc}")

    local_files = existing_documents()
    embedding_service = get_embedding_service()
    qdrant_service = get_qdrant_service()
    pending = files_needing_ingestion(local_files, qdrant_service)

    indexed = []
    errors = {}
    for filename in pending:
        dest_path = os.path.join(settings.PDF_FOLDER, filename)
        try:
            ingest_document_by_extension(dest_path, embedding_service, qdrant_service)
            indexed.append(filename)
        except Exception as exc:
            errors[filename] = str(exc)

    return {
        "downloaded": downloaded,
        "indexed": indexed,
        "errors": errors,
        "up_to_date": not downloaded and not pending,
    }


@app.post("/api/transcript", response_model=TranscriptResponse)
def generate_transcript():
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

    saved_path = save_narrative_script(script, "All Documents")

    entry = {
        "id": str(uuid.uuid4()),
        "label": f"Transcript {len(_transcripts) + 1}",
        "script": script,
        "saved_path": saved_path,
        "created_at": datetime.now().isoformat(),
    }
    _transcripts.append(entry)
    return TranscriptResponse(**entry)


@app.get("/api/transcripts")
def list_transcripts():
    return {
        "transcripts": [
            TranscriptSummary(id=t["id"], label=t["label"], created_at=t["created_at"])
            for t in _transcripts
        ]
    }


@app.get("/api/transcripts/{transcript_id}", response_model=TranscriptResponse)
def get_transcript(transcript_id: str):
    for t in _transcripts:
        if t["id"] == transcript_id:
            return TranscriptResponse(**t)
    raise HTTPException(404, "Transcript not found.")


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
                    questions = generate_suggested_questions(summary_llm, answer, "All Documents")
                    return ChatResponse(
                        answer=answer, type="summary", filename="All Documents", questions=questions
                    )
                return ChatResponse(
                    answer="No indexed content found. Please upload and process at least one document first.",
                    type="info",
                )

            available = ", ".join(docs) or "none"
            return ChatResponse(
                answer=f"I couldn't uniquely match that document name. Available documents: {available}",
                type="info",
            )

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


# ---------------------------------------------------------------------------
# Static frontend (static/index.html) — served last so /api/* takes priority.
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))