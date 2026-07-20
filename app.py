"""
Streamlit app for the RAG bot — lives inside bot/, next to config/ and
services/.

Streamlit is the entire application: there is no dependency on
scripts/*.py. All orchestration (ingestion, intent routing, summarization,
Q&A) is implemented right here, directly on top of the reusable building
blocks in services/:

    services/pdf_parser.py      -> PDFParser            (PDF -> pages)
    services/chunking.py        -> DocumentChunker,
                                    SemanticChunkingService (pages -> chunks)
    services/embeddings.py      -> EmbeddingService      (BAAI/bge-m3)
    services/qdrant_db.py       -> QdrantService          (vector store)
    services/retriever.py       -> Retriever              (search + intent)
    services/openrouter_llm.py  -> OpenRouterLLMService    (Qwen via OpenRouter)

Folder layout expected:

    bot/
        app.py      <- this file (run this)
        config/
        services/
        scripts/
        data/pdfs/
        .env

Run with (from inside the bot/ folder):

    streamlit run app.py
"""

import logging
import os
import re
import uuid
from datetime import datetime
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

# config/settings.py uses python-dotenv + relative paths (e.g.
# PDF_FOLDER="data/pdfs"), both of which assume the working directory is
# this folder (bot/) — which it is, since app.py lives here too.

from config.settings import settings  # noqa: E402
from services.pdf_parser import PDFParser  # noqa: E402
from services.docx_parser import DocxParser  # noqa: E402
from services.excel_parser import ExcelParser  # noqa: E402
from services.pptx_parser import PptxParser  # noqa: E402
from services.image_parser import ImageParser, IMAGE_EXTENSIONS  # noqa: E402
from services.markdown_parser import MarkdownParser  # noqa: E402
from services.xml_parser import XMLParser  # noqa: E402
from services.json_parser import JSONParser  # noqa: E402
from services.transcript_parser import TranscriptParser  # noqa: E402
from services.reranker import ReRankerService  # noqa: E402
from services.s3_storage import S3Storage  # noqa: E402
from services.chunking import (
    Chunk,
    chunk_extracted_pages,
    chunk_pages_legacy,
)
from services.embeddings import EmbeddingService  # noqa: E402
from services.qdrant_db import QdrantService  # noqa: E402
from services.retriever import Retriever, SUMMARY_KEYWORDS  # noqa: E402
from services.llm_service import (  # noqa: E402
    FALLBACK_ANSWER,
    OpenRouterLLMService,
    BedrockLLMService,
)
from services.document_resolver import (  # noqa: E402
    resolve_pdf_reference,
    resolve_summary_request,
    ambiguous_candidates,
)
from services.canonical_naming import canonical_display_name  # noqa: E402


# ===========================================================================
# BACKEND — orchestration logic (this is what scripts/*.py used to do,
# now implemented directly in Streamlit on top of services/ only).
# ===========================================================================

# File types the pipeline can ingest as training content. Each maps to its
# own parser (PDFParser / DocxParser / ExcelParser) but all of them emit the
# same PageContent objects, so everything downstream (chunking, embedding,
# Qdrant storage, retrieval) is identical regardless of source format.
SUPPORTED_EXTENSIONS = (
    ".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".pptx", ".md", ".xml", ".json", ".txt",
) + IMAGE_EXTENSIONS

# ---------------------------------------------------------------------------
# Cached resources — heavy objects loaded once per process.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (BAAI/bge-m3)...")
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )


@st.cache_resource(show_spinner="Connecting to Qdrant...")
def get_qdrant_service() -> QdrantService:
    return QdrantService(
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        api_key=settings.QDRANT_API_KEY,
    )


@st.cache_resource(show_spinner=False)
def get_s3_storage() -> Optional[S3Storage]:
    """Return an S3Storage, or None if S3_BUCKET_NAME isn't configured —
    in which case the app runs local-folder-only, exactly as before."""
    if not settings.S3_BUCKET_NAME:
        return None
    return S3Storage(
        bucket_name=settings.S3_BUCKET_NAME,
        prefix=settings.S3_PREFIX,
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


@st.cache_resource(show_spinner=False)
def get_reranker_service() -> Optional[ReRankerService]:
    if not settings.USE_RERANKER:
        return None
    return ReRankerService(
        model_name=settings.RERANKER_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )


@st.cache_resource(show_spinner=False)
def get_retriever(_embedding_service, _qdrant_service) -> Retriever:
    return Retriever(
        _embedding_service,
        _qdrant_service,
        top_k=settings.TOP_K,
        summary_top_k=settings.TOP_K_SUMMARY,
        min_relevance_score=settings.MIN_RELEVANCE_SCORE,
        reranker_service=get_reranker_service(),
    )


@st.cache_resource(show_spinner=False)
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


@st.cache_resource(show_spinner=False)
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


@st.cache_resource(show_spinner=False)
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


# ---------------------------------------------------------------------------
# Filename resolution (was scripts/summarize.py::_resolve_filename)
# ---------------------------------------------------------------------------
def resolve_filename(name: str, pdf_folder: str):
    try:
        candidates = [
            f for f in os.listdir(pdf_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
    except OSError:
        return None
    resolved = resolve_pdf_reference(name, candidates)
    if resolved:
        return resolved
    normalized = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
    matches = [
        candidate for candidate in candidates
        if " ".join(re.findall(r"[a-z0-9]+", canonical_display_name(candidate).lower())) in normalized
    ]
    return matches[0] if len(matches) == 1 else None


def display_name(filename: str) -> str:
    return canonical_display_name(filename)


# ---------------------------------------------------------------------------
# PDF -> pages -> semantic boundaries -> document chunks -> semantic chunks
# (shared by both ingestion and summarization)
# ---------------------------------------------------------------------------
def parse_and_chunk(
    file_path: str,
    embedding_service: EmbeddingService,
    parser: PDFParser = None,
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

    return chunks or None


# ---------------------------------------------------------------------------
# Ingestion: parse -> chunk -> embed -> upsert into Qdrant
# (was scripts/ingest.py, scoped to a single file)
# ---------------------------------------------------------------------------
def _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service) -> None:
    """Embed and persist bounded batches to keep large documents memory-safe."""
    filename = chunks[0].filename
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata.update({
            "original_filename": filename,
            "canonical_name": canonical_display_name(filename),
            "s3_key": metadata.get("s3_key", filename),
            "local_path": os.path.join(settings.PDF_FOLDER, filename),
        })
        chunk.metadata = metadata
    batch_size = max(1, settings.INDEX_BATCH_SIZE)
    first_batch = chunks[:batch_size]
    if not first_batch:
        return

    # Prove embedding works before replacing an existing document.
    embeddings = embedding_service.embed_documents(
        [chunk.text for chunk in first_batch]
    )
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


def ingest_single_pdf(file_path: str, embedding_service: EmbeddingService,
                      qdrant_service: QdrantService) -> int:
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


def ingest_single_docx(file_path: str, embedding_service: EmbeddingService,
                        qdrant_service: QdrantService) -> int:
    """Ingest a .docx training-content file the same way ingest_single_pdf
    ingests a PDF. Word docs have no PDF-style page-level TOC, so there is
    no TOC-record step here (see parse_and_chunk / DocxParser.extract_pages)."""
    parser = DocxParser(docx_folder=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0

    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)

    return len(chunks)


def ingest_single_excel(file_path: str, embedding_service: EmbeddingService,
                         qdrant_service: QdrantService, restructure_llm=None) -> int:
    """Ingest an .xlsx/.xlsm/.xls/.csv file through the shared RAG contract.

    Each emitted PageContent is already a structured 20-30-row table chunk;
    see ExcelParser.extract_pages.
    The sheet name is stored as toc_section so a query naming a sheet can
    be narrowed to it. Like .docx, there's no PDF-style page-level TOC
    step here.

    ``restructure_llm`` is accepted for backward compatibility; structured
    parsing itself is deterministic and requires no LLM call."""
    parser = ExcelParser(excel_folder=settings.PDF_FOLDER, llm_service=restructure_llm)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser, use_semantic_chunking=False)
    if not chunks:
        return 0

    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)

    return len(chunks)


def ingest_single_pptx(file_path: str, embedding_service: EmbeddingService,
                        qdrant_service: QdrantService) -> int:
    """Ingest a .pptx training-content file the same way ingest_single_docx
    ingests a Word doc. Slides give a natural per-page unit (see
    PptxParser.extract_pages), but there's no PDF-style TOC step here."""
    parser = PptxParser(pptx_folder=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0

    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)

    return len(chunks)


def ingest_single_image(file_path: str, embedding_service: EmbeddingService,
                         qdrant_service: QdrantService) -> int:
    """Ingest a standalone image via Docker Tesseract OCR (see
    ImageParser), the same OCR path used for scanned PDF pages."""
    parser = ImageParser(image_folder=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0

    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)

    return len(chunks)


def ingest_single_markdown(file_path: str, embedding_service: EmbeddingService,
                           qdrant_service: QdrantService) -> int:
    parser = MarkdownParser(folder_path=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)
    return len(chunks)


def ingest_single_xml(file_path: str, embedding_service: EmbeddingService,
                       qdrant_service: QdrantService) -> int:
    parser = XMLParser(folder_path=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)
    return len(chunks)


def ingest_single_json(file_path: str, embedding_service: EmbeddingService,
                        qdrant_service: QdrantService) -> int:
    parser = JSONParser(folder_path=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)
    return len(chunks)


def ingest_single_transcript(file_path: str, embedding_service: EmbeddingService,
                            qdrant_service: QdrantService) -> int:
    parser = TranscriptParser(folder_path=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0
    _embed_and_upsert_in_batches(chunks, embedding_service, qdrant_service)
    return len(chunks)


def ingest_document_by_extension(
    dest_path: str,
    embedding_service: EmbeddingService,
    qdrant_service: QdrantService,
) -> int:
    file_ext = os.path.splitext(dest_path)[1].lower()

    if file_ext == ".docx":
        return ingest_single_docx(dest_path, embedding_service, qdrant_service)
    elif file_ext in (".xlsx", ".xlsm", ".xls", ".csv"):
        return ingest_single_excel(dest_path, embedding_service, qdrant_service)
    elif file_ext == ".pptx":
        return ingest_single_pptx(dest_path, embedding_service, qdrant_service)
    elif file_ext in IMAGE_EXTENSIONS:
        return ingest_single_image(dest_path, embedding_service, qdrant_service)
    elif file_ext == ".pdf":
        return ingest_single_pdf(dest_path, embedding_service, qdrant_service)
    elif file_ext == ".md":
        return ingest_single_markdown(dest_path, embedding_service, qdrant_service)
    elif file_ext == ".xml":
        return ingest_single_xml(dest_path, embedding_service, qdrant_service)
    elif file_ext == ".json":
        return ingest_single_json(dest_path, embedding_service, qdrant_service)
    elif file_ext == ".txt":
        return ingest_single_transcript(dest_path, embedding_service, qdrant_service)
    else:
        logger.warning("Skipping '%s': unsupported extension '%s'.", dest_path, file_ext)
        return 0


def files_needing_ingestion(filenames: list, qdrant_service: QdrantService) -> list:
    """Filter filenames down to those with no chunks yet indexed in Qdrant.

    Covers two cases at once: files just downloaded by sync_down() (never
    indexed), and files that were already sitting in the local folder from
    an earlier sync but never made it into Qdrant (e.g. ingestion failed
    previously, or the collection was reset). retrieve_document() is a
    read-only payload scroll, so this never re-parses/re-embeds anything.

    If the Qdrant collection doesn't exist yet at all, every file needs
    ingestion — that lookup failure is treated as "not indexed" rather
    than surfaced as an error here.
    """
    pending = []
    for filename in filenames:
        try:
            already_indexed = bool(qdrant_service.retrieve_document(filename))
        except Exception:
            already_indexed = False
        if not already_indexed:
            pending.append(filename)
    return pending


# ---------------------------------------------------------------------------
# Summarization after upload: read the chunks already persisted in Qdrant.
# ---------------------------------------------------------------------------
def _get_all_chunks(name: str, qdrant_service):
    """Resolve name to indexed chunks. Returns None if file not found, [] if no chunks."""
    resolved = resolve_filename(name, settings.PDF_FOLDER)
    if resolved is None:
        return None
    raw = qdrant_service.retrieve_document(resolved)
    return [{"text": c["text"], "filename": c["filename"], "page_label": c["page_label"]} for c in raw]


def save_narrative_script(script_text: str, label: str) -> str:
    """Save a generated training script as a .txt file in the local
    Narrative_scripts folder (settings.NARRATIVE_SCRIPTS_DIR) and return the
    path it was written to. Filenames are timestamped so repeated
    generations never overwrite each other.
    """
    os.makedirs(settings.NARRATIVE_SCRIPTS_DIR, exist_ok=True)

    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "script"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_script_{safe_label}_{timestamp}.txt"
    path = os.path.join(settings.NARRATIVE_SCRIPTS_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(script_text)

    logger.info("Saved training script to '%s'.", path)
    return path


def summarize_indexed_document(
    name: str,
    qdrant_service: QdrantService,
    summary_llm: OpenRouterLLMService,
) -> str:
    """Summarize stored chunks without reopening or rechunking the PDF."""
    chunks = _get_all_chunks(name, qdrant_service)
    if chunks is None:
        available = ", ".join(
            f for f in os.listdir(settings.PDF_FOLDER)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ) if os.path.isdir(settings.PDF_FOLDER) else "(folder not found)"
        return (
            f"File not found for '{name}' in {settings.PDF_FOLDER}.\n"
            f"Available documents: {available or 'none'}"
        )
    if not chunks:
        return (
            f"No indexed content was found for: {name}. "
            "Upload it with “Process & Summarize” first."
        )
    summary = summary_llm.generate_summary(chunks)
    if summary.strip() == FALLBACK_ANSWER:
        raise RuntimeError(
            "The summary model returned no content. Please try the request again."
        )
    return summary


# ---------------------------------------------------------------------------
# Query routing + answering (was scripts/query.py)
# ---------------------------------------------------------------------------
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
    if re.search(r"\bunit\s*\d+\b", q) and re.search(
        r"\b(summary|summarize|overview|brief)\b", q
    ):
        return True
    return False


def answer_question(
    retriever: Retriever,
    qa_llm: OpenRouterLLMService,
    summary_llm: OpenRouterLLMService,
    question: str,
) -> str:
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


# ---------------------------------------------------------------------------
# Suggested follow-up questions, generated from the summary text.
# ---------------------------------------------------------------------------
def generate_suggested_questions(llm_service: OpenRouterLLMService,
                                  summary_text: str, filename: str) -> list:
    prompt = (
        f"You produced this executive summary of the document \"{filename}\":\n\n"
        f"{summary_text}\n\n"
        "Suggest exactly 3 short, natural follow-up questions a reader might "
        "ask about this document, based only on the summary above. "
        "Return ONLY the 3 questions, one per line. "
        "Do not number them, do not use bullets or quotes, no extra text."
    )
    try:
        raw = llm_service._call_llm(prompt)  # reuse existing HTTP call helper
    except Exception:
        return []

    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip().strip('"') for l in raw.splitlines()]
    lines = [l for l in lines if l]
    return lines[:3]


# ===========================================================================
# UI — Streamlit only.
# ===========================================================================
st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄", layout="wide")

st.title("📄 Document Q&A Assistant")
st.caption("Upload a PDF, get an instant summary with suggested questions, then chat with your documents.")

if settings.LLM_PROVIDER == "bedrock":
    if not (settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY):
        st.info(
            "LLM_PROVIDER=bedrock with no AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in .env — "
            "falling back to boto3's default credential chain (IAM role, AWS CLI config, etc.). "
            "If that's not configured either, uploading/summarizing/querying will fail.",
            icon="ℹ️",
        )
elif not settings.OPENROUTER_API_KEY:
    st.warning(
        "OPENROUTER_API_KEY is not set in .env — uploading/summarizing/"
        "querying will fail until it's configured.",
        icon="⚠️",
    )

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("summary_text", None)
st.session_state.setdefault("summary_filename", None)
st.session_state.setdefault("suggested_questions", [])
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("summary_notice", None)
# Cache: filename -> {"summary": str, "questions": list}
st.session_state.setdefault("summary_cache", {})
st.session_state.setdefault("presentation_text", None)
st.session_state.setdefault("presentation_filename", None)
st.session_state.setdefault("presentation_saved_path", None)

os.makedirs(settings.PDF_FOLDER, exist_ok=True)

s3_storage = get_s3_storage()

# S3 sync (download + extract + chunk + embed) only runs when the user
# explicitly clicks "🔄 Sync from S3" in the sidebar -- see that button's
# handler below. Nothing here runs automatically on startup/restart, so
# opening or restarting the app never triggers a network call or loads
# the embedding model on its own.


def _existing_documents():
    return sorted(
        f for f in os.listdir(settings.PDF_FOLDER) if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )


# ---------------------------------------------------------------------------
# Sidebar — upload & document picker
# ---------------------------------------------------------------------------
with st.sidebar:
    if s3_storage:
        if st.button("🔄 Sync from S3", use_container_width=True):
            try:
                with st.spinner("Fetching documents from S3..."):
                    downloaded = s3_storage.sync_down(settings.PDF_FOLDER)

                qdrant_service = get_qdrant_service()
                local_files = [
                    f for f in os.listdir(settings.PDF_FOLDER) if f.lower().endswith(SUPPORTED_EXTENSIONS)
                ]
                embedding_service = get_embedding_service()
                pending = files_needing_ingestion(local_files, qdrant_service)

                if pending:
                    progress_placeholder = st.empty()
                    for i, filename in enumerate(pending, start=1):
                        progress_placeholder.info("Indexing updates…", icon="⏳")
                        dest_path = os.path.join(settings.PDF_FOLDER, filename)
                        try:
                            ingest_document_by_extension(dest_path, embedding_service, qdrant_service)
                        except Exception as exc:
                            logger.warning("Could not index '%s': %s", filename, exc)
                    progress_placeholder.empty()

                if downloaded or pending:
                    st.toast("Sync completed successfully.", icon="✅")
                else:
                    st.toast("Already up to date with S3.", icon="📦")
            except Exception as exc:
                st.warning(f"Could not sync from S3: {exc}", icon="⚠️")

    st.header("📤 Upload a document")
    uploaded_file = st.file_uploader(
        "Choose a PDF, Word, PowerPoint, Excel, CSV, or image document",
        type=["pdf", "docx", "pptx", "xlsx", "xlsm", "xls", "csv",
              "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        if st.button("Process", type="primary", use_container_width=True):
            # The upload name is the storage identity in both local cache and
            # S3. Canonical names are display metadata only.
            original_filename = os.path.basename(uploaded_file.name)
            dest_path = os.path.join(settings.PDF_FOLDER, original_filename)

            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            embedding_service = get_embedding_service()
            qdrant_service = get_qdrant_service()

            # Qdrant-aware check (same as files_needing_ingestion used in the
            # S3 sync flow) rather than just "does the file exist on disk" --
            # a file can exist locally but have never made it into Qdrant
            # (e.g. a previous ingestion attempt failed, or the collection
            # was reset), in which case it still needs to be chunked.
            already_ingested = original_filename not in files_needing_ingestion(
                [original_filename], qdrant_service
            )

            if s3_storage:
                try:
                    if s3_storage.file_exists(original_filename):
                        st.info("This filename already exists in S3; using the existing authoritative object.", icon="ℹ️")
                    else:
                        s3_storage.upload_file(dest_path, original_filename)
                        st.toast("Document backed up to S3.", icon="📦")
                except Exception as exc:
                    st.warning(f"Saved locally, but S3 upload failed: {exc}", icon="⚠️")

            try:
                if not already_ingested:
                    with st.spinner("Reading and indexing document..."):
                        n_chunks = ingest_document_by_extension(
                            dest_path, embedding_service, qdrant_service
                        )
                    if n_chunks == 0:
                        st.error("No extractable content found in this document.")
                        st.stop()
                    st.success(
                        "Document indexed. Ask a question or request a summary in the chat below.",
                        icon="✅",
                    )
                else:
                    st.info("This document is already indexed.", icon="ℹ️")
            except Exception as exc:
                st.error(f"Something went wrong while processing the document: {exc}")

    existing = _existing_documents()
    # Document inventory is intentionally not shown in the chat UI.
    if existing:
        st.divider()
        st.header("📋 Generate Training Script")
        st.caption(
            "Generates a cinematic, story-driven training video script from the "
            "knowledge base. The format is fixed; the story is freshly "
            "invented every time you click — click again for a different one."
        )

        if st.button("Generate Training Script", use_container_width=True):
            qdrant_service = get_qdrant_service()
            presentation_llm = get_presentation_llm()
            try:
                with st.spinner("Gathering content from the knowledge base..."):
                    all_chunks = []
                    for pdf in existing:
                        chunks = _get_all_chunks(pdf, qdrant_service)
                        if chunks:
                            all_chunks.extend(chunks)
                if not all_chunks:
                    st.error("No indexed content found. Please upload and process at least one document first.")
                else:
                    with st.spinner("Generating training script… this may take a moment."):
                        presentation_text = presentation_llm.generate_presentation(all_chunks)
                    st.session_state.presentation_text = presentation_text
                    st.session_state.presentation_filename = "Training Script"
                    st.session_state.presentation_saved_path = save_narrative_script(
                        presentation_text, st.session_state.presentation_filename
                    )
            except Exception as exc:
                st.error(f"Something went wrong while generating the training script: {exc}")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.summary_text = None
        st.session_state.summary_filename = None
        st.session_state.suggested_questions = []
        st.session_state.summary_cache = {}
        st.session_state.presentation_text = None
        st.session_state.presentation_filename = None
        st.session_state.presentation_saved_path = None
        st.rerun()


# ---------------------------------------------------------------------------
# Summary card + suggested questions
# ---------------------------------------------------------------------------
if st.session_state.summary_text:
    with st.container(border=True):
        # Cite the original S3 filename only when a specific document was summarized.
        source_label = st.session_state.summary_filename or ""
        st.subheader(f"🧾 Summary{f' — {source_label}' if source_label else ''}")
        st.write(st.session_state.summary_text)

        if st.session_state.suggested_questions:
            st.write("**Suggested questions:**")
            cols = st.columns(len(st.session_state.suggested_questions))
            for col, q in zip(cols, st.session_state.suggested_questions):
                if col.button(q, use_container_width=True, key=f"sugg_{q}"):
                    st.session_state.pending_question = q

if st.session_state.presentation_text:
    with st.container(border=True):
        st.subheader("📋 Training Script")
        st.markdown(st.session_state.presentation_text)
        if st.session_state.presentation_saved_path:
            st.caption("Script saved locally.")
        st.download_button(
            label="⬇️ Download Training Script (.txt)",
            data=st.session_state.presentation_text,
            file_name="training_script.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

typed_question = st.chat_input("Ask a question…")
question_to_answer = st.session_state.pending_question or typed_question
st.session_state.pending_question = None

if question_to_answer:
    st.session_state.chat_history.append(("user", question_to_answer))
    with st.chat_message("user"):
        st.write(question_to_answer)

    embedding_service = get_embedding_service()
    qdrant_service = get_qdrant_service()
    retriever = get_retriever(embedding_service, qdrant_service)
    qa_llm = get_qa_llm()
    summary_llm = get_summary_llm()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if is_summary_question(question_to_answer):
                    target = resolve_filename(question_to_answer, settings.PDF_FOLDER)
                    if not target:
                        target = resolve_summary_request(
                            question_to_answer,
                            _existing_documents(),
                            st.session_state.summary_filename,
                        )
                    if target:
                        # ✅ Check cache first — no Qdrant scroll or LLM call needed
                        if target in st.session_state.summary_cache:
                            cached = st.session_state.summary_cache[target]
                            answer = cached["summary"]
                            questions = cached["questions"]
                        else:
                            answer = summarize_indexed_document(
                                target,
                                qdrant_service,
                                summary_llm,
                            )
                            questions = generate_suggested_questions(
                                summary_llm, answer, target
                            )
                            # Store in cache for next time
                            st.session_state.summary_cache[target] = {
                                "summary": answer,
                                "questions": questions,
                            }
                        st.session_state.summary_text = answer
                        st.session_state.summary_filename = target
                        st.session_state.suggested_questions = questions
                        st.session_state.summary_notice = target
                    else:
                        # Check if user wants all documents summarized
                        q_lower = question_to_answer.lower()
                        all_keywords = ["all", "every", "each", "entire", "whole", "combined", "together", "documents", "all doc", "all pdf"]
                        wants_all = any(kw in q_lower for kw in all_keywords)
                        existing = _existing_documents()
                        if wants_all and existing:
                            all_chunks = []
                            for pdf in existing:
                                chunks = _get_all_chunks(pdf, qdrant_service)
                                if chunks:
                                    all_chunks.extend(chunks)
                            if all_chunks:
                                answer = summary_llm.generate_summary(all_chunks)
                                questions = generate_suggested_questions(
                                    summary_llm, answer, "All Documents"
                                )
                                st.session_state.summary_text = answer
                                st.session_state.summary_filename = "All Documents"
                                st.session_state.suggested_questions = questions
                            else:
                                answer = "No indexed content found. Please upload and process at least one PDF first."
                        else:
                            tied = ambiguous_candidates(question_to_answer, existing)
                            if tied:
                                options = ", ".join(tied)
                                answer = (
                                    "More than one document shares that unit number. "
                                    f"Please specify which one by its unique id: {options}"
                                )
                            else:
                                available = ", ".join(display_name(item) for item in existing) or "none"
                                answer = (
                                    "I couldn't uniquely match that document name. "
                                    f"Available documents: {available}"
                                )
                else:
                    answer = answer_question(
                        retriever, qa_llm, summary_llm, question_to_answer
                    )
            except Exception as exc:
                answer = f"An error occurred while answering: {exc}"
        st.write(answer)

    st.session_state.chat_history.append(("assistant", answer))
    if st.session_state.summary_notice:
        st.session_state.summary_notice = None
        st.rerun()
