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

import streamlit as st

logger = logging.getLogger(__name__)

# config/settings.py uses python-dotenv + relative paths (e.g.
# PDF_FOLDER="data/pdfs"), both of which assume the working directory is
# this folder (bot/) — which it is, since app.py lives here too.

from config.settings import settings  # noqa: E402
from services.pdf_parser import PDFParser  # noqa: E402
from services.docx_parser import DocxParser  # noqa: E402
from services.chunking import DocumentChunker, SemanticChunkingService  # noqa: E402
from services.embeddings import EmbeddingService  # noqa: E402
from services.qdrant_db import QdrantService  # noqa: E402
from services.retriever import Retriever, SUMMARY_KEYWORDS  # noqa: E402
from services.openrouter_llm import (  # noqa: E402
    FALLBACK_ANSWER,
    OpenRouterLLMService,
)
from services.document_resolver import (  # noqa: E402
    resolve_pdf_reference,
    resolve_summary_request,
)


# ===========================================================================
# BACKEND — orchestration logic (this is what scripts/*.py used to do,
# now implemented directly in Streamlit on top of services/ only).
# ===========================================================================

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
def get_retriever(_embedding_service, _qdrant_service) -> Retriever:
    return Retriever(
        _embedding_service,
        _qdrant_service,
        top_k=settings.TOP_K,
        summary_top_k=settings.TOP_K_SUMMARY,
        min_relevance_score=settings.MIN_RELEVANCE_SCORE,
    )


@st.cache_resource(show_spinner=False)
def get_qa_llm() -> OpenRouterLLMService:
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
            f for f in os.listdir(pdf_folder) if f.lower().endswith((".pdf", ".docx"))
        ]
    except OSError:
        return None
    return resolve_pdf_reference(name, candidates)


# ---------------------------------------------------------------------------
# PDF -> pages -> document chunks -> semantic chunks
# (shared by both ingestion and summarization)
# ---------------------------------------------------------------------------
def parse_and_chunk(
    file_path: str,
    embedding_service: EmbeddingService,
    parser: PDFParser = None,
):
    parser = parser or PDFParser(pdf_folder=settings.PDF_FOLDER)
    pages = parser.extract_pages(file_path)
    if not pages:
        return None

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
    chunks = semantic_chunker.chunk_documents(document_chunks)
    return chunks or None


# ---------------------------------------------------------------------------
# Ingestion: parse -> chunk -> embed -> upsert into Qdrant
# (was scripts/ingest.py, scoped to a single file)
# ---------------------------------------------------------------------------
def ingest_single_pdf(file_path: str, embedding_service: EmbeddingService,
                      qdrant_service: QdrantService) -> int:
    parser = PDFParser(pdf_folder=settings.PDF_FOLDER)
    chunks = parse_and_chunk(file_path, embedding_service, parser=parser)
    if not chunks:
        return 0

    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = embedding_service.embed_documents(chunk_texts)

    qdrant_service.ensure_collection(vector_size=len(chunk_embeddings[0]))
    qdrant_service.upsert_chunks(chunks, chunk_embeddings)

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

    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = embedding_service.embed_documents(chunk_texts)

    qdrant_service.ensure_collection(vector_size=len(chunk_embeddings[0]))
    qdrant_service.upsert_chunks(chunks, chunk_embeddings)

    return len(chunks)


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
            if f.lower().endswith((".pdf", ".docx"))
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

if not settings.OPENROUTER_API_KEY:
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

os.makedirs(settings.PDF_FOLDER, exist_ok=True)


def _existing_documents():
    return sorted(
        f for f in os.listdir(settings.PDF_FOLDER) if f.lower().endswith((".pdf", ".docx"))
    )


# ---------------------------------------------------------------------------
# Sidebar — upload & document picker
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📤 Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF or Word document", type=["pdf", "docx"])

    if uploaded_file is not None:
        if st.button("Process & Summarize", type="primary", use_container_width=True):
            dest_path = os.path.join(settings.PDF_FOLDER, uploaded_file.name)
            already_ingested = os.path.isfile(dest_path)
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()

            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            embedding_service = get_embedding_service()
            qdrant_service = get_qdrant_service()
            summary_llm = get_summary_llm()

            try:
                if not already_ingested:
                    with st.spinner(f"Reading and indexing '{uploaded_file.name}'..."):
                        if file_ext == ".docx":
                            n_chunks = ingest_single_docx(dest_path, embedding_service, qdrant_service)
                        else:
                            n_chunks = ingest_single_pdf(dest_path, embedding_service, qdrant_service)
                    if n_chunks == 0:
                        st.error("No extractable content found in this document.")
                        st.stop()
                    st.toast(f"Indexed {n_chunks} chunks.", icon="✅")

                with st.spinner("Generating summary..."):
                    summary = summarize_indexed_document(
                        uploaded_file.name,
                        qdrant_service,
                        summary_llm,
                    )

                with st.spinner("Coming up with follow-up questions..."):
                    questions = generate_suggested_questions(summary_llm, summary, uploaded_file.name)

                st.session_state.summary_text = summary
                st.session_state.summary_filename = uploaded_file.name
                st.session_state.suggested_questions = questions
                st.session_state.summary_cache[uploaded_file.name] = {
                    "summary": summary,
                    "questions": questions,
                }
            except Exception as exc:
                st.error(f"Something went wrong while processing the document: {exc}")

    existing = _existing_documents()
    if existing:
        st.divider()
        st.header("📚 Previously uploaded")
        pick = st.selectbox("Re-summarize an existing document", ["—"] + existing)
        if pick != "—" and st.button("Summarize selected", use_container_width=True):
            # Use cached summary if available — skip Qdrant + LLM entirely
            if pick in st.session_state.summary_cache:
                cached = st.session_state.summary_cache[pick]
                summary = cached["summary"]
                questions = cached["questions"]
            else:
                qdrant_service = get_qdrant_service()
                summary_llm = get_summary_llm()
                try:
                    with st.spinner("Generating summary..."):
                        summary = summarize_indexed_document(
                            pick,
                            qdrant_service,
                            summary_llm,
                        )
                    with st.spinner("Coming up with follow-up questions..."):
                        questions = generate_suggested_questions(summary_llm, summary, pick)
                    st.session_state.summary_cache[pick] = {
                        "summary": summary,
                        "questions": questions,
                    }
                except Exception as exc:
                    st.error(f"Something went wrong while summarizing: {exc}")
                    summary = None
                    questions = []
            if summary:
                st.session_state.summary_text = summary
                st.session_state.summary_filename = pick
                st.session_state.suggested_questions = questions

        st.divider()
        st.header("📋 Generate Training Script")
        st.caption(
            "Generates a cinematic, story-driven training video script from all "
            "uploaded documents. The format is fixed; the story is freshly "
            "invented every time you click — click again for a different one."
        )

        if st.button("Generate Training Script", use_container_width=True):
            qdrant_service = get_qdrant_service()
            presentation_llm = get_presentation_llm()
            try:
                with st.spinner("Gathering content from all documents..."):
                    all_chunks = []
                    for pdf in existing:
                        chunks = _get_all_chunks(pdf, qdrant_service)
                        if chunks:
                            all_chunks.extend(chunks)
                if not all_chunks:
                    st.error("No indexed content found. Please upload and process at least one document first.")
                else:
                    with st.spinner(f"Generating training script from {len(existing)} document(s)... this may take a moment."):
                        presentation_text = presentation_llm.generate_presentation(all_chunks)
                    st.session_state.presentation_text = presentation_text
                    st.session_state.presentation_filename = "All Documents"
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
        st.rerun()


# ---------------------------------------------------------------------------
# Summary card + suggested questions
# ---------------------------------------------------------------------------
if st.session_state.summary_text:
    with st.container(border=True):
        st.subheader(f"🧾 Summary — {st.session_state.summary_filename}")
        st.write(st.session_state.summary_text)

        if st.session_state.suggested_questions:
            st.write("**Suggested questions:**")
            cols = st.columns(len(st.session_state.suggested_questions))
            for col, q in zip(cols, st.session_state.suggested_questions):
                if col.button(q, use_container_width=True, key=f"sugg_{q}"):
                    st.session_state.pending_question = q

if st.session_state.presentation_text:
    with st.container(border=True):
        st.subheader(f"📋 Training Script — {st.session_state.presentation_filename}")
        st.markdown(st.session_state.presentation_text)
        st.download_button(
            label="⬇️ Download Training Script (.txt)",
            data=st.session_state.presentation_text,
            file_name=f"training_script_{st.session_state.presentation_filename}.txt",
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

typed_question = st.chat_input("Ask a question about your documents...")
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
                            available = ", ".join(existing) or "none"
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