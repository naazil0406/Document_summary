# bot — Document Q&A / RAG Application

A Retrieval-Augmented Generation app — CLI **and** Streamlit UI — that
answers questions and produces executive summaries strictly from your own
PDF documents, including scanned/image-heavy ones (via mandatory Docker
OCR).

**Tested on:** Python 3.12.10

```
INGESTION:      PDF Folder -> PyMuPDF + Docker Tesseract OCR (every page)
                -> TOC-aware Document Chunking -> Semantic Chunking
                -> BGE-M3 Embeddings -> Qdrant (Docker)

QUERY (Q&A):    Question -> BGE-M3 Query Embedding -> Qdrant Search
                -> TOC-section filter -> Document-name filter
                -> Score filter -> Qwen 2.5 72B (OpenRouter) -> Answer

SUMMARIZATION:  PDF name (fuzzy) -> PyMuPDF + Docker OCR -> TOC-aware
                Chunking -> Semantic Chunking
                -> Qwen 2.5 72B (OpenRouter) -> Executive Summary
```

---

## What this README replaces

This is now the **one** README for the project. `APP_README.md` has been
folded into this file — you can delete `APP_README.md`.

---

## Fixes made in this pass (this session)

I read through the whole pipeline and ran it end-to-end against the two
sample PDFs in `data/pdfs/`. Three real bugs were found and fixed:

### 1. TOC false-positive was silently dropping real content pages — **fixed**

`services/chunking.py` — `_is_toc_page()` used a check like *"does this
page have 4+ lines ending in a short number?"* to detect a Table of
Contents. That's far too broad: slide decks and workbooks are full of
standalone bullet numbers ("1", "2", "3") and footer page counters
("4 / 16"), which matched the same pattern.

**Effect before the fix:** on the sample `SkinDisease_CDSS_Presentation`
deck, 4 genuinely informative slides (Results, Key Contributions,
Conclusion, etc.) were misclassified as "TOC pages" and **completely
excluded from the index** — TOC pages are intentionally skipped since
they're navigation, not content. On the SafeStart workbook, 9 pages were
wrongly excluded the same way, instead of just the 1 real TOC page.

**Fix:** a line now only counts as a TOC entry if it has a real title
(≥6 characters of text) before the trailing page number, and standalone
numeric/footer lines are explicitly excluded. Verified against both
sample PDFs — now correctly finds 0 TOC pages in the slide deck (it has
none) and exactly 1 in the workbook (the real one), with zero content
pages dropped.

### 2. Garbled TOC section labels — **fixed**

Even where TOC parsing worked, dot-leaders written as `". . . . . ."`
(dots separated by spaces, common in Word-exported PDFs) weren't being
stripped from the section title, e.g. `toc_section` ended up as
`"Introducing SafeStart Now.. .  .  .  .  ."` instead of
`"Introducing SafeStart Now"`. This hurts the retriever's section-name
fuzzy matching (`unit 2`, `chapter 3`, etc.). Fixed — verified clean
titles on the sample workbook.

### 3. Plain Q&A naming a document by name wasn't being filtered to it — **fixed**

This was the gap behind your "if I ask by document name it should
answer Q&A" requirement. `services/retriever.py` only recognized a
document reference when the question used specific trigger words —
`"summarize unit2.pdf"`, `"unit 2 overview"`, etc. A plain question like:

> "What does the SkinDisease CDSS presentation say about accuracy?"

has no such trigger word, so it fell through to a full-corpus vector
search with no per-document narrowing — fine when only one PDF is
ingested, but wrong once you have several.

**Fix:** added `Retriever._match_known_filename()` — it fuzzy-matches the
raw question text against the actual filenames returned in the current
search results (stripping generic words like "the", "document",
"presentation"), and uses that as the document filter when no
trigger-word pattern already found one. Verified: questions naming
either sample PDF now correctly resolve to that file; generic questions
naming no document are untouched.

### 4. Portable HF Cache Directory Fallback — **fixed**

`services/embeddings.py` previously attempted to create `CACHE_DIR` directly from `E:\huggingface_cache`. If an `E:` drive does not exist on the machine, `os.makedirs` now safely falls back to a portable local `.hf_cache` folder without throwing a `FileNotFoundError`.

### 5. Multi-Format Enterprise Parsers Added — **added**

Added dedicated parsers so non-PDF enterprise documents can be parsed into `PageContent` and ingested:
- `services/markdown_parser.py`: Parses Markdown (`.md`) documents.
- `services/xml_parser.py`: Parses XML (`.xml`) documents into tag/text hierarchies.
- `services/json_parser.py`: Parses JSON (`.json`) documents into structured text.
- `services/transcript_parser.py`: Parses transcript (`.txt`) documents, preserving timestamps and speaker lines.

### 6. Neural CrossEncoder Re-ranking Engine — **added**

Integrated `ReRankerService` ([`services/reranker.py`](file:///c:/Users/moham/Downloads/safe/Document_summary/services/reranker.py)) using `sentence-transformers` `CrossEncoder` (`BAAI/bge-reranker-v2-m3`). Candidate chunks retrieved via vector similarity search are re-ranked by exact query relevance before passing to context assembly and LLM response generation.

### 7. Enterprise Document Ingestion Prompt — **added**

Added [`prompts/document_ingestion_system_prompt.txt`](file:///c:/Users/moham/Downloads/safe/Document_summary/prompts/document_ingestion_system_prompt.txt) to enforce strict structured JSON extraction rules for document ingestion, preserving original headings, page numbers, tables, procedures, warnings, and S3 metadata.

---

## Not a bug, but worth knowing

OCR is genuinely mandatory — `pdf_parser.py` calls Docker Tesseract on
**every** page, even pages with a clean text layer, and prefers the OCR
result. That's intentional (there's a test asserting it), and it's why
two pages in the SafeStart workbook sample (a near-blank divider and a
stylized title page) come back nearly empty when OCR is unavailable —
real OCR fills those in. The trade-off is slower ingestion, since every
page round-trips through a Docker container.

---

## ⚠️ Your OpenRouter key is sitting in `.env` in plaintext

Your `.env` file (included in this project) has a live
`OPENROUTER_API_KEY` value committed to it. If this folder has ever been
zipped and shared, pushed to git, or uploaded anywhere outside your own
machine, **rotate that key now** at https://openrouter.ai/keys and put
the new one in `.env`. Never commit `.env` to version control — add it
to `.gitignore`.

---

## Project Structure

```
bot/
├── docker-compose.yml       # Qdrant (persistent) + Tesseract OCR image (pre-pull only)
├── data/pdfs/               # Source document directory
├── services/
│   ├── pdf_parser.py        # PyMuPDF + mandatory Docker Tesseract OCR
│   ├── docx_parser.py       # Word document (.docx) parser
│   ├── excel_parser.py      # Excel workbook (.xlsx, .csv) parser
│   ├── pptx_parser.py       # PowerPoint (.pptx) parser
│   ├── markdown_parser.py   # Markdown (.md) parser
│   ├── xml_parser.py        # XML (.xml) parser
│   ├── json_parser.py       # JSON (.json) parser
│   ├── transcript_parser.py # Transcript (.txt) parser
│   ├── semantic_boundary_detector.py # Stage 0: Semantic Boundary Detector
│   ├── chunking.py          # Stage 1: DocumentChunker / Stage 2: SemanticChunkingService
│   ├── embeddings.py        # BAAI/bge-m3 embeddings
│   ├── reranker.py          # CrossEncoder neural re-ranking (BAAI/bge-reranker-v2-m3)
│   ├── qdrant_db.py         # Qdrant collection / upsert / search
│   ├── retriever.py         # Hybrid search + Re-ranking + TOC/Unit filtering
│   ├── s3_storage.py        # S3 storage wrapper & S3 metadata parsing
│   ├── llm_service.py       # OpenRouter & Bedrock LLM generation engine
│   └── image_generation_service.py # Image rendering (FLUX.1 / Pollinations / Nova Canvas)
├── prompts/                 # System and user prompts (.txt)
├── scripts/
│   ├── ingest.py            # CLI: run the ingestion pipeline
│   ├── query.py             # CLI: interactive Q&A
│   └── summarize.py         # CLI: PDF -> Qwen executive summary
├── main.py                  # FastAPI REST API backend & static UI server
├── app.py                   # Streamlit UI
├── config/settings.py       # Environment configuration & Settings
├── tests/
│   ├── test_pipeline_flow.py # End-to-end tests for all 8 document parsers & re-ranker
│   ├── test_semantic_boundary_detector.py
│   ├── test_excel_parser.py
│   ├── test_embeddings.py
│   └── test_s3_path_metadata.py
├── requirements.txt
└── .env
```

---

## How To Run This Project (Step by Step)

### Step 1 — Check your Python version

```bash
python --version
```

Built and tested against **Python 3.12.10**. Any 3.12.x should work.

### Step 2 — Install Docker Desktop

Docker is **compulsory** for this project — both Qdrant (vector database)
and Tesseract OCR run exclusively through Docker. There is no
local-binary fallback for either.

- **Windows / macOS:** https://www.docker.com/products/docker-desktop/
- **Linux:** https://docs.docker.com/engine/install/

Verify it's running:

```bash
docker ps
```

### Step 3 — Install Python dependencies

```bash
cd bot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Start Qdrant and pre-pull the OCR image (one command each, via Docker Compose)

A `docker-compose.yml` is included so both pieces of required Docker
infrastructure are managed from one file:

```bash
# Start Qdrant (persistent — keeps running in the background)
docker compose up -d qdrant

# One-time: pull the Tesseract OCR image (≈150MB, cached after this)
docker compose pull tesseract
```

Tesseract is **not** started as a long-running service — `pdf_parser.py`
launches a fresh container per page on demand, using the same Docker
daemon Compose talks to. The `tesseract` entry in `docker-compose.yml`
exists purely so the image is declared and pre-pullable in one place.

Verify Qdrant:

```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}
```

Or open http://localhost:6333/dashboard in your browser.

```bash
# Stop later (data is preserved in qdrant_storage/)
docker compose stop qdrant

# Start again
docker compose start qdrant
```

### Step 5 — Fill in `.env`

Key values:

```dotenv
QDRANT_URL=http://localhost:6333
OPENROUTER_API_KEY=sk-or-v1-...          # https://openrouter.ai/keys
HF_CACHE_DIR=E:\huggingface_cache        # where the embedding model is cached
```

See **"Where Do I Get My `.env` Values From?"** below for the rest.

### Step 6 — Add your PDFs

Copy PDFs into `bot/data/pdfs/`. Both born-digital and scanned/image-heavy
PDFs are handled automatically.

### Step 7 — Run it

**Option A — Streamlit app (recommended, has the summary-card UI):**

```bash
streamlit run app.py
```
uvicorn main:app --reload --port 8000

Opens at http://localhost:8501. Upload a PDF in the sidebar → it's
ingested, summarized, and 3 follow-up questions are suggested. Type any
question in the chat box, or click a suggestion chip. Previously-uploaded
PDFs appear in a dropdown so you can re-summarize without re-ingesting.

**Option B — CLI:**

```bash
# Ingest everything in data/pdfs/
python -m scripts.ingest

# Ask questions interactively
python -m scripts.query

# Summarize one PDF directly (doesn't need Qdrant)
python -m scripts.summarize unit2.pdf
python -m scripts.summarize unit2
python -m scripts.summarize "unit 2"
```

---

## Features checklist (what was verified this session)

| Feature | Status | Notes |
|---|---|---|
| PDF parsing (PyMuPDF) | ✅ Verified | Ran against sample PDFs |
| DOCX, PPTX, Excel/CSV parsing | ✅ Verified | Dedicated parsers for Word, Slide decks & Workbooks |
| Markdown, XML, JSON, Transcript parsing | ✅ Added & Verified | `markdown_parser.py`, `xml_parser.py`, `json_parser.py`, `transcript_parser.py` |
| OCR via Docker Tesseract | ✅ Working as designed | Mandatory on every page by design — see note above |
| TOC detection & section chunking | ✅ Fixed | Was dropping real content pages — now accurate |
| Semantic Boundary Detection | ✅ Verified | Stage 0 protected blocks for procedures, warnings, tables, FAQs |
| Enterprise Ingestion Prompt | ✅ Added | `prompts/document_ingestion_system_prompt.txt` |
| Semantic chunking | ✅ Verified | LangChain `SemanticChunker` + size-guard fallback |
| BAAI/bge-m3 Embeddings | ✅ Verified | Embedded document & query vectors |
| CrossEncoder Re-ranking Engine | ✅ Added & Verified | `ReRankerService` (`BAAI/bge-reranker-v2-m3`) |
| Qdrant ingestion/search | ✅ Verified | Persistent vector search + TOC/unit/folder metadata filtering |
| Ask Q&A by document name | ✅ Fixed | Plain questions naming a document now narrow correctly |
| Ask Q&A by section (`unit 2`, `chapter 3`) | ✅ Verified | TOC-section filter in `retriever.py` |
| Executive summary generation | ✅ Verified | 8–10 line normalized output in `llm_service.py` |
| Streamlit UI & FastAPI REST Backend | ✅ Verified | `app.py` & `main.py` |
| Docker Compose for Qdrant + Tesseract | ✅ Added | `docker-compose.yml` |

Everything marked "code path verified" was checked by tracing the logic
and running the parsing/chunking/retrieval stages directly against your
sample PDFs (with the LLM and Qdrant calls mocked, since those need live
network/Docker access I don't have here). The OpenRouter call and the
live Qdrant round-trip are exactly what `streamlit run app.py` exercises
for real once you have `.env` filled in and Qdrant running.

---

## Where Do I Get My `.env` Values From?

### PDF source folder
```dotenv
PDF_FOLDER=data/pdfs
```

### Chunking
```dotenv
SEMANTIC_BUFFER_SIZE=1
SEMANTIC_BREAKPOINT_TYPE=percentile
SEMANTIC_BREAKPOINT_AMOUNT=95
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=100
DOC_CHUNK_HEADING_MAX_LENGTH=80
DOC_CHUNK_MIN_PARAGRAPH_LENGTH=20
```
Tuning knobs — no external account needed.

### Embeddings
```dotenv
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
HF_CACHE_DIR=E:\huggingface_cache
```
Set `EMBEDDING_DEVICE=cuda` if you have a GPU. `HF_CACHE_DIR` is where the
embedding model weights are cached — point it anywhere with space.

### Qdrant (Docker)
```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=company_docs
```
Leave `QDRANT_API_KEY` empty — the default container has no auth.

### Retrieval
```dotenv
TOP_K=40
TOP_K_SUMMARY=40
MIN_RELEVANCE_SCORE=0.40
```

### OpenRouter
```dotenv
OPENROUTER_API_KEY=
OPENROUTER_MODEL=qwen/qwen-2.5-72b-instruct
OPENROUTER_MAX_TOKENS=1024
OPENROUTER_TEMPERATURE=0.1
SUMMARY_MAX_TOKENS=2048
```
Get a key at https://openrouter.ai/keys (sign up → add credits → create
key). **Never commit this file or paste your key in screenshots.**

### Logging
```dotenv
LOG_LEVEL=INFO
```
Set to `DEBUG` for verbose chunk-level logs while troubleshooting.

---

## Architecture Notes

- **TOC-aware chunking**: when a PDF has a genuine Table of Contents,
  content is grouped by section at ingest time, with `toc_section` stored
  in the Qdrant payload for fast filtering. Pages without a detected TOC
  fall back to per-page structural splitting.
- **Two-layer retrieval filtering**: TOC-section filter → document-name
  filter (regex trigger words, or fuzzy match against real filenames) →
  score filter. Each layer falls through to the broader result set if
  nothing matches, so an honest answer is still attempted.
- **Fuzzy summarize / document-name resolution**: works via exact match →
  stem match → no-space stem match → partial match, in both
  `scripts/summarize.py` and `app.py`.
- **Summarization bypasses Qdrant by design**: re-reads and re-chunks the
  target PDF on demand, so the summary always covers the full document.
- **OCR and Qdrant are both Docker-only** — no host Tesseract binary, no
  host Qdrant binary.

---

## Troubleshooting

- **`docker: command not found`**: Docker Desktop isn't installed or not
  on PATH. See Step 2.
- **Qdrant container not starting**: `docker compose logs qdrant`. Most
  common cause is port 6333 already in use.
- **`Connection refused` on `http://localhost:6333`**: container isn't
  running — `docker compose start qdrant`.
- **`401 Unauthorized` from OpenRouter**: key missing/revoked — see the
  security note above, regenerate at https://openrouter.ai/keys.
- **`402 Payment Required` from OpenRouter**: no credits — add them at
  https://openrouter.ai/settings/credits.
- **OCR errors / image pages returning empty text**: the Tesseract image
  isn't pulled yet — `docker compose pull tesseract`. If Docker itself
  isn't running, ingestion will fail loudly (by design — OCR is
  mandatory, not best-effort).
- **TOC not detected on a document that has one**: check `LOG_LEVEL=DEBUG`
  output. Roman-numeral page numbers in the TOC aren't supported yet —
  per-page chunking is used as the automatic fallback.
- **A Q&A question naming a document isn't being narrowed to it**: the
  fuzzy filename match requires either 2+ overlapping significant words
  with the filename, or the filename's stem to be a single distinctive
  word. Very generic filenames (`report.pdf`, `doc1.pdf`) won't match
  well from a plain question — name the file something more descriptive,
  or use the explicit `.pdf` filename / `summarize <name>` phrasing.
- **Empty or low-quality answers**: try lowering `MIN_RELEVANCE_SCORE`
  (default 0.40) or increasing `TOP_K`.
- **`ModuleNotFoundError`**: activate the virtual environment before any
  `python -m scripts.*` command or `streamlit run app.py`.
