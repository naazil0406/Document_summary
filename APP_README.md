# bot — Document Q&A / Training Script Generator (RAG Application)

A Retrieval-Augmented Generation app — CLI **and** Streamlit UI — that
answers questions, produces executive summaries, and generates
story-driven corporate training video scripts strictly from your own
documents (PDF, Word, Excel, CSV — including scanned/image-heavy PDFs via
mandatory Docker OCR).

**Tested on:** Python 3.12.10

```
STORAGE:        Upload -> saved locally (data/pdfs/) AND mirrored to S3
                (optional) -> fetched back down from S3 on session start
                / manual sync, so S3 is the durable source of truth

INGESTION:      PDF -> PyMuPDF + Docker Tesseract OCR (every page)
                DOCX -> python-docx
                XLSX/XLSM/XLS/CSV -> structured 25-row table chunks
                -> TOC-aware Document Chunking -> Semantic Chunking
                -> BGE-M3 Embeddings -> Qdrant (Docker)

QUERY (Q&A):    Question -> BGE-M3 Query Embedding -> Qdrant Search
                -> TOC-section filter -> Document-name filter
                -> Score filter -> LLM (OpenRouter or AWS Bedrock) -> Answer

SUMMARIZATION:  Document name (fuzzy) -> re-parse + re-chunk fresh
                -> LLM (OpenRouter or AWS Bedrock) -> Executive Summary

TRAINING SCRIPT: All ingested documents' chunks -> batched if large
                -> LLM invents a fresh illustrative story around a random
                seed character/workplace, teaches the real source-document
                concepts -> saved to Narrative_scripts/*.txt + downloadable
```

---

## What's in this version

- **Multi-format ingestion**: PDF, Word (`.docx`), Excel (`.xlsx`/`.xlsm`),
  and CSV — all flow through the same chunk/embed/store pipeline.
- **Two LLM providers, switchable in `.env`**: OpenRouter (any model on
  their catalog) or AWS Bedrock (native `boto3` Converse API — no
  OpenRouter account needed at all). Same prompts, same behavior, just a
  different transport underneath.
- **Optional AWS S3 document storage**: uploads are mirrored to S3 as a
  durable backup and fetched back down automatically — entirely optional,
  the app runs local-folder-only if `S3_BUCKET_NAME` is unset.
- **Externalized prompts**: every prompt (Q&A, summary, presentation
  script, per-batch extraction, script-editing) lives as a plain `.txt`
  file in `prompts/`, split into a `_system_prompt.txt` (fixed persona/
  rules) and `_user_prompt.txt` (the dynamic input) — edit wording without
  touching code.
- **Training video script generation**: turns your documents into a
  cinematic, story-driven narration script for internal training videos,
  with a fresh randomly-seeded character/workplace every run so it never
  reuses the same illustrative story twice. Saved automatically to
  `Narrative_scripts/`.

---

## Project Structure

```
bot/
├── docker-compose.yml        # Qdrant (persistent) + Tesseract OCR image (pre-pull only)
├── data/pdfs/                 # Local working cache of your documents (PDF/Word/Excel/CSV)
├── Narrative_scripts/          # Generated training scripts saved here (.gitignored)
├── prompts/                   # Every prompt as a plain .txt file (system + user pairs)
│   ├── qa_system_prompt.txt / qa_user_prompt.txt
│   ├── summary_system_prompt.txt / summary_user_prompt.txt
│   ├── presentation_system_prompt.txt / presentation_user_prompt.txt
│   ├── batch_extraction_system_prompt.txt / batch_extraction_user_prompt.txt
│   └── edit_presentation_system_prompt.txt / edit_presentation_user_prompt.txt
├── services/
│   ├── pdf_parser.py          # PyMuPDF + mandatory Docker Tesseract OCR
│   ├── docx_parser.py         # Word document parsing
│   ├── excel_parser.py        # Multi-sheet/table Excel + CSV structured parsing
│   ├── chunking.py            # Stage 1: TOC-aware DocumentChunker
│   │                          # Stage 2: SemanticChunkingService
│   ├── embeddings.py          # BAAI/bge-m3 embeddings (documents + queries)
│   ├── qdrant_db.py           # Qdrant collection / upsert / search
│   ├── retriever.py           # Query -> embedding -> Qdrant search
│   │                          # -> TOC section filter -> filename filter -> score filter
│   ├── s3_storage.py          # Optional: mirror uploads to S3, fetch/sync back down
│   ├── llm_service.py         # BaseLLMService (shared prompt logic) +
│   │                          # OpenRouterLLMService + BedrockLLMService
│   └── document_resolver.py   # Fuzzy filename matching helper
├── scripts/
│   ├── ingest.py               # CLI: run the ingestion pipeline (PDF only)
│   ├── query.py                # CLI: interactive Q&A
│   └── summarize.py            # CLI: document -> executive summary (no Qdrant needed)
├── app.py                     # Streamlit UI — the real entry point (upload, summarize,
│                               # chat, generate training scripts) — all services/* based
├── config/settings.py         # Environment-driven configuration + logging
├── requirements.txt
└── .env
```

> **Note:** `scripts/ingest.py` is a legacy PDF-only CLI utility, kept for
> reference. `app.py` (Streamlit) is the actual application and supports
> all four document types plus every feature described here.

---

## How To Run This Project (Step by Step)

### Step 1 — Check your Python version
```bash
python --version
```
Built and tested against **Python 3.12.10**. Any 3.12.x should work.

### Step 2 — Install Docker Desktop
Docker is **compulsory** — both Qdrant (vector database) and Tesseract
OCR run exclusively through Docker.
- **Windows / macOS:** https://www.docker.com/products/docker-desktop/
- **Linux:** https://docs.docker.com/engine/install/

Verify it's running: `docker ps`

### Step 3 — Install Python dependencies
```bash
cd bot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Start Qdrant and pre-pull the OCR image
```bash
docker compose up -d qdrant
docker compose pull tesseract     # one-time, ≈150MB
```
Tesseract is **not** a long-running service — `pdf_parser.py` launches a
fresh container per page on demand.

Verify Qdrant: `curl http://localhost:6333/healthz` or open
http://localhost:6333/dashboard.

```bash
docker compose stop qdrant     # stop later (data preserved in qdrant_storage/)
docker compose start qdrant    # start again
```

### Step 5 — Fill in `.env`

At minimum, pick **one** LLM provider and fill in its credentials:

**Option A — OpenRouter (simplest, no AWS account needed):**
```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...          # https://openrouter.ai/keys
```

**Option B — AWS Bedrock (runs models directly on AWS, no OpenRouter):**
```dotenv
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```
See **"AWS Bedrock setup"** below — model access has to be explicitly
enabled in the Bedrock console before this will work.

Also set:
```dotenv
QDRANT_URL=http://localhost:6333
HF_CACHE_DIR=/path/to/cache             # where the embedding model is cached
```

See **"Full `.env` reference"** below for every setting.

### Step 6 — Add your documents
Copy PDF, Word, Excel, or CSV files into `bot/data/pdfs/` (the folder name
is legacy — it now holds all supported formats), or just upload them
through the Streamlit sidebar once it's running.

### Step 7 — Run it

**Streamlit app (recommended — the real entry point):**
```bash
streamlit run app.py
```
Opens at http://localhost:8501. Upload a document → it's ingested,
summarized, and follow-up questions are suggested. Chat with it, generate
an executive summary, or generate a full training video script — all from
the sidebar.

**CLI (PDF-only, legacy):**
```bash
python -m scripts.ingest
python -m scripts.query
python -m scripts.summarize unit2.pdf
```

---
docker compose up -d qdrant
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
## AWS Bedrock setup (only if `LLM_PROVIDER=bedrock`)

Two things live in the **AWS console**, not `.env`:

1. **Model access** — Bedrock console → *Model access* → request/enable
   access to `amazon.nova-micro-v1:0` and `amazon.nova-2-lite-v1:0` in
   your chosen region. This is a separate opt-in from IAM permissions —
   valid credentials with `bedrock:InvokeModel` will still fail with
   `AccessDeniedException` if the model itself isn't enabled.
2. **IAM permissions** on the user/role in `.env`: `bedrock:InvokeModel`
   (or `bedrock:Converse`) for the two models above.

Nova 2 Lite needs a **region-prefixed inference profile ID** for
on-demand access, not the bare model ID:
```dotenv
BEDROCK_PRESENTATION_MODEL=us.amazon.nova-2-lite-v1:0      # US regions
BEDROCK_PRESENTATION_MODEL=global.amazon.nova-2-lite-v1:0  # outside the US
```

## AWS S3 setup (optional — only if `S3_BUCKET_NAME` is set)

- The bucket must already exist in your AWS account — the app doesn't
  create it.
- IAM permissions needed: `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`.
- If left blank, the app runs entirely local-folder-only — no boto3 calls
  are made and no AWS credentials are required for storage.

Bedrock and S3 share the same `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
/ `AWS_REGION` — one set of credentials covers both.

---

## Full `.env` reference

```dotenv
# ----- Document source folder (local working cache) -----
PDF_FOLDER=data/pdfs

# ----- Semantic chunking -----
SEMANTIC_BUFFER_SIZE=1
SEMANTIC_BREAKPOINT_TYPE=percentile
SEMANTIC_BREAKPOINT_AMOUNT=95
MAX_CHUNK_SIZE=600
CHUNK_OVERLAP=60

# ----- Document chunking (headings / sections / paragraphs) -----
DOC_CHUNK_HEADING_MAX_LENGTH=80
DOC_CHUNK_MIN_PARAGRAPH_LENGTH=20

# ----- Embeddings -----
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cpu                    # "cuda" if you have a GPU
HF_CACHE_DIR=/path/to/cache

# ----- Qdrant (Docker) -----
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                         # empty — default container has no auth
QDRANT_COLLECTION_NAME=company_docs

# ----- Retrieval -----
TOP_K=8                                 # chunks retrieved per Q&A query
TOP_K_SUMMARY=15                        # chunks retrieved per summary
MIN_RELEVANCE_SCORE=0.40

# ----- LLM provider switch: "openrouter" or "bedrock" -----
LLM_PROVIDER=openrouter

# ----- AWS (shared by S3 storage AND Bedrock inference) -----
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1

# ----- S3 document storage (optional — blank disables it entirely) -----
S3_BUCKET_NAME=
S3_PREFIX=documents/

# ----- Bedrock models (used only when LLM_PROVIDER=bedrock) -----
BEDROCK_REGION=us-east-1
BEDROCK_MODEL=amazon.nova-micro-v1:0             # Q&A + summary
BEDROCK_PRESENTATION_MODEL=us.amazon.nova-2-lite-v1:0   # training scripts

# ----- OpenRouter (used only when LLM_PROVIDER=openrouter) -----
OPENROUTER_API_KEY=
OPENROUTER_MODEL=qwen/qwen-2.5-72b-instruct
OPENROUTER_SITE_URL=
OPENROUTER_SITE_NAME=bot
PRESENTATION_MODEL=qwen/qwen-2.5-72b-instruct

# ----- Token / temperature settings (used by whichever provider is active) -----
OPENROUTER_MAX_TOKENS=512               # Q&A answer length cap
OPENROUTER_TEMPERATURE=0.1
SUMMARY_MAX_TOKENS=800
PRESENTATION_MAX_TOKENS=4096
PRESENTATION_TEMPERATURE=0.9            # higher — the story should vary run to run

# ----- Where generated training scripts are saved locally -----
NARRATIVE_SCRIPTS_DIR=Narrative_scripts

# ----- Logging -----
LOG_LEVEL=INFO                          # DEBUG for verbose chunk-level logs
```

**Never commit `.env` to version control** — it's already in
`.gitignore`. If a real API key or AWS secret ever ends up committed or
shared, rotate it immediately.

---

## Architecture Notes

- **TOC-aware chunking**: PDFs with a genuine Table of Contents are
  grouped by section at ingest time (`toc_section` stored in the Qdrant
  payload). Excel/CSV sheets use their sheet name as `toc_section` the
  same way, so "what's on the Pricing sheet" narrows correctly. Documents
  without a detected structure fall back to per-page/per-sheet splitting.
- **Two-layer retrieval filtering**: TOC-section filter → document-name
  filter (regex trigger words, or fuzzy match against real filenames) →
  score filter. Each layer falls through to the broader result set if
  nothing matches, so an honest answer is still attempted.
- **Summarization bypasses Qdrant by design**: re-reads and re-chunks the
  target document on demand, so the summary always covers the full thing.
- **LLM provider abstraction**: `services/llm_service.py`'s
  `BaseLLMService` owns every prompt-building/batching/post-processing
  method; `OpenRouterLLMService` and `BedrockLLMService` each implement
  only `_call_llm()` — the actual network call. Switching providers is a
  one-line `.env` change, never a code change.
- **Storage abstraction**: S3 is the source of truth for "what documents
  exist" when configured; `data/pdfs/` is a local working cache of it.
  Upload → saved locally + mirrored to S3. Fetch → anything in S3 but
  missing locally is downloaded automatically once per session, or on
  demand via the sidebar's "Sync from S3" button.
- **Training script generation**: the script's fixed shape (header line,
  `TRT:` runtime, scene directions, `Narrator (Trainer):` blocks) comes
  from `prompts/presentation_*_prompt.txt`; the illustrative story is
  freshly invented every run using a random seed character name +
  workplace, so back-to-back generations never converge on the same
  example even at low temperature. Documents too large for one call are
  first compressed batch-by-batch via `batch_extraction_*_prompt.txt`,
  then assembled into the final script.
- **OCR and Qdrant are both Docker-only** for PDFs — no host Tesseract
  binary, no host Qdrant binary. Word/Excel/CSV parsing needs no Docker at
  all (`python-docx`/`openpyxl`/stdlib `csv` run natively).

---

## Troubleshooting

- **`docker: command not found`**: Docker Desktop isn't installed or not
  on PATH.
- **Qdrant container not starting**: `docker compose logs qdrant`. Most
  common cause is port 6333 already in use.
- **`Connection refused` on `http://localhost:6333`**: container isn't
  running — `docker compose start qdrant`.
- **`401 Unauthorized` from OpenRouter**: key missing/revoked — regenerate
  at https://openrouter.ai/keys.
- **`402 Payment Required` from OpenRouter**: no credits — add them at
  https://openrouter.ai/settings/credits.
- **`AccessDeniedException` from Bedrock**: almost always means the model
  hasn't been enabled in *Model access* in the Bedrock console for your
  region — this is separate from IAM permissions.
- **Bedrock error asking for an "inference profile"**: you're using a
  bare model ID that requires a region-prefixed one (Nova 2 Lite does) —
  use `us.amazon.nova-2-lite-v1:0` / `global.amazon.nova-2-lite-v1:0`.
- **S3 sync fails on startup**: check `S3_BUCKET_NAME` is a real,
  existing bucket (not a placeholder value) and that your AWS credentials
  have `s3:ListBucket`/`s3:GetObject`/`s3:PutObject` on it.
- **OCR errors / image PDF pages returning empty text**: the Tesseract
  image isn't pulled yet — `docker compose pull tesseract`.
- **TOC not detected on a document that has one**: check
  `LOG_LEVEL=DEBUG` output. Roman-numeral TOC page numbers aren't
  supported — per-page chunking is the automatic fallback.
- **A Q&A question naming a document isn't being narrowed to it**: the
  fuzzy filename match needs either 2+ overlapping significant words with
  the filename, or a single distinctive stem word. Very generic filenames
  (`report.pdf`, `doc1.pdf`) won't match well from a plain question.
- **Excel/CSV sheet not showing up**: fully-empty sheets are skipped
  automatically; check the sheet actually has data in `openpyxl`/Excel
  itself.
