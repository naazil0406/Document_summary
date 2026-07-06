"""
LLM service — single file covering everything model-related.

Contains:
  1. Shared logic used by every provider: prompt loading from prompts/*.txt,
     chunk-context formatting, story-seed randomization, summary cleanup,
     and chunk batching for long presentation generation (BaseLLMService).
  2. OpenRouterLLMService — transport over OpenRouter's chat-completions API.
  3. BedrockLLMService — transport over AWS Bedrock's Converse API.

Both provider classes only implement _call_llm(system_prompt, user_prompt)
-> str; every other method (build_prompt, generate_answer, generate_summary,
build_presentation_prompt, edit_presentation, generate_presentation) is
inherited from BaseLLMService, so both providers behave identically from
app.py's point of view — app.py picks which one to instantiate based on
settings.LLM_PROVIDER ("openrouter" or "bedrock").
"""

import json
import logging
import os
import random
import re
from typing import List, Tuple

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = (
    "The information is not available in the provided documents."
)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


# ===========================================================================
# 1. SHARED LOGIC — prompt loading, formatting, batching
# ===========================================================================

# Prompts are NOT hardcoded in code. Every prompt used below (Q&A,
# summarisation, presentation/narration script, per-batch extraction, and
# script-editing) is split into a SYSTEM half (persona + fixed rules/format,
# never changes call-to-call) and a USER half (the dynamic input: retrieved
# chunks, the question, the seed, etc.). Each half lives as its own .txt file
# in the prompts/ directory so it can be edited without touching code.
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
)

def _resolve_prompt_path(primary: str, fallback: str) -> str:
    """Return the first existing prompt path, falling back to the simple prompt file name."""
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    return primary


_QA_SYSTEM_PROMPT_PATH = _resolve_prompt_path(
    os.path.join(_PROMPTS_DIR, "qa_system_prompt.txt"),
    os.path.join(_PROMPTS_DIR, "qa_prompt.txt"),
)
_QA_USER_PROMPT_PATH = _resolve_prompt_path(
    os.path.join(_PROMPTS_DIR, "qa_user_prompt.txt"),
    os.path.join(_PROMPTS_DIR, "qa_prompt.txt"),
)

_SUMMARY_SYSTEM_PROMPT_PATH = _resolve_prompt_path(
    os.path.join(_PROMPTS_DIR, "summary_system_prompt.txt"),
    os.path.join(_PROMPTS_DIR, "summary_prompt.txt"),
)
_SUMMARY_USER_PROMPT_PATH = _resolve_prompt_path(
    os.path.join(_PROMPTS_DIR, "summary_user_prompt.txt"),
    os.path.join(_PROMPTS_DIR, "summary_prompt.txt"),
)

_BATCH_EXTRACTION_SYSTEM_PROMPT_PATH = _resolve_prompt_path(
    os.path.join(_PROMPTS_DIR, "batch_extraction_system_prompt.txt"),
    os.path.join(_PROMPTS_DIR, "batch_extraction_prompt.txt"),
)
_BATCH_EXTRACTION_USER_PROMPT_PATH = _resolve_prompt_path(
    os.path.join(_PROMPTS_DIR, "batch_extraction_user_prompt.txt"),
    os.path.join(_PROMPTS_DIR, "batch_extraction_prompt.txt"),
)

_EDIT_PRESENTATION_SYSTEM_PROMPT_PATH = _resolve_prompt_path(
    os.path.join(_PROMPTS_DIR, "edit_presentation_system_prompt.txt"),
    os.path.join(_PROMPTS_DIR, "edit_presentation_prompt.txt"),
)
_EDIT_PRESENTATION_USER_PROMPT_PATH = _resolve_prompt_path(
    os.path.join(_PROMPTS_DIR, "edit_presentation_user_prompt.txt"),
    os.path.join(_PROMPTS_DIR, "edit_presentation_prompt.txt"),
)

_PRESENTATION_SYSTEM_PROMPT_PATH = _resolve_prompt_path(
    os.getenv(
        "PRESENTATION_SYSTEM_PROMPT_PATH",
        os.path.join(_PROMPTS_DIR, "presentation_system_prompt.txt"),
    ),
    os.path.join(_PROMPTS_DIR, "presentation_prompt.txt"),
)
_PRESENTATION_USER_PROMPT_PATH = _resolve_prompt_path(
    os.getenv(
        "PRESENTATION_USER_PROMPT_PATH",
        os.path.join(_PROMPTS_DIR, "presentation_user_prompt.txt"),
    ),
    os.path.join(_PROMPTS_DIR, "presentation_prompt.txt"),
)

_EXCEL_RESTRUCTURING_SYSTEM_PROMPT_PATH = os.path.join(
    _PROMPTS_DIR, "excel_restructuring_system_prompt.txt"
)

# Seed pool used to force a different illustrative character/workplace into
# each generated script, so back-to-back runs don't converge on the same
# story even at low sampling temperature.
_SEED_FIRST_NAMES = [
    "Maria", "David", "Aisha", "Wei", "Carlos", "Emily", "Tunde", "Sofia",
    "James", "Priya", "Liam", "Noor", "Diego", "Grace", "Hiro", "Fatima",
    "Marcus", "Elena", "Kwame", "Anya",
]
_SEED_LAST_NAMES = [
    "Santos", "Bennett", "Khan", "Chen", "Ramirez", "Novak", "Adeyemi",
    "Rossi", "Turner", "Patel", "O'Connor", "Haddad", "Silva", "Okafor",
    "Tanaka", "Kowalski", "Nguyen", "Fischer", "Osei", "Bergstrom",
]
_SEED_WORKPLACES = [
    "a food-processing plant", "a construction site", "a chemical warehouse",
    "a hospital maintenance department", "an automotive assembly line",
    "a logistics distribution center", "an oil refinery",
    "a commercial kitchen", "a data-center facility", "a mining operation",
    "a shipyard", "a pharmaceutical packaging plant", "a paper mill",
    "an electrical substation", "a cold-storage facility",
]


def _load_prompt_template(path: str) -> str:
    """Load a prompt template from a plain .txt file in the prompts/ directory.

    Raised errors are intentionally not swallowed — prompts are no longer
    hardcoded in code, so a missing/empty file should fail loudly rather
    than silently falling back to some other copy of the prompt.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Prompt template not found at '{path}'. "
            f"Edit that file in the prompts/ directory (or check the relevant *_PATH env var)."
        ) from exc

    if not template or not template.strip():
        raise RuntimeError(f"'{path}' is empty.")
    return template


def load_qa_prompts() -> Tuple[str, str]:
    return (
        _load_prompt_template(_QA_SYSTEM_PROMPT_PATH),
        _load_prompt_template(_QA_USER_PROMPT_PATH),
    )


def load_summary_prompts() -> Tuple[str, str]:
    return (
        _load_prompt_template(_SUMMARY_SYSTEM_PROMPT_PATH),
        _load_prompt_template(_SUMMARY_USER_PROMPT_PATH),
    )


def load_batch_extraction_prompts() -> Tuple[str, str]:
    return (
        _load_prompt_template(_BATCH_EXTRACTION_SYSTEM_PROMPT_PATH),
        _load_prompt_template(_BATCH_EXTRACTION_USER_PROMPT_PATH),
    )


def load_edit_presentation_prompts() -> Tuple[str, str]:
    return (
        _load_prompt_template(_EDIT_PRESENTATION_SYSTEM_PROMPT_PATH),
        _load_prompt_template(_EDIT_PRESENTATION_USER_PROMPT_PATH),
    )


def load_presentation_prompts() -> Tuple[str, str]:
    return (
        _load_prompt_template(_PRESENTATION_SYSTEM_PROMPT_PATH),
        _load_prompt_template(_PRESENTATION_USER_PROMPT_PATH),
    )


def load_excel_restructuring_system_prompt() -> str:
    return _load_prompt_template(_EXCEL_RESTRUCTURING_SYSTEM_PROMPT_PATH)


def random_story_seed() -> Tuple[str, str]:
    """Pick a random character name + workplace to seed a unique story."""
    name = f"{random.choice(_SEED_FIRST_NAMES)} {random.choice(_SEED_LAST_NAMES)}"
    workplace = random.choice(_SEED_WORKPLACES)
    return name, workplace


def format_context(chunks: List[dict]) -> str:
    """Flatten retrieved chunks into the '[1] Source:...\\nPage:...\\n<text>'
    block format every prompt's {retrieved_chunks} placeholder expects."""
    if not chunks:
        return "No relevant context found."

    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        # Include any stored metadata (author, title, etc.) so the LLM
        # can answer metadata queries even if that information isn't in
        # the chunk text itself.
        meta = chunk.get("metadata") or {}
        meta_str = ""
        if isinstance(meta, dict) and meta:
            items = [f"{k}: {v}" for k, v in meta.items() if v is not None and str(v).strip()]
            if items:
                meta_str = "Metadata: " + " | ".join(items) + "\n"

        formatted.append(
            f"[{i}]\n"
            f"Source: {chunk.get('filename', 'unknown')}\n"
            f"Page: {chunk.get('page_label', 'N/A')}\n"
            f"{meta_str}\n"
            f"{chunk.get('text', '').strip()}"
        )
    return "\n\n---\n\n".join(formatted)


def normalize_summary(text: str) -> str:
    """Clean up summary output without truncating it."""
    text = re.sub(r"\r\n|\r", "\n", text).strip()

    # Remove leading bullet/numbering markers from any line
    lines = text.split("\n")
    cleaned = [re.sub(r"^[-*#\d.\s]+", "", line) for line in lines]
    return "\n".join(cleaned).strip()


def batch_chunks(chunks: List[dict], max_chars: int = 60000) -> List[List[dict]]:
    """Split chunks into batches that each fit within max_chars of context text."""
    batches, current, current_len = [], [], 0
    for chunk in chunks:
        chunk_len = len(chunk.get("text", ""))
        if current and current_len + chunk_len > max_chars:
            batches.append(current)
            current, current_len = [], 0
        current.append(chunk)
        current_len += chunk_len
    if current:
        batches.append(current)
    return batches


def render_structured_document(doc: dict) -> str:
    """Flatten the excel-restructuring JSON schema (document_title,
    document_type, sections[{title, content, subsections}]) back into
    heading-marked plain text, so it can flow through the existing
    DocumentChunker (which splits on heading-like lines) exactly like any
    other page's text — the JSON's hierarchy is preserved as Markdown-style
    '#'/'##'/'###' heading depth rather than lost.
    """
    lines: List[str] = []

    title = doc.get("document_title") or ""
    doc_type = doc.get("document_type") or ""
    if title:
        lines.append(f"# {title}")
    if doc_type:
        lines.append(f"Document type: {doc_type}")
    if title or doc_type:
        lines.append("")

    def render_section(section: dict, depth: int) -> None:
        heading_marker = "#" * min(depth, 6)
        section_title = section.get("title") or section.get("section_id") or ""
        if section_title:
            lines.append(f"{heading_marker} {section_title}")

        for item in section.get("content", []) or []:
            lines.append(str(item))

        if section.get("content"):
            lines.append("")

        for subsection in section.get("subsections", []) or []:
            render_section(subsection, depth + 1)

    for section in doc.get("sections", []) or []:
        render_section(section, 2)

    return "\n".join(lines).strip()


class BaseLLMService:
    """Base class for any LLM transport (OpenRouter, Bedrock, ...).

    A subclass only needs to implement _call_llm(system_prompt, user_prompt)
    -> str — how to actually reach its provider and get text back.
    Everything else (loading/building the five prompt pairs, summary
    post-processing, and batching long presentation content across
    multiple calls) is shared here, so every transport behaves identically
    from app.py's point of view.
    """

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    @staticmethod
    def _format_context(chunks: List[dict]) -> str:
        return format_context(chunks)

    def build_prompt(self, chunks: List[dict], question: str) -> Tuple[str, str]:
        """Return (system_prompt, user_prompt) for the Q&A call."""
        system_template, user_template = load_qa_prompts()
        user_prompt = user_template.format(
            retrieved_chunks=self._format_context(chunks),
            user_question=question,
        )
        return system_template, user_prompt

    def build_summary_prompt(self, chunks: List[dict]) -> Tuple[str, str]:
        """Return (system_prompt, user_prompt) for the summarization call."""
        system_template, user_template = load_summary_prompts()
        user_prompt = user_template.format(
            retrieved_chunks=self._format_context(chunks),
        )
        return system_template, user_prompt

    @staticmethod
    def _normalize_summary(text: str) -> str:
        return normalize_summary(text)

    def generate_answer(self, chunks: List[dict], question: str) -> str:
        """Generate an answer to *question* using retrieved *chunks*."""
        system_prompt, user_prompt = self.build_prompt(chunks, question)
        return self._call_llm(system_prompt, user_prompt)

    def generate_summary(self, chunks: List[dict]) -> str:
        """Generate a rich executive summary from *chunks*."""
        system_prompt, user_prompt = self.build_summary_prompt(chunks)
        result = self._call_llm(system_prompt, user_prompt)
        return self._normalize_summary(result)

    def build_presentation_prompt(self, chunks: List[dict], seed_character_name: str = None,
                                   seed_workplace: str = None) -> Tuple[str, str]:
        """Return (system_prompt, user_prompt) for the presentation-script call."""
        if not seed_character_name or not seed_workplace:
            seed_character_name, seed_workplace = random_story_seed()
        system_template, user_template = load_presentation_prompts()
        user_prompt = user_template.format(
            retrieved_chunks=self._format_context(chunks),
            seed_character_name=seed_character_name,
            seed_workplace=seed_workplace,
        )
        return system_template, user_prompt

    def edit_presentation(self, current_script: str, instruction: str) -> str:
        """Revise an already-generated training script per a follow-up chat
        instruction (e.g. "make it shorter", "change the character's name
        to Priya", "add a line about PPE"). Applies ONLY the requested
        change and preserves the fixed script format — it does not
        regenerate a new story or invent a new seed.
        """
        system_template, user_template = load_edit_presentation_prompts()
        user_prompt = user_template.format(
            current_script=current_script,
            instruction=instruction,
        )
        return self._call_llm(system_template, user_prompt)

    def _batch_chunks(self, chunks: List[dict], max_chars: int = 60000) -> List[List[dict]]:
        return batch_chunks(chunks, max_chars)

    def generate_presentation(self, chunks: List[dict]) -> str:
        """Generate a full training script, batching chunks to avoid context limits.

        Script SHAPE is fixed (prompts/presentation_system_prompt.txt); the
        illustrative STORY is randomized fresh on every call via a seeded
        character/workplace.
        """
        seed_character_name, seed_workplace = random_story_seed()
        batches = self._batch_chunks(chunks, max_chars=60000)

        if len(batches) == 1:
            # Fits in one call — generate directly
            system_prompt, user_prompt = self.build_presentation_prompt(
                batches[0], seed_character_name, seed_workplace
            )
            return self._call_llm(system_prompt, user_prompt)

        # Multiple batches: summarize each batch first, then do a final pass
        logger.info("Presentation: %d batches detected, using batch strategy.", len(batches))

        batch_summaries = []
        extraction_system, extraction_user_template = load_batch_extraction_prompts()
        for i, batch in enumerate(batches, 1):
            logger.info("Summarizing batch %d/%d for presentation.", i, len(batches))
            extraction_user = extraction_user_template.format(
                retrieved_chunks=self._format_context(batch),
            )
            batch_summary = self._call_llm(extraction_system, extraction_user)
            batch_summaries.append(f"=== Source Batch {i} ===\n{batch_summary}")

        # Final pass: generate full script from the combined batch summaries
        combined = "\n\n".join(batch_summaries)
        system_template, user_template = load_presentation_prompts()
        final_user_prompt = user_template.format(
            retrieved_chunks=combined,
            seed_character_name=seed_character_name,
            seed_workplace=seed_workplace,
        )
        return self._call_llm(system_template, final_user_prompt)

    def restructure_excel_content(self, raw_content: str, file_name: str, sheet_name: str) -> dict:
        """Send raw flattened spreadsheet rows through the excel-restructuring
        prompt and return the parsed JSON (document_title, document_type,
        sections[{title, content, subsections, metadata}], metadata).

        This is a pure reorganization pass — no summarizing, no answering,
        no invented content — meant to run BEFORE chunking/embedding, so
        messy spreadsheet layout (merged-cell repeats, inconsistent
        grouping) becomes a clean hierarchy the chunker can split on
        sensibly. Raises RuntimeError if the model doesn't return valid
        JSON, rather than silently falling back to the raw, unstructured
        text — a caller can catch that and use the raw text instead if it
        wants a soft-fail behavior.

        Unlike every other prompt pair in this file, this one has no
        separate user-prompt .txt template — the system prompt alone fully
        specifies the task, so the user turn is just the raw data.
        """
        system_prompt = load_excel_restructuring_system_prompt()
        user_prompt = (
            f"File name: {file_name}\n"
            f"Sheet name: {sheet_name}\n\n"
            f"Extracted content:\n{raw_content}"
        )
        response = self._call_llm(system_prompt, user_prompt)

        # Models occasionally wrap JSON in ```json ... ``` even when told
        # not to — strip that defensively before parsing.
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Excel restructuring for '{file_name}' / sheet '{sheet_name}' "
                f"did not return valid JSON: {exc}\nRaw response: {response[:500]}"
            ) from exc


# ===========================================================================
# 2. OpenRouter transport
# ===========================================================================

class OpenRouterLLMService(BaseLLMService):
    """LLM transport over OpenRouter's chat-completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen-2.5-72b-instruct",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        base_url: str = OPENROUTER_CHAT_COMPLETIONS_URL,
        site_url: str = "",
        site_name: str = "",
        timeout: int = 90,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required.")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if site_url:
            self.headers["HTTP-Referer"] = site_url
        if site_name:
            self.headers["X-Title"] = site_name

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a system + user message pair to OpenRouter and return the response text."""
        logger.info(
            "Sending prompt to OpenRouter model '%s' (system=%d chars, user=%d chars).",
            self.model, len(system_prompt), len(user_prompt),
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"{response.status_code} error from OpenRouter: {response.text[:500]}"
            )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return FALLBACK_ANSWER

        answer = (
            choices[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return answer if answer else FALLBACK_ANSWER


# ===========================================================================
# 3. AWS Bedrock transport
# ===========================================================================

class BedrockLLMService(BaseLLMService):
    """LLM transport over AWS Bedrock's unified Converse API
    (bedrock-runtime.converse) — the same call shape works across every
    Bedrock-hosted model family (Nova, Claude, Llama, Mistral, ...), so
    switching model IDs later needs no code change here.

    Credentials: if AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are supplied
    they are used directly; otherwise boto3 falls back to its normal
    credential chain (env vars, ~/.aws/credentials, or an IAM role if
    running on EC2/ECS/Lambda) — the same pattern used by
    services/s3_storage.py, so one set of AWS credentials in .env covers
    both S3 and Bedrock.

    Note on Nova 2 model IDs: some Nova 2 models require a region-prefixed
    Cross-Region Inference (CRIS) profile ID for on-demand access rather
    than the bare model ID — e.g. "us.amazon.nova-2-lite-v1:0" instead of
    "amazon.nova-2-lite-v1:0" when calling from a US region. If a call
    fails with a "use an inference profile" error, that's the fix.
    """

    def __init__(
        self,
        model: str = "amazon.nova-micro-v1:0",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        region_name: str = "us-east-1",
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        client_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client = boto3.client("bedrock-runtime", **client_kwargs)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a system + user message pair to Bedrock and return the response text."""
        logger.info(
            "Sending prompt to Bedrock model '%s' (system=%d chars, user=%d chars).",
            self.model, len(system_prompt), len(user_prompt),
        )

        try:
            response = self.client.converse(
                modelId=self.model,
                system=[{"text": system_prompt}],
                messages=[
                    {"role": "user", "content": [{"text": user_prompt}]},
                ],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(
                f"Bedrock inference failed for model '{self.model}': {exc}"
            ) from exc

        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        answer = "".join(
            block["text"] for block in content_blocks if "text" in block
        ).strip()
        return answer if answer else FALLBACK_ANSWER