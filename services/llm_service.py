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
from typing import List, Optional, Tuple

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

_IMAGE_PROMPT_SYSTEM_PROMPT_PATH = os.path.join(
    _PROMPTS_DIR, "image_prompt_system.txt"
)

_CONTENT_GENERATION_SYSTEM_PROMPT_PATH = os.path.join(
    _PROMPTS_DIR, "content_generation_system.txt"
)

# The exact set of content types the Learning Content Generation Engine
# supports (see prompts/content_generation_system.txt). Kept here, not just
# in main.py, so llm_service can validate/normalize independent of the API
# layer.
CONTENT_TYPES = [
    "Recall Card",
    "AI Image",
    "Infographic",
    "Flashcard",
    "Scenario",
    "Spot the Mistake Challenge",
    "Daily Quiz",
    "Fun Fact",
    "Reflection Question",
    "Safety / Best Practice Tip",
    "Daily Tip",
]

# Daily Tip has its own word-count contract (40-80 words), distinct from
# the 50-80/100-word ceiling every other Content Type follows — see
# prompts/content_generation_system.txt's Daily Tip section.
_DAILY_TIP_MIN_WORDS = 40
_DAILY_TIP_MAX_WORDS = 80
_DAILY_TIP_MAX_ATTEMPTS = 3

# Must stay in sync with the "STATES AND ERRORS REFERENCE FRAMEWORK"
# section of prompts/content_generation_system.txt — same four States,
# same four Errors, same wording. Each Daily Tip is anchored to exactly
# one of these eight (see generate_daily_tips()'s rotation), so a batch
# request ("give me 10 daily tips") systematically covers the framework
# instead of the model freely picking whatever idea it likes each time.
_STATES = ["Rushing", "Frustration", "Fatigue", "Complacency"]
_ERRORS = ["Eyes not on task", "Mind not on task", "Line of fire", "Balance, traction, grip"]
_DAILY_TIP_FOCUS_ROTATION = [("State", s) for s in _STATES] + [("Error", e) for e in _ERRORS]

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


def load_image_prompt_system_prompt() -> str:
    return _load_prompt_template(_IMAGE_PROMPT_SYSTEM_PROMPT_PATH)


def load_content_generation_system_prompt() -> str:
    return _load_prompt_template(_CONTENT_GENERATION_SYSTEM_PROMPT_PATH)


def load_video_story_dual_prompt() -> str:
    """Dual Video Script + Story mode lives inside presentation_prompt.txt
    itself (the 'DUAL-OUTPUT MODE' section at the bottom), reusing the same
    file as generate_presentation() rather than a separate prompt file."""
    return _load_prompt_template(_PRESENTATION_SYSTEM_PROMPT_PATH)


def load_video_story_dual_prompt() -> str:
    return _load_prompt_template(_PRESENTATION_SYSTEM_PROMPT_PATH)


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

    def generate_image_prompt(self, user_query: str, chunks: List[dict], mode: str = "infographic") -> str:
        """Nova Lite step of the image pipeline: turn a user's image request
        plus retrieved RAG chunks into ONE optimized natural-language prompt
        for Nova Canvas.

        mode: "infographic" (flat vector/iconographic explainer visual) or
        "scene" (photorealistic cinematic single moment, no text/icons).
        This is passed through explicitly as an "Output Mode:" line so
        image_prompt_system.txt never has to infer/guess the mode from the
        wording of user_query — see that file's "OUTPUT MODE" section.
        Content types that describe a real situation (Scenario, Spot the
        Mistake Challenge, AI Image) should use "scene"; conceptual/
        explainer content types (Recall Card, Infographic, Flashcard,
        Daily Quiz, Fun Fact, Reflection Question, Safety/Best Practice
        Tip) should use "infographic".

        This must return a single continuous descriptive paragraph — no
        markdown, no JSON, no bullet points — per the image_prompt_system.txt
        rules, so unlike restructure_excel_content() there's no parsing step;
        the raw stripped model text IS the Nova Canvas prompt.

        Unlike every other prompt pair in this file, this one has no
        separate user-prompt .txt template — the system prompt alone fully
        specifies the task, so the user turn is just the request + context.
        """
        if mode not in ("infographic", "scene"):
            mode = "infographic"

        system_prompt = load_image_prompt_system_prompt()
        retrieved_chunks = format_context(chunks)
        user_prompt = (
            f"Output Mode: {mode}\n\n"
            f"User Request: {user_query}\n\n"
            f"Retrieved Context:\n{retrieved_chunks}"
        )
        image_prompt = self._call_llm(system_prompt, user_prompt).strip()

        # Defensive cleanup: models occasionally wrap output in quotes or
        # code fences even when told not to.
        if image_prompt.startswith("```"):
            image_prompt = re.sub(r"^```(?:\w+)?\s*", "", image_prompt)
            image_prompt = re.sub(r"\s*```$", "", image_prompt)
        image_prompt = image_prompt.strip().strip('"').strip()

        if not image_prompt:
            raise RuntimeError("Nova Lite returned an empty image prompt.")
        return image_prompt

    def generate_learning_content(
        self,
        content_type: str,
        topic: str,
        chunks: List[dict],
        monthly_topic_content: Optional[str] = None,
        common_data: Optional[str] = None,
        web_results: Optional[str] = None,
        avoid_repeating: Optional[List[str]] = None,
        daily_tip_focus: Optional[Tuple[str, str]] = None,
    ) -> str:
        """Content Generation Agent: turn (Content Type + Topic + retrieved
        Knowledge Base chunks) into one short, original piece of
        learner-facing content — never copied from the Knowledge Base or
        the internet, per prompts/content_generation_system.txt.

        monthly_topic_content / common_data / web_results are optional
        extra sources described in the system prompt:
          - monthly_topic_content: this month's featured lesson material,
            treated as a primary fact source alongside Retrieved Context.
          - common_data: general human-performance background used only
            for understanding, never quoted.
          - web_results: supplementary internet research used only to
            enrich with a real-world example/best practice, never a
            source of wording. Callers are responsible for fetching this
            (e.g. via a web search step) before calling this method.
        avoid_repeating: previously-generated pieces of content for this
          same (content_type, topic) pair in the current batch (e.g. tip
          #1-4 of a 10-tip request). Passed back to the model so tip #5
          picks a different angle/fact/phrasing instead of converging on
          the same idea. Optional — a single one-off generation doesn't
          need it.
        daily_tip_focus: only meaningful when content_type == "Daily Tip".
          A ("State", "Rushing") or ("Error", "Eyes not on task") pair —
          see _DAILY_TIP_FOCUS_ROTATION — that this specific tip must be
          built around, per the "STATES AND ERRORS REFERENCE FRAMEWORK"
          rule that the State/Error is shown only through the learner's
          actions/thoughts, never named outright in the tip itself.
        All default to None/omitted so existing callers that only pass
        (content_type, topic, chunks) keep working unchanged.

        Like generate_image_prompt(), this prompt pair has no separate
        user-prompt .txt template; the system prompt fully specifies the
        task and the user turn is just the dynamic inputs.
        """
        if content_type not in CONTENT_TYPES:
            raise ValueError(
                f"Unsupported content type '{content_type}'. "
                f"Supported types: {', '.join(CONTENT_TYPES)}"
            )

        system_prompt = load_content_generation_system_prompt()
        retrieved_chunks = format_context(chunks)
        user_prompt = (
            f"Content Type: {content_type}\n"
            f"Topic: {topic}\n\n"
            f"Retrieved Context:\n{retrieved_chunks}"
        )
        if daily_tip_focus:
            kind, name = daily_tip_focus
            user_prompt += (
                f"\n\nDaily Tip Focus ({kind}): {name}\n"
                f"Build this tip around a moment where this {kind.lower()} "
                f"leads (or nearly leads) to an incident related to the "
                f"Topic/Retrieved Context. Per the framework rules above, "
                f"show it only through the learner's actions/thoughts — "
                f"do not write \"{name}\" or any other State/Error name in "
                f"the tip text itself."
            )
        if avoid_repeating:
            already_generated = "\n".join(f"- {t}" for t in avoid_repeating)
            user_prompt += (
                f"\n\nAlready generated in this batch — pick a different fact, "
                f"angle, or example; do not repeat or lightly reword any of "
                f"these:\n{already_generated}"
            )
        if monthly_topic_content and monthly_topic_content.strip():
            user_prompt += f"\n\nMonthly Topic Content:\n{monthly_topic_content.strip()}"
        if common_data and common_data.strip():
            user_prompt += f"\n\nCommon Knowledge:\n{common_data.strip()}"
        if web_results and web_results.strip():
            user_prompt += f"\n\nInternet Research:\n{web_results.strip()}"

        content_text = self._call_llm(system_prompt, user_prompt).strip()

        # Defensive cleanup: models occasionally wrap output in quotes,
        # markdown headings, or code fences even when told not to.
        if content_text.startswith("```"):
            content_text = re.sub(r"^```(?:\w+)?\s*", "", content_text)
            content_text = re.sub(r"\s*```$", "", content_text)
        content_text = re.sub(rf"^{re.escape(content_type)}\s*:\s*", "", content_text, flags=re.IGNORECASE)
        content_text = content_text.strip().strip('"').strip()

        if not content_text:
            raise RuntimeError(f"The model returned no content for '{content_type}' on topic '{topic}'.")
        return content_text

    def _generate_one_daily_tip(
        self,
        chunks: List[dict],
        topic: str,
        common_data: Optional[str] = None,
        web_results: Optional[str] = None,
        avoid_repeating: Optional[List[str]] = None,
        focus: Optional[Tuple[str, str]] = None,
    ) -> str:
        """Generate a single 40-80 word Daily Tip, retrying a bounded
        number of times if the model's output falls outside that word
        range. Internal helper — see generate_daily_tips() for the public,
        batch-aware entry point."""
        last_text = ""
        for attempt in range(1, _DAILY_TIP_MAX_ATTEMPTS + 1):
            last_text = self.generate_learning_content(
                content_type="Daily Tip",
                topic=topic,
                chunks=chunks,
                common_data=common_data,
                web_results=web_results,
                avoid_repeating=avoid_repeating,
                daily_tip_focus=focus,
            )
            word_count = len(last_text.split())
            if _DAILY_TIP_MIN_WORDS <= word_count <= _DAILY_TIP_MAX_WORDS:
                return last_text
            logger.warning(
                "Daily Tip attempt %d/%d was %d words (need %d-%d). Retrying.",
                attempt, _DAILY_TIP_MAX_ATTEMPTS, word_count,
                _DAILY_TIP_MIN_WORDS, _DAILY_TIP_MAX_WORDS,
            )
        # All attempts missed the target length — return the closest one
        # rather than failing the request outright.
        return last_text

    def generate_daily_tip(
        self,
        chunks: List[dict],
        topic: str = "",
        common_data: Optional[str] = None,
        web_results: Optional[str] = None,
    ) -> str:
        """Backward-compatible single-tip entry point. Equivalent to
        generate_daily_tips(..., count=1)[0]. See that method's docstring
        for the full behavior."""
        return self.generate_daily_tips(
            chunks, topic=topic, count=1,
            common_data=common_data, web_results=web_results,
        )[0]

    def generate_daily_tips(
        self,
        chunks: List[dict],
        topic: str = "",
        count: int = 1,
        common_data: Optional[str] = None,
        web_results: Optional[str] = None,
    ) -> List[str]:
        """Daily Tip: one or more 40-80 word, conversational, practical
        learning tips — "advice from an experienced safety coach", per
        prompts/content_generation_system.txt's Daily Tip section.

        topic="" is valid and expected for "give me today's daily tip"
        (no specific subject named) — the model picks the best-supported
        angle from whatever chunks were retrieved. Which chunks those are
        is entirely the caller's choice: pass Retriever.retrieve_for_daily_tip()
        output for the global "no dedicated folder" behavior, or
        Retriever.retrieve(query, folders=[...]) output to scope the tips
        to specific folder(s) (e.g. "10 daily tips from video_scripts").

        count: how many distinct tips to generate in one call (e.g. 10 for
        "give me 10 daily tips"). Each tip is anchored to one State or
        Error from the framework (_DAILY_TIP_FOCUS_ROTATION), starting at
        a random point in the 8-item rotation and advancing one per tip —
        so a batch systematically covers different States/Errors instead
        of the model freely picking whatever idea it likes each time, and
        back-to-back single-tip requests don't all land on the same one.
        Each tip after the first is also generated with the previous ones
        passed back to the model as avoid_repeating, for extra insurance
        against near-duplicates even when two tips share a focus (count > 8).
        Capped implicitly by how much distinct material is actually in
        `chunks` — if the source material only supports a few genuinely
        different tips, the model may still converge somewhat; that's a
        content-availability limit, not a bug in this loop.
        """
        effective_topic = topic.strip() if topic and topic.strip() else "today's safety learning"
        count = max(1, count)
        rotation = _DAILY_TIP_FOCUS_ROTATION
        start = random.randrange(len(rotation))
        tips: List[str] = []
        for i in range(count):
            focus = rotation[(start + i) % len(rotation)]
            tip = self._generate_one_daily_tip(
                chunks,
                effective_topic,
                common_data=common_data,
                web_results=web_results,
                avoid_repeating=tips or None,
                focus=focus,
            )
            tips.append(tip)
        return tips

    def generate_video_and_story(self, learning_objectives: str, chunks: List[dict]) -> Tuple[str, str, str]:
        """Generate the dual Video Script + Story training output
        (the 'DUAL-OUTPUT MODE' section of prompts/presentation_prompt.txt)
        in a single LLM call, then split the response into
        (video_script, story, seed_character_name). seed_character_name is
        returned as-is (not re-parsed from the story text) so callers can
        reliably label saved files after the invented storyteller, since
        the model's self-introduction phrasing in the story can vary.

        No topic is passed in — the model selects whatever topic/lesson the
        Knowledge Base content best supports on its own.

        Unlike the QA/summary prompt pairs, this template embeds learning
        objectives and retrieved Knowledge Base content directly via
        .format() and is sent as one filled-in system prompt, with a short
        static user turn — same "single self-contained prompt" pattern
        used by restructure_excel_content().
        """
        template = load_video_story_dual_prompt()
        seed_character_name, seed_workplace = random_story_seed()
        system_prompt = template.format(
            topic="Not specified — choose whichever topic the Knowledge Base Content best supports.",
            learning_objectives=learning_objectives.strip() if learning_objectives and learning_objectives.strip()
            else "Not explicitly given — infer sensible objectives from the Knowledge Base Content.",
            retrieved_chunks=format_context(chunks),
            # seed_character_name/seed_workplace are used directly by the
            # DUAL-OUTPUT MODE instructions so the LEFT PANEL script and the
            # RIGHT PANEL story share the same invented character/incident.
            seed_character_name=seed_character_name,
            seed_workplace=seed_workplace,
        )
        user_prompt = "Generate the LEFT PANEL video script and the RIGHT PANEL story now, following the required format exactly."
        raw_output = self._call_llm(system_prompt, user_prompt)
        video_script, story = self._split_video_story_panels(raw_output)
        return video_script, story, seed_character_name

    @staticmethod
    def _split_video_story_panels(raw_output: str) -> Tuple[str, str]:
        """Split a raw 'LEFT PANEL ... RIGHT PANEL ...' response into
        (video_script, story), tolerant of minor header formatting drift
        (extra dashes, different casing, a trailing colon)."""
        left_marker = re.compile(r"^\s*[-=]*\s*LEFT\s+PANEL\s*[-=:]*\s*$", re.IGNORECASE | re.MULTILINE)
        right_marker = re.compile(r"^\s*[-=]*\s*RIGHT\s+PANEL\s*[-=:]*\s*$", re.IGNORECASE | re.MULTILINE)

        right_match = right_marker.search(raw_output)
        if not right_match:
            # Model didn't use the exact header — safest fallback is to
            # return the whole thing as the video script and flag the story
            # as missing, rather than silently guessing at a split point.
            return raw_output.strip(), (
                "The model did not return a separately labeled story section. "
                "Try regenerating."
            )

        video_part = raw_output[: right_match.start()]
        story_part = raw_output[right_match.end():]

        left_match = left_marker.search(video_part)
        if left_match:
            video_part = video_part[left_match.end():]

        return video_part.strip(), story_part.strip()


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