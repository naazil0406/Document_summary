import json
import logging
import os
import random
import re
from typing import List

import requests

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"

FALLBACK_ANSWER = (
    "The information is not available in the provided documents."
)

# ---------------------------------------------------------------------------
# Main Q&A prompt — used by the query flow
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """\
You are a precise document assistant. Answer questions strictly from the \
context below — never invent or infer facts not present in the context.

Context:
{retrieved_chunks}

Question:
{user_question}

Instructions:
* Answer ONLY using the provided context.
* Answer the question directly and concisely using only the provided context.
* Do not invent, infer, or add information that is not present in the context.
* Do not use section headings, bullets, numbered lists, labels, or extra titles.
* Preserve important facts, dates, names, numbers, and findings exactly as they appear in the context.


Answer:"""

# ---------------------------------------------------------------------------
# Summarisation prompt — used by the summarize flow (more context, richer output)
# ---------------------------------------------------------------------------
SUMMARY_PROMPT_TEMPLATE = """\
You are a clear and concise document analyst. Produce an easy-to-understand
executive summary of the document excerpts below. Use ONLY the content provided
— never invent or infer information not present in the excerpts.

Document content:
{retrieved_chunks}

Task:
Write a clear executive summary in plain language using exactly 2 paragraphs
separated by a blank line. The total summary should be 10 to 15 sentences.
Cover the key purpose, main topics, important ideas, and takeaways. Do not use
section headings, bullet points, numbered lists, labels, or introductory titles.
Keep the writing fluent, readable, and free of padding, repetition, or vague filler.

Executive Summary:"""


# ---------------------------------------------------------------------------
# Presentation / narration script prompt — fixed standard format, but the
# template TEXT itself is NOT hardcoded here. It lives in
# prompts/presentation_prompt.json so it can be edited without touching code.
#
# This is a story-driven, scene-scripted training video narration (opening
# scene, "Narrator (Trainer):" spoken blocks, an illustrative real-life-style
# incident, a concept break-down, a closing reflection prompt) rather than a
# slide-deck outline. The SHAPE of the script is fixed — it is never derived
# from an uploaded Word document. The illustrative STORY inside it, however,
# must be freshly invented every time (different character, workplace, and
# incident) rather than pulled from the training documents themselves.
# ---------------------------------------------------------------------------
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
)
_PRESENTATION_PROMPT_PATH = os.getenv(
    "PRESENTATION_PROMPT_PATH",
    os.path.join(_PROMPTS_DIR, "presentation_prompt.json"),
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


def _load_presentation_prompt_template() -> str:
    """Load the presentation prompt template from prompts/presentation_prompt.json.

    Raised errors are intentionally not swallowed — the template is no longer
    hardcoded in this file, so a missing/broken JSON file should fail loudly
    rather than silently falling back to some other copy of the prompt.
    """
    try:
        with open(_PRESENTATION_PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Presentation prompt template not found at '{_PRESENTATION_PROMPT_PATH}'. "
            "Edit prompts/presentation_prompt.json (or set PRESENTATION_PROMPT_PATH)."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Presentation prompt template at '{_PRESENTATION_PROMPT_PATH}' is not valid JSON: {exc}"
        ) from exc

    template = data.get("presentation_prompt")
    if not template or not template.strip():
        raise RuntimeError(
            f"'{_PRESENTATION_PROMPT_PATH}' is missing a non-empty 'presentation_prompt' field."
        )
    return template


def _random_story_seed() -> tuple[str, str]:
    """Pick a random character name + workplace to seed a unique story."""
    name = f"{random.choice(_SEED_FIRST_NAMES)} {random.choice(_SEED_LAST_NAMES)}"
    workplace = random.choice(_SEED_WORKPLACES)
    return name, workplace


class OpenRouterLLMService:

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

    @staticmethod
    def _format_context(chunks: List[dict]) -> str:
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

    def build_prompt(self, chunks: List[dict], question: str) -> str:
        return PROMPT_TEMPLATE.format(
            retrieved_chunks=self._format_context(chunks),
            user_question=question,
        )

    def build_summary_prompt(self, chunks: List[dict]) -> str:
        return SUMMARY_PROMPT_TEMPLATE.format(
            retrieved_chunks=self._format_context(chunks),
        )

    def _call_llm(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the response text."""
        logger.info("Sending prompt to LLM (%d chars).", len(prompt))

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
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

    @staticmethod
    def _normalize_summary(text: str) -> str:
        """Clean up summary output without truncating it."""
        text = re.sub(r"\r\n|\r", "\n", text).strip()

        # Remove leading bullet/numbering markers from any line
        lines = text.split("\n")
        cleaned = [re.sub(r"^[-*#\d.\s]+", "", line) for line in lines]
        return "\n".join(cleaned).strip()

    def generate_answer(self, chunks: List[dict], question: str) -> str:
        """Generate an answer to *question* using retrieved *chunks*."""
        prompt = self.build_prompt(chunks, question)
        return self._call_llm(prompt)

    def generate_summary(self, chunks: List[dict]) -> str:
        """Generate a rich executive summary from *chunks*."""
        prompt = self.build_summary_prompt(chunks)
        result = self._call_llm(prompt)
        return self._normalize_summary(result)

    def build_presentation_prompt(self, chunks: List[dict], seed_character_name: str = None,
                                   seed_workplace: str = None) -> str:
        if not seed_character_name or not seed_workplace:
            seed_character_name, seed_workplace = _random_story_seed()
        template = _load_presentation_prompt_template()
        return template.format(
            retrieved_chunks=self._format_context(chunks),
            seed_character_name=seed_character_name,
            seed_workplace=seed_workplace,
        )

    def edit_presentation(self, current_script: str, instruction: str) -> str:
        """Revise an already-generated training script per a follow-up chat
        instruction (e.g. "make it shorter", "change the character's name
        to Priya", "add a line about PPE"). Applies ONLY the requested
        change and preserves the fixed script format — it does not
        regenerate a new story or invent a new seed.
        """
        prompt = (
            "You are revising an existing corporate training video script. "
            "Apply ONLY the requested change below and return the FULL "
            "revised script — do not summarize, explain, or add commentary "
            "about the change; output nothing but the revised script "
            "itself, and keep the exact same structure as the original "
            "(the 'Story - <name>.mp4' header line, the 'TRT: <MM:SS>' "
            "line, the 'Narrative Script:' label, bracketed scene/visual "
            "directions, and 'Narrator (Trainer):' spoken blocks). Do not "
            "change anything the instruction didn't ask you to change.\n\n"
            f"CURRENT SCRIPT:\n{current_script}\n\n"
            f"REQUESTED CHANGE:\n{instruction}\n\n"
            "Revised script:"
        )
        return self._call_llm(prompt)

    def _batch_chunks(self, chunks: List[dict], max_chars: int = 60000) -> List[List[dict]]:
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

    def generate_presentation(self, chunks: List[dict]) -> str:
        """Generate a full training script, batching chunks to avoid context limits.

        The script's SHAPE is fixed (loaded from prompts/presentation_prompt.json)
        and never derived from an uploaded document. The illustrative STORY inside
        it is freshly randomized on every call — a different character name and
        workplace are seeded in each time, and the prompt itself instructs the
        model to invent the incident from general knowledge rather than reuse
        anything from the training documents.
        """
        seed_character_name, seed_workplace = _random_story_seed()
        batches = self._batch_chunks(chunks, max_chars=60000)

        if len(batches) == 1:
            # Fits in one call — generate directly
            prompt = self.build_presentation_prompt(batches[0], seed_character_name, seed_workplace)
            return self._call_llm(prompt)

        # Multiple batches: summarize each batch first, then do a final pass
        logger.info("Presentation: %d batches detected, using batch strategy.", len(batches))

        batch_summaries = []
        for i, batch in enumerate(batches, 1):
            logger.info("Summarizing batch %d/%d for presentation.", i, len(batches))
            summary_prompt = (
                "You are a document analyst. Extract and preserve ALL key information, "
                "topics, procedures, safety rules, definitions, and examples from the "
                "content below. Write in detailed bullet points. Do not omit anything "
                "important. This output will be used to generate a corporate training script.\n\n"
                f"Content:\n{self._format_context(batch)}\n\nDetailed extraction:"
            )
            batch_summary = self._call_llm(summary_prompt)
            batch_summaries.append(f"=== Source Batch {i} ===\n{batch_summary}")

        # Final pass: generate full script from the combined batch summaries
        combined = "\n\n".join(batch_summaries)
        template = _load_presentation_prompt_template()
        final_prompt = template.format(
            retrieved_chunks=combined,
            seed_character_name=seed_character_name,
            seed_workplace=seed_workplace,
        )
        return self._call_llm(final_prompt)