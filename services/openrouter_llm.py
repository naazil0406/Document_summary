import logging
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
# Presentation / narration script prompt — fixed standard format.
#
# This is a story-driven, scene-scripted training video narration (opening
# scene, "Narrator (Trainer):" spoken blocks, an illustrative real-life-style
# incident, a concept break-down, a closing reflection prompt) rather than a
# slide-deck outline. The format is fixed and built into the prompt — it is
# NOT derived from an uploaded Word document.
# ---------------------------------------------------------------------------
PRESENTATION_PROMPT_TEMPLATE = """\
You are an expert corporate training scriptwriter who writes cinematic, story-driven narration scripts for internal training videos.

TRAINING CONTENT (the only source of facts, terminology, statistics, and procedures below — never invent facts that contradict it, and never invent statistics that are not present in it):
{retrieved_chunks}

====================================================
REQUIRED SCRIPT FORMAT — follow this structure and style exactly
====================================================

Line 1 — a short header naming the story/segment, in this exact form:
Story - <a short, memorable title or the illustrative character's name>.mp4

Line 2 — an estimated runtime for a video of this length, in this exact form:
TRT: <MM:SS>

Line 3 — the section label, on its own line:
Narrative Script:

Then the script body itself, written as a video narration script with:

* Bracketed stage/visual directions such as [Opening Scene: ...] and [Visual: ...] at the opening, at key transitions, and at the close — describing what appears on screen, never spoken aloud.
* "Narrator (Trainer):" on its own line before each spoken block, followed by the narration in quotation marks, written in a warm, conversational, first-person storytelling voice — as if a trainer is narrating a short film to a live audience.
* Open with a short, vivid, illustrative scenario: invent a plausible named person and a specific, concrete workplace situation, and a specific incident or near-miss, that makes the training content's core concept concrete. The lesson, terminology, statistics, and facts embedded in the story must come from the TRAINING CONTENT — only the illustrative character and scene are invented.
* After the story, include a short analytical "let's break this down" passage that names the underlying concept(s), definitions, or terminology the story illustrates, using the TRAINING CONTENT's own vocabulary and framework (its own named concepts, models, or terms — not generic safety language invented for this script).
* Weave in real statistics, definitions, or procedures from the TRAINING CONTENT naturally, as the trainer explaining them mid-narration — never as a bullet list.
* Close each major beat with a smooth spoken transition into the next idea, and include at least one further [Visual: ...] direction partway through (for example a checklist, diagram, or transition graphic that appears on screen).
* End with a closing scene direction and a short reflective question posed directly to the audience, inviting them to share their own experience with the group.
* Keep the tone engaging, natural, and human throughout. Do not use headings, bullet points, or numbered lists anywhere inside the narration body itself — the only structural markers are the bracketed scene/visual directions and the "Narrator (Trainer):" labels.

Do not add any closing meta-commentary about the script itself (no summary paragraph explaining what the script does or how it aligns with the source material). End on the final scene direction or the discussion question — nothing after it.

Training Script:"""


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

    def build_presentation_prompt(self, chunks: List[dict]) -> str:
        return PRESENTATION_PROMPT_TEMPLATE.format(
            retrieved_chunks=self._format_context(chunks),
        )

    @staticmethod
    def _batch_chunks(chunks: List[dict], max_chars: int = 60000) -> List[List[dict]]:
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

        Always uses the fixed cinematic narrative format (PRESENTATION_PROMPT_TEMPLATE)
        — the script's structure is standard and is never derived from an
        uploaded document.
        """
        batches = self._batch_chunks(chunks, max_chars=60000)

        if len(batches) == 1:
            # Fits in one call — generate directly
            prompt = self.build_presentation_prompt(batches[0])
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
        final_prompt = PRESENTATION_PROMPT_TEMPLATE.format(retrieved_chunks=combined)
        return self._call_llm(final_prompt)