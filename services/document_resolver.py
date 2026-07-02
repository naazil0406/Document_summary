"""Helpers for resolving casual document references to uploaded PDF names."""

import os
import re
from typing import Iterable, Optional


_REQUEST_WORDS = {
    "a",
    "about",
    "brief",
    "can",
    "could",
    "current",
    "document",
    "executive",
    "file",
    "for",
    "give",
    "me",
    "now",
    "of",
    "overview",
    "pdf",
    "please",
    "show",
    "summary",
    "summarise",
    "summarize",
    "that",
    "the",
    "this",
    "to",
    "would",
    "you",
}


_DOC_EXTENSIONS = (".pdf", ".docx")


def _words(value: str) -> list[str]:
    value = re.sub(r"\.(?:pdf|docx)\b", " ", value.lower())
    return re.findall(r"[a-z0-9]+", value)


def _compact(value: str) -> str:
    return "".join(_words(value))


def _reference_words(value: str) -> list[str]:
    return [word for word in _words(value) if word not in _REQUEST_WORDS]


def resolve_pdf_reference(reference: str, filenames: Iterable[str]) -> Optional[str]:
    """Return the unique best PDF matching a full or abbreviated reference.

    Matching ignores case, spaces, punctuation, and the ``.pdf`` suffix. A
    meaningful part of a filename is enough (for example ``"safe start"`` or
    ``"unit 2"``), but ambiguous references deliberately return ``None``.
    """
    candidates = [name for name in filenames if name.lower().endswith(_DOC_EXTENSIONS)]
    reference_words = _reference_words(reference)
    if not reference_words:
        return None

    reference_phrase = " ".join(reference_words)
    reference_compact = "".join(reference_words)
    scored = []

    for filename in candidates:
        stem = os.path.splitext(filename)[0]
        stem_words = _words(stem)
        stem_phrase = " ".join(stem_words)
        stem_compact = "".join(stem_words)

        score = 0
        if reference.lower().strip() == filename.lower():
            score = 1000
        elif reference_phrase == stem_phrase:
            score = 950
        elif reference_compact == stem_compact:
            score = 900
        elif reference_phrase in stem_phrase:
            score = 700 + len(reference_compact)
        elif reference_compact and reference_compact in stem_compact:
            score = 650 + len(reference_compact)
        elif all(word in stem_words for word in reference_words):
            score = 500 + sum(len(word) for word in reference_words)

        if score:
            scored.append((score, filename))

    if not scored:
        return None

    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def resolve_summary_request(
    question: str,
    filenames: Iterable[str],
    current_filename: Optional[str] = None,
) -> Optional[str]:
    """Resolve the document named in a summary request.

    Generic requests such as ``"summarize this document"`` use the document
    currently shown in the summary card. If there is only one uploaded PDF,
    that PDF is the natural fallback.
    """
    candidates = [name for name in filenames if name.lower().endswith(_DOC_EXTENSIONS)]
    resolved = resolve_pdf_reference(question, candidates)
    if resolved:
        return resolved

    if not _reference_words(question):
        if current_filename in candidates:
            return current_filename
        if len(candidates) == 1:
            return candidates[0]
    return None