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


_DOC_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv")

_UNIT_PATTERN = re.compile(r"\bunit\s*0*(\d+)\b", re.IGNORECASE)
_PART_PATTERN = re.compile(r"\bpart\s*0*(\d+)\b", re.IGNORECASE)


def _words(value: str) -> list[str]:
    value = re.sub(r"\.(?:pdf|docx|xlsx|xlsm|xls|csv)\b", " ", value.lower())
    return re.findall(r"[a-z0-9]+", value)


def _compact(value: str) -> str:
    return "".join(_words(value))


def _reference_words(value: str) -> list[str]:
    return [word for word in _words(value) if word not in _REQUEST_WORDS]


def _unit_number(value: str) -> Optional[str]:
    match = _UNIT_PATTERN.search(value)
    return match.group(1) if match else None


def _part_number(value: str) -> Optional[str]:
    match = _PART_PATTERN.search(value)
    return match.group(1) if match else None


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
    ref_unit = _unit_number(reference)
    ref_part = _part_number(reference)

    scored = []

    for filename in candidates:
        stem = os.path.splitext(filename)[0]
        stem_words = _words(stem)
        stem_phrase = " ".join(stem_words)
        stem_compact = "".join(stem_words)

        file_unit = _unit_number(stem)
        file_part = _part_number(stem)

        # If the reference names a specific unit number, filenames with a
        # *different* unit number (or no unit number at all, when others do
        # match) are disqualified outright rather than fuzzy-scored.
        if ref_unit is not None and file_unit is not None and file_unit != ref_unit:
            continue

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

        if not score:
            continue

        # Bonus for exact unit-number match; this is what breaks the
        # "Unit 1" vs "Unit 1 Part 1" tie in favor of the plain unit file
        # when the reference itself has no part number.
        if ref_unit is not None and file_unit == ref_unit:
            score += 50
            # Penalize files that carry an *extra* part number the
            # reference didn't ask for — they're more specific than what
            # was requested, so they shouldn't outrank a bare unit match.
            if ref_part is None and file_part is not None:
                score -= 20
            # Reward exact part match when the reference asked for one.
            if ref_part is not None and file_part == ref_part:
                score += 30

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