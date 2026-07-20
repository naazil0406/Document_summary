"""Helpers for resolving casual document references to indexed document names.

Documents are canonically named ``"<original name> - 123456.ext"`` -- see
``services/canonical_naming.py``. Resolution happens in two tiers:

  1. An explicit 6-digit unique id anywhere in the reference (e.g.
     "100235", or "the report - 100235") always resolves unambiguously --
     it's the one thing guaranteed unique per document.
  2. Legacy fuzzy word matching against the filename stem, applied to
     every candidate (canonical or not) since the canonical stem now
     carries the full original name rather than just a unit number.

Note: because a reference like "the safe start workbook" is matched
against the *original name* portion of the canonical filename, it
continues to work after a file has been renamed to canonical form -- the
original name isn't discarded the way a bare unit number would be.
"""

import os
import re
from typing import Iterable, List, Optional

from services.canonical_naming import parse_canonical


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


_DOC_EXTENSIONS = (
    ".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv",
    ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
)

_UNIQUE_ID_PATTERN = re.compile(r"\b(\d{6})\b")
_PART_PATTERN = re.compile(r"\bpart\s*0*(\d+)\b", re.IGNORECASE)

# Mirrors canonical_naming._LABEL_WORDS: words that typically act as a
# labeling/numbering scheme (Unit 4, Part 2, ...). canonical_filename()
# merges these onto an adjacent number when building the short label
# (e.g. "Unit 4" -> "Unit4"), so a reference has to be merged the same
# way here or "unit"/"4" as separate tokens will never line up with the
# single "unit4" token stored in the canonical name.
_LABEL_WORDS = {"unit", "part", "module", "chapter", "section", "lesson", "week", "day", "vol", "volume"}


def _words(value: str) -> list[str]:
    value = re.sub(
        r"\.(?:pdf|docx|xlsx|xlsm|xls|csv|pptx|png|jpe?g|webp|bmp|tiff)\b",
        " ",
        value.lower(),
    )
    tokens = re.findall(r"[a-z0-9]+", value)

    # Add a merged "unit4"-style token wherever a label word is
    # immediately followed by a bare number, without discarding the
    # original separate tokens (so exact phrase/compact comparisons
    # elsewhere still see the untouched word sequence too).
    merged = []
    for i, token in enumerate(tokens):
        merged.append(token)
        if token in _LABEL_WORDS and i + 1 < len(tokens) and tokens[i + 1].isdigit():
            merged.append(token + tokens[i + 1])
    return merged


def _compact(value: str) -> str:
    return "".join(_words(value))


def _reference_words(value: str) -> list[str]:
    return [word for word in _words(value) if word not in _REQUEST_WORDS]


def _reference_unique_id(reference: str) -> Optional[str]:
    match = _UNIQUE_ID_PATTERN.search(reference)
    return match.group(1) if match else None


def _part_number(value: str) -> Optional[str]:
    match = _PART_PATTERN.search(value)
    return match.group(1) if match else None


def _legacy_fuzzy_match(reference: str, reference_words: list[str], filenames: List[str]) -> Optional[str]:
    """Fuzzy word-overlap matching against every candidate's filename stem
    (canonical or not) -- for canonical names this matches against the
    original-name portion carried in "<original name> - 123456"."""
    if not reference_words or not filenames:
        return None

    reference_phrase = " ".join(reference_words)
    reference_compact = "".join(reference_words)
    ref_part = _part_number(reference)

    scored = []
    for filename in filenames:
        stem = os.path.splitext(filename)[0]
        info = parse_canonical(filename)
        # Match against just the original-name portion when canonical,
        # so a stale unique id elsewhere in the reference doesn't dilute
        # word overlap scoring.
        match_stem = info.original_name if info else stem

        stem_words = _words(match_stem)
        stem_phrase = " ".join(stem_words)
        stem_compact = "".join(stem_words)

        file_part = _part_number(stem)

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
        else:
            # Canonical labels are shortened to a single keyword (e.g. the
            # original "CERT 2.0_Unit 4 Master Sheet" becomes just
            # "Unit4"), so a query word like "mastersheet" can be lost
            # entirely even for the obviously-correct file. Give partial
            # credit for whatever words DO overlap, so a query still
            # resolves as long as no other candidate matches as well or
            # better (handled by the tie-break below).
            overlap_words = [word for word in reference_words if word in stem_words]
            if overlap_words:
                score = 200 + sum(len(word) for word in overlap_words)

        if not score:
            continue

        if ref_part is not None and file_part == ref_part:
            score += 30

        scored.append((score, filename))

    if not scored:
        return None

    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def resolve_pdf_reference(reference: str, filenames: Iterable[str]) -> Optional[str]:
    """Return the unique best document matching a full or abbreviated reference.

    Ambiguous references deliberately return ``None`` so the caller can
    ask the user to pick by unique id."""
    candidates = [name for name in filenames if name.lower().endswith(_DOC_EXTENSIONS)]
    if not candidates:
        return None

    # Tier 1: an explicit unique id is the most specific thing a reference
    # can contain, and is guaranteed unique per document.
    ref_id = _reference_unique_id(reference)
    if ref_id:
        for filename in candidates:
            info = parse_canonical(filename)
            if info and info.unique_id == ref_id:
                return filename
        # A specific id was given but matched nothing -- don't guess further.
        return None

    # Tier 2: fuzzy word matching against every candidate's (original) name.
    reference_words = _reference_words(reference)
    if not reference_words:
        return None
    return _legacy_fuzzy_match(reference, reference_words, candidates)


def ambiguous_candidates(reference: str, filenames: Iterable[str]) -> List[str]:
    """Kept for backward compatibility with callers. Since resolution no
    longer groups documents by a shared unit number, there is no longer a
    structural "genuine tie" case to detect here -- ties are already
    handled (and reported as unresolved) inside resolve_pdf_reference /
    _legacy_fuzzy_match. Always returns an empty list."""
    return []


def resolve_summary_request(
    question: str,
    filenames: Iterable[str],
    current_filename: Optional[str] = None,
) -> Optional[str]:
    """Resolve the document named in a summary request.

    Generic requests such as ``"summarize this document"`` use the document
    currently shown in the summary card. If there is only one uploaded
    document, that document is the natural fallback.
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