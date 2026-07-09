"""Helpers for resolving casual document references to indexed document names.

Documents are canonically named ``"Unit N - 123456.ext"`` (or ``"Misc -
123456.ext"`` when no unit number applies) -- see
``services/canonical_naming.py``. Resolution now happens in three tiers:

  1. An explicit 6-digit unique id anywhere in the reference (e.g. "Unit 20
     - 100235", or just "100235") always resolves unambiguously -- it's the
     one thing guaranteed unique per document.
  2. A bare unit number ("unit 1", "Unit_5") is matched against every
     candidate's parsed unit number. Exactly one match resolves; two or
     more is a genuine tie and is left to the caller to disambiguate (see
     `ambiguous_candidates`) instead of guessing.
  3. Legacy fuzzy word matching against the filename stem, applied only to
     filenames that aren't (yet) in canonical form -- e.g. immediately
     after an upload and before a rename step has run, or before the
     canonical-naming migration has been applied to older documents.

Note: because canonical names carry only "Unit N" and a unique id, a
descriptive reference like "the safe start workbook" has nothing to match
once a file is fully canonical -- the unit number or unique id is the
supported way to refer to a specific document going forward.
"""

import os
import re
from typing import Iterable, List, Optional

from services.canonical_naming import extract_unit_number, parse_canonical


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

_UNIQUE_ID_PATTERN = re.compile(r"\b(\d{6})\b")
_PART_PATTERN = re.compile(r"\bpart\s*0*(\d+)\b", re.IGNORECASE)


def _words(value: str) -> list[str]:
    value = re.sub(r"\.(?:pdf|docx|xlsx|xlsm|xls|csv)\b", " ", value.lower())
    return re.findall(r"[a-z0-9]+", value)


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


def _file_unit(filename: str) -> Optional[str]:
    """Unit number for a filename, whether it's already canonical or not."""
    info = parse_canonical(filename)
    if info is not None:
        return info.unit
    return extract_unit_number(os.path.splitext(filename)[0])


def _candidates_matching_unit(unit: str, filenames: Iterable[str]) -> List[str]:
    return [f for f in filenames if _file_unit(f) == unit]


def _legacy_fuzzy_match(reference: str, reference_words: list[str], filenames: List[str]) -> Optional[str]:
    """Fuzzy word-overlap matching, applied only to filenames that aren't
    (yet) in canonical form. Preserved for backward compatibility with
    documents uploaded before a rename step runs."""
    if not reference_words or not filenames:
        return None

    reference_phrase = " ".join(reference_words)
    reference_compact = "".join(reference_words)
    ref_unit = extract_unit_number(reference)
    ref_part = _part_number(reference)

    scored = []
    for filename in filenames:
        stem = os.path.splitext(filename)[0]
        stem_words = _words(stem)
        stem_phrase = " ".join(stem_words)
        stem_compact = "".join(stem_words)

        file_unit = extract_unit_number(stem)
        file_part = _part_number(stem)

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

        if ref_unit is not None and file_unit == ref_unit:
            score += 50
            if ref_part is None and file_part is not None:
                score -= 20
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

    Ambiguous references deliberately return ``None`` -- see
    `ambiguous_candidates` for listing the specific documents tied on a
    shared unit number so the caller can ask the user to pick by unique id.
    """
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

    # Tier 2: a bare unit number.
    ref_unit = extract_unit_number(reference)
    if ref_unit is not None:
        matches = _candidates_matching_unit(ref_unit, candidates)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None  # genuine tie -- see ambiguous_candidates()
        # No canonical/raw file carries this unit number -- fall through.

    # Tier 3: legacy fuzzy matching, only against filenames not yet canonical.
    reference_words = _reference_words(reference)
    if not reference_words:
        return None
    non_canonical = [f for f in candidates if parse_canonical(f) is None]
    return _legacy_fuzzy_match(reference, reference_words, non_canonical)


def ambiguous_candidates(reference: str, filenames: Iterable[str]) -> List[str]:
    """If `reference` names a unit number shared by 2+ documents, return
    those documents so the caller can present their unique ids for the user
    to pick from. Returns an empty list when there's no genuine tie."""
    candidates = [name for name in filenames if name.lower().endswith(_DOC_EXTENSIONS)]
    ref_unit = extract_unit_number(reference)
    if ref_unit is None:
        return []
    matches = _candidates_matching_unit(ref_unit, candidates)
    return matches if len(matches) > 1 else []


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