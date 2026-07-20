"""Human-readable display names for documents.

Storage names are never generated or changed here.  The S3 object key is
authoritative and its basename is also the local cache filename.  This module
only derives the ``canonical_name`` metadata shown to people in the UI.
"""

import hashlib
import os
import re
from datetime import datetime
from typing import NamedTuple, Optional


_NOISE = {
    "final", "draft", "copy", "updated", "update", "version", "ver",
    "document", "file", "v", "rev",
}

_STOPWORDS = _NOISE | {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on",
    "workbook", "report", "school", "back", "now",
}
_LABEL_WORDS = {"unit", "part", "module", "chapter", "section", "lesson", "week", "day", "vol", "volume"}

_LEGACY_CANONICAL_PATTERN = re.compile(r"^(.+) - (\d{6})$")


class CanonicalInfo(NamedTuple):
    """Compatibility shape for records produced by the retired scheme."""
    original_name: str
    unique_id: str


def parse_canonical(name: str) -> Optional[CanonicalInfo]:
    """Read an old generated filename without generating or renaming one."""
    stem = os.path.splitext(os.path.basename(name))[0].strip()
    match = _LEGACY_CANONICAL_PATTERN.match(stem)
    if not match:
        return None
    return CanonicalInfo(match.group(1), match.group(2))


def canonical_display_name(original_filename: str) -> str:
    """Create a concise label without ever using it as a filename.

    The rules intentionally remove common upload suffixes such as ``Final``
    and version numbers.  A small safety-topic refinement keeps long safety
    filenames readable (``How_to_Prevent_Heat_Stress...`` -> ``Heat Stress
    Prevention``) while all other documents retain their meaningful title.
    """
    # Old canonical storage aliases already carry their stable id. Preserve
    # it for display during migration rather than deriving a second one.
    legacy = parse_canonical(original_filename)
    if legacy:
        return f"{legacy.original_name} - {legacy.unique_id}"

    stem = os.path.splitext(os.path.basename(original_filename))[0]
    words = re.findall(r"[A-Za-z0-9]+", stem.replace("_", " "))
    # Strip explicit version suffixes (v3 / rev2) but keep meaningful
    # numbering such as the ``1`` in "Unit 1".
    while words and (words[-1].lower() in _NOISE or re.fullmatch(r"v\d+|rev\d+", words[-1], re.I)):
        words.pop()
    if not words:
        return f"Document - {unique_id_for(original_filename)}"
    lower = [word.lower() for word in words]
    if "heat" in lower and "stress" in lower and any(word in lower for word in ("prevent", "prevention")):
        label = "Heat Stress Prevention"
        return f"{label} - {unique_id_for(original_filename)}"
    # Leading imperative boilerplate is rarely useful in a display title.
    if lower[:3] == ["how", "to", "prevent"]:
        words = words[3:] + ["Prevention"]
    # Retain the concise, single-keyword convention from the existing UI
    # (e.g. "Unit 1" -> "Unit1") and append a deterministic ID so same
    # label documents are never ambiguous.
    meaningful = [word for word in words if word.lower() not in _STOPWORDS] or words
    digits = [word for word in words if word.isdigit()]
    label_word = next((word for word in meaningful if word.lower() in _LABEL_WORDS), None)
    if label_word and digits:
        label_index = words.index(label_word)
        following = words[label_index + 1] if label_index + 1 < len(words) else ""
        label = f"{label_word}{following if following.isdigit() else digits[0]}"
    elif len(meaningful) <= 4:
        label = " ".join(meaningful)
    else:
        label = max(meaningful, key=len)
    return f"{label.strip() or 'Document'} - {unique_id_for(original_filename)}"


def unique_id_for(original_filename: str) -> str:
    """Stable six-digit UI ID; it is never used in a storage path."""
    digest = hashlib.sha256(original_filename.encode("utf-8")).hexdigest()
    return str(100000 + (int(digest, 16) % 900000))


def canonical_filename(original_name: str) -> str:
    """Backward-compatible alias. Storage identity is the original name."""
    return original_name


def is_canonical(name: str) -> bool:
    """Legacy compatibility; generated canonical storage filenames are obsolete."""
    return False


def reshorten_canonical(name: str) -> str:
    """Legacy compatibility; filenames must no longer be transformed."""
    return name


def current_month_folder(dt: Optional[datetime] = None) -> str:
    return (dt or datetime.now()).strftime("%B").lower()
