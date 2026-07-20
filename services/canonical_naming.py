"""Canonical document naming: ``"<original stem> - 123456.ext"``.

Every document gets a stable, unambiguous name at ingestion time so that
casual references never collide the way raw uploaded filenames can (e.g.
two different uploads both named "report.pdf").

The unique 6-digit id is a deterministic hash of the *original* filename,
not a random or sequential counter -- so re-running ingestion, or migrating
the same source file twice, always produces the same canonical name instead
of minting a new one.
"""

import hashlib
import os
import re
from datetime import datetime
from typing import NamedTuple, Optional

# Matches the canonical stem itself, e.g. "Quarterly Report - 100235".
_CANONICAL_PATTERN = re.compile(r"^(.+) - (\d{6})$")

_ID_MIN = 100000
_ID_RANGE = 900000  # 100000..999999 inclusive

_MAX_LABEL_LEN = 50


# Generic filler words stripped when picking the one keyword to keep as
# the label, so labels favor the distinctive term (e.g. "Unit1",
# "SafeStart") over generic noise (e.g. "workbook", "the", "new").
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "new",
    "now", "final", "version", "copy", "updated", "draft", "workbook",
    "document", "file", "report", "back", "school",
}
_MAX_LABEL_LEN = 20


class CanonicalInfo(NamedTuple):
    original_name: str  # original filename stem, sanitized
    unique_id: str  # 6-digit string


def _sanitize_stem(stem: str) -> str:
    """Strip characters that are unsafe in filenames (path separators,
    etc.) while keeping the name human-readable."""
    stem = re.sub(r'[\\/:*?"<>|]', "", stem).strip()
    return stem or "Document"


# Short words that typically act as a labeling/numbering scheme in
# filenames (Unit 1, Part 2, Module 3) -- preferred over longer generic
# nouns when picking which word to merge a trailing digit onto.
_LABEL_WORDS = {"unit", "part", "module", "chapter", "section", "lesson", "week", "day", "vol", "volume"}


def _shorten_label(label: str, max_len: int = _MAX_LABEL_LEN) -> str:
    """Reduce a label to a single short, distinctive keyword instead of
    the full original phrase -- e.g. "NOW SafeStart workbook Unit 1
    Introduction" -> "Unit1". The unique id that follows is what keeps the
    full name unambiguous, so shortening here is purely cosmetic:

    1. Split into words, drop generic filler/stopwords.
    2. If a bare number is present, attach it to a nearby labeling word
       (Unit/Part/Module/...) if one exists, else the longest remaining
       distinctive word.
    3. Otherwise prefer a word that already contains a digit, else the
       longest remaining word.
    4. Fall back to the first word if nothing survives filtering.
    """
    words = re.findall(r"[A-Za-z0-9]+", label)
    if not words:
        return "Document"

    non_stop = [w for w in words if w.lower() not in _STOPWORDS]
    pool = non_stop or words

    digits = [w for w in words if w.isdigit()]
    core_words = [w for w in pool if not w.isdigit()]
    if digits and core_words:
        label_word = next((w for w in core_words if w.lower() in _LABEL_WORDS), None)
        if label_word:
            # Use the digit that immediately follows the label word in the
            # ORIGINAL word order (e.g. "Unit 4" -> 4), not just
            # digits[0] -- a leading version number like "CERT 2.0" would
            # otherwise be picked instead of the real unit number,
            # silently mislabeling the file (e.g. "CERT 2.0_Unit 4..."
            # becoming "Unit2" instead of "Unit4").
            try:
                label_idx = words.index(label_word)
            except ValueError:
                label_idx = -1
            if 0 <= label_idx < len(words) - 1 and words[label_idx + 1].isdigit():
                chosen_digit = words[label_idx + 1]
            else:
                chosen_digit = digits[0]
            base = label_word
        else:
            base = max(core_words, key=len)
            chosen_digit = digits[0]
        keyword = base + chosen_digit
        return keyword[:max_len]

    alnum_with_digit = [w for w in pool if w.isalnum() and any(c.isdigit() for c in w) and not w.isdigit()]
    if alnum_with_digit:
        keyword = alnum_with_digit[0]
    else:
        keyword = max(pool, key=len)

    return keyword[:max_len]


def reshorten_canonical(name: str) -> str:
    """Given a name that's *already* canonical (has a " - 123456" id
    suffix from an earlier, pre-shortening run), reapply the current
    label-shortening rules to just the label -- keeping the existing
    unique id untouched.

    This matters because re-running canonical_filename() on an
    already-canonical name would hash the *entire* string (id included)
    and mint a brand-new id, silently orphaning the file's existing
    Qdrant points. Returns `name` unchanged if it isn't canonical, or if
    the label is already fully shortened.
    """
    info = parse_canonical(name)
    if info is None:
        return name
    ext = os.path.splitext(name)[1]
    short_label = _shorten_label(info.original_name)
    return f"{short_label} - {info.unique_id}{ext}"


def unique_id_for(original_name: str) -> str:
    """Deterministic 6-digit id derived from the original filename.

    Using a hash (rather than a random or sequential counter) means the
    same source file always resolves to the same canonical name, even if
    ingestion or the migration script is re-run.
    """
    digest = hashlib.sha256(original_name.encode("utf-8")).hexdigest()
    return str(_ID_MIN + (int(digest, 16) % _ID_RANGE))


def is_canonical(name: str) -> bool:
    """True if `name` is already in canonical "<name> - 123456" form
    (extension ignored)."""
    stem = os.path.splitext(name)[0]
    return bool(_CANONICAL_PATTERN.match(stem.strip()))


def parse_canonical(name: str) -> Optional[CanonicalInfo]:
    """Return (original_name, unique_id) if `name` is canonical, else None."""
    stem = os.path.splitext(name)[0]
    match = _CANONICAL_PATTERN.match(stem.strip())
    if not match:
        return None
    original_name, uid = match.group(1), match.group(2)
    return CanonicalInfo(original_name=original_name, unique_id=uid)


def current_month_folder(dt: Optional[datetime] = None) -> str:
    """Lowercase month-name folder for the given date (today by default),
    e.g. "july" -- matches the bucket's existing monthly-folder convention.
    Used to place new uploads (and their local mirror) in the right
    month's folder without any manual per-month config."""
    return (dt or datetime.now()).strftime("%B").lower()


def canonical_filename(original_name: str) -> str:
    """Return the canonical storage/display name for a source filename.

    Idempotent: if `original_name` is already canonical, it is returned
    unchanged (so re-running this on an already-migrated file is a no-op).
    """
    if is_canonical(original_name):
        return original_name

    stem, ext = os.path.splitext(original_name)
    uid = unique_id_for(original_name)
    label = _shorten_label(_sanitize_stem(stem))
    return f"{label} - {uid}{ext}"