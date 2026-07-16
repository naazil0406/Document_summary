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
from typing import NamedTuple, Optional

# Matches the canonical stem itself, e.g. "Quarterly Report - 100235".
_CANONICAL_PATTERN = re.compile(r"^(.+) - (\d{6})$")

_ID_MIN = 100000
_ID_RANGE = 900000  # 100000..999999 inclusive


class CanonicalInfo(NamedTuple):
    original_name: str  # original filename stem, sanitized
    unique_id: str  # 6-digit string


def _sanitize_stem(stem: str) -> str:
    """Strip characters that are unsafe in filenames (path separators,
    etc.) while keeping the name human-readable."""
    stem = re.sub(r'[\\/:*?"<>|]', "", stem).strip()
    return stem or "Document"


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


def canonical_filename(original_name: str) -> str:
    """Return the canonical storage/display name for a source filename.

    Idempotent: if `original_name` is already canonical, it is returned
    unchanged (so re-running this on an already-migrated file is a no-op).
    """
    if is_canonical(original_name):
        return original_name

    stem, ext = os.path.splitext(original_name)
    uid = unique_id_for(original_name)
    label = _sanitize_stem(stem)
    return f"{label} - {uid}{ext}"