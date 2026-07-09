"""Canonical document naming: ``"Unit N - 123456.ext"`` (or ``"Misc -
123456.ext"`` when no unit number is found in the original filename).

Every document gets a stable, unambiguous name at ingestion time so that
casual references like "Unit 1" never collide the way raw uploaded
filenames can (e.g. "...Unit 1_09122023.pdf" vs "...Unit 1 Part 1.xlsx").

The unique 6-digit id is a deterministic hash of the *original* filename,
not a random or sequential counter -- so re-running ingestion, or migrating
the same source file twice, always produces the same canonical name instead
of minting a new one.
"""

import hashlib
import os
import re
from typing import NamedTuple, Optional

_UNIT_PATTERN = re.compile(r"(?<![a-zA-Z])unit[\s_]*0*(\d+)(?!\d)", re.IGNORECASE)

# Matches the canonical stem itself, e.g. "Unit 20 - 100235" or "Misc - 483920".
_CANONICAL_PATTERN = re.compile(r"^(?:Unit (\d+)|Misc) - (\d{6})$", re.IGNORECASE)

_ID_MIN = 100000
_ID_RANGE = 900000  # 100000..999999 inclusive


class CanonicalInfo(NamedTuple):
    unit: Optional[str]  # unit number as a string, or None for "Misc"
    unique_id: str  # 6-digit string


def extract_unit_number(name: str) -> Optional[str]:
    """Pull a unit number out of a raw (non-canonical) filename or reference."""
    match = _UNIT_PATTERN.search(name)
    return match.group(1) if match else None


def unique_id_for(original_name: str) -> str:
    """Deterministic 6-digit id derived from the original filename.

    Using a hash (rather than a random or sequential counter) means the
    same source file always resolves to the same canonical name, even if
    ingestion or the migration script is re-run.
    """
    digest = hashlib.sha256(original_name.encode("utf-8")).hexdigest()
    return str(_ID_MIN + (int(digest, 16) % _ID_RANGE))


def is_canonical(name: str) -> bool:
    """True if `name` is already in canonical "Unit N - 123456" / "Misc -
    123456" form (extension ignored)."""
    stem = os.path.splitext(name)[0]
    return bool(_CANONICAL_PATTERN.match(stem.strip()))


def parse_canonical(name: str) -> Optional[CanonicalInfo]:
    """Return (unit, unique_id) if `name` is canonical, else None."""
    stem = os.path.splitext(name)[0]
    match = _CANONICAL_PATTERN.match(stem.strip())
    if not match:
        return None
    unit, uid = match.group(1), match.group(2)
    return CanonicalInfo(unit=unit, unique_id=uid)


def canonical_filename(original_name: str) -> str:
    """Return the canonical storage/display name for a source filename.

    Idempotent: if `original_name` is already canonical, it is returned
    unchanged (so re-running this on an already-migrated file is a no-op).
    """
    if is_canonical(original_name):
        return original_name

    stem, ext = os.path.splitext(original_name)
    unit = extract_unit_number(stem)
    uid = unique_id_for(original_name)
    label = f"Unit {unit}" if unit else "Misc"
    return f"{label} - {uid}{ext}"