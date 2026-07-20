"""Local mapping between the display/local filename and the *actual*
S3 object key.

Canonical renaming (see ``services/canonical_naming.py``) is only ever
applied locally and in the UI now -- S3 object keys are never renamed or
uploaded under the canonical name, so the bucket keeps whatever name the
file arrived under.

That means the local/display name and the S3 key can differ, so this
module persists a small JSON side-table (``<PDF_FOLDER>/.s3_name_map.json``)
recording, for every local file that isn't stored in S3 under its own
name, what key it actually lives at. Anything not present in the map is
assumed to use its own name as the S3 key (the common case).
"""

import json
import logging
import os
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_MAP_FILENAME = ".s3_name_map.json"
_lock = threading.Lock()


def _map_path(pdf_folder: str) -> str:
    return os.path.join(pdf_folder, _MAP_FILENAME)


def _load(pdf_folder: str) -> Dict[str, str]:
    path = _map_path(pdf_folder)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read S3 name map at '%s': %s", path, exc)
        return {}


def _save(pdf_folder: str, mapping: Dict[str, str]) -> None:
    os.makedirs(pdf_folder, exist_ok=True)
    path = _map_path(pdf_folder)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def set_s3_key(pdf_folder: str, local_name: str, s3_key_name: str) -> None:
    """Record that `local_name` (the local/display filename) is stored in
    S3 under `s3_key_name` (the basename actually used as the S3 key).

    No-op (and clears any stale entry) if they're already identical, since
    that's the default assumption when nothing is in the map.
    """
    with _lock:
        mapping = _load(pdf_folder)
        if local_name == s3_key_name:
            mapping.pop(local_name, None)
        else:
            mapping[local_name] = s3_key_name
        _save(pdf_folder, mapping)


def get_s3_key(pdf_folder: str, local_name: str) -> str:
    """Return the S3 key basename for `local_name`, falling back to
    `local_name` itself if there's no recorded mapping."""
    with _lock:
        mapping = _load(pdf_folder)
    return mapping.get(local_name, local_name)


def rename_local(pdf_folder: str, old_local_name: str, new_local_name: str) -> None:
    """Update the map when a local/display name changes (e.g. the
    canonical-rename step) without the underlying S3 key changing."""
    with _lock:
        mapping = _load(pdf_folder)
        s3_key = mapping.pop(old_local_name, old_local_name)
        if new_local_name == s3_key:
            mapping.pop(new_local_name, None)
        else:
            mapping[new_local_name] = s3_key
        _save(pdf_folder, mapping)


def remove(pdf_folder: str, local_name: str) -> None:
    with _lock:
        mapping = _load(pdf_folder)
        if mapping.pop(local_name, None) is not None:
            _save(pdf_folder, mapping)
            