"""Migrate local document names to the authoritative Amazon S3 names.

This command never uploads, downloads, renames, deletes, re-parses, or
re-embeds documents.  It renames only an already-present local cache file and
enriches the payload of its existing Qdrant points in place, preserving vector
IDs and embeddings.  It is safe to run repeatedly.

Run: ``python -m scripts.migrate_canonical_names [--dry-run]``.
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from services.canonical_naming import canonical_display_name
from services.qdrant_db import QdrantService
from services.s3_storage import S3Storage
from services import name_mapping

logger = logging.getLogger(__name__)


def _mapping(folder: str) -> dict[str, str]:
    """Read the legacy side table solely to identify old local aliases."""
    return name_mapping._load(folder)  # compatibility data; no new entries are written


def migrate(dry_run: bool = False) -> list[tuple[str, str]]:
    if not settings.S3_BUCKET_NAME:
        raise RuntimeError("S3_BUCKET_NAME is required: S3 is the migration source of truth.")
    storage = S3Storage(settings.S3_BUCKET_NAME, settings.S3_PREFIX, settings.AWS_REGION,
                        settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY)
    qdrant = QdrantService(settings.QDRANT_URL, settings.QDRANT_COLLECTION_NAME, settings.QDRANT_API_KEY)
    aliases = _mapping(settings.PDF_FOLDER)
    changed: list[tuple[str, str]] = []

    for obj in storage.list_objects():
        filename, key = obj["filename"], obj["key"]
        local_dir = os.path.join(settings.PDF_FOLDER, obj["folder_name"]) if obj["folder_name"] else settings.PDF_FOLDER
        target = os.path.join(local_dir, filename)
        # Existing exact mirror is a no-op; still enrich vector metadata.
        old_name = filename
        if not os.path.isfile(target):
            aliases_for_key = [local for local, s3_name in aliases.items() if s3_name == filename]
            candidates = [name for name in aliases_for_key if os.path.isfile(os.path.join(local_dir, name))]
            if len(candidates) != 1:
                # Do not guess between same-content/duplicate-looking files.
                logger.warning("No safe local match for S3 object '%s'; skipped.", key)
                continue
            old_name = candidates[0]
            old_path = os.path.join(local_dir, old_name)
            if dry_run:
                changed.append((old_name, filename))
                continue
            os.makedirs(local_dir, exist_ok=True)
            os.replace(old_path, target)
            changed.append((old_name, filename))
            # Existing points retain IDs/vectors; only their identifying payload changes.
            qdrant.rename_document(old_name, filename)
            name_mapping.remove(settings.PDF_FOLDER, old_name)

        if not dry_run:
            qdrant.enrich_document_metadata(
                filename, canonical_display_name(filename), key, obj["folder_name"], target, obj["upload_date"]
            )

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    renamed = migrate(args.dry_run)
    logger.info("%s %d local file(s).", "Would rename" if args.dry_run else "Renamed", len(renamed))
    for old, new in renamed:
        logger.info("%s -> %s", old, new)


if __name__ == "__main__":
    main()
