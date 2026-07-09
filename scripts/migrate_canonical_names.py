"""
One-off migration: rename every already-indexed document to canonical
"Unit N - 123456.ext" / "Misc - 123456.ext" form.

This does NOT re-parse, re-chunk, or re-embed anything -- it only:
  1. renames the local file in PDF_FOLDER
  2. renames the matching object in S3 (if S3_BUCKET_NAME is configured)
  3. updates the `filename` payload field on the document's existing
     Qdrant points (chunks + TOC entries) in place

Safe to re-run: documents already in canonical form are skipped, and the
canonical name for a given source file is always the same (deterministic
hash), so re-running this never produces a different name for the same file.

Run with:
    python -m scripts.migrate_canonical_names
    python -m scripts.migrate_canonical_names --dry-run
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from services.qdrant_db import QdrantService
from services.s3_storage import S3Storage
from services.canonical_naming import canonical_filename, is_canonical

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv")


def migrate(dry_run: bool = False) -> list[tuple[str, str]]:
    """Rename every non-canonical document in PDF_FOLDER (and S3, and
    Qdrant) to canonical form. Returns the list of (old_name, new_name)
    pairs that were renamed (or would be, if dry_run)."""

    folder = settings.PDF_FOLDER
    if not os.path.isdir(folder):
        logger.warning("PDF_FOLDER '%s' does not exist; nothing to migrate.", folder)
        return []

    s3_storage = None
    if settings.S3_BUCKET_NAME:
        s3_storage = S3Storage(
            bucket_name=settings.S3_BUCKET_NAME,
            prefix=settings.S3_PREFIX,
        )
    else:
        logger.info("S3_BUCKET_NAME not set; migration will only touch local files and Qdrant.")

    qdrant_service = QdrantService(
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        api_key=settings.QDRANT_API_KEY,
    )

    renamed: list[tuple[str, str]] = []

    for old_name in sorted(os.listdir(folder)):
        old_path = os.path.join(folder, old_name)
        if not os.path.isfile(old_path):
            continue
        if os.path.splitext(old_name)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue
        if is_canonical(old_name):
            logger.info("Skipping '%s' (already canonical).", old_name)
            continue

        new_name = canonical_filename(old_name)
        new_path = os.path.join(folder, new_name)

        if os.path.exists(new_path):
            logger.warning(
                "Skipping '%s': target '%s' already exists locally.",
                old_name, new_name,
            )
            continue

        if dry_run:
            logger.info("[dry-run] Would rename '%s' -> '%s'.", old_name, new_name)
            renamed.append((old_name, new_name))
            continue

        try:
            os.rename(old_path, new_path)
            logger.info("Renamed local file '%s' -> '%s'.", old_name, new_name)

            if s3_storage:
                s3_storage.rename_file(old_name, new_name)
                logger.info("Renamed S3 object '%s' -> '%s'.", old_name, new_name)

            count = qdrant_service.rename_document(old_name, new_name)
            logger.info(
                "Re-keyed %d Qdrant point(s) for '%s' -> '%s'.",
                count, old_name, new_name,
            )

            renamed.append((old_name, new_name))
        except Exception as exc:
            logger.error(
                "Failed to fully migrate '%s' -> '%s': %s. "
                "Check local/S3/Qdrant state manually before re-running.",
                old_name, new_name, exc,
            )

    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without changing anything.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    renamed = migrate(dry_run=args.dry_run)

    if not renamed:
        logger.info("Nothing to migrate -- all documents already canonical.")
        return

    logger.info("=" * 60)
    logger.info(
        "%s %d document(s):",
        "Would rename" if args.dry_run else "Renamed",
        len(renamed),
    )
    for old_name, new_name in renamed:
        logger.info("  %s  ->  %s", old_name, new_name)


if __name__ == "__main__":
    main()