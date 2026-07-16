"""
One-off migration: rename every already-indexed document to canonical
"Unit N - 123456.ext" / "Misc - 123456.ext" form.

This does NOT re-parse, re-chunk, or re-embed anything, and it does NOT
touch S3 -- it only:
  1. renames the local file in PDF_FOLDER (flat, or inside a month
     subfolder, e.g. PDF_FOLDER/july/<file>)
  2. updates the `filename` payload field on the document's existing
     Qdrant points (chunks + TOC entries) in place, so the UI's
     "Indexed Documents" list shows the new short name

S3 object keys are left exactly as they are.

Safe to re-run: documents already in fully-shortened canonical form are
skipped, and the canonical name for a given source file is always the
same (deterministic hash), so re-running this never produces a different
name for the same file.

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
from services.canonical_naming import canonical_filename, is_canonical, reshorten_canonical

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (
    ".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".pptx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
)


def _iter_documents(folder: str):
    """Yield (full_path, filename) for every supported document directly
    in `folder` (legacy flat layout) or one level down inside a month
    subfolder (e.g. folder/july/<file>, current layout)."""
    if not os.path.isdir(folder):
        return
    for entry in sorted(os.listdir(folder)):
        full = os.path.join(folder, entry)
        if os.path.isfile(full):
            if os.path.splitext(entry)[1].lower() in SUPPORTED_EXTENSIONS:
                yield full, entry
        elif os.path.isdir(full):
            for sub_entry in sorted(os.listdir(full)):
                sub_full = os.path.join(full, sub_entry)
                if os.path.isfile(sub_full) and os.path.splitext(sub_entry)[1].lower() in SUPPORTED_EXTENSIONS:
                    yield sub_full, sub_entry


def migrate(dry_run: bool = False) -> list[tuple[str, str]]:
    """Rename every local document (and its Qdrant `filename` payload) to
    canonical form. S3 is never touched. Handles two cases:

      - not yet canonical: assign a fresh "<label> - 123456.ext" name
      - already canonical but with an old, un-shortened label: reshorten
        just the label, keeping the existing unique id (and therefore
        the existing Qdrant points) intact

    Returns the list of (old_name, new_name) pairs that were renamed (or
    would be, if dry_run).
    """

    folder = settings.PDF_FOLDER
    if not os.path.isdir(folder):
        logger.warning("PDF_FOLDER '%s' does not exist; nothing to migrate.", folder)
        return []

    qdrant_service = QdrantService(
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        api_key=settings.QDRANT_API_KEY,
    )

    renamed: list[tuple[str, str]] = []

    for old_path, old_name in _iter_documents(folder):
        if is_canonical(old_name):
            new_name = reshorten_canonical(old_name)
            if new_name == old_name:
                logger.info("Skipping '%s' (already fully shortened).", old_name)
                continue
        else:
            new_name = canonical_filename(old_name)

        new_path = os.path.join(os.path.dirname(old_path), new_name)

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

            count = qdrant_service.rename_document(old_name, new_name)
            logger.info(
                "Re-keyed %d Qdrant point(s) for '%s' -> '%s' (UI will show the new name).",
                count, old_name, new_name,
            )

            renamed.append((old_name, new_name))
        except Exception as exc:
            logger.error(
                "Failed to fully migrate '%s' -> '%s': %s. "
                "Check local/Qdrant state manually before re-running.",
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