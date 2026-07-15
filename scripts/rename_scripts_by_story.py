"""
One-off: rename already-saved video scripts (e.g. "Transcript_2_script.txt",
"All_Documents_script.txt") to the invented character's name pulled from
their own content -- the same auto-naming new scripts get from their
"Story - <name>.mp4" header line (see prompts/presentation_prompt.txt).

Scripts whose content has no parseable "Story - <name>.mp4" line, or that
are already named correctly, are left untouched. Safe to re-run.

Run with:
    python -m scripts.rename_scripts_by_story
    python -m scripts.rename_scripts_by_story --dry-run
"""

import argparse
import logging
import os
import re
import sys
from typing import List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

logger = logging.getLogger(__name__)

_STORY_HEADER_PATTERN = re.compile(r"^Story\s*-\s*(.+?)\.mp4\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_story_name(script_text: str) -> Optional[str]:
    match = _STORY_HEADER_PATTERN.search(script_text)
    if not match:
        return None
    name = match.group(1).strip()
    return name or None


def rename_scripts(dry_run: bool = False) -> List[Tuple[str, str]]:
    folder = settings.NARRATIVE_SCRIPTS_DIR
    if not os.path.isdir(folder):
        logger.warning("NARRATIVE_SCRIPTS_DIR '%s' does not exist; nothing to rename.", folder)
        return []

    renamed: List[Tuple[str, str]] = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith("_script.txt"):
            continue
        old_path = os.path.join(folder, filename)
        try:
            with open(old_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as exc:
            logger.warning("Could not read '%s': %s", filename, exc)
            continue

        story_name = _extract_story_name(content)
        if not story_name:
            logger.info("Skipping '%s' (no 'Story - <name>.mp4' header found).", filename)
            continue

        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", story_name).strip("_") or "script"
        new_name = f"{safe_label}_script.txt"
        if new_name == filename:
            logger.info("Skipping '%s' (already matches its story name).", filename)
            continue

        new_path = os.path.join(folder, new_name)
        if os.path.exists(new_path):
            counter = 2
            while os.path.exists(new_path):
                new_name = f"{safe_label}_script_{counter}.txt"
                new_path = os.path.join(folder, new_name)
                counter += 1

        if dry_run:
            logger.info("[dry-run] Would rename '%s' -> '%s'.", filename, new_name)
            renamed.append((filename, new_name))
            continue

        try:
            os.rename(old_path, new_path)
            logger.info("Renamed '%s' -> '%s'.", filename, new_name)
            renamed.append((filename, new_name))
        except OSError as exc:
            logger.error("Could not rename '%s' -> '%s': %s", filename, new_name, exc)

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

    renamed = rename_scripts(dry_run=args.dry_run)

    if not renamed:
        logger.info("Nothing to rename.")
        return

    logger.info("=" * 60)
    logger.info(
        "%s %d script(s):",
        "Would rename" if args.dry_run else "Renamed",
        len(renamed),
    )
    for old_name, new_name in renamed:
        logger.info("  %s  ->  %s", old_name, new_name)


if __name__ == "__main__":
    main()