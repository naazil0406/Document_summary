"""
Markdown document (.md) parser.

Extracts content and structure from Markdown files into PageContent objects
so they flow seamlessly through the semantic boundary detector, chunker,
embedding, and Qdrant storage pipeline.
"""

import logging
import os
from typing import List

from services.pdf_parser import PageContent

logger = logging.getLogger(__name__)


class MarkdownParser:
    """Extracts content from Markdown (.md) files."""

    def __init__(self, folder_path: str = ""):
        self.folder_path = folder_path

    def extract_pages(self, file_path: str) -> List[PageContent]:
        """Read a Markdown file and return PageContent."""
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as exc:
            logger.error("Failed to read Markdown file '%s': %s", file_path, exc)
            return []

        return [
            PageContent(
                filename=filename,
                page_number=1,
                page_label="1",
                text=content.strip(),
            )
        ]
