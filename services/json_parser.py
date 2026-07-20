"""
JSON document (.json) parser.

Extracts content and structure from JSON files into human-readable PageContent objects.
"""

import json
import logging
import os
from typing import Any, List

from services.pdf_parser import PageContent

logger = logging.getLogger(__name__)


class JSONParser:
    """Extracts content from JSON (.json) files."""

    def __init__(self, folder_path: str = ""):
        self.folder_path = folder_path

    def _format_json_val(self, val: Any, level: int = 0) -> str:
        indent = "  " * level
        lines = []
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{indent}{k}:")
                    lines.append(self._format_json_val(v, level + 1))
                else:
                    lines.append(f"{indent}{k}: {v}")
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, (dict, list)):
                    lines.append(self._format_json_val(item, level + 1))
                else:
                    lines.append(f"{indent}- {item}")
        else:
            lines.append(f"{indent}{val}")
        return "\n".join(lines)

    def extract_pages(self, file_path: str) -> List[PageContent]:
        """Read a JSON file and return PageContent."""
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            formatted_text = self._format_json_val(data)
        except Exception as exc:
            logger.warning("JSON parsing failed for '%s' (%s), falling back to raw text read.", file_path, exc)
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    formatted_text = f.read()
            except Exception as e:
                logger.error("Failed to read JSON file '%s': %s", file_path, e)
                return []

        return [
            PageContent(
                filename=filename,
                page_number=1,
                page_label="1",
                text=formatted_text.strip(),
            )
        ]
