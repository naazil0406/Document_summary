"""
XML document (.xml) parser.

Extracts content and structure from XML files into PageContent objects.
"""

import logging
import os
import xml.etree.ElementTree as ET
from typing import List

from services.pdf_parser import PageContent

logger = logging.getLogger(__name__)


class XMLParser:
    """Extracts content from XML (.xml) files."""

    def __init__(self, folder_path: str = ""):
        self.folder_path = folder_path

    def _element_to_text(self, elem: ET.Element, level: int = 0) -> str:
        lines = []
        indent = "  " * level
        tag = elem.tag.strip()
        text = (elem.text or "").strip()
        tail = (elem.tail or "").strip()
        attrs = " ".join([f'{k}="{v}"' for k, v in elem.attrib.items()])
        attr_str = f" ({attrs})" if attrs else ""

        if text:
            lines.append(f"{indent}{tag}{attr_str}: {text}")
        elif tag:
            lines.append(f"{indent}{tag}{attr_str}")

        for child in elem:
            lines.append(self._element_to_text(child, level + 1))

        if tail:
            lines.append(f"{indent}{tail}")

        return "\n".join(filter(None, lines))

    def extract_pages(self, file_path: str) -> List[PageContent]:
        """Read an XML file and return PageContent."""
        filename = os.path.basename(file_path)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            text_content = self._element_to_text(root)
        except Exception as exc:
            logger.warning("XML parsing failed for '%s' (%s), falling back to raw text read.", file_path, exc)
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text_content = f.read()
            except Exception as e:
                logger.error("Failed to read XML file '%s': %s", file_path, e)
                return []

        return [
            PageContent(
                filename=filename,
                page_number=1,
                page_label="1",
                text=text_content.strip(),
            )
        ]
