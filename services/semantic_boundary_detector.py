"""
Semantic Boundary Detection — Stage between Document Restructuring and
DocumentChunker.

Analyzes a structured document (document_title + sections hierarchy) and
identifies logical semantic boundaries without modifying, summarizing, or
rewriting any content.

Pipeline position:
    Document Restructuring -> SemanticBoundaryDetector -> DocumentChunker
    -> Embedding Generation -> Qdrant
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

ContentItem = Union[str, dict]

_PROTECTED_CONTENT_TYPES = frozenset(
    {
        "procedure",
        "warning",
        "danger",
        "caution",
        "table",
        "list",
        "definition",
        "faq",
        "code_block",
        "image_caption",
        "formula",
    }
)

_WARNING_RE = re.compile(
    r"^(?:warning|danger|caution|danger notice)\s*[:\-]?\s*",
    re.IGNORECASE,
)
_NOTE_RE = re.compile(r"^note\s*[:\-]\s*", re.IGNORECASE)
_FAQ_START_RE = re.compile(
    r"^(?:faq|frequently asked questions?)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
_FAQ_Q_RE = re.compile(r"^(?:q(?:uestion)?|faq)\s*[:\.]?\s+", re.IGNORECASE)
_PROCEDURE_START_RE = re.compile(
    r"^(?:procedure|emergency procedure|steps?)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
_STEP_RE = re.compile(
    r"^step\s+\d+[\.):]\s+",
    re.IGNORECASE,
)
_NUMBERED_LIST_RE = re.compile(r"^\d+[\.)]\s+")
_BULLET_RE = re.compile(r"^[-*•]\s+")
_TABLE_ROW_RE = re.compile(r"\|.+\|")
_DEFINITION_RE = re.compile(
    r"^(?:definition|def\.?)\s*[:\-]\s*",
    re.IGNORECASE,
)
_TERM_DEFINITION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\s\-/]{0,60}:\s+\S")
_IMAGE_CAPTION_RE = re.compile(
    r"^(?:figure|fig\.?|image|photo|illustration)\s+\d+",
    re.IGNORECASE,
)
_FORMULA_RE = re.compile(r"^\$\$.+\$\$$|^[A-Za-z]\s*=\s*.+$")
_APPENDIX_RE = re.compile(r"^appendix\s+[a-z0-9]+", re.IGNORECASE)
_GLOSSARY_RE = re.compile(r"^glossary\s*$", re.IGNORECASE)
_EXAMPLE_RE = re.compile(r"^example\s*[:\-]?\s*", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```")


@dataclass
class SemanticBlock:
    block_id: str
    heading_path: List[str]
    parent_heading: str
    section_id: str
    content: str
    page_start: Optional[int]
    page_end: Optional[int]
    content_type: str
    protected: bool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "heading_path": self.heading_path,
            "parent_heading": self.parent_heading,
            "section_id": self.section_id,
            "content": self.content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "content_type": self.content_type,
            "protected": self.protected,
            "metadata": dict(self.metadata),
        }


class SemanticBoundaryDetector:
    """Identify semantic chunk boundaries in a restructured document."""

    def detect(self, document: dict) -> dict:
        """Return ``{"semantic_blocks": [...]}`` for a structured document."""
        doc_metadata = self._extract_document_metadata(document)
        heading_root = self._document_heading_root(document)
        blocks: List[SemanticBlock] = []
        block_counter = 0

        for section in document.get("sections", []) or []:
            block_counter = self._walk_section(
                section=section,
                heading_path=heading_root,
                parent_heading=heading_root[-1] if heading_root else "",
                doc_metadata=doc_metadata,
                blocks=blocks,
                block_counter=block_counter,
            )

        return {"semantic_blocks": [block.to_dict() for block in blocks]}

    def detect_json(self, document: dict) -> str:
        """Return JSON string of semantic blocks only."""
        return json.dumps(self.detect(document), ensure_ascii=False)

    @staticmethod
    def _document_heading_root(document: dict) -> List[str]:
        title = (document.get("document_title") or "").strip()
        folder = ""
        meta = document.get("metadata") or {}
        if isinstance(meta, dict):
            folder = (meta.get("folder") or "").strip()
        if folder and title:
            return [folder, title]
        if title:
            return [title]
        if folder:
            return [folder]
        return []

    @staticmethod
    def _extract_document_metadata(document: dict) -> dict:
        meta = document.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}

        filename = (
            meta.get("filename")
            or meta.get("file_name")
            or document.get("filename")
            or ""
        )
        return {
            "filename": filename,
            "folder": meta.get("folder", ""),
            "subfolder": meta.get("subfolder", ""),
            "s3_key": meta.get("s3_key", ""),
            "content_type": meta.get("content_type", ""),
        }

    @staticmethod
    def _page_range(
        section_meta: dict,
        item_meta: Optional[dict] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        sources = [item_meta or {}, section_meta or {}]
        page_start: Optional[int] = None
        page_end: Optional[int] = None
        for source in sources:
            if page_start is None and source.get("page_start") is not None:
                page_start = int(source["page_start"])
            if page_end is None and source.get("page_end") is not None:
                page_end = int(source["page_end"])
        if page_start is not None and page_end is None:
            page_end = page_start
        if page_end is not None and page_start is None:
            page_start = page_end
        return page_start, page_end

    def _walk_section(
        self,
        section: dict,
        heading_path: List[str],
        parent_heading: str,
        doc_metadata: dict,
        blocks: List[SemanticBlock],
        block_counter: int,
    ) -> int:
        section_id = (section.get("section_id") or "").strip()
        title = (section.get("title") or "").strip()
        section_meta = section.get("metadata") or {}
        if not isinstance(section_meta, dict):
            section_meta = {}

        current_path = list(heading_path)
        if title:
            current_path.append(title)

        current_parent = parent_heading or (heading_path[-1] if heading_path else "")

        for item in section.get("content", []) or []:
            block_counter = self._process_content_item(
                item=item,
                heading_path=current_path,
                parent_heading=current_parent,
                section_id=section_id,
                section_meta=section_meta,
                doc_metadata=doc_metadata,
                blocks=blocks,
                block_counter=block_counter,
            )

        for subsection in section.get("subsections", []) or []:
            block_counter = self._walk_section(
                section=subsection,
                heading_path=current_path,
                parent_heading=title or parent_heading,
                doc_metadata=doc_metadata,
                blocks=blocks,
                block_counter=block_counter,
            )

        return block_counter

    def _process_content_item(
        self,
        item: ContentItem,
        heading_path: List[str],
        parent_heading: str,
        section_id: str,
        section_meta: dict,
        doc_metadata: dict,
        blocks: List[SemanticBlock],
        block_counter: int,
    ) -> int:
        if isinstance(item, dict):
            return self._process_structured_item(
                item=item,
                heading_path=heading_path,
                parent_heading=parent_heading,
                section_id=section_id,
                section_meta=section_meta,
                doc_metadata=doc_metadata,
                blocks=blocks,
                block_counter=block_counter,
            )

        text = str(item)
        if not text.strip():
            return block_counter

        page_start, page_end = self._page_range(section_meta)
        for segment_text, content_type in self._split_text_segments(text):
            block_counter += 1
            blocks.append(
                self._make_block(
                    block_counter=block_counter,
                    heading_path=heading_path,
                    parent_heading=parent_heading,
                    section_id=section_id,
                    content=segment_text,
                    page_start=page_start,
                    page_end=page_end,
                    content_type=content_type,
                    doc_metadata=doc_metadata,
                    section_meta=section_meta,
                )
            )
        return block_counter

    def _process_structured_item(
        self,
        item: dict,
        heading_path: List[str],
        parent_heading: str,
        section_id: str,
        section_meta: dict,
        doc_metadata: dict,
        blocks: List[SemanticBlock],
        block_counter: int,
    ) -> int:
        item_meta = item.get("metadata") or {}
        if not isinstance(item_meta, dict):
            item_meta = {}

        explicit_type = (item.get("content_type") or item.get("type") or "").strip().lower()
        content = item.get("content", item.get("text", ""))

        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)
        content = str(content)

        if not content.strip() and not explicit_type:
            return block_counter

        page_start, page_end = self._page_range(section_meta, item_meta)

        if explicit_type:
            content_type = self._normalize_content_type(explicit_type)
            block_counter += 1
            blocks.append(
                self._make_block(
                    block_counter=block_counter,
                    heading_path=heading_path,
                    parent_heading=parent_heading,
                    section_id=section_id,
                    content=content,
                    page_start=page_start,
                    page_end=page_end,
                    content_type=content_type,
                    doc_metadata=doc_metadata,
                    section_meta=section_meta,
                    item_meta=item_meta,
                )
            )
            return block_counter

        for segment_text, content_type in self._split_text_segments(content):
            block_counter += 1
            blocks.append(
                self._make_block(
                    block_counter=block_counter,
                    heading_path=heading_path,
                    parent_heading=parent_heading,
                    section_id=section_id,
                    content=segment_text,
                    page_start=page_start,
                    page_end=page_end,
                    content_type=content_type,
                    doc_metadata=doc_metadata,
                    section_meta=section_meta,
                    item_meta=item_meta,
                )
            )
        return block_counter

    def _make_block(
        self,
        block_counter: int,
        heading_path: List[str],
        parent_heading: str,
        section_id: str,
        content: str,
        page_start: Optional[int],
        page_end: Optional[int],
        content_type: str,
        doc_metadata: dict,
        section_meta: dict,
        item_meta: Optional[dict] = None,
    ) -> SemanticBlock:
        metadata = dict(doc_metadata)
        metadata["section_id"] = section_id
        metadata["heading_path"] = list(heading_path)
        metadata["page_start"] = page_start
        metadata["page_end"] = page_end
        metadata["content_type"] = content_type

        for extra in (section_meta, item_meta or {}):
            for key in ("filename", "folder", "subfolder", "s3_key", "content_type", "toc_section"):
                if extra.get(key):
                    metadata[key] = extra[key]

        return SemanticBlock(
            block_id=f"block_{block_counter:06d}",
            heading_path=list(heading_path),
            parent_heading=parent_heading,
            section_id=section_id,
            content=content,
            page_start=page_start,
            page_end=page_end,
            content_type=content_type,
            protected=content_type in _PROTECTED_CONTENT_TYPES,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_content_type(content_type: str) -> str:
        aliases = {
            "bulleted_list": "list",
            "numbered_list": "list",
            "code": "code_block",
            "caption": "image_caption",
            "formula_block": "formula",
        }
        normalized = aliases.get(content_type, content_type)
        if normalized == "danger" or normalized == "caution":
            return "warning"
        return normalized

    def _split_text_segments(self, text: str) -> List[Tuple[str, str]]:
        """Split raw text into semantically distinct segments."""
        lines = text.splitlines()
        if not lines:
            return []

        segments: List[Tuple[str, str]] = []
        current_lines: List[str] = []
        current_type: Optional[str] = None
        in_code_block = False

        def flush() -> None:
            nonlocal current_lines, current_type
            if not current_lines:
                return
            segment_text = "\n".join(current_lines)
            if segment_text.strip():
                segments.append((segment_text, current_type or "paragraph"))
            current_lines = []
            current_type = None

        for line in lines:
            stripped = line.strip()

            if _CODE_FENCE_RE.match(stripped):
                if in_code_block:
                    current_lines.append(line)
                    flush()
                    current_type = "code_block"
                    in_code_block = False
                    continue
                flush()
                in_code_block = True
                current_type = "code_block"
                current_lines.append(line)
                continue

            if in_code_block:
                current_lines.append(line)
                continue

            line_type = self._classify_line(stripped)

            if line_type == "appendix" or line_type == "glossary":
                flush()
                current_type = line_type
                current_lines.append(line)
                flush()
                continue

            if line_type in _PROTECTED_CONTENT_TYPES:
                if current_type != line_type and current_lines:
                    flush()
                current_type = line_type
                current_lines.append(line)
                continue

            if current_type in _PROTECTED_CONTENT_TYPES:
                if self._continues_protected_block(current_type, stripped):
                    current_lines.append(line)
                    continue
                flush()

            if not stripped:
                if current_type in (None, "paragraph") and current_lines:
                    flush()
                elif current_lines:
                    current_lines.append(line)
                continue

            if current_type and current_type != "paragraph" and line_type != current_type:
                flush()

            if not current_type:
                current_type = line_type

            current_lines.append(line)

        flush()
        return segments

    @staticmethod
    def _continues_protected_block(block_type: str, stripped: str) -> bool:
        if not stripped:
            return True
        if block_type == "procedure":
            return bool(_STEP_RE.match(stripped) or _NUMBERED_LIST_RE.match(stripped))
        if block_type == "list":
            return bool(_BULLET_RE.match(stripped) or _NUMBERED_LIST_RE.match(stripped))
        if block_type == "table":
            return bool(_TABLE_ROW_RE.search(stripped) or "\t" in stripped)
        if block_type == "faq":
            return bool(
                _FAQ_Q_RE.match(stripped)
                or stripped.lower().startswith("a:")
                or stripped.lower().startswith("answer:")
            )
        if block_type == "warning":
            return not _NOTE_RE.match(stripped)
        if block_type in ("definition", "note", "example", "image_caption", "formula"):
            return True
        return False

    @staticmethod
    def _classify_line(stripped: str) -> str:
        if not stripped:
            return "paragraph"
        if _CODE_FENCE_RE.match(stripped):
            return "code_block"
        if _WARNING_RE.match(stripped):
            return "warning"
        if _NOTE_RE.match(stripped):
            return "note"
        if _FAQ_START_RE.match(stripped) or _FAQ_Q_RE.match(stripped):
            return "faq"
        if _PROCEDURE_START_RE.match(stripped) or _STEP_RE.match(stripped):
            return "procedure"
        if _TABLE_ROW_RE.search(stripped):
            return "table"
        if _BULLET_RE.match(stripped):
            return "list"
        if _NUMBERED_LIST_RE.match(stripped):
            return "list"
        if _DEFINITION_RE.match(stripped):
            return "definition"
        if _TERM_DEFINITION_RE.match(stripped):
            return "definition"
        if _IMAGE_CAPTION_RE.match(stripped):
            return "image_caption"
        if _FORMULA_RE.match(stripped):
            return "formula"
        if _APPENDIX_RE.match(stripped):
            return "appendix"
        if _GLOSSARY_RE.match(stripped):
            return "glossary"
        if _EXAMPLE_RE.match(stripped):
            return "example"
        return "paragraph"


def detect_semantic_boundaries(document: dict) -> dict:
    """Convenience wrapper around :class:`SemanticBoundaryDetector`."""
    return SemanticBoundaryDetector().detect(document)
