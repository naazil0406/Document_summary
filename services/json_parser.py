"""
JSON document (.json) parser.

Extracts content and structure from JSON files into human-readable PageContent
objects, and — since JSON has no headings or blank-line paragraphs for the
generic DocumentChunker/SemanticChunker stages to split on — performs its own
record-boundary-aware chunking here, at extraction time.

Each PageContent returned by extract_pages() is already a complete, final
chunk: one record group (e.g. one "department" with its "employees"), never
sliced mid-record. main.py's parse_and_chunk() detects JSONParser and passes
these pages straight through to embedding, bypassing the prose-oriented
chunking stages (see the JSONParser branch added in parse_and_chunk).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from services.pdf_parser import PageContent

logger = logging.getLogger(__name__)

try:
    from config.settings import settings
    _DEFAULT_MAX_CHARS = settings.JSON_CHUNK_MAX_CHARS
except Exception:  # settings unavailable (e.g. standalone/test usage)
    _DEFAULT_MAX_CHARS = 1500


# ---------------------------------------------------------------------------
# MongoDB shell / mongoexport syntax sanitizer
# ---------------------------------------------------------------------------
# Real-world .json exports often come straight out of `mongo` shell or
# `mongoexport` (without --jsonArray / --canonical) and contain BSON type
# wrappers that are NOT valid JSON on their own, e.g.:
#   "moduleId": ObjectId("3431")
#   "category": NumberInt(1)
#   "views": NumberLong(15234)
#   "score": NumberDecimal("4.75")
#   "createdAt": ISODate("2026-07-01T10:00:00Z")
#   "flag": BinData(0, "...")
#   "ts": Timestamp(1234567890, 1)
# json.load() raises "Expecting value" on these, which previously sent the
# WHOLE file down the raw-text fallback path (no record-aware chunking).
# This sanitizer rewrites them into plain JSON equivalents (numbers/strings)
# before parsing, so structured extraction + chunking still applies.
_MONGO_NUMERIC_WRAPPER_RE = re.compile(
    r"\b(?:NumberInt|NumberLong)\(\s*\"?(-?\d+)\"?\s*\)"
)
_MONGO_DECIMAL_WRAPPER_RE = re.compile(
    r"\bNumberDecimal\(\s*\"?(-?\d+(?:\.\d+)?)\"?\s*\)"
)
_MONGO_STRING_WRAPPER_RE = re.compile(
    r"\b(?:ObjectId|ISODate|UUID)\(\s*\"([^\"]*)\"\s*\)"
)
_MONGO_TIMESTAMP_RE = re.compile(
    r"\bTimestamp\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
)
_MONGO_BINDATA_RE = re.compile(
    r"\bBinData\(\s*\d+\s*,\s*\"([^\"]*)\"\s*\)"
)


def _sanitize_mongo_shell_json(raw_text: str) -> str:
    """
    Rewrite MongoDB shell/export BSON-wrapper syntax into plain JSON so
    json.loads() can parse it. Idempotent no-op on files that don't contain
    any of these patterns (normal JSON passes through unchanged).
    """
    text = raw_text
    text = _MONGO_NUMERIC_WRAPPER_RE.sub(r"\1", text)
    text = _MONGO_DECIMAL_WRAPPER_RE.sub(r'"\1"', text)
    text = _MONGO_STRING_WRAPPER_RE.sub(r'"\1"', text)
    text = _MONGO_TIMESTAMP_RE.sub(r'{"seconds": \1, "increment": \2}', text)
    text = _MONGO_BINDATA_RE.sub(r'"\1"', text)
    return text


# ---------------------------------------------------------------------------
# Missing-colon repair (separate from Mongo-shell syntax)
# ---------------------------------------------------------------------------
# Manually-edited or buggy-export JSON sometimes drops the colon between a
# key and its value, e.g.:
#   "placeholderText" "I'm tired",
# instead of:
#   "placeholderText": "I'm tired",
#
# This is NOT a Mongo shell wrapper — it's a plain syntax typo.
#
# IMPORTANT: the match is anchored so the "key" quoted-string must be
# immediately preceded (ignoring whitespace) by `{` or `,` — i.e. a position
# where a JSON object key is structurally expected. Without this anchor, a
# naive "quoted string, then value token" regex can misfire by pairing the
# CLOSING quote of one key with the OPENING quote of the next key's value
# across an already-valid colon (e.g. wrongly treating `"text": "` as if it
# were one quoted string, corrupting well-formed JSON). Anchoring on the
# preceding `{`/`,` guarantees we only ever match an actual key position,
# so correctly formed JSON is never touched.
_MISSING_COLON_RE = re.compile(
    r'([{,]\s*)'
    r'("(?:[^"\\]|\\.)*")'
    r'([ \t]*\n?[ \t]*)'
    r'("(?:[^"\\]|\\.)*"|\[|\{|-?\d|true\b|false\b|null\b)'
)


def _fix_missing_colons(text: str) -> str:
    """Insert a colon between a key and value when one is missing."""
    return _MISSING_COLON_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}: {m.group(4)}", text)


class JSONParser:
    """Extracts content from JSON (.json) files and chunks it at record boundaries."""

    def __init__(self, folder_path: str = "", max_chunk_chars: int = _DEFAULT_MAX_CHARS):
        self.folder_path = folder_path
        # Cap per chunk. JSON records (e.g. one employee/one order) are kept
        # whole; only a *list* of many small records gets grouped up to this
        # size, and only a genuinely oversized single record gets recursed
        # into its own nested record-lists for further splitting.
        self.max_chunk_chars = max_chunk_chars

    # ------------------------------------------------------------------
    # Legacy full-document formatter (kept for the exception fallback path
    # and for small JSON files where a single chunk is already fine).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Record-boundary-aware chunking (new)
    # ------------------------------------------------------------------
    def _format_record_shallow(self, item: Any) -> Tuple[str, Dict[str, list]]:
        """
        Format a record's own scalar/nested-dict fields into indented text,
        but DON'T expand any field that is itself a list-of-dicts (a
        "repeating record list", e.g. an employee's own sub-list) — those
        are returned separately so the caller can decide whether to inline
        them (record fits under the cap) or recurse into them as their own
        chunk(s) (record is too big on its own).
        """
        if not isinstance(item, dict):
            return self._format_json_val(item), {}

        lines = []
        nested_record_lists: Dict[str, list] = {}
        for k, v in item.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                nested_record_lists[k] = v
                continue
            if isinstance(v, (dict, list)):
                lines.append(f"{k}:")
                lines.append(self._format_json_val(v, level=1))
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines), nested_record_lists

    def _chunk_json_node(
        self,
        data: Any,
        context_path: str = "",
    ) -> List[str]:
        """
        Recursively walk a JSON node, producing a list of text chunks where
        each chunk is capped at self.max_chunk_chars and never splits a
        single record in half.

        - dict: each top-level key becomes its own line/chunk contribution;
          a key whose value is a list-of-dicts is treated as a "table" of
          repeating records and handled via _chunk_record_list.
        - list-of-dicts at the root: same record-list handling directly.
        - anything else (small dict/list/scalar): formatted as one chunk.
        """
        chunks: List[str] = []

        if isinstance(data, dict):
            scalar_lines = []
            for k, v in data.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    if scalar_lines:
                        chunks.append(
                            self._prefix(context_path, "\n".join(scalar_lines))
                        )
                        scalar_lines = []
                    chunks.extend(
                        self._chunk_record_list(
                            v, f"{context_path} > {k}" if context_path else k
                        )
                    )
                else:
                    text = self._format_json_val(v, level=0)
                    scalar_lines.append(f"{k}: {text}" if "\n" not in text else f"{k}:\n{text}")
            if scalar_lines:
                chunks.append(self._prefix(context_path, "\n".join(scalar_lines)))
            return chunks

        if isinstance(data, list) and data and isinstance(data[0], dict):
            return self._chunk_record_list(data, context_path)

        # Small/scalar/simple-list root
        return [self._prefix(context_path, self._format_json_val(data))]

    def _chunk_record_list(self, items: List[dict], context_path: str) -> List[str]:
        """
        Pack sibling records from a repeating-record list (e.g. all
        "employees") into chunks up to max_chunk_chars, grouping several
        small records per chunk, splitting a group when the cap would be
        exceeded, and recursing into any single record that is itself too
        big (e.g. a "department" record whose own "employees" sub-list is
        large) — recursion happens on that record's nested lists only, so
        the record's own scalar fields never get separated from it.
        """
        chunks: List[str] = []
        buffer: List[str] = []
        buffer_len = 0

        def flush():
            if buffer:
                body = "\n---\n".join(buffer)
                chunks.append(self._prefix(context_path, body))

        for idx, item in enumerate(items):
            item_text, nested_lists = self._format_record_shallow(item)
            nested_len = sum(
                len(self._format_json_val(v)) for v in nested_lists.values()
            )
            total_len = len(item_text) + nested_len

            if total_len > self.max_chunk_chars:
                # This single record is too big even alone (e.g. a department
                # with 40 employees) -> flush what's buffered, emit this
                # record's own fields as their own chunk, then recurse into
                # each of its nested record-lists as separate sub-chunks.
                flush()
                buffer, buffer_len = [], 0
                label = item.get("name") or item.get("id") or item.get("title") or f"item_{idx}"
                chunks.append(
                    self._prefix(f"{context_path} > {label}", item_text)
                )
                for nested_key, nested_val in nested_lists.items():
                    chunks.extend(
                        self._chunk_record_list(
                            nested_val, f"{context_path} > {label} > {nested_key}"
                        )
                    )
                continue

            # Record (with its nested lists inlined) fits within the cap.
            if nested_lists:
                nested_text = "\n".join(
                    f"{k}:\n{self._format_json_val(v, level=1)}"
                    for k, v in nested_lists.items()
                )
                item_text = f"{item_text}\n{nested_text}" if item_text else nested_text

            if buffer_len + len(item_text) > self.max_chunk_chars and buffer:
                flush()
                buffer, buffer_len = [], 0

            buffer.append(item_text)
            buffer_len += len(item_text)

        flush()
        return chunks

    @staticmethod
    def _prefix(context_path: str, body: str) -> str:
        if not context_path:
            return body
        return f"[{context_path}]\n{body}"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def extract_pages(self, file_path: str) -> List[PageContent]:
        """
        Read a JSON file and return one PageContent PER CHUNK.

        - Small JSON (flattened text already <= max_chunk_chars): returned
          as a single PageContent, same as before — no behavior change for
          small files.
        - Large JSON (10k+ lines and beyond): split at record boundaries via
          _chunk_json_node, so no chunk ever cuts a record's fields apart.
        - Malformed JSON: falls back to a single raw-text PageContent, same
          as the original behavior.
        """
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()

            data = None
            last_exc = None
            # Cascade of increasingly aggressive repair attempts. Each is
            # tried only after the previous one fails, and we stop at the
            # first one that parses — so a clean file always takes the
            # first (zero-repair) branch with no behavior change.
            attempts = [
                ("no repair needed", raw_text),
                ("sanitized MongoDB shell syntax (ObjectId/NumberInt/etc.)",
                 _sanitize_mongo_shell_json(raw_text)),
                ("repaired missing colons (key/value typo)",
                 _fix_missing_colons(raw_text)),
                ("sanitized MongoDB syntax + repaired missing colons",
                 _fix_missing_colons(_sanitize_mongo_shell_json(raw_text))),
            ]
            tried_texts = set()
            for description, candidate_text in attempts:
                if candidate_text in tried_texts:
                    continue
                tried_texts.add(candidate_text)
                try:
                    data = json.loads(candidate_text)
                    if description != "no repair needed":
                        logger.info("JSON '%s': %s before parsing.", filename, description)
                    break
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    continue
            if data is None:
                raise last_exc
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

        full_text = self._format_json_val(data).strip()

        if len(full_text) <= self.max_chunk_chars:
            # Small file: keep old single-chunk behavior.
            return [
                PageContent(
                    filename=filename,
                    page_number=1,
                    page_label="1",
                    text=full_text,
                    metadata={"chunk_type": "json_full_document"},
                )
            ]

        record_chunks = self._chunk_json_node(data)
        if not record_chunks:
            # Defensive fallback — shouldn't normally happen.
            record_chunks = [full_text]

        logger.info(
            "JSON '%s': %d chars flattened -> %d record-aware chunk(s) (cap=%d chars).",
            filename, len(full_text), len(record_chunks), self.max_chunk_chars,
        )

        return [
            PageContent(
                filename=filename,
                page_number=i + 1,
                page_label=str(i + 1),
                text=chunk_text.strip(),
                metadata={"chunk_type": "json_record_group", "chunk_index": i},
            )
            for i, chunk_text in enumerate(record_chunks)
            if chunk_text and chunk_text.strip()
        ]