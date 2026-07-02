"""
Two-stage chunking service.

Stage 1 — Document Chunking (DocumentChunker):
    Splits each page's raw text along structural boundaries — headings,
    sections, paragraphs — while respecting page boundaries (a document
    chunk never spans two pages, since chunking runs per PageContent).

    TOC/Index-based chunking:
    When a Table of Contents page is detected, the document pages are
    re-grouped by TOC entries so each section (e.g. "Unit 2 – Variables")
    becomes a labelled set of DocumentChunks.  Inside each TOC section the
    normal per-page structural splitter still runs, so sub-headings within
    the section are respected — the section is NOT collapsed into one giant
    chunk.  metadata["toc_section"] is set on every chunk that came from a
    named section so the retriever can filter by it.

    TOC pages themselves are skipped (not stored as chunks) since they
    contain only navigation data that would pollute retrieval results.

Stage 2 — Semantic Chunking (SemanticChunkingService):
    Runs LangChain's SemanticChunker (using BGE-M3 embeddings for
    similarity-based boundary detection) *inside* each document chunk
    produced by Stage 1.  A size-based safeguard
    (RecursiveCharacterTextSplitter) is applied to any resulting chunk
    that exceeds max_chunk_size.

Pipeline: pages -> DocumentChunker -> SemanticChunkingService -> Chunk[]
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.pdf_parser import PageContent

logger = logging.getLogger(__name__)

# Heading regexes — only explicit structural markers qualify
_HEADING_NUMBERED_RE = re.compile(
    r"^(\d+(\.\d+)*[\.)]?|section\s+\d+[:.]?)\s+\S", re.IGNORECASE
)
_HEADING_MARKDOWN_RE = re.compile(r"^#{1,6}\s+\S")
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")


@dataclass
class DocumentChunk:
    """A structural (heading/section/paragraph/page-bounded) chunk, pre-semantic-chunking."""
    text: str
    filename: str
    page_number: int
    page_label: str
    page_start: int
    page_end: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a single semantically-coherent text chunk, ready for embedding/storage."""
    chunk_id: str
    text: str
    filename: str
    page_number: int
    page_label: str
    page_start: int
    page_end: int
    metadata: dict = field(default_factory=dict)
    toc_section: str = ""


# ---------------------------------------------------------------------------
# TOC detection and parsing helpers
# ---------------------------------------------------------------------------

def _is_toc_page(text: str) -> bool:
    """
    Return True if the page looks like a genuine Table of Contents page.

    A real TOC entry is "<title text> <leader> <page number>" — it always
    has a substantial title before the trailing number. Pages from slide
    decks or workbooks often contain *standalone* numeric lines (bullet
    badges like "1", "2", "3", or footer page counters like "4 / 16") which
    are NOT TOC entries on their own, even though a naive "line ends with a
    short number" check would match them. Counting those caused real
    content pages to be misclassified as TOC pages and silently dropped
    from the index.
    """
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for ln in lines[:5]:
        lowered = ln.lower()
        if "table of contents" in lowered or lowered.strip() in ("contents", "toc", "index"):
            return True

    page_number_like_lines = 0
    for ln in lines:
        # Skip standalone numbers / footer page counters like "4 / 16" —
        # these have no real title text and are not TOC entries.
        if re.fullmatch(r"\d{1,4}", ln):
            continue
        if re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", ln):
            continue

        # Require a real title (enough non-numeric, non-leader text) before
        # the trailing page number, so single bullet/number lines never count.
        #
        # Three dot-leader styles are accepted:
        #   - solid dots:           "Introduction.......... 3"
        #   - dots separated by single spaces (common PDF rendering quirk,
        #     e.g. real SafeStart workbook PDFs): "Introduction.. . . . . 3"
        #   - double-space leader:  "Introduction        3"
        m = (
            re.search(r"(?:\.\s?){3,}\s*(\d{1,4})\s*$", ln)
            or re.search(r"\s{2,}(\d{1,4})\s*$", ln)
        )
        if not m:
            continue

        title_part = ln[: m.start()].rstrip(" .")
        if len(title_part) < 6:
            continue

        page_number_like_lines += 1

    return page_number_like_lines >= 4


def _parse_toc_entries(toc_text: str, max_pages: int) -> List[Tuple[str, int]]:
    """
    Return a list of (title, page_number) pairs parsed from raw TOC text.

    Accepted formats:
      - Dot-leader (solid):           "Unit 2 – Variables ........ 5"
      - Dot-leader (space-separated): "Unit 2 – Variables .. . . . 5"
        (this is how some PDFs, including the SafeStart workbooks, actually
        render dot leaders — dots separated by single spaces rather than a
        solid run of dots)
      - Double-space:                 "Unit 2 – Variables       5"

    The single-space fallback is intentionally kept last and least
    preferred because on its own it matches too many body-text lines
    (e.g. "Version 2", "See item 3") — it only runs once the two dot-leader
    patterns above have already had a chance to match.
    Lines that are purely numeric or too short are skipped.
    """
    entries: List[Tuple[str, int]] = []
    seen: Set[int] = set()  # avoid duplicate page numbers from header repetition

    for line in toc_text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = None
        for pattern in [
            r"^(?P<title>.+?)\s+\.{2,}\s*(?P<p>\d+)\s*$",
            r"^(?P<title>.+?)(?:\.\s?){3,}\s*(?P<p>\d+)\s*$",
            r"^(?P<title>.+?)\s{2,}(?P<p>\d+)\s*$",
            r"^(?P<title>.+?)\s+(?P<p>\d+)\s*$",
        ]:
            m = re.match(pattern, line)
            if m:
                break

        if not m:
            continue

        title = m.group("title").strip()
        # Strip any trailing dot-leader remnants the regex didn't fully
        # consume — e.g. "Introducing SafeStart Now.. . . . . . ." where the
        # leader uses dots separated by spaces rather than a solid run of
        # dots. Without this, toc_section labels end up polluted with
        # leader characters, hurting fuzzy section matching at query time.
        title = re.sub(r"(?:[.\s]){3,}$", "", title).strip(" .")
        try:
            p = int(m.group("p"))
        except ValueError:
            continue

        # Skip implausible page numbers and duplicate entries
        if not (1 <= p <= max_pages):
            continue
        if not title or len(title) < 2:
            continue
        # Skip lines where the "title" is itself just a number (e.g. "3 .... 3")
        if re.fullmatch(r"[\d\s.]+", title):
            continue
        if p in seen:
            continue

        seen.add(p)
        entries.append((title, p))

    return entries


def _toc_chunk_pages(
    pages: List[PageContent],
    toc_entries: List[Tuple[str, int]],
    toc_page_numbers: Set[int],
    document_chunker: "DocumentChunker",
) -> Tuple[List[DocumentChunk], Set[int]]:
    """
    Group pages by TOC entries and apply the structural splitter within each
    section.  Returns (doc_chunks, covered_page_numbers).

    Key design decisions:
    - Each section's pages are chunked individually by document_chunker.chunk_page()
      so sub-headings inside the section are preserved.
    - metadata["toc_section"] is injected into every produced chunk.
    - TOC pages themselves are excluded from the output.
    - total_pages is derived from the highest page_number in `pages`, not
      len(pages), so non-contiguous page lists work correctly.
    """
    doc_chunks: List[DocumentChunk] = []
    covered: Set[int] = set()

    if not pages or not toc_entries:
        return doc_chunks, covered

    # Use actual max page_number, not len(pages), to handle gaps in page numbering
    max_page_number = max(p.page_number for p in pages)
    page_by_number: Dict[int, PageContent] = {p.page_number: p for p in pages}

    entries = sorted(toc_entries, key=lambda x: x[1])

    for i, (title, start_pg) in enumerate(entries):
        end_pg = entries[i + 1][1] - 1 if i + 1 < len(entries) else max_page_number

        section_pages = [
            page_by_number[pn]
            for pn in range(start_pg, end_pg + 1)
            if pn in page_by_number and pn not in toc_page_numbers
        ]

        if not section_pages:
            logger.debug("TOC section '%s' (pages %d–%d): no pages found.", title, start_pg, end_pg)
            continue

        section_meta_extra = {"toc_section": title}
        section_chunk_count = 0

        for pg in section_pages:
            try:
                page_chunks = document_chunker.chunk_page(pg)
            except Exception as exc:
                logger.error(
                    "Structural chunking failed for page %d in section '%s': %s",
                    pg.page_number, title, exc,
                )
                continue

            for dc in page_chunks:
                # Inject toc_section into the chunk's metadata
                merged_meta = {**(dc.metadata or {}), **section_meta_extra}
                doc_chunks.append(
                    DocumentChunk(
                        text=dc.text,
                        filename=dc.filename,
                        page_number=dc.page_number,
                        page_label=dc.page_label,
                        page_start=dc.page_start,
                        page_end=dc.page_end,
                        metadata=merged_meta,
                    )
                )
                section_chunk_count += 1

            covered.add(pg.page_number)

        logger.debug(
            "TOC section '%s': pages %d–%d → %d structural chunk(s).",
            title, start_pg, end_pg, section_chunk_count,
        )

    return doc_chunks, covered


class DocumentChunker:
    """
    Stage 1: splits page text into structural chunks along headings,
    sections, and paragraphs, never crossing a page boundary.
    TOC-based grouping is attempted first; uncovered pages (and TOC pages
    themselves) fall back to per-page structural splitting, with TOC pages
    skipped entirely.
    """

    def __init__(self, heading_max_length: int = 80, min_paragraph_length: int = 20):
        self.heading_max_length = heading_max_length
        self.min_paragraph_length = min_paragraph_length

    def _is_heading(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > self.heading_max_length:
            return False
        if _SENTENCE_END_RE.search(stripped):
            return False
        if _HEADING_MARKDOWN_RE.match(stripped):
            return True
        if _HEADING_NUMBERED_RE.match(stripped):
            return True
        letters = [c for c in stripped if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.80:
            if len(stripped) >= 4:
                return True
        return False

    def _split_into_sections(self, text: str) -> List[str]:
        lines = text.splitlines()
        sections: List[List[str]] = []
        current: List[str] = []
        for line in lines:
            if self._is_heading(line) and current:
                sections.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append(current)
        return ["\n".join(s).strip() for s in sections if "\n".join(s).strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_section_into_paragraphs(self, section_text: str) -> List[str]:
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        if not raw_paragraphs:
            return []
        if len(raw_paragraphs) == 1 and len(raw_paragraphs[0]) > self.min_paragraph_length * 20:
            raw_paragraphs = self._split_into_sentences(raw_paragraphs[0])
        merged: List[str] = []
        buffer = ""
        for para in raw_paragraphs:
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(buffer) >= self.min_paragraph_length:
                merged.append(buffer)
                buffer = ""
        if buffer:
            if merged:
                merged[-1] = f"{merged[-1]}\n\n{buffer}"
            else:
                merged.append(buffer)
        return merged

    def chunk_page(self, page: PageContent) -> List[DocumentChunk]:
        """Split a single page's text into heading/section/paragraph-bounded chunks."""
        document_chunks: List[DocumentChunk] = []
        for section in self._split_into_sections(page.text):
            for paragraph in self._split_section_into_paragraphs(section):
                stripped = paragraph.strip()
                word_count = len(stripped.split())
                if word_count < 3:
                    logger.debug(
                        "Skipping label-only chunk (%d words) on page %d: %r",
                        word_count, page.page_number, stripped[:60],
                    )
                    continue
                document_chunks.append(
                    DocumentChunk(
                        text=paragraph,
                        filename=page.filename,
                        page_number=page.page_number,
                        page_label=page.page_label,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        metadata=getattr(page, "metadata", {}) or {},
                    )
                )
        return document_chunks

    def chunk_pages(self, pages: List[PageContent]) -> List[DocumentChunk]:
        """
        Split every page into document chunks, preserving page boundaries.

        TOC-based chunking strategy:
          1. Detect TOC pages (explicit title or ≥4 dot-leader lines).
          2. Parse (title, start_page) entries from them.
          3. For each TOC section, collect the corresponding pages and run the
             per-page structural splitter on each — injecting toc_section into
             metadata.  This preserves sub-headings within sections.
          4. TOC pages themselves are skipped (not stored as chunks).
          5. Pages not covered by any TOC entry are processed individually
             with the normal structural splitter.

        If no TOC is found, every page is processed individually.
        """
        all_chunks: List[DocumentChunk] = []
        total_pages = len(pages)
        covered_page_nums: Set[int] = set()

        # ── Step 1: identify TOC pages ───────────────────────────────────────
        toc_pages = [p for p in pages if _is_toc_page(p.text)]
        toc_page_numbers: Set[int] = {p.page_number for p in toc_pages}

        if toc_pages:
            combined_toc_text = "\n".join(p.text for p in toc_pages)
            max_page_number = max(p.page_number for p in pages)
            entries = _parse_toc_entries(combined_toc_text, max_page_number)

            if entries:
                logger.info(
                    "TOC detected (%d page(s), %d entries). Using TOC-based chunking.",
                    len(toc_pages), len(entries),
                )
                toc_chunks, covered_page_nums = _toc_chunk_pages(
                    pages, entries, toc_page_numbers, self
                )
                all_chunks.extend(toc_chunks)
                logger.info(
                    "TOC chunking produced %d structural chunks covering %d content pages.",
                    len(toc_chunks), len(covered_page_nums),
                )
            else:
                logger.info(
                    "TOC page(s) detected but no parseable entries found. "
                    "Falling back to per-page chunking."
                )
        else:
            logger.info("No TOC page detected. Using per-page structural chunking.")

        # ── Step 2: per-page fallback — skip TOC pages entirely ─────────────
        uncovered = [
            p for p in pages
            if p.page_number not in covered_page_nums
            and p.page_number not in toc_page_numbers
        ]
        if uncovered:
            logger.info(
                "%d page(s) not covered by TOC — chunking individually.",
                len(uncovered),
            )
        skipped_toc = [p for p in pages if p.page_number in toc_page_numbers]
        if skipped_toc:
            logger.info(
                "Skipping %d TOC page(s) (not stored as chunks).",
                len(skipped_toc),
            )

        for page in uncovered:
            try:
                all_chunks.extend(self.chunk_page(page))
            except Exception as exc:
                logger.error(
                    "Document chunking failed for page %d of %s: %s",
                    page.page_number, page.filename, exc,
                )

        logger.info(
            "Document Chunking: produced %d structural chunks from %d pages.",
            len(all_chunks), total_pages,
        )
        return all_chunks


class SemanticChunkingService:
    """
    Stage 2: applies semantic chunking (BGE-M3-driven boundary detection)
    inside each document chunk produced by DocumentChunker.
    """

    def __init__(
        self,
        embeddings,
        buffer_size: int = 1,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        self._semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            buffer_size=buffer_size,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

        self._size_guard = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _split_text(self, text: str) -> List[str]:
        try:
            semantic_chunks = self._semantic_chunker.split_text(text)
        except Exception as exc:
            logger.warning(
                "Semantic chunking failed (%s). Falling back to size-based splitting.", exc,
            )
            return self._size_guard.split_text(text)

        final_chunks: List[str] = []
        for chunk in semantic_chunks:
            if len(chunk) > self.max_chunk_size:
                final_chunks.extend(self._size_guard.split_text(chunk))
            else:
                final_chunks.append(chunk)
        return final_chunks

    def chunk_documents(self, document_chunks: List[DocumentChunk]) -> List[Chunk]:
        """Apply semantic chunking inside each document chunk, producing final Chunk objects."""
        all_chunks: List[Chunk] = []
        for doc_chunk in document_chunks:
            try:
                texts = self._split_text(doc_chunk.text)
            except Exception as exc:
                logger.error(
                    "Semantic chunking failed for a document chunk on page %d of %s: %s",
                    doc_chunk.page_number, doc_chunk.filename, exc,
                )
                continue

            for text in texts:
                cleaned = text.strip()
                if not cleaned:
                    continue
                # Preserve toc_section and any other metadata from the document chunk
                chunk_meta = dict(doc_chunk.metadata) if doc_chunk.metadata else {}
                all_chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=cleaned,
                        filename=doc_chunk.filename,
                        page_number=doc_chunk.page_number,
                        page_label=doc_chunk.page_label,
                        page_start=doc_chunk.page_start,
                        page_end=doc_chunk.page_end,
                        metadata=chunk_meta,
                        toc_section=chunk_meta.get("toc_section", ""),
                    )
                )

        logger.info(
            "Semantic Chunking: produced %d final chunks from %d document chunks.",
            len(all_chunks), len(document_chunks),
        )
        return all_chunks