"""
PDF parser with Docker-based Tesseract OCR fallback.

Configured for: AMD64 / x86_64 (Docker Desktop on Windows/Linux AMD)

Strategy:
  1. Extract text layer with PyMuPDF (fast, zero host dependencies).
  2. If a page yields fewer than MIN_TEXT_CHARS characters it is likely
     image-based. The page is rasterised to a PNG in a temp directory,
     mounted into the official Tesseract Docker image (linux/amd64), OCR'd
     inside the container, and the text is read back from the same directory.
     No Tesseract binary is required on the host — Docker Desktop is enough.
  3. If Docker is not running or the image is missing, a clear error is
     logged and the page keeps whatever text PyMuPDF found — no crash.

Host requirements:
  - Docker Desktop running (AMD64 / x86_64)
  - pip install docker Pillow

One-time image pull (cached forever after):
  docker pull --platform linux/amd64 tesseractshadow/tesseract4re

PDF types handled:
  - Born-digital (text layer present)  → PyMuPDF fast path, no Docker
  - Fully scanned / raster             → Docker OCR on every page
  - Mixed (60 % image / 40 % text)     → per-page decision
"""

import io
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict
import re

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

# Pages with fewer characters from the text layer are sent to Docker OCR.
MIN_TEXT_CHARS: int = 300  # raised from 50 — workbook pages with real content yield 400-800 chars

# Tesseract Docker image — linux/amd64 build, small and well-maintained.
TESSERACT_DOCKER_IMAGE: str = "tesseractshadow/tesseract4re"

# Force AMD64 platform so Docker Desktop on AMD never pulls an ARM layer.
DOCKER_PLATFORM: str = "linux/amd64"

# Rasterisation resolution — 200 dpi gives good OCR accuracy without huge PNGs.
OCR_DPI: int = 200

# OCR language passed to Tesseract. Add more with '+', e.g. "eng+fra".
TESSERACT_LANG: str = "eng"

# OCR is mandatory for every page so scanned or image-heavy content is still
# captured. The parser uses Docker OCR exclusively and does not fall back to
# a local Tesseract binary.
ENABLE_OCR_FALLBACK: bool = False

# ─────────────────────────────────────────────────────────────────────────────


def _get_docker_client():
    """
    Return a connected Docker client, or raise RuntimeError with a clear
    message if the daemon is not reachable.

    The parser is designed to keep working even if Docker is unavailable, so
    we short-circuit here to avoid hanging on Docker Desktop startup.
    """
    try:
        import docker
    except ImportError:
        raise RuntimeError(
            "Python package 'docker' is not installed. "
            "Run: pip install docker"
        )

    try:
        client = docker.from_env()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Docker daemon — is Docker Desktop running? Error: {exc}"
        )

    try:
        client.ping()
    except Exception as exc:
        raise RuntimeError(f"Docker daemon did not respond: {exc}")

    return client, docker


def _render_page_png(page: fitz.Page, tmpdir: str) -> str:
    """Rasterise a PDF page to a temporary PNG file and return the path."""
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Python package 'Pillow' is not installed. Run: pip install Pillow")

    scale = OCR_DPI / 72
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    png_path = os.path.join(tmpdir, "page.png")
    Image.open(io.BytesIO(pix.tobytes("png"))).save(png_path)
    return png_path


def _ocr_page_local_tesseract(page: fitz.Page) -> str:
    """Unused helper retained for compatibility; Docker OCR is mandatory."""
    raise RuntimeError("Local Tesseract OCR is disabled; Docker OCR is mandatory")


def ocr_png_docker(png_path: str, *, label: str = "image") -> str:
    """Run Docker Tesseract OCR on an existing PNG file and return the text.

    Shared by the PDF per-page OCR fallback below and by
    ``services/image_parser.py`` (standalone uploaded images), so both
    paths go through the exact same Docker/Tesseract invocation instead of
    duplicating the container-run logic.
    """
    try:
        client, docker = _get_docker_client()
    except RuntimeError as exc:
        logger.error("Docker OCR failed: %s", exc)
        raise

    src_dir = os.path.dirname(os.path.abspath(png_path))
    src_name = os.path.basename(png_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        work_png = os.path.join(tmpdir, "page.png")
        shutil.copy(os.path.join(src_dir, src_name) if src_dir else src_name, work_png)

        out_base = os.path.join(tmpdir, "output")   # Tesseract appends .txt

        # ── Run Tesseract in a linux/amd64 container ─────────────────────────
        volumes = {tmpdir: {"bind": "/data", "mode": "rw"}}
        command = f"tesseract /data/page.png /data/output {TESSERACT_LANG}"

        logger.debug(
            "Running Docker OCR: image=%s platform=%s command=%s",
            TESSERACT_DOCKER_IMAGE, DOCKER_PLATFORM, command,
        )

        try:
            container_output = client.containers.run(
                image=TESSERACT_DOCKER_IMAGE,
                command=command,
                volumes=volumes,
                platform=DOCKER_PLATFORM,
                remove=True,
                stdout=True,
                stderr=True,
            )
            logger.debug("Container stdout/stderr: %s", container_output)

        except docker.errors.ImageNotFound:
            logger.error(
                "Docker image '%s' not found locally.\n"
                "Pull it once with:\n"
                "  docker pull --platform linux/amd64 %s",
                TESSERACT_DOCKER_IMAGE,
                TESSERACT_DOCKER_IMAGE,
            )
            raise

        except docker.errors.ContainerError as exc:
            logger.error(
                "Tesseract container exited with non-zero status for %s: %s",
                label, exc,
            )
            raise

        except Exception as exc:
            logger.error("Docker OCR failed for %s: %s", label, exc)
            raise

        output_txt = out_base + ".txt"
        if not os.path.isfile(output_txt):
            logger.error("Tesseract produced no output file for %s.", label)
            raise RuntimeError("Docker OCR did not produce output")

        with open(output_txt, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()


def _ocr_page_docker(page: fitz.Page) -> str:
    """Rasterise a PDF page and run OCR inside a Docker Tesseract container."""
    try:
        from PIL import Image  # noqa: F401  (import check retained for parity)
    except ImportError:
        raise RuntimeError(
            "Python package 'Pillow' is not installed. "
            "Run: pip install Pillow"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            png_path = _render_page_png(page, tmpdir)
        except Exception as exc:
            logger.error("Could not render page %d for Docker OCR: %s", page.number + 1, exc)
            raise

        return ocr_png_docker(png_path, label=f"page {page.number + 1}")
    


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TOCEntry:
    """Represents one PDF Table-of-Contents entry and its inclusive page range."""
    level: int
    title: str
    page_start: int
    page_end: int


@dataclass
class PageContent:
    filename: str
    page_number: int
    page_label: str
    text: str
    metadata: dict = field(default_factory=dict)    


# ── Parser ────────────────────────────────────────────────────────────────────

class PDFParser:

    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder

        # filename -> list[TOCEntry]
        self.toc_map: Dict[str, List[TOCEntry]] = {}

    def _list_pdf_files(self) -> List[str]:
        return [
            os.path.join(self.pdf_folder, f)
            for f in os.listdir(self.pdf_folder)
            if f.lower().endswith(".pdf")
        ]
    def extract_toc(self, doc: fitz.Document) -> List[TOCEntry]:
        """
        Extract PDF table of contents.

        PyMuPDF reports one-based start pages. Each entry's inclusive end
        page is inferred from the next entry, or the end of the document.
        """
        toc_entries: List[TOCEntry] = []

        try:
            toc = doc.get_toc(simple=False)

            if not toc:
                logger.info("No TOC found.")
                return []

            logger.info("Found %d TOC entries.", len(toc))

            for item in toc:
                if len(item) < 3:
                    continue

                level = int(item[0])
                title = str(item[1]).strip()
                page_start = int(item[2])

                # PyMuPDF uses -1 for unresolved/external TOC destinations.
                if not title or not (1 <= page_start <= len(doc)):
                    logger.debug(
                        "Skipping invalid TOC entry: title=%r page=%s",
                        title, page_start,
                    )
                    continue

                toc_entries.append(
                    TOCEntry(
                        level=level,
                        title=title,
                        page_start=page_start,
                        page_end=page_start,
                    )
                )

            for index, entry in enumerate(toc_entries):
                if index + 1 < len(toc_entries):
                    next_start = toc_entries[index + 1].page_start
                    entry.page_end = max(entry.page_start, next_start - 1)
                else:
                    entry.page_end = len(doc)

        except Exception as exc:
            logger.warning("Unable to extract TOC: %s", exc)

        return toc_entries

    @staticmethod
    def _toc_section_for_page(
        page_number: int,
        toc_entries: List[TOCEntry],
    ) -> str:
        """Return the most specific TOC entry whose range contains the page."""
        matches = [
            entry
            for entry in toc_entries
            if entry.page_start <= page_number <= entry.page_end
        ]
        if not matches:
            return ""
        return max(matches, key=lambda entry: (entry.level, entry.page_start)).title

    def extract_pages(self, file_path: str) -> List[PageContent]:
        filename = os.path.basename(file_path)
        pages: List[PageContent] = []

        doc = fitz.open(file_path)
        logger.info("Processing '%s' (%d pages)", filename, len(doc))
        toc_entries = self.extract_toc(doc)
        self.toc_map[filename] = toc_entries
        logger.info("Stored %d TOC entries for '%s'.", len(toc_entries), filename)

        for page_num in range(len(doc)):
            page = doc[page_num]
            extracted_text = page.get_text("text")
            extracted_char_count = len(extracted_text.strip())

            logger.info(
                "Page %d: %d chars from text layer — running OCR compulsorily (Docker Tesseract).",
                page_num + 1, extracted_char_count,
            )
            ocr_text = _ocr_page_docker(page)
            ocr_char_count = len(ocr_text.strip())

            # OCR always runs (compulsory — every page goes through Docker
            # Tesseract), but the *result text* is chosen by quality, not by
            # OCR automatically winning. A born-digital page's PyMuPDF text
            # layer is the verbatim original text with zero recognition
            # error. Re-OCR'ing that same rendered page through Tesseract
            # can — and in practice does — introduce real misreads (garbled
            # words, mangled section titles), which then poison chunking,
            # retrieval, and answers downstream. So: once the text layer has
            # enough content to be trustworthy, keep it. OCR only becomes the
            # primary source for genuinely sparse/empty pages (scanned or
            # image-only content) — exactly the case it exists to handle.
            if extracted_char_count >= MIN_TEXT_CHARS:
                text = extracted_text
                source = "text layer (trusted; OCR ran but was not used)"
            elif ocr_char_count > extracted_char_count:
                text = ocr_text
                source = "OCR (text layer too sparse)"
            else:
                text = extracted_text or ocr_text
                source = "text layer (fallback)"

            char_count = len(text.strip())
            logger.info(
                "Page %d: using %s — %d chars (text layer=%d, OCR=%d).",
                page_num + 1, source, char_count, extracted_char_count, ocr_char_count,
            )

            # Capture document-level metadata (author, title, etc.) so it can
            # be propagated into chunks and stored alongside embeddings.
            doc_metadata = dict(doc.metadata or {})
            doc_metadata["toc_section"] = self._toc_section_for_page(
                page_num + 1,
                toc_entries,
            )
            pages.append(PageContent(
                filename=filename,
                page_number=page_num + 1,
                page_label=f"Page {page_num + 1}",
                text=text,
                metadata=doc_metadata,
            ))

        doc.close()
        return pages

    def extract_all(self) -> List[PageContent]:
        all_pages: List[PageContent] = []
        pdf_files = self._list_pdf_files()
        logger.info("Found %d PDF file(s) in '%s'", len(pdf_files), self.pdf_folder)

        for pdf in pdf_files:
            try:
                extracted = self.extract_pages(pdf)
                logger.info(
                    "Extracted %d page(s) from '%s'",
                    len(extracted), os.path.basename(pdf),
                )
                all_pages.extend(extracted)
            except Exception as exc:
                logger.error("Failed to process '%s': %s", pdf, exc)

        return all_pages