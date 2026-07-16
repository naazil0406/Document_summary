"""
Standalone image (.png / .jpg / .jpeg / .webp) parser.

Uploaded photos/screenshots (e.g. a photo of a whiteboard, a scanned
worksheet, a screenshot of a slide) are OCR'd via the same Docker
Tesseract path pdf_parser.py already uses for scanned PDF pages --
services.pdf_parser.ocr_png_docker() -- so there is exactly one OCR
implementation in the codebase, not two.

extract_pages() returns a single PageContent per image (same dataclass
PDFParser/DocxParser/PptxParser produce), so images flow through the
existing DocumentChunker / SemanticChunkingService / Qdrant pipeline
unchanged.

Host requirements are identical to the PDF OCR fallback:
  - Docker Desktop running (AMD64 / x86_64)
  - pip install docker Pillow
  - docker pull --platform linux/amd64 tesseractshadow/tesseract4re
"""

import logging
import os
import tempfile
from typing import List

from services.pdf_parser import PageContent, ocr_png_docker

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

# Below this many OCR'd characters, an image is treated as having no
# extractable text (e.g. a decorative photo, a logo) rather than an error.
MIN_TEXT_CHARS: int = 10


class ImageParser:
    """Extracts OCR'd text content from standalone image files."""

    def __init__(self, image_folder: str):
        self.image_folder = image_folder

    def _list_image_files(self) -> List[str]:
        return [
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

    @staticmethod
    def _as_ocr_ready_png(file_path: str, tmpdir: str) -> str:
        """Tesseract's Docker image is driven with a plain PNG in this
        codebase (see pdf_parser._render_page_png) -- convert non-PNG
        inputs (jpg/webp/bmp/tiff) to PNG so ocr_png_docker() always
        receives the format it was built for."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".png":
            return file_path

        from PIL import Image

        png_path = os.path.join(tmpdir, "converted.png")
        with Image.open(file_path) as im:
            im.convert("RGB").save(png_path, "PNG")
        return png_path

    def extract_pages(self, file_path: str) -> List[PageContent]:
        """OCR a standalone image and return it as a single-item PageContent list."""
        filename = os.path.basename(file_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                png_path = self._as_ocr_ready_png(file_path, tmpdir)
                text = ocr_png_docker(png_path, label=filename).strip()
            except Exception as exc:
                logger.error("OCR failed for image '%s': %s", filename, exc)
                raise

        if len(text) < MIN_TEXT_CHARS:
            logger.warning(
                "No meaningful extractable text found in image '%s' (%d chars OCR'd).",
                filename, len(text),
            )
            return []

        metadata = {
            "title": "",
            "author": "",
            "toc_section": "",
        }

        return [
            PageContent(
                filename=filename,
                page_number=1,
                page_label="Image",
                text=text,
                metadata=metadata,
            )
        ]

    def extract_all(self) -> List[PageContent]:
        all_pages: List[PageContent] = []
        image_files = self._list_image_files()
        logger.info("Found %d image file(s) in '%s'", len(image_files), self.image_folder)

        for path in image_files:
            try:
                extracted = self.extract_pages(path)
                logger.info("Extracted %d page(s) from '%s'", len(extracted), os.path.basename(path))
                all_pages.extend(extracted)
            except Exception as exc:
                logger.error("Failed to process '%s': %s", path, exc)

        return all_pages