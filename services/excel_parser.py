"""
Tabular data parser: Excel workbooks (.xlsx / .xlsm) and CSV files (.csv).

Spreadsheets/CSVs have no notion of "pages" the way PDFs do — the natural
unit is the SHEET (a CSV file has exactly one, implicit sheet). Each sheet
becomes one PageContent, exactly like each PDF page becomes one
PageContent, so the file flows through the existing DocumentChunker /
SemanticChunkingService / Qdrant pipeline unchanged.

Each row is flattened into a single readable line rather than kept as a
raw grid, since that's what both the structural chunker (which splits on
paragraph-like boundaries) and the LLM downstream can actually reason
about:

    Header row (first non-empty row) supplies column names -> each
    following row becomes "ColumnA: value | ColumnB: value | ...",
    skipping empty cells so sparse spreadsheets don't produce noisy
    "ColumnC: " fragments.

    If a sheet/CSV has no usable header row, rows fall back to plain
    "value | value | ..." lines instead.

.xlsx/.xlsm formulas are read as their last-calculated VALUE (openpyxl
data_only=True), not as formula text — a trainee reading "Total: 4820" is
what matters, not "=SUM(B2:B40)".

CSV is read with Python's built-in csv module — no extra dependency,
since a CSV file is just delimited plain text (unlike .xlsx, which is a
zipped XML package that genuinely needs openpyxl to parse).
"""

import csv
import logging
import os
from typing import List, Sequence

from openpyxl import load_workbook

from services.pdf_parser import PageContent

logger = logging.getLogger(__name__)

_EXCEL_EXTENSIONS = (".xlsx", ".xlsm")
_CSV_EXTENSIONS = (".csv",)
_TABULAR_EXTENSIONS = _EXCEL_EXTENSIONS + _CSV_EXTENSIONS


def _cell_display(value) -> str:
    """Render a cell value as clean display text ('' for empty/None)."""
    if value is None:
        return ""
    return str(value).strip()


def _flatten_rows(rows: Sequence[Sequence]) -> str:
    """Turn a grid of rows (worksheet rows or CSV rows) into readable
    'Column: value' lines, shared by both the .xlsx/.xlsm and .csv paths."""
    # Drop fully-empty rows (common at the end of exported sheets/CSVs).
    rows = [r for r in rows if any(_cell_display(v) for v in r)]
    if not rows:
        return ""

    header = [_cell_display(v) for v in rows[0]]
    has_header = any(header)

    lines: List[str] = []
    data_rows = rows[1:] if has_header else rows
    for row in data_rows:
        if has_header:
            pairs = [
                f"{header[i]}: {_cell_display(v)}"
                for i, v in enumerate(row)
                if i < len(header) and header[i] and _cell_display(v)
            ]
        else:
            pairs = [_cell_display(v) for v in row if _cell_display(v)]
        if pairs:
            lines.append(" | ".join(pairs))

    return "\n".join(lines)


class ExcelParser:
    """Extracts content from .xlsx / .xlsm / .csv files, one PageContent per sheet."""

    def __init__(self, excel_folder: str):
        self.excel_folder = excel_folder

    def _list_excel_files(self) -> List[str]:
        return [
            os.path.join(self.excel_folder, f)
            for f in os.listdir(self.excel_folder)
            if f.lower().endswith(_TABULAR_EXTENSIONS)
        ]

    @staticmethod
    def _extract_workbook_pages(file_path: str, filename: str) -> List[PageContent]:
        """One PageContent per sheet in an .xlsx/.xlsm workbook."""
        pages: List[PageContent] = []
        try:
            workbook = load_workbook(file_path, data_only=True, read_only=True)
        except Exception as exc:
            logger.error("Could not open '%s': %s", filename, exc)
            return []

        for sheet_index, sheet_name in enumerate(workbook.sheetnames, start=1):
            sheet = workbook[sheet_name]
            rows = list(sheet.iter_rows(values_only=True))
            text = _flatten_rows(rows)
            if not text.strip():
                logger.info("Sheet '%s' in '%s' is empty — skipping.", sheet_name, filename)
                continue

            pages.append(
                PageContent(
                    filename=filename,
                    page_number=sheet_index,
                    page_label=f"Sheet: {sheet_name}",
                    text=text,
                    # toc_section = sheet name so the retriever's lightweight
                    # section-hint filtering (see services/retriever.py) can
                    # narrow a query like "what's on the Pricing sheet" to
                    # just that sheet's rows.
                    metadata={"toc_section": sheet_name},
                )
            )

        workbook.close()
        return pages

    @staticmethod
    def _extract_csv_pages(file_path: str, filename: str) -> List[PageContent]:
        """A CSV file has exactly one implicit sheet -> one PageContent."""
        try:
            # utf-8-sig quietly strips a BOM if Excel added one on export.
            with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
                # Sniff the delimiter (comma vs semicolon vs tab) instead of
                # assuming comma, since exports from non-English locales
                # commonly use ';'.
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel  # comma-delimited fallback
                rows = list(csv.reader(f, dialect))
        except Exception as exc:
            logger.error("Could not open '%s': %s", filename, exc)
            return []

        text = _flatten_rows(rows)
        if not text.strip():
            logger.info("'%s' has no extractable rows — skipping.", filename)
            return []

        return [
            PageContent(
                filename=filename,
                page_number=1,
                page_label="CSV Data",
                text=text,
                metadata={"toc_section": ""},
            )
        ]

    def extract_pages(self, file_path: str) -> List[PageContent]:
        """Read an .xlsx/.xlsm/.csv as training content, one PageContent per sheet."""
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext in _CSV_EXTENSIONS:
            return self._extract_csv_pages(file_path, filename)
        return self._extract_workbook_pages(file_path, filename)

    def extract_all(self) -> List[PageContent]:
        all_pages: List[PageContent] = []
        excel_files = self._list_excel_files()
        logger.info("Found %d tabular file(s) in '%s'", len(excel_files), self.excel_folder)

        for path in excel_files:
            try:
                extracted = self.extract_pages(path)
                logger.info("Extracted %d sheet(s) from '%s'", len(extracted), os.path.basename(path))
                all_pages.extend(extracted)
            except Exception as exc:
                logger.error("Failed to process '%s': %s", path, exc)

        return all_pages