"""
Retriever service.

Combines the embedding service and the Qdrant service: converts a user
query into a BGE-M3 embedding, then retrieves the most relevant stored
chunks via similarity search.

Matching strategy:
  - Document/filename-based matching has been intentionally removed.
    Filenames in this deployment are inconsistent/messy in the UI, so they
    are not a reliable signal.
  - Instead, the retriever matches purely on a "unit / chapter / section +
    number" hint extracted from the query (e.g. "unit 2", "chapter 3",
    "section 1"). This hint is matched against each TOC entry's own title
    (for the fast TOC-lookup path) and against each chunk's toc_section
    payload field (for the semantic-search fallback path) — never against
    filename. This correctly distinguishes "Unit 1" from "Unit 2" even when
    every unit lives inside the same physical file.
  - Summary/overview/infographic/image queries fetch a larger candidate set
    (TOP_K_SUMMARY) and skip the TOC-first shortcut in favor of broader
    semantic retrieval.
  - A minimum relevance score threshold (MIN_SCORE) is enforced, with a
    keyword-overlap fallback for cases where semantic scores run low
    (e.g. tabular/workbook content).
"""

import logging
import os
import re
from typing import List, Optional

from config.settings import settings
from services.embeddings import EmbeddingService
from services.qdrant_db import QdrantService

logger = logging.getLogger(__name__)

SUMMARY_KEYWORDS = {
    "summary",
    "summarize",
    "summarise",
    "overview",
    "brief",
    "document summary",
    "current pdf",
    "entire document",
    "whole document",
    "what is this document",
    "what does this document",
    "describe this document",
    "tell me about this document",
    "infographic",
    "visual summary",
    "visual overview",
    "image for",
    "image of",
    "picture for",
    "picture of",
}

TOC_MATCH_THRESHOLD = 0.80


class Retriever:
    """Retrieves the most relevant chunks for a given user query."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        top_k: int = settings.TOP_K,
        summary_top_k: int = settings.TOP_K_SUMMARY,
        min_relevance_score: float = settings.MIN_RELEVANCE_SCORE,
        toc_match_threshold: float = TOC_MATCH_THRESHOLD,
    ):
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.top_k = top_k
        self.summary_top_k = summary_top_k
        self.min_relevance_score = min_relevance_score
        self.toc_match_threshold = toc_match_threshold

    # ── Normalisation ────────────────────────────────────────────────────────

    def _normalize_query(self, query: str) -> str:
        normalized = query.lower().strip()
        normalized = re.sub(r"\bsumarize\b",  "summarize", normalized)
        normalized = re.sub(r"\bsummarise\b", "summarize", normalized)
        # Only expand "unit2" → "unit 2" when NOT followed by a file extension
        # (to avoid corrupting "unit2.pdf" → "unit 2.pdf")
        normalized = re.sub(r"\bunit\s*(\d+)\b(?![\.\w])", r"unit \1", normalized)
        return normalized

    # ── Intent detection ─────────────────────────────────────────────────────

    def _is_summary_query(self, query: str) -> bool:
        q = self._normalize_query(query)
        if any(keyword in q for keyword in SUMMARY_KEYWORDS):
            return True
        # "summarize unit 2", "unit 2 summary", "infographic of unit 2",
        # "image for unit 2", etc.
        if re.search(r"\bunit\s*\d+\b", q) and re.search(
            r"\b(summary|summarize|overview|brief|infographic|image|picture)\b", q
        ):
            return True
        return False

    def _get_top_k(self, query: str) -> int:
        return self.summary_top_k if self._is_summary_query(query) else self.top_k

    # ── Unit / chapter / section hint extraction ─────────────────────────────

    def _extract_unit_hint(self, query: str) -> str:
        """
        Extract a "unit N" / "chapter N" / "section N" hint from the query.
        This is the ONLY scoping signal this retriever uses — there is no
        filename/document-name matching, since filenames are inconsistent
        in this deployment and not a reliable way to identify content.

        Examples:
          - "give me an image for unit2"   → "unit 2"
          - "infographic of unit 2"        → "unit 2"
          - "summarize chapter 3"          → "chapter 3"
          - "what is in section 1"         → "section 1"
        """
        normalized = self._normalize_query(query)
        m = re.search(r"\b(unit\s*\d+|chapter\s*\d+|section\s*\d+)\b", normalized)
        if m:
            raw = m.group(1)
            # normalise spacing so "unit2" → "unit 2"
            return re.sub(r"(unit|chapter|section)\s*(\d+)", r"\1 \2", raw).strip()
        return ""

    def extract_unit_hint(self, query: str) -> str:
        """Public wrapper around _extract_unit_hint, for callers outside this
        class (e.g. the image-generation pipeline) that want to know whether
        a query names a specific unit/chapter/section.
        """
        return self._extract_unit_hint(query)

    def _rewrite_query_for_retrieval(self, query: str) -> str:
        """Remove summary-style wording before embedding for retrieval."""
        normalized = self._normalize_query(query)
        if not self._is_summary_query(normalized):
            return normalized
        rewritten = re.sub(
            r"\b(summary|summarize|summarise|overview|brief|executive summary|"
            r"document summary|infographic|visual summary|visual overview)\b",
            "",
            normalized,
        )
        rewritten = re.sub(r"\s{2,}", " ", rewritten).strip()
        return rewritten or normalized

    # ── Matching helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _title_matches(title: str, hint: str) -> bool:
        """Return True when a TOC entry's *title* is consistent with a
        unit/chapter/section hint extracted from the query (e.g. "unit 2").

        This is what distinguishes "Unit 1" from "Unit 2" when every unit
        lives in the same file — only the TOC entry's own title carries
        that distinction, since filename is no longer used for matching.

        Titles in this deployment follow the format
        "Unit <number> - <unique number>" (e.g. "Unit 2 - 100234"). Matching
        is done by extracting the keyword ("unit"/"chapter"/"section") and
        its number from both the hint and the title and comparing them
        exactly — NOT by substring matching. Substring matching on a hint
        like "unit 2" would incorrectly match a title like
        "Unit 20 - 100235" (since "unit2" is a substring of "unit20"), so
        exact number comparison is required to tell "Unit 2" apart from
        "Unit 20", "Unit 21", "Unit 200", etc. The trailing unique number
        after the dash is ignored entirely for matching purposes.
        """
        if not title or not hint:
            return False
        hint_keyword, hint_number = Retriever._extract_keyword_number(hint)
        title_keyword, title_number = Retriever._extract_keyword_number(title)
        if hint_keyword and title_keyword:
            return hint_keyword == title_keyword and hint_number == title_number
        # Hint or title didn't parse as "keyword number" (unexpected format) —
        # fall back to a plain substring check rather than silently failing.
        return hint.lower() in title.lower()

    @staticmethod
    def _toc_section_matches(toc_section: str, hint: str) -> bool:
        """Return True when a chunk's *toc_section* payload field is
        consistent with the unit/chapter/section hint extracted from the
        query. Uses the same exact keyword+number comparison as
        _title_matches, for the same reason (avoid "unit 2" matching
        "unit 20").
        """
        if not toc_section or not hint:
            return False
        hint_keyword, hint_number = Retriever._extract_keyword_number(hint)
        section_keyword, section_number = Retriever._extract_keyword_number(toc_section)
        if hint_keyword and section_keyword:
            return hint_keyword == section_keyword and hint_number == section_number
        return hint.lower() in toc_section.lower()

    @staticmethod
    def _extract_keyword_number(text: str):
        """Extract a (keyword, number) pair like ("unit", "2") from a string
        such as "unit 2", "Unit 2 - 100234", or "Unit2". Returns (None, None)
        if no unit/chapter/section + number pattern is found. The number is
        compared as a normalized string (leading zeros stripped) so "unit 02"
        and "unit 2" are still treated as the same unit.
        """
        if not text:
            return None, None
        m = re.search(r"\b(unit|chapter|section)\s*0*(\d+)\b", text.lower())
        if m:
            return m.group(1), m.group(2)
        return None, None

    # ── Main retrieval ────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[dict]:
        """Embed the query and return the most relevant chunks above MIN_RELEVANCE_SCORE."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        normalized_query = self._normalize_query(query)
        retrieval_query = self._rewrite_query_for_retrieval(normalized_query)

        logger.info(
            "Embedding query: %s | normalized: %s | retrieval: %s",
            query, normalized_query, retrieval_query,
        )

        query_vector = self.embedding_service.embed_query(retrieval_query)

        top_k = self._get_top_k(query)
        unit_hint = self._extract_unit_hint(query)

        logger.info(
            "Query intent: %s | top_k=%d | unit_hint=%r",
            "SUMMARY" if self._is_summary_query(query) else "Q&A",
            top_k,
            unit_hint or "none",
        )

        # Summary requests retain the existing summary retrieval behavior.
        # For Q&A, TOC lookup gets first refusal before semantic retrieval.
        if not self._is_summary_query(query):
            logger.info("Searching TOC...")
            try:
                toc_results = self.qdrant_service.search_toc(query_vector, top_k=10)
            except Exception as exc:
                logger.warning(
                    "TOC search unavailable (%s). Using semantic retrieval.",
                    exc,
                )
                toc_results = []

            # Narrow by unit/chapter/section title. This is the ONLY
            # scoping applied here — no filename matching. Only fall back
            # to the unfiltered set if nothing actually matches the hint,
            # so we never silently prefer a wrong-unit result just because
            # it scored higher semantically.
            if unit_hint:
                title_filtered = [
                    result
                    for result in toc_results
                    if self._title_matches(result.get("title", ""), unit_hint)
                ]
                if title_filtered:
                    toc_results = title_filtered
                else:
                    logger.info(
                        "TOC title filter '%s' matched no TOC entries — "
                        "falling back to unfiltered TOC candidates.",
                        unit_hint,
                    )

            eligible_toc_results = [
                result
                for result in toc_results
                if result.get("score", 0.0) >= self.toc_match_threshold
            ]
            toc_match = (
                max(
                    eligible_toc_results,
                    key=lambda result: result.get("score", 0.0),
                )
                if eligible_toc_results
                else None
            )

            if toc_match:
                logger.info(
                    "TOC Match Found:\n%s\n\nPages:\n%d-%d",
                    toc_match.get("title", ""),
                    toc_match.get("page_start", -1),
                    toc_match.get("page_end", -1),
                )
                try:
                    section_chunks = self.qdrant_service.retrieve_section(
                        filename=toc_match.get("filename", ""),
                        page_start=toc_match.get("page_start", -1),
                        page_end=toc_match.get("page_end", -1),
                    )
                except Exception as exc:
                    logger.warning(
                        "TOC section retrieval failed (%s). "
                        "Using semantic retrieval.",
                        exc,
                    )
                    section_chunks = []
                if section_chunks:
                    logger.info(
                        "Retrieved %d chunks from TOC section.",
                        len(section_chunks),
                    )
                    return section_chunks

                logger.warning(
                    "TOC match had no stored chunks. Using semantic retrieval."
                )
            else:
                logger.info("No TOC match.\n\nUsing semantic retrieval.")
        else:
            logger.info("Summary query detected. Skipping TOC search.")

        results = self.qdrant_service.search(query_vector, top_k=top_k)
        # TOC points intentionally share the collection but are navigation
        # records, not answer context. Existing chunks always contain text.
        results = [result for result in results if result.get("text", "").strip()]
        logger.info("Retrieved %d chunks from Qdrant.", len(results))

        # ── Filter by minimum relevance score ────────────────────────────────
        filtered = [r for r in results if r.get("score", 0.0) >= self.min_relevance_score]

        # The Bedrock embeddings can yield low absolute similarity scores for
        # tabular/workbook content, so add a keyword overlap fallback before
        # giving up. This keeps Excel questions from failing when the semantic
        # scores are weak but the chunk text still clearly matches the query.
        if not filtered and results:
            query_terms = set(re.findall(r"[a-z0-9]+", retrieval_query.lower()))
            query_terms = {term for term in query_terms if len(term) > 2}
            keyword_matches = []
            for result in results:
                text = (result.get("text", "") or "").lower()
                terms = set(re.findall(r"[a-z0-9]+", text))
                overlap = len(query_terms & terms)
                if overlap:
                    keyword_matches.append((overlap, result))
            if keyword_matches:
                keyword_matches.sort(key=lambda item: item[0], reverse=True)
                filtered = [result for _, result in keyword_matches[: min(8, len(keyword_matches))]]
                logger.info(
                    "Keyword fallback matched %d chunk(s) using %d shared term(s).",
                    len(filtered),
                    max(overlap for overlap, _ in keyword_matches),
                )

        if len(filtered) < len(results):
            logger.info(
                "Dropped %d low-relevance chunks (score < %.2f). Kept %d.",
                len(results) - len(filtered), self.min_relevance_score, len(filtered),
            )

        # Fall back to raw results if everything was below threshold
        if not filtered and results:
            logger.warning(
                "No chunk passed the relevance threshold %.2f. "
                "Falling back to top %d retrieved chunks.",
                self.min_relevance_score, len(results),
            )
            filtered = results

        # ── Unit / chapter / section filtering (the only scoping signal) ─────
        if unit_hint:
            unit_filtered = [
                r for r in filtered
                if self._toc_section_matches(r.get("toc_section", ""), unit_hint)
            ]
            if unit_filtered:
                logger.info(
                    "Unit hint '%s' matched %d chunk(s).",
                    unit_hint, len(unit_filtered),
                )
                filtered = unit_filtered
            else:
                logger.info(
                    "Unit hint '%s' matched no chunks — "
                    "returning all relevance-filtered chunks.",
                    unit_hint,
                )

        if filtered:
            logger.info(
                "Top result — score: %.4f | section: %r | page: %s | file: %s",
                filtered[0].get("score", 0.0),
                filtered[0].get("toc_section", ""),
                filtered[0].get("page_label", "N/A"),
                filtered[0].get("filename", "Unknown"),
            )
        else:
            logger.info("No chunks met the minimum relevance threshold.")

        return filtered