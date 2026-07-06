"""
Retriever service.

Combines the embedding service and the Qdrant service: converts a user
query into a BGE-M3 embedding, then retrieves the most relevant stored
chunks via similarity search.

Improvements:
  - TOC-aware filtering: when a query mentions a document section name
    (e.g. "unit 2", "chapter 3", "introduction") the retriever tries to
    narrow results to chunks whose toc_section matches, before falling
    back to the full result set.
  - Document-name filtering: "summarize unit2.pdf" or "summarize unit 2"
    correctly isolates chunks from that file.
  - Summary/overview queries fetch a larger candidate set (TOP_K_SUMMARY).
  - A minimum relevance score threshold (MIN_SCORE) is enforced.
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
}

TOC_MATCH_THRESHOLD = 0.80

# Generic words stripped out before fuzzy-matching a query against real
# filenames in the corpus — these carry no identifying signal.
_FILENAME_STOPWORDS = {
    "the", "a", "an", "of", "and", "to", "in", "for", "on", "at", "by",
    "is", "are", "this", "that", "document", "file", "pdf", "presentation",
    "workbook", "report", "doc", "according", "about", "say", "says",
    "what", "does", "unit",
}


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
        # "summarize unit 2", "unit 2 summary", etc.
        if re.search(r"\bunit\s*\d+\b", q) and re.search(
            r"\b(summary|summarize|overview|brief)\b", q
        ):
            return True
        return False

    def _get_top_k(self, query: str) -> int:
        return self.summary_top_k if self._is_summary_query(query) else self.top_k

    # ── Document / section hint extraction ───────────────────────────────────

    def _extract_document_hint(self, query: str) -> str:
        """
        Extract a likely filename stem or document name from the query.

        Handles patterns like:
          - "summarize unit2.pdf"          → "unit2.pdf"
          - "summarize unit 2"             → "unit 2"
          - "summarize unit2"              → "unit2"
          - "give me a summary of unit 2"  → "unit 2"
          - "what is in chapter 3"         → "chapter 3"
          - "tell me about the introduction section" → "introduction"
        """
        normalized = self._normalize_query(query)

        # Explicit PDF filename — no spaces allowed in filename stem
        m = re.search(r"\b([a-z0-9][a-z0-9._-]*\.pdf)\b", normalized)
        if m:
            return m.group(1).strip()

        # "unit N" / "chapter N" / "section N" patterns (with or without space)
        m = re.search(r"\b(unit\s*\d+|chapter\s*\d+|section\s*\d+)\b", normalized)
        if m:
            # normalise spacing so "unit2" → "unit 2"
            raw = m.group(1)
            cleaned = re.sub(r"(unit|chapter|section)\s*(\d+)", r"\1 \2", raw)
            return cleaned.strip()

        # "summarize <word>", "summary of <word>", "overview of <word>"
        for pattern in [
            r"\b(?:summarize|summary of|overview of|brief on|describe)\s+([a-z0-9][a-z0-9_.-]+)\b",
            r"\b([a-z0-9][a-z0-9_.-]+)\s+(?:summary|overview|brief)\b",
        ]:
            m = re.search(pattern, normalized)
            if m:
                candidate = m.group(1).strip(" .")
                # Skip generic filler words
                if candidate not in {"the", "a", "an", "this", "that", "document", "file", "pdf"}:
                    return candidate

        return ""

    def _extract_toc_section_hint(self, query: str) -> str:
        """
        Return a section label hint to filter toc_section payload field.
        E.g. "summarize unit 2" → "unit 2"
        """
        normalized = self._normalize_query(query)
        m = re.search(r"\b(unit\s*\d+|chapter\s*\d+|section\s*\d+)\b", normalized)
        if m:
            raw = m.group(1)
            return re.sub(r"(unit|chapter|section)\s*(\d+)", r"\1 \2", raw).strip()
        return ""

    def _rewrite_query_for_retrieval(self, query: str) -> str:
        """Remove summary-style wording before embedding for retrieval."""
        normalized = self._normalize_query(query)
        if not self._is_summary_query(normalized):
            return normalized
        rewritten = re.sub(
            r"\b(summary|summarize|summarise|overview|brief|executive summary|document summary)\b",
            "",
            normalized,
        )
        rewritten = re.sub(r"\s{2,}", " ", rewritten).strip()
        return rewritten or normalized

    # ── Matching helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _filename_matches(filename: str, hint: str) -> bool:
        """
        Return True when *filename* is consistent with the hint extracted
        from the query.  Matching is fuzzy: the hint can be a bare stem
        ("unit2"), a normalised form ("unit 2"), or a full name
        ("unit2.pdf").
        """
        if not filename or not hint:
            return True
        fn = filename.lower()
        h = hint.lower()

        # Exact or suffix match for explicit PDF filename
        if h.endswith(".pdf"):
            return fn == h or fn.endswith(h)

        # Normalise: collapse spaces and remove extension for comparison
        fn_stem = re.sub(r"\.pdf$", "", fn)
        fn_nospace = re.sub(r"\s+", "", fn_stem)
        h_nospace = re.sub(r"\s+", "", h)

        return (
            h in fn_stem
            or fn_stem.endswith(h)
            or h_nospace in fn_nospace
            or fn_nospace == h_nospace
        )

    @staticmethod
    def _toc_section_matches(toc_section: str, hint: str) -> bool:
        """Return True when *toc_section* contains the section hint."""
        if not toc_section or not hint:
            return False
        ts = toc_section.lower()
        h = hint.lower()
        # normalise "unit 2" ↔ "unit2"
        ts_nospace = re.sub(r"\s+", "", ts)
        h_nospace = re.sub(r"\s+", "", h)
        return h in ts or h_nospace in ts_nospace

    @staticmethod
    def _match_known_filename(query: str, candidate_filenames) -> str:
        """
        Fuzzy-match the raw query text against actual filenames present in
        the candidate result set (no trigger word required).

        This covers plain Q&A that names a document directly, e.g.
        "What does the SkinDisease CDSS presentation say about accuracy?"
        — which has no "summarize"/"unit N" keyword for
        `_extract_document_hint()` to latch onto, but clearly identifies a
        specific document by name. Each filename stem is split into
        significant words (stopwords and short tokens removed); if enough
        of those words appear in the query, that filename becomes the hint.
        """
        q_norm = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
        if not q_norm:
            return ""

        best_match, best_score = "", 0.0
        for fname in candidate_filenames:
            if not fname:
                continue
            stem = re.sub(r"\.pdf$", "", fname.lower())
            words = [
                w for w in re.split(r"[^a-z0-9]+", stem)
                if w and w not in _FILENAME_STOPWORDS and len(w) > 2
            ]
            if not words:
                continue
            hits = sum(1 for w in words if w in q_norm)
            if hits == 0:
                continue
            score = hits / len(words)
            # Require either multiple matching words, or a single-word stem
            # that matches outright — avoids matching on one generic token.
            if score > best_score and (hits >= 2 or len(words) == 1):
                best_score = score
                best_match = fname

        return best_match if best_score >= 0.5 else ""

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
        document_hint = self._extract_document_hint(query)
        toc_hint = self._extract_toc_section_hint(query)

        logger.info(
            "Query intent: %s | top_k=%d | document_hint=%r | toc_hint=%r",
            "SUMMARY" if self._is_summary_query(query) else "Q&A",
            top_k,
            document_hint or "none",
            toc_hint or "none",
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

            if document_hint:
                document_toc_results = [
                    result
                    for result in toc_results
                    if self._filename_matches(
                        result.get("filename", ""),
                        document_hint,
                    )
                ]
                toc_results = document_toc_results

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

        # If no trigger-word pattern matched a document name (e.g. plain Q&A
        # like "What does the SkinDisease CDSS presentation say about X?"),
        # try fuzzy-matching the query directly against filenames actually
        # present in this candidate set.
        if not document_hint:
            candidate_filenames = {r.get("filename", "") for r in filtered}
            document_hint = self._match_known_filename(query, candidate_filenames)
            if document_hint:
                logger.info("Matched document by name from query text: '%s'", document_hint)

        # ── TOC section filtering (most specific, try first) ─────────────────
        if toc_hint:
            toc_filtered = [
                r for r in filtered
                if self._toc_section_matches(r.get("toc_section", ""), toc_hint)
            ]
            if toc_filtered:
                logger.info(
                    "TOC section filter '%s' matched %d chunk(s).",
                    toc_hint, len(toc_filtered),
                )
                filtered = toc_filtered
            else:
                logger.info(
                    "TOC section filter '%s' matched nothing — "
                    "falling through to filename filter.",
                    toc_hint,
                )

        # ── Filename / document-name filtering ───────────────────────────────
        if document_hint:
            doc_filtered = [
                r for r in filtered
                if self._filename_matches(r.get("filename", ""), document_hint)
            ]
            if doc_filtered:
                logger.info(
                    "Document hint '%s' matched %d chunk(s).",
                    document_hint, len(doc_filtered),
                )
                filtered = doc_filtered
            else:
                logger.info(
                    "Document hint '%s' matched no filenames — "
                    "returning all relevance-filtered chunks.",
                    document_hint,
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
