"""
Re-ranking Service using sentence-transformers CrossEncoder (e.g. BAAI/bge-reranker-v2-m3).

Takes candidate retrieved chunks from hybrid vector/keyword search and re-ranks
them using a dedicated neural CrossEncoder model to produce the final top relevant chunks.
"""

import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class ReRankerService:
    """Neural re-ranker for ordering candidate RAG chunks by exact query relevance."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                logger.info("Loading CrossEncoder re-ranker model '%s' on %s...", self.model_name, self.device)
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                logger.warning(
                    "Could not initialize CrossEncoder '%s': %s. Re-ranker will pass through original scores.",
                    self.model_name,
                    exc,
                )
                self._model = False
        return self._model if self._model is not False else None

    def rerank(
        self,
        query: str,
        chunks: List[dict],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Re-rank candidate chunk dicts against query using CrossEncoder."""
        if not chunks or not query:
            return chunks

        model = self._load_model()
        if model is None:
            return chunks[:top_k] if top_k else chunks

        try:
            pairs = [(query, c.get("text", "")) for c in chunks]
            scores = model.predict(pairs)

            reranked = []
            for chunk, score in zip(chunks, scores):
                chunk_copy = dict(chunk)
                chunk_copy["rerank_score"] = float(score)
                # Keep original score in original_score
                chunk_copy["vector_score"] = chunk.get("score", 0.0)
                chunk_copy["score"] = float(score)
                reranked.append(chunk_copy)

            reranked.sort(key=lambda x: x["score"], reverse=True)

            logger.info(
                "Re-ranked %d candidates for query '%s'. Top score: %.4f",
                len(chunks), query[:40], reranked[0]["score"] if reranked else 0.0
            )

            return reranked[:top_k] if top_k else reranked

        except Exception as exc:
            logger.error("Error during re-ranking: %s", exc)
            return chunks[:top_k] if top_k else chunks
