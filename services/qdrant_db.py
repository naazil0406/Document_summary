"""
Qdrant vector database service.

Handles collection creation, upserting chunk embeddings with their
metadata (filename, page number, chunk id, chunk text, toc_section),
and similarity search against the `company_docs` collection.

Qdrant is run via Docker Desktop — see the README for the one-liner
`docker run` command.  No local Qdrant binary is required.
"""

import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from config.settings import settings
from services.chunking import Chunk

logger = logging.getLogger(__name__)


class QdrantService:
    """Wraps Qdrant client operations needed by the RAG pipeline."""

    def __init__(self, url: str, collection_name: str, api_key: Optional[str] = None):
        self.collection_name = collection_name
        try:
            self.client = QdrantClient(url=url, api_key=api_key or None)
        except Exception as exc:
            logger.error("Failed to connect to Qdrant at %s: %s", url, exc)
            raise

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it does not already exist."""
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection_name in existing:
                logger.info("Collection '%s' already exists.", self.collection_name)
                return

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info(
                "Created collection '%s' with vector size %d.",
                self.collection_name, vector_size,
            )
        except Exception as exc:
            logger.error("Failed to ensure collection '%s': %s", self.collection_name, exc)
            raise

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Store chunk embeddings together with their text and metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        points = [
            qdrant_models.PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "type": "chunk",
                    "text": chunk.text,
                    "filename": chunk.filename,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "page_label": chunk.page_label,
                    "chunk_id": chunk.chunk_id,
                    # toc_section is stored at the top level for easy filtering
                    "toc_section": chunk.toc_section or (chunk.metadata or {}).get(
                        "toc_section", ""
                    ),
                    "metadata": chunk.metadata or {},
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info("Upserted %d points into '%s'.", len(points), self.collection_name)
        except Exception as exc:
            logger.error("Failed to upsert points into Qdrant: %s", exc)
            raise

    def delete_document(self, filename: str) -> None:
        """Remove all existing content/TOC points for one document.

        Ingestion uses this before writing replacement chunks so re-indexing a
        workbook cannot leave stale vectors from an older parsing strategy.
        """
        if not filename:
            raise ValueError("filename is required")
        selector = qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="filename",
                        match=qdrant_models.MatchValue(value=filename),
                    )
                ]
            )
        )
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=selector,
                wait=True,
            )
            logger.info(
                "Removed existing indexed points for document '%s'.",
                filename,
            )
        except Exception as exc:
            logger.error("Failed to replace indexed document '%s': %s", filename, exc)
            raise

    def upsert_toc_entries(
        self,
        toc_entries: List[dict],
        embeddings: List[List[float]],
    ) -> None:
        """Store embedded TOC entries in the same collection as semantic chunks."""
        if len(toc_entries) != len(embeddings):
            raise ValueError("Number of TOC entries and embeddings must match.")

        points = []
        for entry, embedding in zip(toc_entries, embeddings):
            identity = (
                f"{entry['filename']}:{entry['level']}:{entry['title']}:"
                f"{entry['page_start']}:{entry['page_end']}"
            )
            points.append(
                qdrant_models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                    vector=embedding,
                    payload={
                        "type": "toc",
                        "title": entry["title"],
                        "page_start": entry["page_start"],
                        "page_end": entry["page_end"],
                        "filename": entry["filename"],
                    },
                )
            )

        if not points:
            return

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(
                "Upserted %d TOC points into '%s'.",
                len(points), self.collection_name,
            )
        except Exception as exc:
            logger.error("Failed to upsert TOC points into Qdrant: %s", exc)
            raise

    def search(self, query_vector: List[float], top_k: int = settings.TOP_K) -> List[dict]:
        """Find the top_k most similar chunks to the given query vector."""
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.error("Qdrant search failed: %s", exc)
            raise

        hits = []
        for r in results:
            payload = r.payload or {}
            hits.append(
                {
                    "score": r.score,
                    "text": payload.get("text", ""),
                    "filename": payload.get("filename", ""),
                    "page_start": payload.get("page_start", -1),
                    "page_end": payload.get("page_end", -1),
                    "page_label": payload.get("page_label", "N/A"),
                    "chunk_id": payload.get("chunk_id", ""),
                    "toc_section": payload.get("toc_section", ""),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return hits

    def search_toc(
        self,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[dict]:
        """Search only embedded Table-of-Contents entries."""
        toc_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="type",
                    match=qdrant_models.MatchValue(value="toc"),
                )
            ]
        )
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=toc_filter,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.error("Qdrant TOC search failed: %s", exc)
            raise

        return [
            {
                "title": (result.payload or {}).get("title", ""),
                "page_start": (result.payload or {}).get("page_start", -1),
                "page_end": (result.payload or {}).get("page_end", -1),
                "filename": (result.payload or {}).get("filename", ""),
                "score": result.score,
            }
            for result in results
        ]

    def retrieve_section(
        self,
        filename: str,
        page_start: int,
        page_end: int,
    ) -> List[dict]:
        """Return every stored content chunk fully inside an inclusive page range."""
        if page_start < 1 or page_end < page_start:
            raise ValueError("Invalid section page range.")

        section_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="filename",
                    match=qdrant_models.MatchValue(value=filename),
                ),
                qdrant_models.FieldCondition(
                    key="page_start",
                    range=qdrant_models.Range(gte=page_start),
                ),
                qdrant_models.FieldCondition(
                    key="page_end",
                    range=qdrant_models.Range(lte=page_end),
                ),
            ],
            # Old chunk payloads have no type field, so excluding only TOC
            # points keeps retrieval backward compatible with existing data.
            must_not=[
                qdrant_models.FieldCondition(
                    key="type",
                    match=qdrant_models.MatchValue(value="toc"),
                )
            ],
        )

        records = []
        offset = None
        try:
            while True:
                batch, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=section_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                records.extend(batch)
                if offset is None:
                    break
        except Exception as exc:
            logger.error(
                "Qdrant section retrieval failed for %s pages %d-%d: %s",
                filename, page_start, page_end, exc,
            )
            raise

        hits = []
        for record in records:
            payload = record.payload or {}
            hits.append(
                {
                    "score": 1.0,
                    "text": payload.get("text", ""),
                    "filename": payload.get("filename", ""),
                    "page_start": payload.get("page_start", -1),
                    "page_end": payload.get("page_end", -1),
                    "page_label": payload.get("page_label", "N/A"),
                    "chunk_id": payload.get("chunk_id", ""),
                    "toc_section": payload.get("toc_section", ""),
                    "metadata": payload.get("metadata", {}),
                }
            )

        hits.sort(
            key=lambda hit: (
                hit["page_start"],
                hit["page_end"],
                hit["chunk_id"],
            )
        )
        return hits

    def retrieve_document(self, filename: str) -> List[dict]:
        """Return all content chunks already indexed for one PDF.

        This reads stored payloads only; it never parses, OCRs, chunks, or
        embeds the source document again.
        """
        document_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="filename",
                    match=qdrant_models.MatchValue(value=filename),
                )
            ],
            # Old chunks may not have a type, so exclude only TOC records.
            must_not=[
                qdrant_models.FieldCondition(
                    key="type",
                    match=qdrant_models.MatchValue(value="toc"),
                )
            ],
        )

        records = []
        offset = None
        try:
            while True:
                batch, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=document_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                records.extend(batch)
                if offset is None:
                    break
        except Exception as exc:
            logger.error(
                "Qdrant document retrieval failed for %s: %s",
                filename,
                exc,
            )
            raise

        hits = []
        for record in records:
            payload = record.payload or {}
            text = payload.get("text", "")
            if not text.strip():
                continue
            hits.append(
                {
                    "score": 1.0,
                    "text": text,
                    "filename": payload.get("filename", ""),
                    "page_start": payload.get("page_start", -1),
                    "page_end": payload.get("page_end", -1),
                    "page_label": payload.get("page_label", "N/A"),
                    "chunk_id": payload.get("chunk_id", ""),
                    "toc_section": payload.get("toc_section", ""),
                    "metadata": payload.get("metadata", {}),
                }
            )

        hits.sort(
            key=lambda hit: (
                hit["page_start"],
                hit["page_end"],
                hit["chunk_id"],
            )
        )
        logger.info(
            "Retrieved %d indexed chunks for document '%s'.",
            len(hits),
            filename,
        )
        return hits
