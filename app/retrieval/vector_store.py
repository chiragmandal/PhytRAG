"""
Qdrant vector store wrapper.

Abstracts the Qdrant client behind a simple interface so:
  - The rest of the app never imports qdrant-client directly
  - Swapping to a different vector DB only requires editing this file
  - Unit tests can mock this class cleanly

Payload schema stored per chunk:
  {
    "pmcid":       "PMC1234567",
    "title":       "Paper title",
    "chunk_index": 3,
    "text":        "the actual chunk text",
    "authors":     "Smith J, Doe A",
  }
"""
import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)

# Batch size for upsert operations. 100 is safe for all-MiniLM-L6-v2 vectors
# (384 floats * 4 bytes * 100 = ~154 KB per batch, well within gRPC limits).
UPSERT_BATCH_SIZE = 100


class VectorStore:
    def __init__(self, url: str, collection_name: str, vector_size: int = 384) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = QdrantClient(url=url, timeout=30)
        logger.info("VectorStore connected to Qdrant at %s", url)

    # ── Collection management ────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """
        Create the collection if it does not exist.
        Idempotent: safe to call on every startup.
        Using Cosine distance because we store normalised vectors.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            logger.info("Creating Qdrant collection: %s", self.collection_name)
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        else:
            logger.info("Qdrant collection already exists: %s", self.collection_name)

    def collection_info(self) -> dict[str, Any]:
        info = self._client.get_collection(self.collection_name)
        return {
            "vectors_count": info.points_count,
            "points_count": info.points_count,
            "status": info.status,
        }

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def upsert(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        """
        Upsert chunks in batches. Vectors and payloads must have the same length.
        Each point gets a UUID so re-ingestion of the same paper creates new points
        rather than overwriting (safe for the demo; a production system would use
        a deterministic ID based on pmcid + chunk_index).
        """
        assert len(vectors) == len(payloads), "vectors and payloads must be same length"

        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload,
            )
            for vec, payload in zip(vectors, payloads)
        ]

        total = len(points)
        logger.info("Upserting %d points to collection '%s'", total, self.collection_name)

        for i in range(0, total, UPSERT_BATCH_SIZE):
            batch = points[i : i + UPSERT_BATCH_SIZE]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,  # Wait for indexing before returning - important for consistency
            )
            logger.debug("Upserted batch %d/%d", min(i + UPSERT_BATCH_SIZE, total), total)

        logger.info("Upsert complete: %d points", total)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k most similar chunks, filtered by score_threshold.
        Result format is a plain list of dicts so the caller (routes.py) has
        no dependency on qdrant-client types.
        """
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {
                "pmcid": r.payload.get("pmcid", "unknown"),
                "title": r.payload.get("title", "Unknown title"),
                "text": r.payload.get("text", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "authors": r.payload.get("authors", ""),
                "score": r.score,
            }
            for r in results
        ]

    def close(self) -> None:
        self._client.close()
