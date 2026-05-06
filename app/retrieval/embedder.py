"""
Embedding model wrapper.

Wraps sentence-transformers with:
  - Thread-safe lazy initialisation
  - Batch embedding support (used during ingestion)
  - Normalised vectors (required for cosine similarity in Qdrant)
"""
import logging
import threading
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Thread-safe wrapper around a SentenceTransformer model.

    One instance lives in app.state for the process lifetime.
    The underlying model is not thread-safe during init, so we use a lock.
    After init, encode() is stateless and safe to call concurrently.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._lock = threading.Lock()
        self._model: SentenceTransformer | None = None
        logger.info("Embedder initialised with model=%s", model_name)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                # Double-checked locking: another thread may have initialised
                # the model while we were waiting for the lock.
                if self._model is None:
                    logger.info("Loading SentenceTransformer: %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """
        Embed a single string. Returns a normalised float list.
        Normalisation is important: Qdrant's cosine distance expects unit vectors
        for correct similarity scores.
        """
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Embed a list of strings in batches.
        batch_size=64 is a sensible default for CPU inference on MiniLM.
        Returns list of normalised float lists.
        """
        logger.info("Embedding %d texts in batches of %d", len(texts), batch_size)
        vecs: np.ndarray = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vecs.tolist()

    def dimension(self) -> int:
        """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()
