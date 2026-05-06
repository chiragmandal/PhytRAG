#!/usr/bin/env python3
"""
Chunk, embed, and upsert the plant biology corpus into Qdrant.

Pipeline:
  corpus.jsonl  -->  chunks  -->  embeddings  -->  Qdrant

Chunking strategy: sliding window over words.
  chunk_size=400 words, overlap=40 words.
  400 words ~ 500 tokens ~ comfortable context window for retrieval.
  Overlap prevents splitting concepts across chunk boundaries.

Embedding: all-MiniLM-L6-v2 (384 dims, CPU-friendly, fast).

Run AFTER `make up` (Qdrant must be running).

Usage:
    python -m ingestion.chunk_and_embed
    python -m ingestion.chunk_and_embed --chunk-size 300 --overlap 30
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = Path(__file__).parent / "data" / "corpus.jsonl"


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> list[str]:
    """
    Sliding window word-level chunker.

    Word-level (not character-level) because:
      - Consistent semantic density per chunk
      - No mid-word splits that confuse the embedding model
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk) > 50:  # skip tiny trailing chunks
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def load_corpus(path: Path) -> list[dict]:
    """Load JSONL corpus. Each line is one paper dict."""
    if not path.exists():
        logger.error(
            "Corpus not found at %s. Run: python -m ingestion.download_corpus", path
        )
        sys.exit(1)

    papers = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    logger.info("Loaded %d papers from %s", len(papers), path)
    return papers


# ── Main ingestion pipeline ───────────────────────────────────────────────────

def ingest(chunk_size: int = 400, overlap: int = 40) -> None:
    settings = get_settings()

    logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
    vector_store = VectorStore(
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
        vector_size=settings.vector_size,
    )
    vector_store.ensure_collection()

    logger.info("Loading embedding model: %s", settings.embed_model)
    embedder = Embedder(model_name=settings.embed_model)

    papers = load_corpus(CORPUS_PATH)

    all_vectors: list[list[float]] = []
    all_payloads: list[dict] = []
    total_chunks = 0

    logger.info("Chunking %d papers (size=%d, overlap=%d words)", len(papers), chunk_size, overlap)
    for paper in tqdm(papers, desc="Chunking papers"):
        text = paper.get("full_text") or paper.get("abstract", "")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        total_chunks += len(chunks)

        for idx, chunk in enumerate(chunks):
            all_payloads.append(
                {
                    "pmcid": paper["pmcid"],
                    "title": paper["title"],
                    "authors": paper.get("authors", ""),
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

    logger.info("Total chunks to embed: %d", total_chunks)

    # Embed in one batch call (shows progress bar internally)
    texts_to_embed = [p["text"] for p in all_payloads]
    all_vectors = embedder.embed_batch(texts_to_embed, batch_size=64)

    # Upsert into Qdrant
    vector_store.upsert(vectors=all_vectors, payloads=all_payloads)

    # Final check
    info = vector_store.collection_info()
    logger.info(
        "Ingestion complete. Collection '%s' now has %d vectors.",
        settings.collection_name,
        info["vectors_count"],
    )
    print(f"\nDone. {info['vectors_count']} chunks indexed in Qdrant.")
    print("Test with: curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' "
          "-d '{\"q\": \"What is GA20ox?\"}' | python3 -m json.tool")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk and embed plant biology corpus into Qdrant")
    parser.add_argument("--chunk-size", type=int, default=400, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=40, help="Overlap between chunks in words")
    args = parser.parse_args()
    ingest(chunk_size=args.chunk_size, overlap=args.overlap)
