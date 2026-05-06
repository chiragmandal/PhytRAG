"""
API route handlers.

/query   POST  - RAG query: retrieve + generate + return with citations
/health  GET   - Liveness + dependency health check
/metrics GET   - Prometheus metrics scrape endpoint (registered by starlette_exporter)
"""
import logging
import time
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.schemas import HealthResponse, QueryRequest, QueryResponse, SourceChunk
from app.config import Settings, get_settings
from app.generation.llm_client import LLMClient
from app.observability.metrics import (
    ERROR_COUNTER,
    QUERY_COUNTER,
    QUERY_LATENCY,
    RETRIEVAL_SCORE,
    TOKENS_COUNTER,
    TTFT_HISTOGRAM,
)
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Dependency helpers ───────────────────────────────────────────────────────

def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(
    body: QueryRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
) -> QueryResponse:
    """
    Retrieve relevant chunks from the plant biology corpus, then generate
    a grounded answer using the local LLM.

    Sources are included in the response so the caller can verify provenance.
    """
    query_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    logger.info("query_start", extra={"query_id": query_id, "question": body.q[:120]})

    try:
        # 1. Embed the query
        query_vector = embedder.embed(body.q)

        # 2. Retrieve top-k chunks
        hits = vector_store.search(
            query_vector=query_vector,
            top_k=body.top_k,
            score_threshold=settings.min_relevance_score,
        )

        if not hits:
            logger.warning("No relevant chunks found", extra={"query_id": query_id})

        # Record retrieval scores for Prometheus
        for hit in hits:
            RETRIEVAL_SCORE.observe(hit["score"])

        # 3. Build prompt
        context_block = "\n\n---\n\n".join(
            f"[Source {i+1}: {h['title']} (PMC{h['pmcid']})]\n{h['text']}"
            for i, h in enumerate(hits)
        )
        prompt = _build_prompt(body.q, context_block)

        # 4. Generate answer (streaming so we can capture TTFT)
        llm = LLMClient(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        answer, ttft_ms, tokens_generated = await llm.generate(prompt)

        t_end = time.perf_counter()
        latency_ms = int((t_end - t_start) * 1000)

        # 5. Record metrics
        QUERY_COUNTER.labels(status="ok").inc()
        QUERY_LATENCY.labels(status="ok").observe(t_end - t_start)
        TTFT_HISTOGRAM.observe(ttft_ms / 1000)
        TOKENS_COUNTER.inc(tokens_generated)

        logger.info(
            "query_complete",
            extra={
                "query_id": query_id,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "tokens": tokens_generated,
                "sources_count": len(hits),
            },
        )

        return QueryResponse(
            query_id=query_id,
            answer=answer,
            sources=[
                SourceChunk(
                    pmcid=h["pmcid"],
                    title=h["title"],
                    score=round(h["score"], 4),
                    excerpt=h["text"][:300] + "..." if len(h["text"]) > 300 else h["text"],
                    chunk_index=h["chunk_index"],
                )
                for h in hits
            ],
            model=settings.llm_model,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tokens_generated=tokens_generated,
        )

    except Exception as exc:
        ERROR_COUNTER.labels(stage="query").inc()
        QUERY_COUNTER.labels(status="error").inc()
        logger.exception("query_error", extra={"query_id": query_id, "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Deep health check: verifies Qdrant connectivity and Ollama reachability.
    Used by Docker HEALTHCHECK and Kubernetes readiness probes.
    """
    qdrant_ok = False
    collection_vectors = 0
    try:
        vs: VectorStore = request.app.state.vector_store
        info = vs.collection_info()
        qdrant_ok = True
        collection_vectors = info.get("vectors_count") or 0
    except Exception:
        pass

    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (qdrant_ok and ollama_ok) else "degraded",
        version=settings.app_version,
        qdrant_ok=qdrant_ok,
        ollama_ok=ollama_ok,
        collection_vectors=collection_vectors,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_prompt(question: str, context: str) -> str:
    """
    System + user prompt structured for factual, citation-aware responses.
    The instruction to cite sources nudges the model toward grounded answers
    and away from hallucination.
    """
    system = (
        "You are a plant biology research assistant. "
        "Answer the user's question using ONLY the provided source excerpts. "
        "If the sources do not contain enough information to answer, say so clearly. "
        "Be concise and scientific. Cite sources by their number [Source N]."
    )
    user = f"Sources:\n{context}\n\nQuestion: {question}"
    return f"<system>{system}</system>\n\n<user>{user}</user>"
