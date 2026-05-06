"""
Request and response schemas for the PhytRAG API.

Keeping schemas in a dedicated module makes them importable by eval/
and tests/ without pulling in the full FastAPI app.
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    q: str = Field(..., min_length=5, max_length=1000, description="Natural language question about plant biology")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source chunks to retrieve")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"q": "What is the role of GA20ox in Arabidopsis stem elongation?", "top_k": 5}
            ]
        }
    }


class SourceChunk(BaseModel):
    pmcid: str = Field(..., description="PubMed Central ID of the source paper")
    title: str = Field(..., description="Paper title")
    score: float = Field(..., description="Cosine similarity score (0-1)")
    excerpt: str = Field(..., description="The retrieved text chunk")
    chunk_index: int = Field(..., description="Chunk position within the original paper")


class QueryResponse(BaseModel):
    query_id: str = Field(..., description="Unique ID for this request (use for log correlation)")
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved sources")
    sources: list[SourceChunk] = Field(..., description="Retrieved chunks used to generate the answer")
    model: str = Field(..., description="LLM model used for generation")
    latency_ms: int = Field(..., description="Total end-to-end latency in milliseconds")
    ttft_ms: int = Field(..., description="Time to first token in milliseconds")
    tokens_generated: int = Field(..., description="Number of tokens in the generated answer")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_id": "3f7a1c2d-...",
                    "answer": "GA20 oxidase (GA20ox) catalyses the penultimate step...",
                    "sources": [],
                    "model": "llama3.2:3b",
                    "latency_ms": 3420,
                    "ttft_ms": 890,
                    "tokens_generated": 148,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant_ok: bool
    ollama_ok: bool
    collection_vectors: int
