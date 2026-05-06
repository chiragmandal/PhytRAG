"""
Application configuration.

All values are read from environment variables with sensible defaults
for local development. The only required override in Docker is QDRANT_URL
and OLLAMA_BASE_URL which are set in docker-compose.yml.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Qdrant ──────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "phytrag"
    vector_size: int = 384  # all-MiniLM-L6-v2 output dimension

    # ── Embedding model ──────────────────────────────────────────────────────
    embed_model: str = "all-MiniLM-L6-v2"

    # ── LLM (Ollama OpenAI-compatible endpoint) ──────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.1      # low temp = grounded, factual responses
    llm_max_tokens: int = 512

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5          # number of chunks to retrieve
    min_relevance_score: float = 0.3  # discard chunks below this score

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "phytrag-retrieval-eval"

    # ── App ──────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    app_name: str = "PhytRAG"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton - settings are read once and shared across the process."""
    return Settings()
