"""
PhytRAG API entrypoint.

Lifespan events initialise shared resources (embedder, Qdrant client)
once at startup so every request shares the same loaded model and
connection pool - critical for low-latency inference.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.observability.logging_config import configure_logging
from app.observability.metrics import setup_metrics
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore

# Disable tokeniser fork-safety warning (harmless on macOS, noisy in logs)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load embedding model + warm up Qdrant connection.
    Shutdown: graceful close of connection pool.
    """
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Load the embedding model once; it lives in app.state for the process lifetime.
    # Loading here (not per-request) avoids a 2-3 second cold start on first query.
    logger.info("Loading embedding model: %s", settings.embed_model)
    app.state.embedder = Embedder(model_name=settings.embed_model)

    logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
    app.state.vector_store = VectorStore(
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
        vector_size=settings.vector_size,
    )
    app.state.vector_store.ensure_collection()

    logger.info("Startup complete. Ready to serve queries.")
    yield

    # Cleanup
    logger.info("Shutting down %s", settings.app_name)
    app.state.vector_store.close()


app = FastAPI(
    title="PhytRAG",
    description=(
        "Retrieval-augmented generation over open-access plant biology literature. "
        "Demonstrates production MLOps patterns: containerised inference serving, "
        "vector database operations, Prometheus/Grafana observability, and "
        "MLflow experiment tracking."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Prometheus /metrics is registered inside setup_metrics
setup_metrics(app)

app.include_router(router)
