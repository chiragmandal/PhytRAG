"""
API endpoint tests using FastAPI's TestClient.

All external dependencies (Qdrant, Ollama) are mocked so tests run
entirely in CI without any running services.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def mock_app_state():
    """Inject mock embedder and vector_store into app.state."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 384

    mock_vs = MagicMock()
    mock_vs.search.return_value = [
        {
            "pmcid": "PMC1234567",
            "title": "Gibberellin biosynthesis in Arabidopsis",
            "text": "GA20ox catalyses the penultimate step in gibberellin biosynthesis...",
            "chunk_index": 0,
            "authors": "Smith J",
            "score": 0.85,
        }
    ]
    mock_vs.collection_info.return_value = {"vectors_count": 500, "points_count": 500}

    with patch.object(app, "state", create=True) as mock_state:
        mock_state.embedder = mock_embedder
        mock_state.vector_store = mock_vs
        yield mock_state


@pytest.fixture()
def client(mock_app_state):
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        with (
            patch("app.api.routes.httpx.AsyncClient") as mock_http,
        ):
            mock_ctx = AsyncMock()
            mock_http.return_value.__aenter__.return_value = mock_ctx
            mock_ctx.get.return_value.status_code = 200

            resp = client.get("/health")
            assert resp.status_code == 200

    def test_health_schema(self, client):
        with patch("app.api.routes.httpx.AsyncClient") as mock_http:
            mock_ctx = AsyncMock()
            mock_http.return_value.__aenter__.return_value = mock_ctx
            mock_ctx.get.return_value.status_code = 200

            data = client.get("/health").json()
            assert "status" in data
            assert "version" in data
            assert "qdrant_ok" in data


class TestQueryEndpoint:
    def test_query_too_short_returns_422(self, client):
        resp = client.post("/query", json={"q": "hi"})
        assert resp.status_code == 422

    def test_query_returns_sources(self, client):
        with patch("app.api.routes.LLMClient") as MockLLM:
            instance = MockLLM.return_value
            instance.generate = AsyncMock(return_value=("GA20ox is an enzyme...", 400, 42))

            resp = client.post("/query", json={"q": "What is the role of GA20ox in Arabidopsis?"})
            assert resp.status_code == 200
            data = resp.json()
            assert "answer" in data
            assert "sources" in data
            assert len(data["sources"]) == 1
            assert data["sources"][0]["pmcid"] == "PMC1234567"

    def test_query_response_has_metrics(self, client):
        with patch("app.api.routes.LLMClient") as MockLLM:
            instance = MockLLM.return_value
            instance.generate = AsyncMock(return_value=("An answer.", 350, 20))

            resp = client.post("/query", json={"q": "How does jasmonic acid affect plant defense?"})
            data = resp.json()
            assert "latency_ms" in data
            assert "ttft_ms" in data
            assert "tokens_generated" in data
            assert data["latency_ms"] > 0

    def test_metrics_endpoint_reachable(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert b"phytrag_queries_total" in resp.content
