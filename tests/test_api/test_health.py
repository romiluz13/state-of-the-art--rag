"""Tests for health check endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock


@pytest.fixture
def mock_mongodb_client():
    """Mock MongoDB client for testing."""
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    return client


@pytest.fixture
def setup_env(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # Clear settings cache
    from src.config.settings import get_settings
    get_settings.cache_clear()


def test_health_check_healthy(mock_mongodb_client, setup_env):
    """Test health check endpoint when MongoDB is healthy."""
    from src.api.main import app, get_mongodb_client

    # Override dependency
    app.dependency_overrides[get_mongodb_client] = lambda: mock_mongodb_client

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mongodb"] is True

    # Clean up
    app.dependency_overrides.clear()


def test_health_check_unhealthy(mock_mongodb_client, setup_env):
    """Test health check endpoint when MongoDB is unhealthy."""
    from src.api.main import app, get_mongodb_client

    mock_mongodb_client.health_check = AsyncMock(return_value=False)

    # Override dependency
    app.dependency_overrides[get_mongodb_client] = lambda: mock_mongodb_client

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["mongodb"] is False

    # Clean up
    app.dependency_overrides.clear()


def test_root_endpoint(setup_env):
    """Test root endpoint."""
    from src.api.main import app

    with TestClient(app) as client:
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "SOTA RAG API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"


def test_query_endpoint(setup_env, mock_mongodb_client):
    """Test query endpoint with mocked dependencies."""
    from src.api.main import app, get_mongodb_client, get_voyage_client
    from src.api.routes import query as query_module
    from src.retrieval import RetrievalPipeline, RetrievalConfig
    from unittest.mock import MagicMock, AsyncMock

    # Mock voyage client
    mock_voyage = MagicMock()

    async def mock_embed(*args, **kwargs):
        return {"data": [{"embedding": [0.1] * 1024}]}

    async def mock_rerank(*args, **kwargs):
        return {"data": []}

    mock_voyage.embed = mock_embed
    mock_voyage.rerank = mock_rerank

    # Mock MongoDB aggregate to return empty results
    async def mock_aggregate(pipeline):
        return
        yield  # Empty async generator

    mock_collection = MagicMock()
    mock_collection.aggregate = mock_aggregate
    mock_mongodb_client.db = {"chunks": mock_collection, "entities": mock_collection, "communities": mock_collection}

    # Directly set the mocked clients and pipeline in the query module
    config = RetrievalConfig(rerank=False)
    query_module._mongodb_client = mock_mongodb_client
    query_module._voyage_client = mock_voyage
    query_module._retrieval_pipeline = RetrievalPipeline(mock_mongodb_client, mock_voyage, config)

    app.dependency_overrides[get_mongodb_client] = lambda: mock_mongodb_client
    app.dependency_overrides[get_voyage_client] = lambda: mock_voyage

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "test query", "strategy": "vector", "rerank": False})

        # Should return 200 with empty results (no data in mock)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["strategy"] == "vector"
        assert isinstance(data["results"], list)

    # Clean up
    app.dependency_overrides.clear()
    query_module._retrieval_pipeline = None
    query_module._mongodb_client = None
    query_module._voyage_client = None


def test_ingest_text_endpoint(setup_env, mock_mongodb_client):
    """Test ingest text endpoint."""
    from src.api.main import app, get_mongodb_client, get_voyage_client
    from unittest.mock import MagicMock, AsyncMock

    # Create mock voyage client that returns different values for float vs binary
    embed_call_count = [0]

    async def mock_embed(*args, **kwargs):
        embed_call_count[0] += 1
        if kwargs.get("output_dtype") == "binary":
            return {"data": [{"embedding": [0, 1] * 512}], "usage": {"total_tokens": 100}}
        return {"data": [{"embedding": [0.1] * 1024}], "usage": {"total_tokens": 100}}

    mock_voyage = MagicMock()
    mock_voyage.embed = mock_embed

    # Create mock for MongoDB db property
    mock_mongodb_client.db = MagicMock()
    mock_mongodb_client.db.__getitem__ = MagicMock(return_value=MagicMock(
        insert_one=AsyncMock(),
        insert_many=AsyncMock()
    ))

    # Override dependencies
    app.dependency_overrides[get_mongodb_client] = lambda: mock_mongodb_client
    app.dependency_overrides[get_voyage_client] = lambda: mock_voyage

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ingest/text",
            json={"text": "This is a test document with some content.", "title": "Test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["document_id"] is not None
        assert data["title"] == "Test"

    # Clean up
    app.dependency_overrides.clear()
