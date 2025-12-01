"""Tests for generate API endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.generation.base import GenerationResult, Citation
from src.generation.pipeline import PipelineResult
from src.retrieval.base import RetrievalResult


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_pipeline_result():
    """Create mock pipeline result."""
    return PipelineResult(
        generation_result=GenerationResult(
            answer="Machine learning is a field of AI [1].",
            query="What is ML?",
            citations=[
                Citation(
                    citation_id="[1]",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    text="ML content",
                    relevance_score=0.9,
                    verified=True,
                )
            ],
            model="claude-sonnet-4-20250514",
            prompt_type="factual",
        ),
        context=[
            RetrievalResult(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="ML content here",
                score=0.9,
            )
        ],
        crag_iterations=0,
        retrieval_strategy="hybrid",
        metrics={"total_time_seconds": 1.5},
    )


class TestGenerateEndpoint:
    """Test /generate endpoint."""

    def test_generate_basic(self, client, mock_pipeline_result):
        """Test basic generation request."""
        with patch("src.api.routes.generate.get_generation_pipeline") as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.generate.return_value = mock_pipeline_result
            mock_get_pipeline.return_value = mock_pipeline

            response = client.post(
                "/generate",
                json={"query": "What is ML?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "What is ML?"
            assert "answer" in data
            assert "citations" in data

    def test_generate_with_strategy(self, client, mock_pipeline_result):
        """Test generation with specific strategy."""
        with patch("src.api.routes.generate.get_generation_pipeline") as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.generate.return_value = mock_pipeline_result
            mock_get_pipeline.return_value = mock_pipeline

            response = client.post(
                "/generate",
                json={
                    "query": "What is ML?",
                    "strategy": "graphrag",
                },
            )

            assert response.status_code == 200

    def test_generate_with_prompt_type(self, client, mock_pipeline_result):
        """Test generation with specific prompt type."""
        with patch("src.api.routes.generate.get_generation_pipeline") as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.generate.return_value = mock_pipeline_result
            mock_get_pipeline.return_value = mock_pipeline

            response = client.post(
                "/generate",
                json={
                    "query": "What is ML?",
                    "prompt_type": "factual",
                },
            )

            assert response.status_code == 200

    def test_generate_with_options(self, client, mock_pipeline_result):
        """Test generation with all options."""
        with patch("src.api.routes.generate.get_generation_pipeline") as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.generate.return_value = mock_pipeline_result
            mock_get_pipeline.return_value = mock_pipeline

            response = client.post(
                "/generate",
                json={
                    "query": "What is ML?",
                    "strategy": "hybrid",
                    "prompt_type": "factual",
                    "top_k": 5,
                    "enable_crag": True,
                    "enable_hallucination_check": True,
                    "verify_citations": True,
                },
            )

            assert response.status_code == 200

    def test_generate_empty_query(self, client):
        """Test generation with empty query."""
        response = client.post(
            "/generate",
            json={"query": ""},
        )

        assert response.status_code == 422  # Validation error

    def test_generate_missing_query(self, client):
        """Test generation with missing query."""
        response = client.post(
            "/generate",
            json={},
        )

        assert response.status_code == 422


class TestGeneratePromptsEndpoint:
    """Test /generate/prompts endpoint."""

    def test_list_prompts(self, client):
        """Test listing available prompts."""
        response = client.get("/generate/prompts")

        assert response.status_code == 200
        data = response.json()
        assert "prompts" in data

        prompt_names = [p["name"] for p in data["prompts"]]
        assert "factual" in prompt_names
        assert "graphrag" in prompt_names
        assert "raptor" in prompt_names


class TestGenerateConfigEndpoint:
    """Test /generate/config endpoint."""

    def test_get_config(self, client):
        """Test getting generation config."""
        with patch("src.api.routes.generate.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()

            response = client.get("/generate/config")

            assert response.status_code == 200
            data = response.json()
            assert "model" in data
            assert "features" in data
            assert "crag" in data


class TestParseStrategy:
    """Test strategy parsing."""

    def test_parse_valid_strategies(self):
        """Test parsing valid strategy strings."""
        from src.api.routes.generate import parse_strategy
        from src.retrieval import RetrievalStrategy

        assert parse_strategy("vector") == RetrievalStrategy.VECTOR
        assert parse_strategy("text") == RetrievalStrategy.TEXT
        assert parse_strategy("hybrid") == RetrievalStrategy.HYBRID
        assert parse_strategy("graphrag") == RetrievalStrategy.GRAPHRAG
        assert parse_strategy("raptor") == RetrievalStrategy.RAPTOR
        assert parse_strategy("auto") == RetrievalStrategy.AUTO

    def test_parse_case_insensitive(self):
        """Test case insensitive parsing."""
        from src.api.routes.generate import parse_strategy
        from src.retrieval import RetrievalStrategy

        assert parse_strategy("HYBRID") == RetrievalStrategy.HYBRID
        assert parse_strategy("Graphrag") == RetrievalStrategy.GRAPHRAG

    def test_parse_unknown_defaults_to_auto(self):
        """Test unknown strategy defaults to auto."""
        from src.api.routes.generate import parse_strategy
        from src.retrieval import RetrievalStrategy

        assert parse_strategy("unknown") == RetrievalStrategy.AUTO
