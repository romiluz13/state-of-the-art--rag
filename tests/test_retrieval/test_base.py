"""Tests for retrieval base classes."""

import pytest
from src.retrieval.base import RetrievalResult, RetrievalConfig


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Test content",
            score=0.95,
        )
        assert result.chunk_id == "chunk_1"
        assert result.score == 0.95
        assert result.metadata == {}

    def test_with_scores(self):
        """Test result with different scores."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Test",
            score=0.9,
            vector_score=0.95,
            text_score=0.85,
            rerank_score=0.92,
        )
        assert result.vector_score == 0.95
        assert result.text_score == 0.85
        assert result.rerank_score == 0.92

    def test_with_metadata(self):
        """Test result with metadata."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Test",
            score=0.9,
            metadata={"source": "docs", "page": 5},
        )
        assert result.metadata["source"] == "docs"
        assert result.metadata["page"] == 5


class TestRetrievalConfig:
    """Tests for RetrievalConfig dataclass."""

    def test_defaults(self):
        """Test default configuration."""
        config = RetrievalConfig()
        assert config.top_k == 10
        assert config.use_binary_quantization is True
        assert config.rerank is True
        assert config.vector_weight == 0.7
        assert config.text_weight == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetrievalConfig(
            top_k=20,
            use_binary_quantization=False,
            vector_weight=0.8,
            text_weight=0.2,
        )
        assert config.top_k == 20
        assert config.use_binary_quantization is False
        assert config.vector_weight == 0.8

    def test_raptor_config(self):
        """Test RAPTOR-specific configuration."""
        config = RetrievalConfig(
            raptor_levels=[0, 1, 2, 3],
            level_weights={0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
        )
        assert 3 in config.raptor_levels
        assert config.level_weights[3] == 0.1

    def test_graphrag_config(self):
        """Test GraphRAG-specific configuration."""
        config = RetrievalConfig(
            graph_depth=3,
            include_communities=False,
        )
        assert config.graph_depth == 3
        assert config.include_communities is False
