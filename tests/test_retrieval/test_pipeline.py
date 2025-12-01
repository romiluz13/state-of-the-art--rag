"""Tests for RetrievalPipeline."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.retrieval.pipeline import RetrievalPipeline, RetrievalStrategy
from src.retrieval.base import RetrievalConfig, RetrievalResult


class TestRetrievalStrategy:
    """Tests for RetrievalStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert RetrievalStrategy.VECTOR.value == "vector"
        assert RetrievalStrategy.TEXT.value == "text"
        assert RetrievalStrategy.HYBRID.value == "hybrid"
        assert RetrievalStrategy.GRAPHRAG.value == "graphrag"
        assert RetrievalStrategy.RAPTOR.value == "raptor"
        assert RetrievalStrategy.AUTO.value == "auto"


class TestRetrievalPipeline:
    """Tests for RetrievalPipeline."""

    @pytest.fixture
    def mock_mongodb(self):
        """Create mock MongoDB client."""
        client = MagicMock()

        async def mock_aggregate(pipeline):
            results = [
                {
                    "chunk_id": "chunk_1",
                    "document_id": "doc_1",
                    "content": "Test content",
                    "metadata": {},
                    "score": 0.9,
                }
            ]
            for r in results:
                yield r

        collection = MagicMock()
        collection.aggregate = mock_aggregate
        client.db = {
            "chunks": collection,
            "entities": collection,
            "communities": collection,
        }
        return client

    @pytest.fixture
    def mock_voyage(self):
        """Create mock Voyage client."""
        client = MagicMock()

        async def mock_embed(texts, model, input_type):
            return {
                "data": [{"embedding": [0.1] * 1024} for _ in texts]
            }

        async def mock_rerank(query, documents, model, top_k, **kwargs):
            return {
                "data": [
                    {"index": i, "relevance_score": 0.9}
                    for i in range(min(top_k, len(documents)))
                ]
            }

        client.embed = mock_embed
        client.rerank = mock_rerank
        return client

    @pytest.fixture
    def pipeline(self, mock_mongodb, mock_voyage):
        """Create RetrievalPipeline with mocks."""
        config = RetrievalConfig(rerank=False)  # Disable rerank for simpler tests
        return RetrievalPipeline(mock_mongodb, mock_voyage, config)

    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.vector_searcher is not None
        assert pipeline.text_searcher is not None
        assert pipeline.hybrid_searcher is not None
        assert pipeline.graphrag_retriever is not None
        assert pipeline.raptor_retriever is not None
        assert pipeline.reranker is not None

    def test_auto_select_graphrag(self, pipeline):
        """Test auto-selection for global questions."""
        queries = [
            "What are all the main themes?",
            "Summarize the document",
            "Give me an overview",
        ]
        for query in queries:
            strategy = pipeline._auto_select_strategy(query)
            assert strategy == RetrievalStrategy.GRAPHRAG, f"Failed for: {query}"

    def test_auto_select_raptor(self, pipeline):
        """Test auto-selection for hierarchical questions."""
        queries = [
            "What does the introduction say?",
            "Read chapter 3 section",
            "What is in the conclusion?",
        ]
        for query in queries:
            strategy = pipeline._auto_select_strategy(query)
            assert strategy == RetrievalStrategy.RAPTOR, f"Failed for: {query}"

    def test_auto_select_text(self, pipeline):
        """Test auto-selection for exact matches."""
        # Quoted queries prefer text search
        query = '"exact phrase"'
        strategy = pipeline._auto_select_strategy(query)
        assert strategy == RetrievalStrategy.TEXT

    def test_auto_select_hybrid_default(self, pipeline):
        """Test auto-selection defaults to hybrid."""
        query = "How does authentication work?"
        strategy = pipeline._auto_select_strategy(query)
        assert strategy == RetrievalStrategy.HYBRID

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, pipeline):
        """Test basic retrieval."""
        results = await pipeline.retrieve(
            query="test query",
            strategy=RetrievalStrategy.VECTOR,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_with_rerank(self, mock_mongodb, mock_voyage):
        """Test retrieval with reranking enabled."""
        config = RetrievalConfig(rerank=True)
        pipeline = RetrievalPipeline(mock_mongodb, mock_voyage, config)

        results = await pipeline.retrieve(
            query="test query",
            strategy=RetrievalStrategy.VECTOR,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_multi_strategy_retrieve(self, pipeline):
        """Test multi-strategy retrieval."""
        strategies = [RetrievalStrategy.VECTOR, RetrievalStrategy.TEXT]

        results = await pipeline.multi_strategy_retrieve(
            query="test query",
            strategies=strategies,
        )

        assert "vector" in results
        assert "text" in results


class TestAutoStrategySelection:
    """Additional tests for auto strategy selection."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for strategy tests."""
        mock_mongodb = MagicMock()
        mock_voyage = MagicMock()
        return RetrievalPipeline(mock_mongodb, mock_voyage)

    def test_global_keywords(self, pipeline):
        """Test all global keywords trigger GraphRAG."""
        global_queries = [
            "list all topics",
            "throughout the document",
            "in general",
            "commonly found",
            "typically used",
        ]
        for query in global_queries:
            assert pipeline._auto_select_strategy(query) == RetrievalStrategy.GRAPHRAG

    def test_hierarchical_keywords(self, pipeline):
        """Test hierarchical keywords trigger RAPTOR."""
        hier_queries = [
            "first section",
            "last chapter",
            "document summary",
            "paper abstract",
        ]
        for query in hier_queries:
            assert pipeline._auto_select_strategy(query) == RetrievalStrategy.RAPTOR
