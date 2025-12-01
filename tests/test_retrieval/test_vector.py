"""Tests for VectorSearcher."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.retrieval.vector import VectorSearcher
from src.retrieval.base import RetrievalConfig


class TestVectorSearcher:
    """Tests for VectorSearcher."""

    @pytest.fixture
    def mock_mongodb(self):
        """Create mock MongoDB client."""
        client = MagicMock()

        # Mock async cursor
        async def mock_aggregate(pipeline):
            results = [
                {
                    "chunk_id": "chunk_1",
                    "document_id": "doc_1",
                    "content": "First result content",
                    "metadata": {"source": "test"},
                    "score": 0.95,
                },
                {
                    "chunk_id": "chunk_2",
                    "document_id": "doc_1",
                    "content": "Second result content",
                    "metadata": {},
                    "score": 0.88,
                },
            ]
            for r in results:
                yield r

        collection = MagicMock()
        collection.aggregate = mock_aggregate
        client.db = {"chunks": collection}

        return client

    @pytest.fixture
    def searcher(self, mock_mongodb):
        """Create VectorSearcher with mock."""
        config = RetrievalConfig(use_binary_quantization=False)
        return VectorSearcher(mock_mongodb, config)

    def test_initialization(self, searcher):
        """Test searcher initialization."""
        assert searcher.COLLECTION_NAME == "chunks"
        assert searcher.VECTOR_INDEX_FULL == "vector_index_full"

    def test_cosine_similarity(self, searcher):
        """Test cosine similarity calculation."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert searcher._cosine_similarity(a, b) == 1.0

        c = [1.0, 0.0, 0.0]
        d = [0.0, 1.0, 0.0]
        assert searcher._cosine_similarity(c, d) == 0.0

    def test_cosine_similarity_normalized(self, searcher):
        """Test cosine similarity with non-unit vectors."""
        a = [2.0, 0.0]
        b = [4.0, 0.0]
        assert abs(searcher._cosine_similarity(a, b) - 1.0) < 0.001

    def test_cosine_similarity_empty(self, searcher):
        """Test cosine similarity with zero vectors."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert searcher._cosine_similarity(a, b) == 0.0

    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self, searcher):
        """Test basic retrieval."""
        query_embedding = [0.1] * 1024
        results = await searcher.retrieve(
            query="test query",
            query_embedding=query_embedding,
            top_k=10,
        )

        assert len(results) == 2
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self, searcher):
        """Test top_k limiting."""
        query_embedding = [0.1] * 1024
        results = await searcher.retrieve(
            query="test",
            query_embedding=query_embedding,
            top_k=1,
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_with_binary(self, mock_mongodb):
        """Test retrieval with binary quantization."""
        config = RetrievalConfig(use_binary_quantization=True)
        searcher = VectorSearcher(mock_mongodb, config)

        # Binary search should attempt two-stage retrieval
        query_embedding = [0.1] * 1024
        results = await searcher.retrieve(
            query="test",
            query_embedding=query_embedding,
        )

        # Results depend on mock; just verify it doesn't error
        assert isinstance(results, list)
