"""Tests for Reranker."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.retrieval.reranker import Reranker
from src.retrieval.base import RetrievalResult


class TestReranker:
    """Tests for Reranker."""

    @pytest.fixture
    def mock_voyage(self):
        """Create mock Voyage client."""
        client = MagicMock()

        async def mock_rerank(query, documents, model, top_k, **kwargs):
            # Return reranked with reversed order (simulating reranking)
            return {
                "data": [
                    {"index": i, "relevance_score": 0.9 - (i * 0.1)}
                    for i in range(min(top_k, len(documents)))
                ]
            }

        client.rerank = mock_rerank
        return client

    @pytest.fixture
    def reranker(self, mock_voyage):
        """Create Reranker with mock."""
        return Reranker(mock_voyage)

    @pytest.fixture
    def sample_results(self):
        """Create sample retrieval results."""
        return [
            RetrievalResult(
                chunk_id=f"chunk_{i}",
                document_id="doc_1",
                content=f"Content {i}",
                score=0.5 + (i * 0.1),
            )
            for i in range(5)
        ]

    def test_initialization(self, reranker):
        """Test reranker initialization."""
        assert reranker.DEFAULT_MODEL == "rerank-2.5"
        assert reranker.MAX_DOCUMENTS == 1000

    @pytest.mark.asyncio
    async def test_rerank_basic(self, reranker, sample_results):
        """Test basic reranking."""
        reranked = await reranker.rerank(
            query="test query",
            results=sample_results,
        )

        assert len(reranked) == 5
        # Scores should be set by reranker
        assert reranked[0].rerank_score == 0.9

    @pytest.mark.asyncio
    async def test_rerank_top_k(self, reranker, sample_results):
        """Test reranking with top_k."""
        reranked = await reranker.rerank(
            query="test query",
            results=sample_results,
            top_k=3,
        )

        assert len(reranked) == 3

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, reranker):
        """Test reranking empty results."""
        reranked = await reranker.rerank(
            query="test query",
            results=[],
        )

        assert reranked == []

    @pytest.mark.asyncio
    async def test_rerank_preserves_scores(self, reranker, sample_results):
        """Test reranking with score preservation."""
        reranked = await reranker.rerank_with_scores(
            query="test query",
            results=sample_results,
        )

        # Should have rerank_score set
        assert all(r.rerank_score is not None for r in reranked)

    def test_compute_score_change(self, reranker, sample_results):
        """Test score change analysis."""
        # Simulate reranked results
        for i, r in enumerate(sample_results):
            r.rerank_score = 0.9 - (i * 0.1)
            r.metadata["original_score"] = r.score

        analysis = reranker.compute_score_change(sample_results)

        assert len(analysis) == 5
        assert "original_score" in analysis[0]
        assert "rerank_score" in analysis[0]
        assert "score_change" in analysis[0]

    @pytest.mark.asyncio
    async def test_batch_rerank(self, reranker, sample_results):
        """Test batch reranking."""
        queries = ["query 1", "query 2"]
        results_list = [sample_results[:3], sample_results[:2]]

        batch_reranked = await reranker.batch_rerank(
            queries=queries,
            results_list=results_list,
        )

        assert len(batch_reranked) == 2
        assert len(batch_reranked[0]) == 3
        assert len(batch_reranked[1]) == 2

    @pytest.mark.asyncio
    async def test_rerank_handles_error(self, sample_results):
        """Test reranking gracefully handles errors."""
        client = MagicMock()

        async def failing_rerank(*args, **kwargs):
            raise Exception("API error")

        client.rerank = failing_rerank
        reranker = Reranker(client)

        # Should return original results on error
        results = await reranker.rerank("test", sample_results)
        assert len(results) == 5
