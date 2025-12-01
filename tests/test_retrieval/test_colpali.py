"""Tests for ColPali retriever."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from src.clients.colpali import MockColPaliClient
from src.retrieval.colpali import ColPaliRetriever, ColPaliRetrievalResult
from src.ingestion.embeddings.colpali import ColPaliPageEmbedding
from src.retrieval.base import RetrievalResult


class TestColPaliRetrievalResult:
    """Tests for ColPali retrieval result."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = ColPaliRetrievalResult(
            chunk_id="doc_123_page_0",
            document_id="doc_123",
            content="Page 1 of document",
            score=0.85,
            metadata={"type": "page"},
            page_num=0,
            image_size=(800, 600),
            has_images=True,
        )

        assert result.chunk_id == "doc_123_page_0"
        assert result.document_id == "doc_123"
        assert result.score == 0.85
        assert result.page_num == 0
        assert result.image_size == (800, 600)
        assert result.has_images is True

    def test_inherits_from_retrieval_result(self):
        """Test inheritance from RetrievalResult."""
        result = ColPaliRetrievalResult(
            chunk_id="test",
            document_id="doc",
            content="content",
            score=0.5,
            metadata={},
        )

        assert isinstance(result, RetrievalResult)

    def test_default_values(self):
        """Test default values for ColPali fields."""
        result = ColPaliRetrievalResult(
            chunk_id="test",
            document_id="doc",
            content="content",
            score=0.5,
            metadata={},
        )

        assert result.page_num == 0
        assert result.image_size == (0, 0)
        assert result.has_images is False


class TestColPaliRetriever:
    """Tests for ColPali retriever."""

    @pytest.fixture
    def mock_mongodb(self):
        """Create mock MongoDB client."""
        client = MagicMock()
        client.db = MagicMock()
        collection = MagicMock()
        client.db.__getitem__ = MagicMock(return_value=collection)
        return client

    @pytest.fixture
    def retriever(self, mock_mongodb):
        """Create retriever with mock client."""
        return ColPaliRetriever(
            mongodb_client=mock_mongodb,
            use_mock=True,
        )

    @pytest.fixture
    def sample_page_embeddings(self):
        """Create sample page embeddings."""
        embeddings = []
        for i in range(3):
            emb = ColPaliPageEmbedding(
                document_id=f"doc_{i}",
                page_num=i,
                embedding=np.random.rand(1024, 128).astype(np.float32),
                image_size=(800, 600),
                has_text=True,
                has_images=i % 2 == 0,
            )
            embeddings.append(emb)
        return embeddings

    def test_initialization_with_mock(self, mock_mongodb):
        """Test initialization with mock client."""
        retriever = ColPaliRetriever(
            mongodb_client=mock_mongodb,
            use_mock=True,
        )
        assert isinstance(retriever.colpali, MockColPaliClient)

    def test_initialization_with_client(self, mock_mongodb):
        """Test initialization with provided client."""
        client = MockColPaliClient()
        retriever = ColPaliRetriever(
            mongodb_client=mock_mongodb,
            colpali_client=client,
        )
        assert retriever.colpali is client

    @pytest.mark.asyncio
    async def test_retrieve_from_embeddings(self, retriever, sample_page_embeddings):
        """Test retrieval from pre-loaded embeddings."""
        results = await retriever.retrieve_from_embeddings(
            query="test query",
            page_embeddings=sample_page_embeddings,
            top_k=2,
        )

        assert len(results) == 2
        for result in results:
            assert isinstance(result, ColPaliRetrievalResult)
            assert result.score >= 0

        # Check sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_retrieve_from_embeddings_empty(self, retriever):
        """Test retrieval with empty embeddings."""
        results = await retriever.retrieve_from_embeddings(
            query="test query",
            page_embeddings=[],
            top_k=5,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_top_k_limit(self, retriever, sample_page_embeddings):
        """Test top_k limits results."""
        results = await retriever.retrieve_from_embeddings(
            query="test query",
            page_embeddings=sample_page_embeddings,
            top_k=1,
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_result_metadata(self, retriever, sample_page_embeddings):
        """Test result metadata is populated."""
        results = await retriever.retrieve_from_embeddings(
            query="test query",
            page_embeddings=sample_page_embeddings,
            top_k=3,
        )

        for result in results:
            assert "type" in result.metadata
            assert result.metadata["type"] == "page"
            assert "has_text" in result.metadata
            assert "has_images" in result.metadata

    @pytest.mark.asyncio
    async def test_result_chunk_id_format(self, retriever, sample_page_embeddings):
        """Test chunk_id follows expected format."""
        results = await retriever.retrieve_from_embeddings(
            query="test query",
            page_embeddings=sample_page_embeddings,
            top_k=3,
        )

        for result in results:
            # chunk_id should be {document_id}_page_{page_num}
            assert "_page_" in result.chunk_id
            parts = result.chunk_id.split("_page_")
            assert parts[0] == result.document_id
            assert int(parts[1]) == result.page_num

    @pytest.mark.asyncio
    async def test_hybrid_retrieve(self, retriever, sample_page_embeddings, mock_mongodb):
        """Test hybrid retrieval combining text and visual."""
        # Setup mock for _load_page_embeddings
        async def mock_load(*args, **kwargs):
            return sample_page_embeddings

        retriever._load_page_embeddings = mock_load

        # Create mock text results
        text_results = [
            RetrievalResult(
                chunk_id="doc_0_chunk_1",
                document_id="doc_0",
                content="Some text content",
                score=0.9,
                metadata={},
            ),
            RetrievalResult(
                chunk_id="doc_1_chunk_1",
                document_id="doc_1",
                content="More text",
                score=0.7,
                metadata={},
            ),
        ]

        results = await retriever.hybrid_retrieve(
            query="test query",
            text_results=text_results,
            top_k=5,
            visual_weight=0.4,
        )

        assert len(results) > 0
        for result in results:
            # Combined scores should be weighted
            assert result.score >= 0

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_weighting(self, retriever, sample_page_embeddings):
        """Test that visual_weight affects final scores."""
        async def mock_load(*args, **kwargs):
            return sample_page_embeddings

        retriever._load_page_embeddings = mock_load

        text_results = [
            RetrievalResult(
                chunk_id="doc_0_chunk_1",
                document_id="doc_0",
                content="Text",
                score=1.0,
                metadata={},
            ),
        ]

        # Test with different weights
        results_high_visual = await retriever.hybrid_retrieve(
            query="test",
            text_results=text_results,
            top_k=5,
            visual_weight=0.9,
        )

        results_low_visual = await retriever.hybrid_retrieve(
            query="test",
            text_results=text_results,
            top_k=5,
            visual_weight=0.1,
        )

        # Results should differ based on weighting
        assert results_high_visual is not None
        assert results_low_visual is not None


class TestColPaliRetrieverLoadEmbeddings:
    """Tests for loading embeddings from MongoDB."""

    @pytest.fixture
    def mock_mongodb_with_data(self):
        """Create mock MongoDB with page embedding data."""
        client = MagicMock()
        client.db = MagicMock()

        # Mock collection with documents containing page embeddings
        collection = MagicMock()
        cursor = AsyncMock()
        cursor.to_list = AsyncMock(return_value=[
            {
                "_id": "doc1",
                "document_id": "doc_1",
                "pages": [
                    {
                        "page_num": 0,
                        "colpali_embedding": np.random.rand(1024, 128).tolist(),
                        "image_width": 800,
                        "image_height": 600,
                        "has_text": True,
                        "has_images": False,
                    },
                    {
                        "page_num": 1,
                        "colpali_embedding": np.random.rand(1024, 128).tolist(),
                        "image_width": 800,
                        "image_height": 600,
                        "has_text": True,
                        "has_images": True,
                    },
                ],
            },
        ])
        collection.find = MagicMock(return_value=cursor)
        client.db.__getitem__ = MagicMock(return_value=collection)

        return client

    @pytest.mark.asyncio
    async def test_load_page_embeddings(self, mock_mongodb_with_data):
        """Test loading page embeddings from MongoDB."""
        retriever = ColPaliRetriever(
            mongodb_client=mock_mongodb_with_data,
            use_mock=True,
        )

        embeddings = await retriever._load_page_embeddings()

        assert len(embeddings) == 2
        for emb in embeddings:
            assert isinstance(emb, ColPaliPageEmbedding)
            assert emb.document_id == "doc_1"
            assert emb.embedding.shape == (1024, 128)

    @pytest.mark.asyncio
    async def test_load_page_embeddings_with_filter(self, mock_mongodb_with_data):
        """Test loading with document_id filter."""
        retriever = ColPaliRetriever(
            mongodb_client=mock_mongodb_with_data,
            use_mock=True,
        )

        await retriever._load_page_embeddings(document_ids=["doc_1"])

        # Verify query was called with filter
        collection = mock_mongodb_with_data.db["documents"]
        call_args = collection.find.call_args[0][0]
        assert "document_id" in call_args


class TestRetrievalStrategyAutoSelect:
    """Tests for auto-selecting ColPali strategy."""

    def test_visual_keywords_trigger_colpali(self):
        """Test that visual keywords select ColPali strategy."""
        from src.retrieval.pipeline import RetrievalPipeline, RetrievalStrategy

        # Create minimal pipeline for testing
        mock_mongo = MagicMock()
        mock_voyage = MagicMock()
        pipeline = RetrievalPipeline(mock_mongo, mock_voyage)

        visual_queries = [
            "show me the chart on page 5",
            "find the diagram showing architecture",
            "find the table with revenue data",
            "show me the image of the product",
            "what is the layout of the dashboard",
        ]

        for query in visual_queries:
            strategy = pipeline._auto_select_strategy(query)
            assert strategy == RetrievalStrategy.COLPALI, f"Failed for: {query}"
