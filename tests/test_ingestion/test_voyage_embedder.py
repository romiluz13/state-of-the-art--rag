"""Tests for VoyageEmbedder with late chunking support."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.ingestion.embeddings.voyage import VoyageEmbedder
from src.ingestion.chunking.base import Chunk


class TestVoyageEmbedder:
    """Tests for VoyageEmbedder."""

    @pytest.fixture
    def mock_voyage_client(self):
        """Create mock Voyage client."""
        client = MagicMock()

        # Mock embed method
        async def mock_embed(texts, model=None, input_type="document", output_dtype="float"):
            if output_dtype == "binary":
                return {
                    "data": [{"embedding": [0, 1] * 64} for _ in texts]
                }
            return {
                "data": [{"embedding": [0.1] * 1024} for _ in texts]
            }

        # Mock contextualized_embed method
        async def mock_contextualized_embed(documents, model="voyage-context-3", input_type="document"):
            # documents is [[doc, chunk1, chunk2, ...]]
            # Return embeddings for all including document
            num_items = len(documents[0])
            return {
                "data": [{
                    "embeddings": [[0.1 * i] * 1024 for i in range(num_items)]
                }]
            }

        client.embed = mock_embed
        client.contextualized_embed = mock_contextualized_embed
        return client

    @pytest.fixture
    def embedder(self, mock_voyage_client):
        """Create embedder with mock client."""
        return VoyageEmbedder(mock_voyage_client)

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            Chunk(content="First chunk content", start_char=0, end_char=19, chunk_index=0, metadata={}),
            Chunk(content="Second chunk content", start_char=20, end_char=40, chunk_index=1, metadata={}),
            Chunk(content="Third chunk content", start_char=41, end_char=60, chunk_index=2, metadata={}),
        ]

    def test_initialization(self, embedder):
        """Test embedder initialization."""
        assert embedder.MAX_BATCH_SIZE == 128
        assert embedder.MAX_TOKENS_PER_BATCH == 120000
        assert embedder.LATE_CHUNKING_MODEL == "voyage-context-3"

    @pytest.mark.asyncio
    async def test_embed_chunks(self, embedder, sample_chunks):
        """Test basic chunk embedding."""
        results = await embedder.embed_chunks(sample_chunks)

        assert len(results) == 3
        assert all("full" in r for r in results)
        assert all("binary" in r for r in results)
        assert len(results[0]["full"]) == 1024

    @pytest.mark.asyncio
    async def test_embed_chunks_no_binary(self, embedder, sample_chunks):
        """Test embedding without binary."""
        results = await embedder.embed_chunks(sample_chunks, include_binary=False)

        assert len(results) == 3
        assert all(r["binary"] is None for r in results)

    @pytest.mark.asyncio
    async def test_embed_query(self, embedder):
        """Test query embedding."""
        embedding = await embedder.embed_query("test query")

        assert len(embedding) == 1024
        assert isinstance(embedding, list)

    @pytest.mark.asyncio
    async def test_embed_empty_chunks(self, embedder):
        """Test embedding empty chunk list."""
        results = await embedder.embed_chunks([])
        assert results == []

    def test_create_batches(self, embedder):
        """Test batch creation respects limits."""
        # Create texts
        texts = ["Hello world"] * 200  # More than MAX_BATCH_SIZE

        batches = embedder._create_batches(texts)

        # Should create multiple batches
        assert len(batches) > 1
        assert all(len(b) <= embedder.MAX_BATCH_SIZE for b in batches)

    def test_create_batches_token_limit(self, embedder):
        """Test batch creation respects token limit."""
        # Create large texts
        large_text = "x" * 10000  # ~2500 tokens
        texts = [large_text] * 100

        batches = embedder._create_batches(texts)

        # Should create batches respecting token limit
        assert len(batches) > 1

    @pytest.mark.asyncio
    async def test_embed_with_late_chunking(self, embedder, sample_chunks):
        """Test late chunking embedding."""
        document = "Full document text. First chunk content. Second chunk content. Third chunk content."

        results = await embedder.embed_with_late_chunking(
            document=document,
            chunks=sample_chunks,
            include_binary=True,
        )

        assert len(results) == 3
        assert all("full" in r for r in results)

    @pytest.mark.asyncio
    async def test_embed_with_late_chunking_empty(self, embedder):
        """Test late chunking with empty chunks."""
        results = await embedder.embed_with_late_chunking(
            document="Some document",
            chunks=[],
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_with_late_chunking_fallback(self, mock_voyage_client, sample_chunks):
        """Test late chunking falls back on error."""
        # Make contextualized_embed fail
        async def failing_contextualized(documents, model, input_type):
            raise Exception("API error")

        mock_voyage_client.contextualized_embed = failing_contextualized
        embedder = VoyageEmbedder(mock_voyage_client)

        document = "Full document text."
        results = await embedder.embed_with_late_chunking(document, sample_chunks)

        # Should fall back to standard embedding
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_embed_texts(self, embedder):
        """Test embedding arbitrary texts."""
        texts = ["Text one", "Text two", "Text three"]

        embeddings = await embedder.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 1024 for e in embeddings)

    @pytest.mark.asyncio
    async def test_embed_texts_empty(self, embedder):
        """Test embedding empty text list."""
        embeddings = await embedder.embed_texts([])
        assert embeddings == []
