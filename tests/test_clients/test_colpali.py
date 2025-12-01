"""Tests for ColPali client."""

import pytest
import numpy as np
from PIL import Image

from src.clients.colpali import ColPaliConfig, MockColPaliClient


class TestColPaliConfig:
    """Tests for ColPali configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ColPaliConfig()
        assert config.model_name == "vidore/colpali-v1.2"
        assert config.device == "cpu"
        assert config.max_image_size == 448
        assert config.batch_size == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = ColPaliConfig(
            device="cuda",
            max_image_size=512,
            batch_size=8,
        )
        assert config.device == "cuda"
        assert config.max_image_size == 512
        assert config.batch_size == 8


class TestMockColPaliClient:
    """Tests for mock ColPali client."""

    MOCK_EMBEDDING_DIM = 128
    MOCK_NUM_PATCHES = 256

    @pytest.fixture
    def client(self):
        """Create mock client."""
        return MockColPaliClient()

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        return Image.new("RGB", (100, 100), color="white")

    def test_initialization(self, client):
        """Test mock client initialization."""
        assert client.config is not None
        assert client._embedding_dim == self.MOCK_EMBEDDING_DIM

    def test_embed_image(self, client, test_image):
        """Test single image embedding."""
        embedding = client.embed_image(test_image)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (self.MOCK_NUM_PATCHES, self.MOCK_EMBEDDING_DIM)
        assert embedding.dtype == np.float32

    def test_embed_images_batch(self, client, test_image):
        """Test batch image embedding."""
        images = [test_image, test_image, test_image]
        embeddings = client.embed_images(images)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (self.MOCK_NUM_PATCHES, self.MOCK_EMBEDDING_DIM)

    def test_embed_images_empty(self, client):
        """Test empty batch."""
        embeddings = client.embed_images([])
        assert embeddings == []

    def test_embed_query(self, client):
        """Test query embedding."""
        embedding = client.embed_query("test query")

        assert isinstance(embedding, np.ndarray)
        # Query has fewer tokens than image patches
        assert embedding.shape[1] == self.MOCK_EMBEDDING_DIM
        assert embedding.dtype == np.float32

    def test_compute_similarity(self, client, test_image):
        """Test similarity computation (MaxSim)."""
        query_emb = client.embed_query("test query")
        page_emb = client.embed_image(test_image)

        score = client.compute_similarity(query_emb, page_emb)

        assert isinstance(score, float)

    def test_rank_pages(self, client, test_image):
        """Test page ranking."""
        pages = [test_image, test_image, test_image]
        page_embeddings = client.embed_images(pages)

        rankings = client.rank_pages("test query", page_embeddings, top_k=2)

        assert len(rankings) == 2
        for page_idx, score in rankings:
            assert 0 <= page_idx < 3
            assert isinstance(score, float)

        # Check sorted by score descending
        scores = [s for _, s in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_different_queries_different_embeddings(self, client):
        """Test that different queries produce different embeddings."""
        emb1 = client.embed_query("first query")
        emb2 = client.embed_query("second query")

        # Different queries should produce different random embeddings
        # (based on hash of query string)
        assert not np.allclose(emb1, emb2)

    def test_same_query_same_embedding(self, client):
        """Test that same query produces same embedding."""
        emb1 = client.embed_query("same query")
        emb2 = client.embed_query("same query")

        np.testing.assert_array_almost_equal(emb1, emb2)


class TestColPaliClientIntegration:
    """Integration tests (skipped unless GPU available)."""

    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_real_client_initialization(self):
        """Test real client initialization."""
        from src.clients.colpali import ColPaliClient

        client = ColPaliClient()
        assert client.model is not None
        assert client.processor is not None
