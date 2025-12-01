"""Tests for ColPali embedder."""

import pytest
import numpy as np
from PIL import Image

from src.clients.colpali import MockColPaliClient
from src.ingestion.embeddings.colpali import ColPaliEmbedder, ColPaliPageEmbedding
from src.ingestion.loaders.pdf import PageImage


class TestColPaliPageEmbedding:
    """Tests for ColPaliPageEmbedding dataclass."""

    @pytest.fixture
    def sample_embedding(self):
        """Create sample embedding array."""
        return np.random.rand(1024, 128).astype(np.float32)

    def test_basic_creation(self, sample_embedding):
        """Test basic embedding creation."""
        emb = ColPaliPageEmbedding(
            document_id="doc_123",
            page_num=0,
            embedding=sample_embedding,
            image_size=(800, 600),
            has_text=True,
            has_images=False,
        )

        assert emb.document_id == "doc_123"
        assert emb.page_num == 0
        assert emb.embedding.shape == (1024, 128)
        assert emb.image_size == (800, 600)
        assert emb.has_text is True
        assert emb.has_images is False

    def test_to_list_conversion(self, sample_embedding):
        """Test conversion to list for MongoDB storage."""
        emb = ColPaliPageEmbedding(
            document_id="doc_123",
            page_num=0,
            embedding=sample_embedding,
            image_size=(800, 600),
            has_text=True,
            has_images=False,
        )

        as_list = emb.to_list()

        assert isinstance(as_list, list)
        assert len(as_list) == 1024
        assert len(as_list[0]) == 128
        assert isinstance(as_list[0][0], float)

    def test_from_list_reconstruction(self, sample_embedding):
        """Test reconstruction from MongoDB-stored list."""
        original = ColPaliPageEmbedding(
            document_id="doc_123",
            page_num=5,
            embedding=sample_embedding,
            image_size=(800, 600),
            has_text=True,
            has_images=True,
        )

        # Convert to list and back
        as_list = original.to_list()
        reconstructed = ColPaliPageEmbedding.from_list(
            document_id="doc_123",
            page_num=5,
            embedding_list=as_list,
            image_size=(800, 600),
            has_text=True,
            has_images=True,
        )

        assert reconstructed.document_id == original.document_id
        assert reconstructed.page_num == original.page_num
        assert reconstructed.image_size == original.image_size
        np.testing.assert_array_almost_equal(
            reconstructed.embedding, original.embedding, decimal=5
        )

    def test_metadata_default(self, sample_embedding):
        """Test default metadata is empty dict."""
        emb = ColPaliPageEmbedding(
            document_id="doc_123",
            page_num=0,
            embedding=sample_embedding,
            image_size=(800, 600),
            has_text=True,
            has_images=False,
        )

        assert emb.metadata == {}

    def test_metadata_custom(self, sample_embedding):
        """Test custom metadata."""
        emb = ColPaliPageEmbedding(
            document_id="doc_123",
            page_num=0,
            embedding=sample_embedding,
            image_size=(800, 600),
            has_text=True,
            has_images=False,
            metadata={"source": "test.pdf"},
        )

        assert emb.metadata == {"source": "test.pdf"}


class TestColPaliEmbedder:
    """Tests for ColPali embedder."""

    @pytest.fixture
    def embedder(self):
        """Create embedder with mock client."""
        return ColPaliEmbedder(use_mock=True)

    @pytest.fixture
    def page_image(self):
        """Create test page image."""
        return PageImage(
            page_num=0,
            image=Image.new("RGB", (800, 600), color="white"),
            width=800,
            height=600,
            has_text=True,
            has_images=False,
        )

    def test_initialization_with_mock(self):
        """Test initialization with mock client."""
        embedder = ColPaliEmbedder(use_mock=True)
        assert isinstance(embedder.client, MockColPaliClient)

    def test_initialization_with_client(self):
        """Test initialization with provided client."""
        client = MockColPaliClient()
        embedder = ColPaliEmbedder(client=client)
        assert embedder.client is client

    def test_embed_page(self, embedder, page_image):
        """Test single page embedding."""
        result = embedder.embed_page("doc_123", page_image)

        assert isinstance(result, ColPaliPageEmbedding)
        assert result.document_id == "doc_123"
        assert result.page_num == 0
        assert result.image_size == (800, 600)
        assert result.has_text is True
        assert result.has_images is False
        assert result.embedding.shape[1] == 128  # embedding_dim

    def test_embed_pages_multiple(self, embedder, page_image):
        """Test batch page embedding."""
        pages = [
            page_image,
            PageImage(
                page_num=1,
                image=Image.new("RGB", (800, 600), color="gray"),
                width=800,
                height=600,
                has_text=True,
                has_images=True,
            ),
        ]

        results = embedder.embed_pages("doc_123", pages)

        assert len(results) == 2
        assert results[0].page_num == 0
        assert results[1].page_num == 1
        assert results[1].has_images is True

    def test_embed_pages_empty(self, embedder):
        """Test embedding empty page list."""
        results = embedder.embed_pages("doc_123", [])
        assert results == []

    def test_embed_query(self, embedder):
        """Test query embedding."""
        embedding = embedder.embed_query("find charts about revenue")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[1] == 128

    def test_compute_similarity(self, embedder, page_image):
        """Test similarity computation."""
        page_emb = embedder.embed_page("doc_123", page_image)
        query_emb = embedder.embed_query("test query")

        score = embedder.compute_similarity(query_emb, page_emb)

        assert isinstance(score, float)
        assert score >= 0.0

    def test_rank_pages(self, embedder, page_image):
        """Test page ranking."""
        pages = [
            page_image,
            PageImage(
                page_num=1,
                image=Image.new("RGB", (800, 600), color="gray"),
                width=800,
                height=600,
                has_text=True,
                has_images=False,
            ),
        ]
        page_embeddings = embedder.embed_pages("doc_123", pages)

        rankings = embedder.rank_pages("test query", page_embeddings, top_k=2)

        assert len(rankings) == 2
        for page_emb, score in rankings:
            assert isinstance(page_emb, ColPaliPageEmbedding)
            assert isinstance(score, float)

        # Check sorted by score descending
        scores = [s for _, s in rankings]
        assert scores == sorted(scores, reverse=True)


class TestPageImage:
    """Tests for PageImage dataclass."""

    def test_basic_creation(self):
        """Test basic PageImage creation."""
        img = Image.new("RGB", (100, 100), color="white")
        page = PageImage(
            page_num=0,
            image=img,
            width=100,
            height=100,
        )

        assert page.page_num == 0
        assert page.width == 100
        assert page.height == 100
        assert page.has_text is True  # default
        assert page.has_images is False  # default

    def test_custom_flags(self):
        """Test custom has_text and has_images."""
        img = Image.new("RGB", (100, 100))
        page = PageImage(
            page_num=5,
            image=img,
            width=100,
            height=100,
            has_text=False,
            has_images=True,
        )

        assert page.has_text is False
        assert page.has_images is True
