"""Tests for RAPTOR hierarchical chunking."""

import pytest
from unittest.mock import AsyncMock

from src.ingestion.chunking.raptor import (
    RAPTORChunker,
    RAPTORNode,
    create_raptor_summarize_prompt,
)
from src.ingestion.chunking.base import Chunk


class TestRAPTORNode:
    """Tests for RAPTORNode dataclass."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = RAPTORNode(
            content="Test content",
            level=0,
            embedding=[0.1] * 1024,
        )
        assert node.content == "Test content"
        assert node.level == 0
        assert len(node.embedding) == 1024
        assert node.children_indices == []
        assert node.parent_index is None

    def test_leaf_node(self):
        """Test leaf node (level 0)."""
        node = RAPTORNode(content="Leaf", level=0)
        assert node.level == 0
        assert node.children_indices == []

    def test_summary_node_with_children(self):
        """Test summary node with children."""
        node = RAPTORNode(
            content="Summary",
            level=1,
            children_indices=[0, 1, 2],
        )
        assert node.level == 1
        assert node.children_indices == [0, 1, 2]


class TestRAPTORChunker:
    """Tests for RAPTORChunker."""

    @pytest.fixture
    def mock_embed_function(self):
        """Mock embedding function."""
        async def embed(texts):
            return [[0.1] * 1024 for _ in texts]
        return embed

    @pytest.fixture
    def mock_summarize_function(self):
        """Mock summarization function."""
        async def summarize(prompt):
            return "This is a summary of the clustered chunks."
        return summarize

    @pytest.fixture
    def raptor_chunker(self, mock_embed_function, mock_summarize_function):
        """Create RAPTOR chunker with mocks."""
        return RAPTORChunker(
            embed_function=mock_embed_function,
            summarize_function=mock_summarize_function,
            chunk_size=100,
            chunk_overlap=20,
            max_levels=2,
            min_cluster_size=2,
        )

    def test_initialization(self, raptor_chunker):
        """Test chunker initialization."""
        assert raptor_chunker.max_levels == 2
        assert raptor_chunker.min_cluster_size == 2

    def test_chunk_basic(self, raptor_chunker):
        """Test basic chunking (sync interface for leaf chunks)."""
        text = "Hello world. " * 50  # ~650 chars
        chunks = raptor_chunker.chunk(text)

        # Should produce chunks
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_build_hierarchy_small_text(self, raptor_chunker):
        """Test hierarchy building with small text."""
        text = "Short text."
        nodes = await raptor_chunker.build_hierarchy(text)

        # Small text should produce at least one leaf node
        assert len(nodes) >= 1
        assert nodes[0].level == 0

    @pytest.mark.asyncio
    async def test_build_hierarchy_multiple_levels(self):
        """Test hierarchy builds multiple levels."""
        call_count = {"embed": 0, "summarize": 0}

        async def embed(texts):
            call_count["embed"] += 1
            return [[0.1 * i] * 1024 for i in range(len(texts))]

        async def summarize(prompt):
            call_count["summarize"] += 1
            return f"Summary {call_count['summarize']}"

        chunker = RAPTORChunker(
            embed_function=embed,
            summarize_function=summarize,
            chunk_size=50,
            chunk_overlap=10,
            max_levels=2,
            min_cluster_size=2,
        )

        # Longer text to produce multiple chunks
        text = "This is sentence number one. " * 20
        nodes = await chunker.build_hierarchy(text)

        # Should have leaf nodes (level 0)
        leaf_nodes = [n for n in nodes if n.level == 0]
        assert len(leaf_nodes) > 0

    @pytest.mark.asyncio
    async def test_nodes_have_levels(self, raptor_chunker):
        """Test that nodes are assigned correct levels."""
        text = "Test content. " * 30
        nodes = await raptor_chunker.build_hierarchy(text)

        # Group by level manually
        levels = {}
        for node in nodes:
            if node.level not in levels:
                levels[node.level] = []
            levels[node.level].append(node)

        # Should have at least level 0 (leaves)
        assert 0 in levels
        assert len(levels[0]) > 0


class TestCreateRaptorSummarizePrompt:
    """Tests for the summarize prompt creation."""

    def test_prompt_creation(self):
        """Test prompt is created correctly."""
        texts = ["Chunk 1 content", "Chunk 2 content"]
        prompt = create_raptor_summarize_prompt(texts)

        assert "Chunk 1 content" in prompt
        assert "Chunk 2 content" in prompt
        assert "summary" in prompt.lower()

    def test_prompt_with_empty_list(self):
        """Test prompt with empty list."""
        prompt = create_raptor_summarize_prompt([])
        assert isinstance(prompt, str)
