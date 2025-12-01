"""Tests for Contextual Chunking (Anthropic method)."""

import pytest
from unittest.mock import AsyncMock

from src.ingestion.chunking.contextual import (
    ContextualChunker,
    BatchContextualChunker,
    CONTEXTUAL_PROMPT,
)
from src.ingestion.chunking.base import Chunk


class TestContextualChunker:
    """Tests for ContextualChunker."""

    @pytest.fixture
    def mock_generate_function(self):
        """Mock LLM generation function."""
        async def generate(prompt):
            return "This chunk discusses MongoDB configuration in the context of setting up a database cluster."
        return generate

    @pytest.fixture
    def chunker(self, mock_generate_function):
        """Create contextual chunker with mock."""
        return ContextualChunker(
            generate_function=mock_generate_function,
            max_document_tokens=8000,
        )

    @pytest.fixture
    def sample_document(self):
        """Sample document for testing."""
        return """MongoDB is a document database designed for ease of application development and scaling.

        Chapter 1: Installation
        MongoDB can be installed on various operating systems. The installation process involves downloading the appropriate package.

        Chapter 2: Configuration
        Configuration involves setting up the data directory, log paths, and network settings. The configuration file uses YAML format.

        Chapter 3: Operations
        Basic operations include insert, find, update, and delete. These operations use the MongoDB query language."""

    @pytest.fixture
    def sample_chunk(self):
        """Sample chunk for testing."""
        return Chunk(
            content="Configuration involves setting up the data directory, log paths, and network settings.",
            start_char=200,
            end_char=290,
            chunk_index=1,
            metadata={},
        )

    def test_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.max_document_tokens == 8000

    @pytest.mark.asyncio
    async def test_add_context_to_chunk(self, chunker, sample_document, sample_chunk):
        """Test adding context to a single chunk."""
        result = await chunker.add_context_to_chunk(sample_document, sample_chunk)

        # Result should contain context + original chunk
        assert sample_chunk.content in result
        assert len(result) > len(sample_chunk.content)

    @pytest.mark.asyncio
    async def test_add_context_to_chunks(self, chunker, sample_document):
        """Test adding context to multiple chunks."""
        chunks = [
            Chunk(content="Chunk 1 content", start_char=0, end_char=15, chunk_index=0, metadata={}),
            Chunk(content="Chunk 2 content", start_char=16, end_char=31, chunk_index=1, metadata={}),
        ]

        results = await chunker.add_context_to_chunks(sample_document, chunks)

        assert len(results) == 2
        assert "Chunk 1 content" in results[0]
        assert "Chunk 2 content" in results[1]

    def test_truncate_document_small(self, chunker, sample_chunk):
        """Test truncation with small document."""
        small_doc = "Short document."
        result = chunker._truncate_document(small_doc, sample_chunk)

        # Small doc should not be truncated
        assert result == small_doc

    def test_truncate_document_large(self, chunker, sample_chunk):
        """Test truncation with large document."""
        # Create document larger than max_chars (8000 * 4 = 32000)
        large_doc = "x" * 50000
        sample_chunk.start_char = 25000
        sample_chunk.end_char = 25100

        result = chunker._truncate_document(large_doc, sample_chunk)

        # Should be truncated
        assert len(result) < len(large_doc)
        assert result.startswith("...")
        assert result.endswith("...")

    @pytest.mark.asyncio
    async def test_context_generation_failure(self, sample_document, sample_chunk):
        """Test graceful handling of context generation failure."""
        async def failing_generate(prompt):
            raise Exception("LLM error")

        chunker = ContextualChunker(
            generate_function=failing_generate,
        )

        result = await chunker.add_context_to_chunk(sample_document, sample_chunk)

        # Should return original content on failure
        assert result == sample_chunk.content


class TestBatchContextualChunker:
    """Tests for BatchContextualChunker."""

    @pytest.fixture
    def mock_generate_function(self):
        """Mock single generation function."""
        async def generate(prompt):
            return "Context for this chunk."
        return generate

    @pytest.fixture
    def mock_batch_generate_function(self):
        """Mock batch generation function."""
        async def batch_generate(prompts):
            return [f"Batch context {i}" for i in range(len(prompts))]
        return batch_generate

    @pytest.fixture
    def batch_chunker(self, mock_generate_function, mock_batch_generate_function):
        """Create batch contextual chunker."""
        return BatchContextualChunker(
            generate_function=mock_generate_function,
            batch_generate_function=mock_batch_generate_function,
            batch_size=3,
        )

    def test_initialization(self, batch_chunker):
        """Test batch chunker initialization."""
        assert batch_chunker.batch_size == 3
        assert batch_chunker.batch_generate_function is not None

    @pytest.mark.asyncio
    async def test_batch_processing(self, batch_chunker):
        """Test batch context generation."""
        document = "Sample document content for testing."
        chunks = [
            Chunk(content=f"Chunk {i}", start_char=i*10, end_char=i*10+8, chunk_index=i, metadata={})
            for i in range(5)
        ]

        results = await batch_chunker.add_context_to_chunks(document, chunks)

        assert len(results) == 5
        # Check batch processing was used (contexts should have "Batch context" prefix)
        assert any("Batch context" in r for r in results)

    @pytest.mark.asyncio
    async def test_fallback_to_sequential(self, mock_generate_function):
        """Test fallback when batch function not provided."""
        chunker = BatchContextualChunker(
            generate_function=mock_generate_function,
            batch_generate_function=None,  # No batch function
            batch_size=3,
        )

        document = "Sample document."
        chunks = [
            Chunk(content="Chunk 1", start_char=0, end_char=7, chunk_index=0, metadata={}),
        ]

        results = await chunker.add_context_to_chunks(document, chunks)

        assert len(results) == 1


class TestContextualPrompt:
    """Tests for the contextual prompt template."""

    def test_prompt_format(self):
        """Test prompt can be formatted correctly."""
        formatted = CONTEXTUAL_PROMPT.format(
            document="Test document",
            chunk="Test chunk",
        )

        assert "Test document" in formatted
        assert "Test chunk" in formatted
        assert "<document>" in formatted
        assert "<chunk>" in formatted

    def test_prompt_instructions(self):
        """Test prompt contains proper instructions."""
        assert "50-100 tokens" in CONTEXTUAL_PROMPT
        assert "context" in CONTEXTUAL_PROMPT.lower()
