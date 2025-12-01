"""Tests for prompt templates."""

import pytest

from src.generation.prompts import (
    PromptTemplate,
    FactualPrompt,
    GraphRAGPrompt,
    RAPTORPrompt,
)
from src.generation.prompts.base import SimplePrompt
from src.retrieval.base import RetrievalResult


@pytest.fixture
def sample_context():
    """Create sample context for testing."""
    return [
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Machine learning is a subset of artificial intelligence.",
            score=0.9,
            metadata={"type": "chunk"},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="Deep learning uses neural networks with multiple layers.",
            score=0.85,
            metadata={"type": "chunk"},
        ),
    ]


@pytest.fixture
def graphrag_context():
    """Create GraphRAG context with entities and communities."""
    return [
        RetrievalResult(
            chunk_id="entity-1",
            document_id="doc-1",
            content="Machine Learning: A field of AI focused on learning from data.",
            score=0.9,
            metadata={"type": "entity", "entity_type": "Concept", "entity_name": "Machine Learning"},
        ),
        RetrievalResult(
            chunk_id="community-1",
            document_id="doc-1",
            content="AI and ML form the foundation of modern data science.",
            score=0.85,
            metadata={"type": "community", "community_level": 2},
        ),
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Detailed explanation of ML algorithms.",
            score=0.8,
            metadata={"type": "chunk"},
        ),
    ]


@pytest.fixture
def raptor_context():
    """Create RAPTOR context with hierarchy levels."""
    return [
        RetrievalResult(
            chunk_id="summary-1",
            document_id="doc-1",
            content="High-level summary of the document about AI.",
            score=0.9,
            level=2,
            metadata={},
        ),
        RetrievalResult(
            chunk_id="summary-2",
            document_id="doc-1",
            content="Section summary about machine learning.",
            score=0.85,
            level=1,
            metadata={},
        ),
        RetrievalResult(
            chunk_id="detail-1",
            document_id="doc-1",
            content="Detailed paragraph about neural networks.",
            score=0.8,
            level=0,
            metadata={},
        ),
    ]


class TestSimplePrompt:
    """Test SimplePrompt class."""

    def test_format(self, sample_context):
        """Test basic prompt formatting."""
        prompt = SimplePrompt()
        result = prompt.format("What is ML?", sample_context)

        assert "What is ML?" in result
        assert "[1]" in result
        assert "[2]" in result
        assert "Machine learning" in result

    def test_system_message(self):
        """Test system message."""
        prompt = SimplePrompt()
        system = prompt.get_system_message()

        assert "helpful assistant" in system
        assert "context" in system.lower()


class TestFactualPrompt:
    """Test FactualPrompt class."""

    def test_format(self, sample_context):
        """Test factual prompt formatting."""
        prompt = FactualPrompt()
        result = prompt.format("What is ML?", sample_context)

        assert "QUESTION:" in result
        assert "SOURCES:" in result
        assert "[1]" in result
        assert "citation" in result.lower() or "INSTRUCTIONS" in result

    def test_system_message_requires_citations(self):
        """Test system message emphasizes citations."""
        prompt = FactualPrompt()
        system = prompt.get_system_message()

        assert "citation" in system.lower()
        assert "unsupported" in system.lower() or "NEVER" in system

    def test_empty_context(self):
        """Test with empty context."""
        prompt = FactualPrompt()
        result = prompt.format("What is ML?", [])

        assert "What is ML?" in result


class TestGraphRAGPrompt:
    """Test GraphRAGPrompt class."""

    def test_format_separates_context_types(self, graphrag_context):
        """Test that context is separated by type."""
        prompt = GraphRAGPrompt()
        result = prompt.format("What is ML?", graphrag_context)

        # Should have entity section
        assert "ENTITIES" in result
        # Should have community section
        assert "COMMUNITY" in result
        # Should have supporting evidence
        assert "SUPPORTING" in result or "EVIDENCE" in result

    def test_entity_citations(self, graphrag_context):
        """Test entity citation format."""
        prompt = GraphRAGPrompt()
        result = prompt.format("What is ML?", graphrag_context)

        # Should use [E1] format for entities
        assert "[E1]" in result
        # Should use [C1] format for communities
        assert "[C1]" in result

    def test_system_message(self):
        """Test system message for synthesis."""
        prompt = GraphRAGPrompt()
        system = prompt.get_system_message()

        assert "synthesis" in system.lower() or "connect" in system.lower()


class TestRAPTORPrompt:
    """Test RAPTORPrompt class."""

    def test_format_separates_levels(self, raptor_context):
        """Test that context is separated by level."""
        prompt = RAPTORPrompt()
        result = prompt.format("What is AI?", raptor_context)

        # Should have summaries section
        assert "SUMMARIES" in result or "HIGH-LEVEL" in result
        # Should have details section
        assert "DETAIL" in result

    def test_summary_citations(self, raptor_context):
        """Test summary citation format."""
        prompt = RAPTORPrompt()
        result = prompt.format("What is AI?", raptor_context)

        # Should use [S1], [S2] for summaries
        assert "[S1]" in result
        # Should use [D1] for details
        assert "[D1]" in result

    def test_level_indicators(self, raptor_context):
        """Test level indicators in output."""
        prompt = RAPTORPrompt()
        result = prompt.format("What is AI?", raptor_context)

        # Should show level info
        assert "Level" in result

    def test_only_summaries(self):
        """Test with only high-level summaries."""
        context = [
            RetrievalResult(
                chunk_id="summary-1",
                document_id="doc-1",
                content="Summary content",
                score=0.9,
                level=1,
            ),
        ]
        prompt = RAPTORPrompt()
        result = prompt.format("Test", context)

        assert "No detailed content" in result or "[S1]" in result

    def test_only_details(self):
        """Test with only detail-level content."""
        context = [
            RetrievalResult(
                chunk_id="detail-1",
                document_id="doc-1",
                content="Detail content",
                score=0.9,
                level=0,
            ),
        ]
        prompt = RAPTORPrompt()
        result = prompt.format("Test", context)

        assert "No high-level" in result or "[D1]" in result


class TestPromptTemplateBase:
    """Test PromptTemplate base class functionality."""

    def test_format_context_with_citations(self, sample_context):
        """Test context formatting with citation markers."""
        prompt = SimplePrompt()
        formatted, citation_map = prompt.format_context(sample_context)

        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "[1]" in citation_map
        assert citation_map["[1]"].content == sample_context[0].content

    def test_format_context_respects_token_limit(self):
        """Test that format_context respects token limits."""
        # Create large context
        large_context = [
            RetrievalResult(
                chunk_id=f"chunk-{i}",
                document_id="doc-1",
                content="x" * 1000,  # ~250 tokens each
                score=0.9 - i * 0.1,
            )
            for i in range(50)
        ]

        prompt = SimplePrompt()
        formatted, citation_map = prompt.format_context(large_context, max_tokens=1000)

        # Should not include all chunks
        assert len(citation_map) < 50

    def test_citation_format(self):
        """Test citation format constant."""
        assert PromptTemplate.CITATION_FORMAT == "[{index}]"
