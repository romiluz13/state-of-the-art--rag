"""Tests for Generator class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.generation.generator import Generator
from src.generation.base import GenerationConfig, GenerationResult
from src.retrieval.base import RetrievalResult


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.anthropic_api_key = "test-api-key"
    return settings


@pytest.fixture
def sample_context():
    """Create sample context."""
    return [
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Machine learning is a field of AI.",
            score=0.9,
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="Deep learning uses neural networks.",
            score=0.85,
        ),
    ]


@pytest.fixture
def mock_claude_response():
    """Create mock Claude API response."""
    response = MagicMock()
    response.content = [MagicMock(text="Machine learning [1] is a field of AI that uses data.")]
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


class TestGeneratorInit:
    """Test Generator initialization."""

    def test_init_with_settings(self, mock_settings):
        """Test initialization with settings."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            assert generator.settings == mock_settings
            assert generator.config is not None

    def test_init_with_config(self, mock_settings):
        """Test initialization with custom config."""
        config = GenerationConfig(model="claude-3-opus", max_tokens=4096)

        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings, config=config)

            assert generator.config.model == "claude-3-opus"
            assert generator.config.max_tokens == 4096

    def test_prompt_registry(self, mock_settings):
        """Test prompt registry initialization."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            assert "factual" in generator.prompts
            assert "graphrag" in generator.prompts
            assert "raptor" in generator.prompts


class TestGeneratorGenerate:
    """Test Generator.generate method."""

    @pytest.mark.asyncio
    async def test_generate_basic(self, mock_settings, sample_context, mock_claude_response):
        """Test basic generation."""
        with patch("src.generation.generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_claude_response
            mock_anthropic.return_value = mock_client

            generator = Generator(settings=mock_settings)
            result = await generator.generate(
                query="What is ML?",
                context=sample_context,
            )

            assert isinstance(result, GenerationResult)
            assert result.answer is not None
            assert result.query == "What is ML?"
            assert "token_usage" in result.__dict__ or result.token_usage is not None

    @pytest.mark.asyncio
    async def test_generate_with_prompt_type(self, mock_settings, sample_context, mock_claude_response):
        """Test generation with specific prompt type."""
        with patch("src.generation.generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_claude_response
            mock_anthropic.return_value = mock_client

            generator = Generator(settings=mock_settings)
            result = await generator.generate(
                query="What is ML?",
                context=sample_context,
                prompt_type="factual",
            )

            assert result.prompt_type == "factual"

    @pytest.mark.asyncio
    async def test_generate_extracts_citations(self, mock_settings, sample_context):
        """Test that citations are extracted from response."""
        response = MagicMock()
        response.content = [MagicMock(text="ML [1] uses data. DL [2] uses networks.")]
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50

        with patch("src.generation.generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            generator = Generator(settings=mock_settings)
            result = await generator.generate(
                query="What is ML?",
                context=sample_context,
            )

            assert len(result.citations) == 2
            assert result.citations[0].citation_id == "[1]"
            assert result.citations[1].citation_id == "[2]"


class TestCitationExtraction:
    """Test citation extraction from generated answers."""

    def test_extract_standard_citations(self, mock_settings):
        """Test extraction of [1], [2] format citations."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(chunk_id="c1", document_id="d1", content="Content 1", score=0.9),
                RetrievalResult(chunk_id="c2", document_id="d1", content="Content 2", score=0.8),
            ]

            citations = generator._extract_citations(
                "Answer with [1] and [2].",
                context,
            )

            assert len(citations) == 2
            assert citations[0].chunk_id == "c1"
            assert citations[1].chunk_id == "c2"

    def test_extract_raptor_citations(self, mock_settings):
        """Test extraction of [S1], [D1] format citations."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(chunk_id="s1", document_id="d1", content="Summary", score=0.9, level=1),
                RetrievalResult(chunk_id="d1", document_id="d1", content="Detail", score=0.8, level=0),
            ]

            citations = generator._extract_citations(
                "Summary [S1] and detail [D1].",
                context,
            )

            assert len(citations) == 2

    def test_extract_graphrag_citations(self, mock_settings):
        """Test extraction of [E1], [C1] format citations."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(
                    chunk_id="e1", document_id="d1", content="Entity",
                    score=0.9, metadata={"type": "entity"}
                ),
                RetrievalResult(
                    chunk_id="c1", document_id="d1", content="Community",
                    score=0.8, metadata={"type": "community"}
                ),
            ]

            citations = generator._extract_citations(
                "Entity [E1] and community [C1].",
                context,
            )

            assert len(citations) == 2

    def test_extract_no_duplicates(self, mock_settings):
        """Test that duplicate citations are not extracted twice."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(chunk_id="c1", document_id="d1", content="Content", score=0.9),
            ]

            citations = generator._extract_citations(
                "Reference [1] here and [1] again.",
                context,
            )

            assert len(citations) == 1


class TestFindContextForCitation:
    """Test _find_context_for_citation method."""

    def test_find_standard_citation(self, mock_settings):
        """Test finding context for standard citation."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(chunk_id="c1", document_id="d1", content="First", score=0.9),
                RetrievalResult(chunk_id="c2", document_id="d1", content="Second", score=0.8),
            ]

            result = generator._find_context_for_citation("", 0, context)
            assert result.chunk_id == "c1"

            result = generator._find_context_for_citation("", 1, context)
            assert result.chunk_id == "c2"

    def test_find_summary_citation(self, mock_settings):
        """Test finding context for summary citation [S1]."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(chunk_id="s1", document_id="d1", content="Summary", score=0.9, level=1),
                RetrievalResult(chunk_id="d1", document_id="d1", content="Detail", score=0.8, level=0),
            ]

            result = generator._find_context_for_citation("S", 0, context)
            assert result.chunk_id == "s1"

    def test_find_detail_citation(self, mock_settings):
        """Test finding context for detail citation [D1]."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(chunk_id="s1", document_id="d1", content="Summary", score=0.9, level=1),
                RetrievalResult(chunk_id="d1", document_id="d1", content="Detail", score=0.8, level=0),
            ]

            result = generator._find_context_for_citation("D", 0, context)
            assert result.chunk_id == "d1"

    def test_find_entity_citation(self, mock_settings):
        """Test finding context for entity citation [E1]."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(
                    chunk_id="e1", document_id="d1", content="Entity",
                    score=0.9, metadata={"type": "entity"}
                ),
                RetrievalResult(
                    chunk_id="c1", document_id="d1", content="Chunk",
                    score=0.8, metadata={"type": "chunk"}
                ),
            ]

            result = generator._find_context_for_citation("E", 0, context)
            assert result.chunk_id == "e1"

    def test_find_community_citation(self, mock_settings):
        """Test finding context for community citation [C1]."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(
                    chunk_id="c1", document_id="d1", content="Community",
                    score=0.9, metadata={"type": "community"}
                ),
            ]

            result = generator._find_context_for_citation("C", 0, context)
            assert result.chunk_id == "c1"

    def test_out_of_range_returns_none(self, mock_settings):
        """Test that out of range index returns None."""
        with patch("src.generation.generator.anthropic.Anthropic"):
            generator = Generator(settings=mock_settings)

            context = [
                RetrievalResult(chunk_id="c1", document_id="d1", content="Content", score=0.9),
            ]

            result = generator._find_context_for_citation("", 10, context)
            assert result is None
