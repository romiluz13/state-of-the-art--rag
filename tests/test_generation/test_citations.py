"""Tests for citation extraction and verification."""

import pytest

from src.generation.citations import CitationExtractor, CitationVerification
from src.generation.base import Citation
from src.retrieval.base import RetrievalResult


@pytest.fixture
def sample_context():
    """Create sample context."""
    return [
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            score=0.9,
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="Deep learning uses neural networks with multiple hidden layers.",
            score=0.85,
        ),
        RetrievalResult(
            chunk_id="chunk-3",
            document_id="doc-2",
            content="Reinforcement learning involves agents learning from rewards.",
            score=0.8,
        ),
    ]


@pytest.fixture
def graphrag_context():
    """Create GraphRAG context."""
    return [
        RetrievalResult(
            chunk_id="entity-1",
            document_id="doc-1",
            content="Entity description for ML",
            score=0.9,
            metadata={"type": "entity"},
        ),
        RetrievalResult(
            chunk_id="community-1",
            document_id="doc-1",
            content="Community summary about AI",
            score=0.85,
            metadata={"type": "community"},
        ),
    ]


@pytest.fixture
def raptor_context():
    """Create RAPTOR context."""
    return [
        RetrievalResult(
            chunk_id="summary-1",
            document_id="doc-1",
            content="High-level summary",
            score=0.9,
            level=1,
        ),
        RetrievalResult(
            chunk_id="detail-1",
            document_id="doc-1",
            content="Detailed content",
            score=0.85,
            level=0,
        ),
    ]


class TestCitationExtractor:
    """Test CitationExtractor class."""

    def test_extract_standard_citations(self, sample_context):
        """Test extraction of standard [1], [2] citations."""
        extractor = CitationExtractor()
        answer = "Machine learning [1] enables AI systems. Deep learning [2] uses neural networks."

        citations = extractor.extract_citations(answer, sample_context)

        assert len(citations) == 2
        assert citations[0].citation_id == "[1]"
        assert citations[0].chunk_id == "chunk-1"
        assert citations[1].citation_id == "[2]"
        assert citations[1].chunk_id == "chunk-2"

    def test_extract_duplicate_citations(self, sample_context):
        """Test that duplicate citations are not repeated."""
        extractor = CitationExtractor()
        answer = "ML [1] is great. As mentioned [1], ML is important. Also [2]."

        citations = extractor.extract_citations(answer, sample_context)

        assert len(citations) == 2  # [1] only counted once

    def test_extract_graphrag_citations(self, graphrag_context):
        """Test extraction of entity and community citations."""
        extractor = CitationExtractor()
        answer = "According to entity [E1] and community [C1], AI is important."

        citations = extractor.extract_citations(answer, graphrag_context)

        # Should find both types
        citation_ids = [c.citation_id for c in citations]
        assert "[E1]" in citation_ids
        assert "[C1]" in citation_ids

    def test_extract_raptor_citations(self, raptor_context):
        """Test extraction of summary and detail citations."""
        extractor = CitationExtractor()
        answer = "The summary [S1] provides overview. Details [D1] explain more."

        citations = extractor.extract_citations(answer, raptor_context)

        citation_ids = [c.citation_id for c in citations]
        assert "[S1]" in citation_ids
        assert "[D1]" in citation_ids

    def test_extract_no_citations(self, sample_context):
        """Test with answer containing no citations."""
        extractor = CitationExtractor()
        answer = "Machine learning is interesting but I won't cite anything."

        citations = extractor.extract_citations(answer, sample_context)

        assert len(citations) == 0

    def test_extract_out_of_range_citation(self, sample_context):
        """Test handling of out-of-range citations."""
        extractor = CitationExtractor()
        answer = "Something [1] and something else [99]."

        citations = extractor.extract_citations(answer, sample_context)

        # Only [1] should be extracted successfully
        assert len(citations) == 1
        assert citations[0].citation_id == "[1]"


class TestCitationVerification:
    """Test citation verification."""

    def test_verify_supported_citation(self, sample_context):
        """Test verification of supported citation."""
        extractor = CitationExtractor()
        answer = "Machine learning is a subset of AI [1]."

        citations = extractor.extract_citations(answer, sample_context)
        verifications = extractor.verify_citations(answer, citations, sample_context)

        assert len(verifications) == 1
        # Should have high support since claim matches source
        assert verifications[0].support_score > 0.3

    def test_verify_unsupported_citation(self, sample_context):
        """Test verification of unsupported citation."""
        extractor = CitationExtractor()
        # Claim not supported by source
        answer = "Python was invented in 1991 [1]."

        citations = [
            Citation(
                citation_id="[1]",
                chunk_id="chunk-1",
                document_id="doc-1",
                text=sample_context[0].content,
                relevance_score=0.9,
            )
        ]

        verifications = extractor.verify_citations(answer, citations, sample_context)

        assert len(verifications) == 1
        # Low support since claim doesn't match source
        assert verifications[0].support_score < 0.5

    def test_extract_claim_for_citation(self, sample_context):
        """Test claim extraction around citation."""
        extractor = CitationExtractor()
        answer = "First sentence. Machine learning is great [1]. Last sentence."

        claim = extractor._extract_claim_for_citation(answer, "[1]")

        assert "Machine learning" in claim
        assert "[1]" in claim


class TestUncitedClaimDetection:
    """Test detection of uncited claims."""

    def test_find_uncited_factual_claims(self, sample_context):
        """Test finding uncited factual statements."""
        extractor = CitationExtractor()
        answer = (
            "Machine learning is used by 90% of companies [1]. "
            "Deep learning requires large datasets. "  # Uncited factual claim
            "Neural networks have multiple layers."  # Another uncited claim
        )

        uncited = extractor.find_uncited_claims(answer, sample_context)

        # Should identify uncited factual claims
        assert len(uncited) >= 1
        # Should contain statements with factual indicators
        assert any("dataset" in claim.lower() or "neural" in claim.lower() for claim in uncited)

    def test_skip_questions(self, sample_context):
        """Test that questions are not flagged as uncited."""
        extractor = CitationExtractor()
        answer = "What is machine learning? It uses data [1]. Why is it useful?"

        uncited = extractor.find_uncited_claims(answer, sample_context)

        # Questions should not be in uncited list
        assert not any("?" in claim for claim in uncited)

    def test_skip_short_sentences(self, sample_context):
        """Test that short sentences are skipped."""
        extractor = CitationExtractor()
        answer = "Yes. ML is important [1]. No. Maybe."

        uncited = extractor.find_uncited_claims(answer, sample_context)

        # Very short sentences should be skipped
        assert "Yes" not in uncited
        assert "No" not in uncited


class TestCitationSummary:
    """Test citation summary statistics."""

    def test_get_citation_summary(self, sample_context):
        """Test citation summary generation."""
        extractor = CitationExtractor()
        citations = [
            Citation(
                citation_id="[1]",
                chunk_id="chunk-1",
                document_id="doc-1",
                text="Content 1",
                relevance_score=0.9,
            ),
            Citation(
                citation_id="[2]",
                chunk_id="chunk-2",
                document_id="doc-1",
                text="Content 2",
                relevance_score=0.8,
            ),
            Citation(
                citation_id="[3]",
                chunk_id="chunk-3",
                document_id="doc-2",
                text="Content 3",
                relevance_score=0.7,
            ),
        ]

        summary = extractor.get_citation_summary(citations)

        assert summary["total_citations"] == 3
        assert summary["unique_sources"] == 3
        assert summary["unique_documents"] == 2

    def test_get_citation_summary_with_verifications(self, sample_context):
        """Test citation summary with verification results."""
        extractor = CitationExtractor()
        citations = [
            Citation(
                citation_id="[1]",
                chunk_id="chunk-1",
                document_id="doc-1",
                text="Content",
                relevance_score=0.9,
            )
        ]

        verifications = [
            CitationVerification(
                citation=citations[0],
                is_valid=True,
                support_score=0.8,
                claim_text="ML is great",
                source_text="ML content",
                issues=[],
            )
        ]

        summary = extractor.get_citation_summary(citations, verifications)

        assert summary["verified_citations"] == 1
        assert summary["verification_rate"] == 1.0
        assert summary["avg_support_score"] == 0.8


class TestSupportScoreCalculation:
    """Test support score calculation."""

    def test_high_overlap_score(self):
        """Test high score for matching content."""
        extractor = CitationExtractor()

        claim = "Machine learning uses data to learn"
        source = "Machine learning is a method that uses data to learn patterns"

        score = extractor._calculate_support_score(claim, source)

        assert score > 0.5

    def test_low_overlap_score(self):
        """Test low score for unrelated content."""
        extractor = CitationExtractor()

        claim = "Python is a programming language"
        source = "Machine learning uses neural networks"

        score = extractor._calculate_support_score(claim, source)

        assert score < 0.4

    def test_numerical_mismatch_detection(self):
        """Test detection of numerical mismatches."""
        extractor = CitationExtractor()

        # Claim has number not in source
        claim = "The model has 95% accuracy"
        source = "The model achieved 87% accuracy"

        has_mismatch = extractor._has_numerical_mismatch(claim, source)

        assert has_mismatch

    def test_numerical_match(self):
        """Test matching numbers."""
        extractor = CitationExtractor()

        claim = "The model has 95% accuracy"
        source = "Achieving 95% accuracy on the test set"

        has_mismatch = extractor._has_numerical_mismatch(claim, source)

        assert not has_mismatch
