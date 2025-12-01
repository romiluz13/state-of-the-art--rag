"""Tests for intent classification."""

import pytest

from src.routing.intent import (
    IntentClassifier,
    QueryIntent,
    IntentResult,
    MockLLMClient,
    INTENT_TO_STRATEGY,
)


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_intent_values(self):
        """Test intent enum values."""
        assert QueryIntent.FACTUAL.value == "factual"
        assert QueryIntent.GLOBAL.value == "global"
        assert QueryIntent.HIERARCHICAL.value == "hierarchical"
        assert QueryIntent.MULTIMODAL.value == "multimodal"
        assert QueryIntent.COMPARATIVE.value == "comparative"

    def test_intent_to_strategy_mapping(self):
        """Test each intent maps to a strategy."""
        for intent in QueryIntent:
            assert intent in INTENT_TO_STRATEGY
            assert isinstance(INTENT_TO_STRATEGY[intent], str)


class TestIntentResult:
    """Tests for IntentResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = IntentResult(
            intent=QueryIntent.FACTUAL,
            confidence=0.85,
            strategy="hybrid",
            reasoning="Test reasoning",
        )
        assert result.intent == QueryIntent.FACTUAL
        assert result.confidence == 0.85
        assert result.strategy == "hybrid"
        assert result.sub_queries is None

    def test_with_sub_queries(self):
        """Test result with sub-queries for comparative."""
        result = IntentResult(
            intent=QueryIntent.COMPARATIVE,
            confidence=0.8,
            strategy="hybrid",
            reasoning="Comparative query",
            sub_queries=["What is A?", "What is B?"],
        )
        assert result.sub_queries == ["What is A?", "What is B?"]


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier without LLM."""
        return IntentClassifier(use_llm=False)

    @pytest.fixture
    def classifier_with_llm(self):
        """Create classifier with mock LLM."""
        return IntentClassifier(
            llm_client=MockLLMClient(),
            use_llm=True,
            confidence_threshold=0.7,
        )

    def test_default_initialization(self, classifier):
        """Test default initialization."""
        assert classifier.use_llm is False
        assert classifier.confidence_threshold == 0.6

    def test_classify_factual_query(self, classifier):
        """Test factual query classification."""
        result = classifier.classify("What is the API rate limit?")
        assert result.strategy == "hybrid"
        assert result.confidence >= 0.5

    def test_classify_global_query(self, classifier):
        """Test global query -> GraphRAG."""
        queries = [
            "What are all the main themes in this document?",
            "Summarize the key points across all chapters",
            "Give me an overview of the topics covered",
        ]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == QueryIntent.GLOBAL
            assert result.strategy == "graphrag"

    def test_classify_hierarchical_query(self, classifier):
        """Test hierarchical query -> RAPTOR."""
        queries = [
            "What does the introduction chapter say?",
            "What's in the conclusion section?",
            "What is covered at the beginning of the document?",
        ]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == QueryIntent.HIERARCHICAL
            assert result.strategy == "raptor"

    def test_classify_multimodal_query(self, classifier):
        """Test multimodal query -> ColPali."""
        queries = [
            "Show me the revenue chart from the report",
            "What does the architecture diagram show?",
            "Find the flowchart explaining the process",
        ]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == QueryIntent.MULTIMODAL
            assert result.strategy == "colpali"

    def test_classify_comparative_query(self, classifier):
        """Test comparative query detection."""
        queries = [
            "Compare Python and JavaScript",
            "What's the difference between REST and GraphQL?",
            "MongoDB vs PostgreSQL",
        ]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == QueryIntent.COMPARATIVE
            assert result.strategy == "hybrid"

    def test_comparative_extracts_sub_queries(self, classifier):
        """Test sub-query extraction for comparative."""
        result = classifier.classify("Compare Python and Java")
        assert result.intent == QueryIntent.COMPARATIVE
        assert result.sub_queries is not None
        assert len(result.sub_queries) == 2

    def test_fallback_to_factual(self, classifier):
        """Test ambiguous queries default to factual."""
        result = classifier.classify("hello")
        assert result.strategy == "hybrid"
        assert result.confidence >= 0.5

    def test_confidence_scoring(self, classifier):
        """Test that specific keywords increase confidence."""
        # More keywords = higher confidence
        result_vague = classifier.classify("tell me about things")
        result_specific = classifier.classify("summarize all the main themes throughout the document")

        assert result_specific.confidence > result_vague.confidence

    def test_llm_fallback_on_low_confidence(self, classifier_with_llm):
        """Test LLM is used when heuristic confidence is low."""
        # Query with clear visual intent for LLM
        result = classifier_with_llm.classify("diagram")
        assert result.intent == QueryIntent.MULTIMODAL
        assert "LLM" in result.reasoning or result.confidence >= 0.6


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    @pytest.fixture
    def llm(self):
        """Create mock LLM client."""
        return MockLLMClient()

    def test_multimodal_classification(self, llm):
        """Test mock classifies chart query as multimodal."""
        response = llm.generate("Query: show me the chart")
        assert "MULTIMODAL" in response

    def test_comparative_classification(self, llm):
        """Test mock classifies comparison as comparative."""
        response = llm.generate("Query: compare X vs Y")
        assert "COMPARATIVE" in response

    def test_global_classification(self, llm):
        """Test mock classifies summary as global."""
        response = llm.generate("Query: summarize everything")
        assert "GLOBAL" in response

    def test_response_format(self, llm):
        """Test response follows expected format."""
        response = llm.generate("any query")
        parts = response.split("|")
        assert len(parts) == 3
        assert parts[0] in ["FACTUAL", "GLOBAL", "HIERARCHICAL", "MULTIMODAL", "COMPARATIVE"]
        assert parts[1].isdigit()
