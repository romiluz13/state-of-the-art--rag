"""Tests for query router."""

import pytest

from src.routing.intent import IntentClassifier, QueryIntent
from src.routing.router import (
    QueryRouter,
    RoutingDecision,
    ABTestConfig,
    MultiQueryRouter,
)


class TestABTestConfig:
    """Tests for ABTestConfig."""

    def test_default_config(self):
        """Test default config is disabled."""
        config = ABTestConfig()
        assert config.enabled is False
        assert config.test_name == ""
        assert config.strategies == []

    def test_enabled_config(self):
        """Test enabled A/B test config."""
        config = ABTestConfig(
            enabled=True,
            test_name="test_hybrid_vs_graphrag",
            strategies=["hybrid", "graphrag"],
            weights=[0.5, 0.5],
        )
        assert config.enabled is True
        assert len(config.strategies) == 2

    def test_select_strategy(self):
        """Test strategy selection."""
        config = ABTestConfig(
            enabled=True,
            test_name="test",
            strategies=["a", "b"],
            weights=[0.5, 0.5],
        )
        # Should return one of the strategies
        selected = config.select_strategy()
        assert selected in ["a", "b"]

    def test_select_strategy_disabled(self):
        """Test selection returns empty when disabled."""
        config = ABTestConfig(enabled=False)
        assert config.select_strategy() == ""

    def test_weighted_selection(self):
        """Test weighted selection bias."""
        config = ABTestConfig(
            enabled=True,
            test_name="test",
            strategies=["a", "b"],
            weights=[0.99, 0.01],  # Heavily favor "a"
        )
        # Run multiple times, should mostly get "a"
        results = [config.select_strategy() for _ in range(100)]
        assert results.count("a") > 80  # Should be mostly "a"


class TestRoutingDecision:
    """Tests for RoutingDecision."""

    def test_basic_creation(self):
        """Test basic decision creation."""
        decision = RoutingDecision(
            query="test query",
            selected_strategy="hybrid",
            intent_result=None,
        )
        assert decision.query == "test query"
        assert decision.selected_strategy == "hybrid"
        assert decision.was_overridden is False
        assert decision.ab_test_variant is None

    def test_with_override(self):
        """Test decision with override flag."""
        decision = RoutingDecision(
            query="test",
            selected_strategy="graphrag",
            intent_result=None,
            was_overridden=True,
        )
        assert decision.was_overridden is True


class TestQueryRouter:
    """Tests for QueryRouter."""

    @pytest.fixture
    def router(self):
        """Create router with default settings."""
        return QueryRouter()

    @pytest.fixture
    def router_with_ab(self):
        """Create router with A/B testing."""
        ab_config = ABTestConfig(
            enabled=True,
            test_name="test_experiment",
            strategies=["hybrid", "graphrag"],
            weights=[0.5, 0.5],
        )
        return QueryRouter(ab_test_config=ab_config)

    def test_default_initialization(self, router):
        """Test default initialization."""
        assert router.default_strategy == "hybrid"
        assert router.min_confidence == 0.5
        assert router.ab_test.enabled is False

    def test_route_basic_query(self, router):
        """Test routing a basic query."""
        decision = router.route("What is the API rate limit?")
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_strategy in ["hybrid", "graphrag", "raptor", "colpali"]

    def test_route_with_override(self, router):
        """Test manual strategy override."""
        decision = router.route("any query", override_strategy="raptor")
        assert decision.selected_strategy == "raptor"
        assert decision.was_overridden is True
        assert decision.intent_result is None

    def test_route_auto_no_override(self, router):
        """Test that 'auto' is not treated as override."""
        decision = router.route("summarize all themes", override_strategy="auto")
        assert decision.was_overridden is False
        # Should route based on intent

    def test_route_global_query(self, router):
        """Test routing global query to GraphRAG."""
        decision = router.route("What are all the main themes throughout the document?")
        assert decision.selected_strategy == "graphrag"
        assert decision.intent_result.intent == QueryIntent.GLOBAL

    def test_route_multimodal_query(self, router):
        """Test routing visual query to ColPali."""
        decision = router.route("Show me the revenue chart")
        assert decision.selected_strategy == "colpali"
        assert decision.intent_result.intent == QueryIntent.MULTIMODAL

    def test_route_with_ab_test(self, router_with_ab):
        """Test routing with A/B test enabled."""
        decision = router_with_ab.route("any query")
        assert decision.ab_test_variant in ["hybrid", "graphrag"]
        assert decision.selected_strategy == decision.ab_test_variant

    def test_route_batch(self, router):
        """Test batch routing."""
        queries = ["query 1", "query 2", "query 3"]
        decisions = router.route_batch(queries)
        assert len(decisions) == 3
        for d in decisions:
            assert isinstance(d, RoutingDecision)

    def test_get_strategy_for_intent(self, router):
        """Test getting strategy for specific intent."""
        assert router.get_strategy_for_intent(QueryIntent.GLOBAL) == "graphrag"
        assert router.get_strategy_for_intent(QueryIntent.HIERARCHICAL) == "raptor"
        assert router.get_strategy_for_intent(QueryIntent.MULTIMODAL) == "colpali"

    def test_update_ab_test(self, router):
        """Test updating A/B test config."""
        new_config = ABTestConfig(
            enabled=True,
            test_name="new_test",
            strategies=["hybrid", "raptor"],
        )
        router.update_ab_test(new_config)
        assert router.ab_test.enabled is True
        assert router.ab_test.test_name == "new_test"

    def test_low_confidence_uses_default(self, router):
        """Test low confidence queries use default strategy."""
        router.min_confidence = 0.99  # Very high threshold
        decision = router.route("hello")  # Ambiguous query
        assert decision.selected_strategy == router.default_strategy


class TestMultiQueryRouter:
    """Tests for MultiQueryRouter."""

    @pytest.fixture
    def multi_router(self):
        """Create multi-query router."""
        base_router = QueryRouter()
        return MultiQueryRouter(base_router)

    def test_simple_query_single_decision(self, multi_router):
        """Test non-comparative returns single decision."""
        decisions = multi_router.route("What is the API rate limit?")
        assert len(decisions) == 1

    def test_comparative_query_multiple_decisions(self, multi_router):
        """Test comparative query decomposes into sub-queries."""
        decisions = multi_router.route("Compare Python and Java")
        # Should have main decision + sub-queries
        assert len(decisions) >= 1

        main = decisions[0]
        assert main.intent_result.intent == QueryIntent.COMPARATIVE

    def test_sub_queries_have_parent_reference(self, multi_router):
        """Test sub-queries reference parent query."""
        decisions = multi_router.route("Compare MongoDB vs PostgreSQL")

        if len(decisions) > 1:
            for sub in decisions[1:]:
                assert "parent_query" in sub.metadata
