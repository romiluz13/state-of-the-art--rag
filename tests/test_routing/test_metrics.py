"""Tests for strategy performance metrics."""

import pytest
from datetime import datetime

from src.routing.metrics import (
    QueryMetric,
    StrategyStats,
    StrategyMetrics,
)


class TestQueryMetric:
    """Tests for QueryMetric dataclass."""

    def test_basic_creation(self):
        """Test basic metric creation."""
        metric = QueryMetric(
            query_id="q-123",
            query="test query",
            strategy="hybrid",
            intent="factual",
            latency_ms=150.5,
            result_count=10,
        )
        assert metric.query_id == "q-123"
        assert metric.strategy == "hybrid"
        assert metric.latency_ms == 150.5
        assert metric.reranked is False
        assert metric.relevance_score is None

    def test_with_optional_fields(self):
        """Test metric with all optional fields."""
        metric = QueryMetric(
            query_id="q-456",
            query="test",
            strategy="graphrag",
            intent="global",
            latency_ms=200.0,
            result_count=5,
            reranked=True,
            relevance_score=0.85,
            user_feedback=1,
            ab_test_variant="graphrag",
        )
        assert metric.reranked is True
        assert metric.relevance_score == 0.85
        assert metric.user_feedback == 1
        assert metric.ab_test_variant == "graphrag"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metric = QueryMetric(
            query_id="q-789",
            query="test query",
            strategy="raptor",
            intent="hierarchical",
            latency_ms=100.0,
            result_count=8,
        )
        d = metric.to_dict()
        assert d["query_id"] == "q-789"
        assert d["strategy"] == "raptor"
        assert d["latency_ms"] == 100.0
        assert "timestamp" in d


class TestStrategyStats:
    """Tests for StrategyStats dataclass."""

    def test_default_stats(self):
        """Test default statistics."""
        stats = StrategyStats(strategy="hybrid")
        assert stats.query_count == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.avg_results == 0.0
        assert stats.avg_relevance == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        stats = StrategyStats(
            strategy="hybrid",
            query_count=10,
            total_latency_ms=1500.0,
        )
        assert stats.avg_latency_ms == 150.0

    def test_avg_results_calculation(self):
        """Test average results calculation."""
        stats = StrategyStats(
            strategy="hybrid",
            query_count=5,
            total_results=50,
        )
        assert stats.avg_results == 10.0

    def test_avg_relevance_calculation(self):
        """Test average relevance calculation."""
        stats = StrategyStats(
            strategy="hybrid",
            relevance_count=4,
            total_relevance=3.2,
        )
        assert stats.avg_relevance == 0.8

    def test_feedback_score_positive(self):
        """Test feedback score with positive bias."""
        stats = StrategyStats(
            strategy="hybrid",
            positive_feedback=8,
            negative_feedback=2,
            neutral_feedback=0,
        )
        # (8 - 2) / 10 = 0.6
        assert stats.feedback_score == 0.6

    def test_feedback_score_negative(self):
        """Test feedback score with negative bias."""
        stats = StrategyStats(
            strategy="hybrid",
            positive_feedback=1,
            negative_feedback=9,
            neutral_feedback=0,
        )
        # (1 - 9) / 10 = -0.8
        assert stats.feedback_score == -0.8

    def test_feedback_score_no_feedback(self):
        """Test feedback score with no feedback."""
        stats = StrategyStats(strategy="hybrid")
        assert stats.feedback_score == 0.0


class TestStrategyMetrics:
    """Tests for StrategyMetrics tracker."""

    @pytest.fixture
    def metrics(self):
        """Create metrics tracker."""
        return StrategyMetrics()

    def test_initialization(self, metrics):
        """Test default initialization."""
        assert metrics.mongodb is None
        assert metrics.collection_name == "queries"

    def test_record_metric(self, metrics):
        """Test recording a single metric."""
        metric = QueryMetric(
            query_id="q-1",
            query="test",
            strategy="hybrid",
            intent="factual",
            latency_ms=100.0,
            result_count=10,
        )
        metrics.record(metric)

        stats = metrics.get_stats("hybrid")
        assert "hybrid" in stats
        assert stats["hybrid"].query_count == 1
        assert stats["hybrid"].total_latency_ms == 100.0

    def test_record_multiple_metrics(self, metrics):
        """Test recording multiple metrics."""
        for i in range(5):
            metric = QueryMetric(
                query_id=f"q-{i}",
                query=f"test {i}",
                strategy="hybrid",
                intent="factual",
                latency_ms=100.0 + i * 10,
                result_count=10,
            )
            metrics.record(metric)

        stats = metrics.get_stats("hybrid")
        assert stats["hybrid"].query_count == 5
        # 100 + 110 + 120 + 130 + 140 = 600
        assert stats["hybrid"].total_latency_ms == 600.0

    def test_record_different_strategies(self, metrics):
        """Test recording metrics for different strategies."""
        strategies = ["hybrid", "graphrag", "raptor"]
        for strategy in strategies:
            metric = QueryMetric(
                query_id=f"q-{strategy}",
                query="test",
                strategy=strategy,
                intent="factual",
                latency_ms=100.0,
                result_count=10,
            )
            metrics.record(metric)

        all_stats = metrics.get_stats()
        assert len(all_stats) == 3
        assert "hybrid" in all_stats
        assert "graphrag" in all_stats
        assert "raptor" in all_stats

    def test_record_with_relevance(self, metrics):
        """Test recording metrics with relevance scores."""
        for i in range(3):
            metric = QueryMetric(
                query_id=f"q-{i}",
                query="test",
                strategy="hybrid",
                intent="factual",
                latency_ms=100.0,
                result_count=10,
                relevance_score=0.8 + i * 0.05,
            )
            metrics.record(metric)

        stats = metrics.get_stats("hybrid")
        # 0.8 + 0.85 + 0.9 = 2.55
        assert stats["hybrid"].total_relevance == pytest.approx(2.55, rel=1e-3)
        assert stats["hybrid"].relevance_count == 3

    def test_record_with_feedback(self, metrics):
        """Test recording metrics with user feedback."""
        # Positive feedback
        metrics.record(QueryMetric(
            query_id="q-1", query="test", strategy="hybrid",
            intent="factual", latency_ms=100, result_count=10,
            user_feedback=1,
        ))
        # Negative feedback
        metrics.record(QueryMetric(
            query_id="q-2", query="test", strategy="hybrid",
            intent="factual", latency_ms=100, result_count=10,
            user_feedback=-1,
        ))
        # Neutral feedback
        metrics.record(QueryMetric(
            query_id="q-3", query="test", strategy="hybrid",
            intent="factual", latency_ms=100, result_count=10,
            user_feedback=0,
        ))

        stats = metrics.get_stats("hybrid")
        assert stats["hybrid"].positive_feedback == 1
        assert stats["hybrid"].negative_feedback == 1
        assert stats["hybrid"].neutral_feedback == 1

    def test_get_comparison(self, metrics):
        """Test getting strategy comparison."""
        # Record different strategies with different performance
        metrics.record(QueryMetric(
            query_id="q-1", query="test", strategy="hybrid",
            intent="factual", latency_ms=100, result_count=10,
        ))
        metrics.record(QueryMetric(
            query_id="q-2", query="test", strategy="hybrid",
            intent="factual", latency_ms=120, result_count=8,
        ))
        metrics.record(QueryMetric(
            query_id="q-3", query="test", strategy="graphrag",
            intent="global", latency_ms=200, result_count=5,
        ))

        comparison = metrics.get_comparison()
        assert len(comparison) == 2

        # Should be sorted by query count
        assert comparison[0]["strategy"] == "hybrid"
        assert comparison[0]["query_count"] == 2
        assert comparison[0]["avg_latency_ms"] == 110.0  # (100 + 120) / 2

    def test_get_ab_test_results_empty(self, metrics):
        """Test A/B test results with no data."""
        results = metrics.get_ab_test_results("test_experiment")
        assert results["test_name"] == "test_experiment"
        assert results["sample_size"] == 0
        assert results["variants"] == {}

    def test_get_ab_test_results(self, metrics):
        """Test A/B test results with data."""
        # Record metrics for A/B test
        for i in range(10):
            variant = "hybrid" if i % 2 == 0 else "graphrag"
            latency = 100.0 if variant == "hybrid" else 150.0
            metric = QueryMetric(
                query_id=f"q-{i}",
                query="test",
                strategy=variant,
                intent="factual",
                latency_ms=latency,
                result_count=10,
                ab_test_variant=variant,
                metadata={"ab_test": "test_experiment"},
            )
            metrics.record(metric)

        results = metrics.get_ab_test_results("test_experiment")
        assert results["sample_size"] == 10
        assert "hybrid" in results["variants"]
        assert "graphrag" in results["variants"]
        assert results["variants"]["hybrid"]["count"] == 5
        assert results["variants"]["graphrag"]["count"] == 5

    def test_record_feedback(self, metrics):
        """Test recording feedback for existing query."""
        metric = QueryMetric(
            query_id="q-feedback-test",
            query="test",
            strategy="hybrid",
            intent="factual",
            latency_ms=100,
            result_count=10,
        )
        metrics.record(metric)

        # Record positive feedback
        result = metrics.record_feedback("q-feedback-test", 1)
        assert result is True

        stats = metrics.get_stats("hybrid")
        assert stats["hybrid"].positive_feedback == 1

    def test_record_feedback_not_found(self, metrics):
        """Test recording feedback for non-existent query."""
        result = metrics.record_feedback("non-existent", 1)
        assert result is False

    def test_clear_stats(self, metrics):
        """Test clearing statistics."""
        metric = QueryMetric(
            query_id="q-1",
            query="test",
            strategy="hybrid",
            intent="factual",
            latency_ms=100,
            result_count=10,
        )
        metrics.record(metric)

        # Verify data exists
        assert len(metrics.get_stats()) > 0

        # Clear stats
        metrics.clear_stats()

        # Verify data is cleared
        assert len(metrics.get_stats()) == 0

    def test_recent_metrics_limit(self, metrics):
        """Test that recent metrics are limited."""
        metrics._max_recent = 10  # Set low limit for testing

        # Record more than limit
        for i in range(15):
            metric = QueryMetric(
                query_id=f"q-{i}",
                query="test",
                strategy="hybrid",
                intent="factual",
                latency_ms=100,
                result_count=10,
            )
            metrics.record(metric)

        # Should only keep last 10
        assert len(metrics._recent_metrics) == 10
        # First should be q-5 (0-4 were dropped)
        assert metrics._recent_metrics[0].query_id == "q-5"
