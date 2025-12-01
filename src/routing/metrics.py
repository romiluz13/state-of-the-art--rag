"""Strategy performance metrics and tracking."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueryMetric:
    """Metrics for a single query execution."""

    query_id: str
    query: str
    strategy: str
    intent: str | None
    latency_ms: float
    result_count: int
    reranked: bool = False
    relevance_score: float | None = None  # From evaluation
    user_feedback: int | None = None  # -1, 0, 1
    ab_test_variant: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "strategy": self.strategy,
            "intent": self.intent,
            "latency_ms": self.latency_ms,
            "result_count": self.result_count,
            "reranked": self.reranked,
            "relevance_score": self.relevance_score,
            "user_feedback": self.user_feedback,
            "ab_test_variant": self.ab_test_variant,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class StrategyStats:
    """Aggregated statistics for a strategy."""

    strategy: str
    query_count: int = 0
    total_latency_ms: float = 0.0
    total_results: int = 0
    total_relevance: float = 0.0
    relevance_count: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    neutral_feedback: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        return self.total_latency_ms / self.query_count if self.query_count > 0 else 0.0

    @property
    def avg_results(self) -> float:
        """Average result count per query."""
        return self.total_results / self.query_count if self.query_count > 0 else 0.0

    @property
    def avg_relevance(self) -> float:
        """Average relevance score."""
        return self.total_relevance / self.relevance_count if self.relevance_count > 0 else 0.0

    @property
    def feedback_score(self) -> float:
        """Net feedback score (-1 to 1)."""
        total = self.positive_feedback + self.negative_feedback + self.neutral_feedback
        if total == 0:
            return 0.0
        return (self.positive_feedback - self.negative_feedback) / total


class StrategyMetrics:
    """Track and analyze strategy performance metrics.

    Supports:
    - Real-time metric collection
    - Strategy comparison
    - A/B test analysis
    - MongoDB persistence
    """

    def __init__(
        self,
        mongodb_client: Any | None = None,
        collection_name: str = "queries",
    ):
        """Initialize metrics tracker.

        Args:
            mongodb_client: MongoDB client for persistence
            collection_name: Collection to store metrics
        """
        self.mongodb = mongodb_client
        self.collection_name = collection_name

        # In-memory stats (for real-time aggregation)
        self._stats: dict[str, StrategyStats] = defaultdict(
            lambda: StrategyStats(strategy="")
        )
        self._recent_metrics: list[QueryMetric] = []
        self._max_recent = 1000  # Keep last N metrics in memory

    def record(self, metric: QueryMetric) -> None:
        """Record a query metric.

        Args:
            metric: QueryMetric to record
        """
        # Update in-memory stats
        stats = self._stats[metric.strategy]
        if not stats.strategy:
            stats.strategy = metric.strategy

        stats.query_count += 1
        stats.total_latency_ms += metric.latency_ms
        stats.total_results += metric.result_count

        if metric.relevance_score is not None:
            stats.total_relevance += metric.relevance_score
            stats.relevance_count += 1

        if metric.user_feedback is not None:
            if metric.user_feedback > 0:
                stats.positive_feedback += 1
            elif metric.user_feedback < 0:
                stats.negative_feedback += 1
            else:
                stats.neutral_feedback += 1

        # Add to recent metrics
        self._recent_metrics.append(metric)
        if len(self._recent_metrics) > self._max_recent:
            self._recent_metrics = self._recent_metrics[-self._max_recent:]

        # Persist to MongoDB if available
        if self.mongodb:
            self._persist_metric(metric)

        logger.debug(
            f"Recorded metric for {metric.strategy}: "
            f"latency={metric.latency_ms:.2f}ms, results={metric.result_count}"
        )

    async def record_async(self, metric: QueryMetric) -> None:
        """Record metric asynchronously."""
        self.record(metric)

        if self.mongodb:
            await self._persist_metric_async(metric)

    def _persist_metric(self, metric: QueryMetric) -> None:
        """Persist metric to MongoDB (sync)."""
        try:
            collection = self.mongodb.db[self.collection_name]
            collection.insert_one(metric.to_dict())
        except Exception as e:
            logger.warning(f"Failed to persist metric: {e}")

    async def _persist_metric_async(self, metric: QueryMetric) -> None:
        """Persist metric to MongoDB (async)."""
        try:
            collection = self.mongodb.db[self.collection_name]
            await collection.insert_one(metric.to_dict())
        except Exception as e:
            logger.warning(f"Failed to persist metric: {e}")

    def get_stats(self, strategy: str | None = None) -> dict[str, StrategyStats]:
        """Get strategy statistics.

        Args:
            strategy: Specific strategy or None for all

        Returns:
            Dict of strategy name to StrategyStats
        """
        if strategy:
            return {strategy: self._stats.get(strategy, StrategyStats(strategy=strategy))}
        return dict(self._stats)

    def get_comparison(self) -> list[dict[str, Any]]:
        """Get strategy comparison data."""
        comparison = []
        for strategy, stats in self._stats.items():
            comparison.append({
                "strategy": strategy,
                "query_count": stats.query_count,
                "avg_latency_ms": round(stats.avg_latency_ms, 2),
                "avg_results": round(stats.avg_results, 2),
                "avg_relevance": round(stats.avg_relevance, 3),
                "feedback_score": round(stats.feedback_score, 3),
            })

        # Sort by query count descending
        comparison.sort(key=lambda x: x["query_count"], reverse=True)
        return comparison

    def get_ab_test_results(self, test_name: str) -> dict[str, Any]:
        """Get A/B test results.

        Args:
            test_name: Name of the A/B test

        Returns:
            Dict with test results per variant
        """
        # Filter metrics for this test
        test_metrics = [
            m for m in self._recent_metrics
            if m.metadata.get("ab_test") == test_name
        ]

        if not test_metrics:
            return {"test_name": test_name, "variants": {}, "sample_size": 0}

        # Group by variant
        variants: dict[str, list[QueryMetric]] = defaultdict(list)
        for m in test_metrics:
            if m.ab_test_variant:
                variants[m.ab_test_variant].append(m)

        # Calculate stats per variant
        results = {"test_name": test_name, "sample_size": len(test_metrics), "variants": {}}

        for variant, metrics in variants.items():
            latencies = [m.latency_ms for m in metrics]
            relevances = [m.relevance_score for m in metrics if m.relevance_score]

            results["variants"][variant] = {
                "count": len(metrics),
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "avg_relevance": sum(relevances) / len(relevances) if relevances else None,
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            }

        return results

    def record_feedback(self, query_id: str, feedback: int) -> bool:
        """Record user feedback for a query.

        Args:
            query_id: Query ID to update
            feedback: -1 (bad), 0 (neutral), 1 (good)

        Returns:
            True if found and updated
        """
        for metric in reversed(self._recent_metrics):
            if metric.query_id == query_id:
                metric.user_feedback = feedback

                # Update stats
                stats = self._stats.get(metric.strategy)
                if stats:
                    if feedback > 0:
                        stats.positive_feedback += 1
                    elif feedback < 0:
                        stats.negative_feedback += 1
                    else:
                        stats.neutral_feedback += 1

                return True

        return False

    def clear_stats(self) -> None:
        """Clear in-memory statistics."""
        self._stats.clear()
        self._recent_metrics.clear()
        logger.info("Cleared strategy metrics")
