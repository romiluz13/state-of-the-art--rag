"""Query router for intelligent strategy selection."""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .intent import IntentClassifier, IntentResult, QueryIntent, INTENT_TO_STRATEGY

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    enabled: bool = False
    test_name: str = ""
    strategies: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)  # Must sum to 1.0

    def select_strategy(self) -> str:
        """Randomly select strategy based on weights."""
        if not self.enabled or not self.strategies:
            return ""

        weights = self.weights if self.weights else [1.0 / len(self.strategies)] * len(self.strategies)
        return random.choices(self.strategies, weights=weights)[0]


@dataclass
class RoutingDecision:
    """Result of query routing decision."""

    query: str
    selected_strategy: str
    intent_result: IntentResult | None
    was_overridden: bool = False
    ab_test_variant: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryRouter:
    """Routes queries to optimal retrieval strategies.

    Features:
    - Intent-based routing using IntentClassifier
    - Manual strategy override support
    - A/B testing for strategy comparison
    - Confidence-based fallbacks
    """

    def __init__(
        self,
        intent_classifier: IntentClassifier | None = None,
        default_strategy: str = "hybrid",
        min_confidence: float = 0.5,
        ab_test_config: ABTestConfig | None = None,
    ):
        """Initialize query router.

        Args:
            intent_classifier: Classifier for query intent
            default_strategy: Fallback strategy when confidence is low
            min_confidence: Minimum confidence to use intent-based routing
            ab_test_config: A/B testing configuration
        """
        self.classifier = intent_classifier or IntentClassifier()
        self.default_strategy = default_strategy
        self.min_confidence = min_confidence
        self.ab_test = ab_test_config or ABTestConfig()

    def route(
        self,
        query: str,
        override_strategy: str | None = None,
    ) -> RoutingDecision:
        """Route query to optimal strategy.

        Args:
            query: The user's query
            override_strategy: Manual strategy override

        Returns:
            RoutingDecision with selected strategy and reasoning
        """
        # Check for manual override first
        if override_strategy and override_strategy != "auto":
            logger.info(f"Using manual override: {override_strategy}")
            return RoutingDecision(
                query=query,
                selected_strategy=override_strategy,
                intent_result=None,
                was_overridden=True,
            )

        # Check A/B test
        if self.ab_test.enabled:
            ab_strategy = self.ab_test.select_strategy()
            if ab_strategy:
                logger.info(f"A/B test selected: {ab_strategy} (test: {self.ab_test.test_name})")
                # Still classify intent for logging/metrics
                intent_result = self.classifier.classify(query)
                return RoutingDecision(
                    query=query,
                    selected_strategy=ab_strategy,
                    intent_result=intent_result,
                    ab_test_variant=ab_strategy,
                    metadata={"ab_test": self.ab_test.test_name},
                )

        # Classify intent
        intent_result = self.classifier.classify(query)

        # Check confidence threshold
        if intent_result.confidence < self.min_confidence:
            logger.info(
                f"Low confidence ({intent_result.confidence:.2f}), "
                f"using default: {self.default_strategy}"
            )
            return RoutingDecision(
                query=query,
                selected_strategy=self.default_strategy,
                intent_result=intent_result,
                metadata={"reason": "low_confidence"},
            )

        # Use intent-based strategy
        logger.info(
            f"Routing to {intent_result.strategy} based on "
            f"{intent_result.intent.value} intent (confidence: {intent_result.confidence:.2f})"
        )

        return RoutingDecision(
            query=query,
            selected_strategy=intent_result.strategy,
            intent_result=intent_result,
        )

    def route_batch(
        self,
        queries: list[str],
        override_strategy: str | None = None,
    ) -> list[RoutingDecision]:
        """Route multiple queries."""
        return [self.route(q, override_strategy) for q in queries]

    def get_strategy_for_intent(self, intent: QueryIntent) -> str:
        """Get recommended strategy for a given intent."""
        return INTENT_TO_STRATEGY.get(intent, self.default_strategy)

    def update_ab_test(self, config: ABTestConfig) -> None:
        """Update A/B test configuration."""
        self.ab_test = config
        if config.enabled:
            logger.info(f"A/B test enabled: {config.test_name}")
        else:
            logger.info("A/B testing disabled")


class MultiQueryRouter:
    """Router that decomposes comparative queries into sub-queries."""

    def __init__(self, base_router: QueryRouter):
        """Initialize multi-query router.

        Args:
            base_router: Base query router for individual queries
        """
        self.router = base_router

    def route(self, query: str) -> list[RoutingDecision]:
        """Route query, potentially decomposing into sub-queries.

        For comparative queries, returns decisions for each sub-query.
        For other queries, returns single-item list.
        """
        main_decision = self.router.route(query)

        # Check if comparative with sub-queries
        if (
            main_decision.intent_result
            and main_decision.intent_result.intent == QueryIntent.COMPARATIVE
            and main_decision.intent_result.sub_queries
        ):
            # Route each sub-query
            sub_decisions = []
            for sub_query in main_decision.intent_result.sub_queries:
                sub_decision = self.router.route(sub_query, override_strategy="hybrid")
                sub_decision.metadata["parent_query"] = query
                sub_decisions.append(sub_decision)

            # Include main decision with sub-decisions
            main_decision.metadata["sub_queries"] = [
                d.query for d in sub_decisions
            ]
            return [main_decision] + sub_decisions

        return [main_decision]
