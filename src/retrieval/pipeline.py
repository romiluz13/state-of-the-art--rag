"""Retrieval pipeline orchestrating all retrieval strategies.

December 2025: 9 strategies for true SOTA status.
"""

import logging
from enum import Enum
from typing import Any

from .base import RetrievalResult, RetrievalConfig
from .vector import VectorSearcher
from .text import TextSearcher
from .hybrid import HybridSearcher
from .graphrag import GraphRAGRetriever
from .leanrag import LeanRAGRetriever  # Dec 2025: Replaces RAPTOR
from .mcts import MCTSRetriever  # Dec 2025: Multi-hop reasoning
from .colpali import ColPaliRetriever
from .reranker import Reranker
from ..routing import QueryRouter, RoutingDecision, IntentClassifier

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies.

    December 2025: Added LEANRAG and MCTS strategies.
    """

    VECTOR = "vector"  # Pure vector search
    TEXT = "text"  # Pure BM25 text search
    HYBRID = "hybrid"  # $rankFusion vector + BM25
    SCORE_FUSION = "score_fusion"  # $scoreFusion weighted combination
    GRAPHRAG = "graphrag"  # $graphLookup + hybrid (Dec 2025: RRF)
    LEANRAG = "leanrag"  # December 2025: Bottom-up hierarchical
    RAPTOR = "raptor"  # Backward compatibility alias for LEANRAG
    MCTS = "mcts"  # December 2025: Multi-hop reasoning
    COLPALI = "colpali"  # Multimodal visual document search (ColQwen2)
    AUTO = "auto"  # Automatic strategy selection


class RetrievalPipeline:
    """Orchestrates retrieval across all strategies.

    Main entry point for the retrieval system. Handles:
    - Strategy selection (manual or automatic)
    - Query embedding
    - Retrieval execution
    - Optional reranking
    """

    def __init__(
        self,
        mongodb_client: Any,
        voyage_client: Any,
        config: RetrievalConfig | None = None,
        colpali_client: Any | None = None,
        query_router: QueryRouter | None = None,
    ):
        """Initialize retrieval pipeline.

        Args:
            mongodb_client: MongoDBClient instance
            voyage_client: VoyageClient instance for embeddings/reranking
            config: Retrieval configuration
            colpali_client: Optional ColPaliClient for visual search
            query_router: Optional QueryRouter for intent-based routing
        """
        self.mongodb = mongodb_client
        self.voyage = voyage_client
        self.config = config or RetrievalConfig()

        # Initialize retrievers
        self.vector_searcher = VectorSearcher(mongodb_client, self.config)
        self.text_searcher = TextSearcher(mongodb_client, self.config)
        self.hybrid_searcher = HybridSearcher(mongodb_client, self.config)
        self.graphrag_retriever = GraphRAGRetriever(mongodb_client, self.config)
        # December 2025: LeanRAG replaces RAPTOR
        self.leanrag_retriever = LeanRAGRetriever(mongodb_client, self.config)
        self.raptor_retriever = self.leanrag_retriever  # Backward compatibility
        # December 2025: MCTS for multi-hop reasoning
        self.mcts_retriever = MCTSRetriever(mongodb_client, self.config)
        self.colpali_retriever = ColPaliRetriever(
            mongodb_client, colpali_client, use_mock=(colpali_client is None)
        )
        self.reranker = Reranker(voyage_client)

        # Query router for intent-based strategy selection
        self.router = query_router or QueryRouter()
        self._last_routing_decision: RoutingDecision | None = None

    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        top_k: int | None = None,
        rerank: bool | None = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        """Main retrieval method.

        Args:
            query: The search query
            strategy: Retrieval strategy to use
            top_k: Number of results to return
            rerank: Whether to rerank results (overrides config)
            **kwargs: Strategy-specific arguments

        Returns:
            List of RetrievalResult sorted by relevance
        """
        top_k = top_k or self.config.top_k
        rerank = rerank if rerank is not None else self.config.rerank

        # Embed query
        query_embedding = await self._embed_query(query)

        # Auto-select strategy if needed
        if strategy == RetrievalStrategy.AUTO:
            strategy = self._auto_select_strategy(query)
            logger.info(f"Auto-selected strategy: {strategy}")

        # Execute retrieval
        results = await self._execute_strategy(
            query, query_embedding, strategy, top_k, **kwargs
        )

        # Rerank if enabled
        if rerank and results:
            rerank_top_k = min(self.config.rerank_top_k, len(results))
            results = await self.reranker.rerank(query, results[:rerank_top_k], top_k)

        return results

    async def _embed_query(self, query: str) -> list[float]:
        """Embed query using Voyage AI."""
        response = await self.voyage.embed(
            texts=[query],
            model="voyage-3.5",
            input_type="query",
        )
        return response["data"][0]["embedding"]

    async def _execute_strategy(
        self,
        query: str,
        query_embedding: list[float],
        strategy: RetrievalStrategy,
        top_k: int,
        **kwargs,
    ) -> list[RetrievalResult]:
        """Execute the selected retrieval strategy."""
        if strategy == RetrievalStrategy.VECTOR:
            return await self.vector_searcher.retrieve(
                query, query_embedding, top_k, **kwargs
            )

        elif strategy == RetrievalStrategy.TEXT:
            return await self.text_searcher.retrieve(
                query, query_embedding, top_k, **kwargs
            )

        elif strategy == RetrievalStrategy.HYBRID:
            return await self.hybrid_searcher.retrieve(
                query, query_embedding, top_k, fusion_type="rank", **kwargs
            )

        elif strategy == RetrievalStrategy.SCORE_FUSION:
            return await self.hybrid_searcher.retrieve(
                query, query_embedding, top_k, fusion_type="score", **kwargs
            )

        elif strategy == RetrievalStrategy.GRAPHRAG:
            return await self.graphrag_retriever.retrieve(
                query, query_embedding, top_k, **kwargs
            )

        elif strategy == RetrievalStrategy.LEANRAG:
            # December 2025: LeanRAG bottom-up hierarchical
            return await self.leanrag_retriever.retrieve(
                query, query_embedding, top_k, **kwargs
            )

        elif strategy == RetrievalStrategy.RAPTOR:
            # Backward compatibility - uses LeanRAG
            return await self.leanrag_retriever.retrieve(
                query, query_embedding, top_k, **kwargs
            )

        elif strategy == RetrievalStrategy.MCTS:
            # December 2025: MCTS for multi-hop reasoning
            return await self.mcts_retriever.retrieve(
                query, query_embedding, top_k, **kwargs
            )

        elif strategy == RetrievalStrategy.COLPALI:
            return await self.colpali_retriever.retrieve(
                query, top_k, **kwargs
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _auto_select_strategy(self, query: str) -> RetrievalStrategy:
        """Automatically select best strategy based on query intent.

        Uses QueryRouter for intent classification and strategy mapping.
        Falls back to hybrid on low confidence.
        """
        # Use router for intent-based selection
        decision = self.router.route(query)
        self._last_routing_decision = decision

        # Map strategy string to enum (December 2025 updated)
        strategy_map = {
            "hybrid": RetrievalStrategy.HYBRID,
            "graphrag": RetrievalStrategy.GRAPHRAG,
            "leanrag": RetrievalStrategy.LEANRAG,  # December 2025 SOTA
            "raptor": RetrievalStrategy.RAPTOR,  # Backward compatibility
            "mcts": RetrievalStrategy.MCTS,  # December 2025: Multi-hop
            "colpali": RetrievalStrategy.COLPALI,
            "vector": RetrievalStrategy.VECTOR,
            "text": RetrievalStrategy.TEXT,
            "score_fusion": RetrievalStrategy.SCORE_FUSION,
        }

        selected = strategy_map.get(decision.selected_strategy, RetrievalStrategy.HYBRID)

        logger.info(
            f"Router selected {selected.value} for query "
            f"(intent: {decision.intent_result.intent.value if decision.intent_result else 'N/A'}, "
            f"confidence: {decision.intent_result.confidence if decision.intent_result else 'N/A'})"
        )

        return selected

    def get_last_routing_decision(self) -> RoutingDecision | None:
        """Get the last routing decision for diagnostics."""
        return self._last_routing_decision

    def route_query(self, query: str) -> RoutingDecision:
        """Route a query without executing retrieval.

        Useful for API endpoint that just returns routing decision.
        """
        return self.router.route(query)

    async def multi_strategy_retrieve(
        self,
        query: str,
        strategies: list[RetrievalStrategy],
        top_k: int | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Execute multiple strategies and return results from each.

        Useful for comparison and ensemble approaches.

        Args:
            query: The search query
            strategies: List of strategies to execute
            top_k: Number of results per strategy

        Returns:
            Dict mapping strategy name to results
        """
        top_k = top_k or self.config.top_k
        query_embedding = await self._embed_query(query)

        results = {}
        for strategy in strategies:
            strategy_results = await self._execute_strategy(
                query, query_embedding, strategy, top_k
            )
            results[strategy.value] = strategy_results

        return results

    async def ensemble_retrieve(
        self,
        query: str,
        top_k: int | None = None,
        weights: dict[RetrievalStrategy, float] | None = None,
    ) -> list[RetrievalResult]:
        """Ensemble retrieval combining multiple strategies.

        Args:
            query: The search query
            top_k: Number of final results
            weights: Strategy weights (default: equal)

        Returns:
            Combined and deduplicated results
        """
        top_k = top_k or self.config.top_k
        strategies = [
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.GRAPHRAG,
            RetrievalStrategy.RAPTOR,
        ]

        if weights is None:
            weights = {s: 1.0 / len(strategies) for s in strategies}

        # Get results from each strategy
        multi_results = await self.multi_strategy_retrieve(query, strategies, top_k * 2)

        # Combine with weights
        combined = []
        seen_chunks = {}

        for strategy in strategies:
            weight = weights.get(strategy, 0.33)
            for result in multi_results.get(strategy.value, []):
                if result.chunk_id in seen_chunks:
                    # Accumulate scores for duplicates
                    seen_chunks[result.chunk_id].score += result.score * weight
                else:
                    result.score *= weight
                    seen_chunks[result.chunk_id] = result
                    combined.append(result)

        # Sort by combined score
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:top_k]
