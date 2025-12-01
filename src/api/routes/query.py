"""Query endpoint for retrieval."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.retrieval import (
    RetrievalPipeline,
    RetrievalStrategy,
    RetrievalConfig,
    RetrievalResult,
)
from src.routing import QueryRouter, RoutingDecision, StrategyMetrics, QueryMetric
from src.clients.mongodb import MongoDBClient
from src.clients.voyage import VoyageClient
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

# Global metrics tracker
_strategy_metrics: StrategyMetrics | None = None


class QueryRequest(BaseModel):
    """Query request."""

    query: str = Field(..., min_length=1, description="The search query")
    strategy: str | None = Field(
        default="hybrid",
        description="Retrieval strategy: vector, text, hybrid, score_fusion, graphrag, raptor, colpali, auto",
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    rerank: bool = Field(default=True, description="Whether to rerank results")
    filters: dict[str, Any] | None = Field(
        default=None, description="Metadata filters for retrieval"
    )
    document_ids: list[str] | None = Field(
        default=None, description="Filter to specific documents (for colpali)"
    )


class SourceInfo(BaseModel):
    """Source information for a retrieved chunk."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    # ColPali-specific fields (optional)
    page_num: int | None = Field(default=None, description="Page number for visual results")
    image_size: tuple[int, int] | None = Field(default=None, description="Image dimensions (w, h)")
    has_images: bool | None = Field(default=None, description="Whether page contains images")


class QueryResponse(BaseModel):
    """Query response."""

    query: str
    strategy: str
    results: list[SourceInfo]
    total_results: int
    reranked: bool


# Global clients (initialized in main.py lifespan)
_mongodb_client: MongoDBClient | None = None
_voyage_client: VoyageClient | None = None
_retrieval_pipeline: RetrievalPipeline | None = None


def get_retrieval_pipeline() -> RetrievalPipeline:
    """Get or create retrieval pipeline."""
    global _retrieval_pipeline, _mongodb_client, _voyage_client

    if _retrieval_pipeline is None:
        settings = get_settings()

        if _mongodb_client is None:
            _mongodb_client = MongoDBClient(settings)

        if _voyage_client is None:
            _voyage_client = VoyageClient(settings.voyage_api_key)

        config = RetrievalConfig(
            top_k=10,
            use_binary_quantization=True,
            rerank=True,
        )
        _retrieval_pipeline = RetrievalPipeline(
            _mongodb_client, _voyage_client, config
        )

    return _retrieval_pipeline


def parse_strategy(strategy_str: str) -> RetrievalStrategy:
    """Parse strategy string to enum."""
    strategy_map = {
        "vector": RetrievalStrategy.VECTOR,
        "text": RetrievalStrategy.TEXT,
        "hybrid": RetrievalStrategy.HYBRID,
        "score_fusion": RetrievalStrategy.SCORE_FUSION,
        "graphrag": RetrievalStrategy.GRAPHRAG,
        "raptor": RetrievalStrategy.RAPTOR,
        "colpali": RetrievalStrategy.COLPALI,
        "auto": RetrievalStrategy.AUTO,
    }
    return strategy_map.get(strategy_str.lower(), RetrievalStrategy.HYBRID)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Execute a retrieval query.

    Retrieves relevant documents using the specified strategy:
    - **vector**: Pure vector similarity search
    - **text**: BM25 keyword search
    - **hybrid**: Combined vector + BM25 via $rankFusion (default)
    - **score_fusion**: Weighted score combination via $scoreFusion
    - **graphrag**: Entity-based retrieval via $graphLookup
    - **raptor**: Hierarchical multi-level retrieval
    - **colpali**: Visual document search (charts, diagrams, images)
    - **auto**: Automatic strategy selection based on query

    Results can be reranked using Voyage rerank-2.5 for improved relevance.
    """
    try:
        pipeline = get_retrieval_pipeline()
        strategy = parse_strategy(request.strategy or "hybrid")

        # Build kwargs for strategy-specific options
        kwargs = {}
        if request.document_ids:
            kwargs["document_ids"] = request.document_ids

        results = await pipeline.retrieve(
            query=request.query,
            strategy=strategy,
            top_k=request.top_k,
            rerank=request.rerank,
            **kwargs,
        )

        # Convert to response format (handle ColPali page results)
        sources = []
        for r in results:
            source = SourceInfo(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            # Add ColPali-specific fields if available
            if hasattr(r, "page_num") and r.page_num is not None:
                source.page_num = r.page_num
            if hasattr(r, "image_size") and r.image_size != (0, 0):
                source.image_size = r.image_size
            if hasattr(r, "has_images"):
                source.has_images = r.has_images
            sources.append(source)

        return QueryResponse(
            query=request.query,
            strategy=strategy.value,
            results=sources,
            total_results=len(sources),
            reranked=request.rerank,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/multi-strategy")
async def multi_strategy_query(
    request: QueryRequest,
    strategies: list[str] = ["hybrid", "graphrag", "raptor"],
) -> dict[str, QueryResponse]:
    """Execute query with multiple strategies for comparison.

    Returns results from each strategy for analysis.
    """
    try:
        pipeline = get_retrieval_pipeline()
        strategy_enums = [parse_strategy(s) for s in strategies]

        multi_results = await pipeline.multi_strategy_retrieve(
            query=request.query,
            strategies=strategy_enums,
            top_k=request.top_k,
        )

        response = {}
        for strategy_name, results in multi_results.items():
            sources = [
                SourceInfo(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ]
            response[strategy_name] = QueryResponse(
                query=request.query,
                strategy=strategy_name,
                results=sources,
                total_results=len(sources),
                reranked=False,
            )

        return response

    except Exception as e:
        logger.error(f"Multi-strategy query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/strategies")
async def list_strategies() -> dict[str, Any]:
    """List available retrieval strategies with descriptions."""
    return {
        "strategies": [
            {
                "name": "vector",
                "description": "Pure vector similarity search using MongoDB $vectorSearch",
                "best_for": "Semantic similarity queries",
            },
            {
                "name": "text",
                "description": "BM25 keyword search using MongoDB $search",
                "best_for": "Exact keyword matching",
            },
            {
                "name": "hybrid",
                "description": "Combined vector + BM25 using $rankFusion (RRF)",
                "best_for": "Most general queries (default)",
            },
            {
                "name": "score_fusion",
                "description": "Weighted score combination using $scoreFusion",
                "best_for": "Custom weight tuning",
            },
            {
                "name": "graphrag",
                "description": "Entity-based retrieval using $graphLookup",
                "best_for": "Global/thematic questions",
            },
            {
                "name": "raptor",
                "description": "Hierarchical multi-level retrieval",
                "best_for": "Document structure questions",
            },
            {
                "name": "colpali",
                "description": "Visual document search using ColPali late interaction",
                "best_for": "Charts, diagrams, tables, visual content",
            },
            {
                "name": "auto",
                "description": "Automatic strategy selection based on query intent classification",
                "best_for": "Unknown query types, intelligent routing",
            },
        ]
    }


class RouteRequest(BaseModel):
    """Request for query routing."""

    query: str = Field(..., min_length=1, description="The query to route")


class RouteResponse(BaseModel):
    """Response from query routing."""

    query: str
    selected_strategy: str
    intent: str | None = None
    confidence: float | None = None
    reasoning: str | None = None
    sub_queries: list[str] | None = None


@router.post("/query/route", response_model=RouteResponse)
async def route_query(request: RouteRequest) -> RouteResponse:
    """Route a query to the optimal retrieval strategy.

    Uses intent classification to determine the best strategy without
    executing retrieval. Useful for debugging or custom pipelines.

    Returns:
        - selected_strategy: The recommended strategy
        - intent: Classified query intent (factual, global, hierarchical, multimodal, comparative)
        - confidence: Classification confidence (0.0-1.0)
        - reasoning: Explanation for the routing decision
        - sub_queries: For comparative queries, the decomposed sub-queries
    """
    try:
        pipeline = get_retrieval_pipeline()
        decision = pipeline.route_query(request.query)

        return RouteResponse(
            query=request.query,
            selected_strategy=decision.selected_strategy,
            intent=decision.intent_result.intent.value if decision.intent_result else None,
            confidence=decision.intent_result.confidence if decision.intent_result else None,
            reasoning=decision.intent_result.reasoning if decision.intent_result else None,
            sub_queries=decision.intent_result.sub_queries if decision.intent_result else None,
        )

    except Exception as e:
        logger.error(f"Routing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_strategy_metrics() -> StrategyMetrics:
    """Get or create strategy metrics tracker."""
    global _strategy_metrics
    if _strategy_metrics is None:
        _strategy_metrics = StrategyMetrics()
    return _strategy_metrics


@router.get("/query/metrics")
async def get_metrics() -> dict[str, Any]:
    """Get strategy performance metrics.

    Returns aggregated statistics for each retrieval strategy including:
    - query_count: Number of queries using this strategy
    - avg_latency_ms: Average response time
    - avg_results: Average number of results returned
    - avg_relevance: Average relevance score (if available)
    - feedback_score: Net user feedback (-1 to 1)
    """
    metrics = get_strategy_metrics()
    return {
        "strategies": metrics.get_comparison(),
        "summary": {
            "total_queries": sum(
                s["query_count"] for s in metrics.get_comparison()
            ),
            "tracked_strategies": len(metrics.get_stats()),
        },
    }


@router.get("/query/metrics/{strategy}")
async def get_strategy_metrics_detail(strategy: str) -> dict[str, Any]:
    """Get detailed metrics for a specific strategy."""
    metrics = get_strategy_metrics()
    stats = metrics.get_stats(strategy)

    if strategy not in stats:
        raise HTTPException(status_code=404, detail=f"No metrics for strategy: {strategy}")

    s = stats[strategy]
    return {
        "strategy": strategy,
        "query_count": s.query_count,
        "avg_latency_ms": round(s.avg_latency_ms, 2),
        "avg_results": round(s.avg_results, 2),
        "avg_relevance": round(s.avg_relevance, 3) if s.relevance_count > 0 else None,
        "feedback": {
            "positive": s.positive_feedback,
            "negative": s.negative_feedback,
            "neutral": s.neutral_feedback,
            "score": round(s.feedback_score, 3),
        },
    }


class FeedbackRequest(BaseModel):
    """User feedback for a query."""

    query_id: str = Field(..., description="The query ID to provide feedback for")
    feedback: int = Field(..., ge=-1, le=1, description="Feedback: -1 (bad), 0 (neutral), 1 (good)")


@router.post("/query/feedback")
async def record_feedback(request: FeedbackRequest) -> dict[str, Any]:
    """Record user feedback for a query.

    Feedback helps improve routing and strategy selection over time.
    """
    metrics = get_strategy_metrics()
    found = metrics.record_feedback(request.query_id, request.feedback)

    if not found:
        raise HTTPException(status_code=404, detail=f"Query not found: {request.query_id}")

    return {"status": "recorded", "query_id": request.query_id, "feedback": request.feedback}
