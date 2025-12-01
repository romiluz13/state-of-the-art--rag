"""Base classes for retrieval system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional fields for different retrieval types
    vector_score: float | None = None
    text_score: float | None = None
    rerank_score: float | None = None
    level: int | None = None  # For RAPTOR hierarchical


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""

    # General settings
    top_k: int = 10
    min_score: float = 0.0

    # Vector search settings
    use_binary_quantization: bool = True
    binary_top_k_multiplier: int = 10  # Fetch 10x more for rescoring
    vector_weight: float = 0.7

    # Text search settings
    text_weight: float = 0.3

    # Hybrid search settings
    rank_fusion_k: int = 60  # RRF k parameter

    # Reranking settings
    rerank: bool = True
    rerank_top_k: int = 50  # How many to send to reranker

    # GraphRAG settings
    graph_depth: int = 2  # $graphLookup depth
    include_communities: bool = True

    # RAPTOR settings
    raptor_levels: list[int] = field(default_factory=lambda: [0, 1, 2])
    level_weights: dict[int, float] = field(
        default_factory=lambda: {0: 0.5, 1: 0.3, 2: 0.2}
    )


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    def __init__(self, mongodb_client: Any, config: RetrievalConfig | None = None):
        """Initialize retriever.

        Args:
            mongodb_client: MongoDBClient instance
            config: Retrieval configuration
        """
        self.mongodb = mongodb_client
        self.config = config or RetrievalConfig()

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents.

        Args:
            query: The query text
            query_embedding: Query vector embedding
            top_k: Number of results to return (overrides config)
            **kwargs: Additional retriever-specific arguments

        Returns:
            List of RetrievalResult sorted by score descending
        """
        pass

    def _apply_min_score(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Filter results below minimum score."""
        return [r for r in results if r.score >= self.config.min_score]

    def _deduplicate(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Remove duplicate chunks, keeping highest score."""
        seen: dict[str, RetrievalResult] = {}
        for result in results:
            if result.chunk_id not in seen or result.score > seen[result.chunk_id].score:
                seen[result.chunk_id] = result
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)
