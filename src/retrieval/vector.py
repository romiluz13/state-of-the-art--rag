"""Vector search using MongoDB $vectorSearch with binary quantization."""

import logging
from typing import Any

from .base import BaseRetriever, RetrievalResult, RetrievalConfig

logger = logging.getLogger(__name__)


class VectorSearcher(BaseRetriever):
    """Vector search using MongoDB Atlas $vectorSearch.

    Supports two-stage retrieval:
    1. Binary quantization for fast initial retrieval (32x smaller)
    2. Full precision rescoring for accuracy
    """

    COLLECTION_NAME = "chunks"
    VECTOR_INDEX_FULL = "vector_index_full"
    VECTOR_INDEX_BINARY = "vector_index_binary"

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        use_binary: bool | None = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        """Retrieve chunks using vector similarity.

        Args:
            query: The query text (for metadata)
            query_embedding: Query vector embedding
            top_k: Number of results to return
            use_binary: Whether to use binary quantization (overrides config)

        Returns:
            List of RetrievalResult sorted by vector score
        """
        top_k = top_k or self.config.top_k
        use_binary = use_binary if use_binary is not None else self.config.use_binary_quantization

        if use_binary:
            results = await self._two_stage_retrieval(query_embedding, top_k)
        else:
            results = await self._full_precision_retrieval(query_embedding, top_k)

        results = self._apply_min_score(results)
        return results[:top_k]

    async def _full_precision_retrieval(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Single-stage full precision vector search."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.VECTOR_INDEX_FULL,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,  # MongoDB recommends 10x
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "chunk_id": 1,
                    "document_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        collection = self.mongodb.db[self.COLLECTION_NAME]
        cursor = collection.aggregate(pipeline)

        results = []
        async for doc in cursor:
            results.append(
                RetrievalResult(
                    chunk_id=doc["chunk_id"],
                    document_id=doc["document_id"],
                    content=doc["content"],
                    score=doc["score"],
                    vector_score=doc["score"],
                    metadata=doc.get("metadata", {}),
                )
            )

        logger.info(f"Full precision search returned {len(results)} results")
        return results

    async def _two_stage_retrieval(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Two-stage retrieval: binary first, then full precision rescore.

        Stage 1: Binary quantization for fast candidate retrieval
        Stage 2: Full precision rescoring on candidates
        """
        # Stage 1: Binary search for candidates
        binary_top_k = top_k * self.config.binary_top_k_multiplier
        candidates = await self._binary_search(query_embedding, binary_top_k)

        if not candidates:
            logger.warning("Binary search returned no candidates")
            return []

        logger.info(f"Binary search returned {len(candidates)} candidates for rescoring")

        # Stage 2: Rescore with full precision
        results = await self._rescore_candidates(
            query_embedding, candidates, top_k
        )

        return results

    async def _binary_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Binary quantization search for fast candidate retrieval."""
        # Convert query to binary for hamming distance
        query_binary = [1 if x > 0 else 0 for x in query_embedding]

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.VECTOR_INDEX_BINARY,
                    "path": "embedding_binary",
                    "queryVector": query_binary,
                    "numCandidates": top_k * 5,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "chunk_id": 1,
                    "document_id": 1,
                    "content": 1,
                    "embedding": 1,  # Need full embedding for rescoring
                    "metadata": 1,
                    "binary_score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        collection = self.mongodb.db[self.COLLECTION_NAME]
        cursor = collection.aggregate(pipeline)

        candidates = []
        async for doc in cursor:
            candidates.append(doc)

        return candidates

    async def _rescore_candidates(
        self,
        query_embedding: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rescore candidates using full precision cosine similarity."""
        results = []

        for doc in candidates:
            # Compute cosine similarity
            if "embedding" in doc and doc["embedding"]:
                score = self._cosine_similarity(query_embedding, doc["embedding"])
            else:
                # Fallback to binary score if no full embedding
                score = doc.get("binary_score", 0.0)

            results.append(
                RetrievalResult(
                    chunk_id=doc["chunk_id"],
                    document_id=doc["document_id"],
                    content=doc["content"],
                    score=score,
                    vector_score=score,
                    metadata=doc.get("metadata", {}),
                )
            )

        # Sort by rescored similarity
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
