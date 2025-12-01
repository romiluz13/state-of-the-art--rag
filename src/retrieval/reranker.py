"""Reranking using Voyage AI rerank-2.5."""

import logging
from typing import Any

from .base import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker using Voyage AI rerank-2.5.

    Improves retrieval quality by re-scoring results with a cross-encoder model.
    """

    DEFAULT_MODEL = "rerank-2.5"
    MAX_DOCUMENTS = 1000  # Voyage API limit

    def __init__(self, voyage_client: Any, model: str | None = None):
        """Initialize reranker.

        Args:
            voyage_client: VoyageClient instance
            model: Reranking model (default: rerank-2.5)
        """
        self.voyage = voyage_client
        self.model = model or self.DEFAULT_MODEL

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results using Voyage reranker.

        Args:
            query: The original query
            results: List of retrieval results to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of RetrievalResult
        """
        if not results:
            return []

        # Limit to API maximum
        if len(results) > self.MAX_DOCUMENTS:
            logger.warning(
                f"Truncating {len(results)} results to {self.MAX_DOCUMENTS} for reranking"
            )
            results = results[: self.MAX_DOCUMENTS]

        # Extract documents for reranking
        documents = [r.content for r in results]

        try:
            # Call Voyage reranking API
            response = await self.voyage.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k or len(results),
            )

            # Map reranked results back
            reranked = []
            for item in response.get("data", []):
                idx = item["index"]
                score = item["relevance_score"]

                result = results[idx]
                result.rerank_score = score
                result.score = score  # Update main score to rerank score
                reranked.append(result)

            logger.info(f"Reranked {len(reranked)} results")
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original results")
            return results[:top_k] if top_k else results

    async def rerank_with_scores(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
        return_documents: bool = False,
    ) -> list[RetrievalResult]:
        """Rerank and preserve both original and rerank scores.

        Args:
            query: The original query
            results: List of retrieval results to rerank
            top_k: Number of results to return
            return_documents: Whether to return document content (can reduce tokens)

        Returns:
            Reranked results with both original score and rerank_score
        """
        if not results:
            return []

        documents = [r.content for r in results]

        try:
            response = await self.voyage.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k or len(results),
                return_documents=return_documents,
            )

            reranked = []
            for item in response.get("data", []):
                idx = item["index"]
                score = item["relevance_score"]

                result = results[idx]
                # Preserve original scores
                original_score = result.score
                result.rerank_score = score
                result.score = score
                result.metadata["original_score"] = original_score
                reranked.append(result)

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k] if top_k else results

    async def batch_rerank(
        self,
        queries: list[str],
        results_list: list[list[RetrievalResult]],
        top_k: int | None = None,
    ) -> list[list[RetrievalResult]]:
        """Rerank multiple query-results pairs efficiently.

        Args:
            queries: List of queries
            results_list: List of result lists (one per query)
            top_k: Number of results per query

        Returns:
            List of reranked result lists
        """
        if len(queries) != len(results_list):
            raise ValueError("Number of queries must match number of result lists")

        reranked_all = []
        for query, results in zip(queries, results_list):
            reranked = await self.rerank(query, results, top_k)
            reranked_all.append(reranked)

        return reranked_all

    def compute_score_change(
        self, results: list[RetrievalResult]
    ) -> list[dict[str, Any]]:
        """Compute score changes after reranking for analysis.

        Args:
            results: Results with both original and rerank scores

        Returns:
            List of dicts with score change analysis
        """
        analysis = []
        for result in results:
            original = result.metadata.get("original_score", result.score)
            rerank = result.rerank_score or result.score

            analysis.append({
                "chunk_id": result.chunk_id,
                "original_score": original,
                "rerank_score": rerank,
                "score_change": rerank - original if original else 0,
                "rank_improved": rerank > original if original else False,
            })

        return analysis
