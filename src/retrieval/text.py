"""Text search using MongoDB Atlas $search for BM25."""

import logging
from typing import Any

from .base import BaseRetriever, RetrievalResult, RetrievalConfig

logger = logging.getLogger(__name__)


class TextSearcher(BaseRetriever):
    """Text search using MongoDB Atlas $search (BM25).

    Provides keyword-based retrieval that complements vector search.
    """

    COLLECTION_NAME = "chunks"
    TEXT_INDEX = "text_search_index"

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        search_fields: list[str] | None = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        """Retrieve chunks using text search (BM25).

        Args:
            query: The query text
            query_embedding: Not used for text search (can be None)
            top_k: Number of results to return
            search_fields: Fields to search (default: content)

        Returns:
            List of RetrievalResult sorted by BM25 score
        """
        top_k = top_k or self.config.top_k
        search_fields = search_fields or ["content"]

        results = await self._text_search(query, top_k, search_fields)
        results = self._apply_min_score(results)

        return results[:top_k]

    async def _text_search(
        self,
        query: str,
        top_k: int,
        search_fields: list[str],
    ) -> list[RetrievalResult]:
        """Perform BM25 text search using $search."""
        pipeline = [
            {
                "$search": {
                    "index": self.TEXT_INDEX,
                    "text": {
                        "query": query,
                        "path": search_fields,
                        "fuzzy": {
                            "maxEdits": 1,  # Allow 1 typo
                            "prefixLength": 3,  # First 3 chars must match
                        },
                    },
                }
            },
            {"$limit": top_k},
            {
                "$project": {
                    "chunk_id": 1,
                    "document_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"},
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
                    text_score=doc["score"],
                    metadata=doc.get("metadata", {}),
                )
            )

        logger.info(f"Text search returned {len(results)} results")
        return results

    async def phrase_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Search for exact phrase matches."""
        top_k = top_k or self.config.top_k

        pipeline = [
            {
                "$search": {
                    "index": self.TEXT_INDEX,
                    "phrase": {
                        "query": query,
                        "path": "content",
                        "slop": 2,  # Allow up to 2 words between terms
                    },
                }
            },
            {"$limit": top_k},
            {
                "$project": {
                    "chunk_id": 1,
                    "document_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"},
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
                    text_score=doc["score"],
                    metadata=doc.get("metadata", {}),
                )
            )

        return results

    async def autocomplete_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Search with autocomplete for partial matches."""
        top_k = top_k or self.config.top_k

        pipeline = [
            {
                "$search": {
                    "index": self.TEXT_INDEX,
                    "autocomplete": {
                        "query": query,
                        "path": "content",
                        "tokenOrder": "sequential",
                    },
                }
            },
            {"$limit": top_k},
            {
                "$project": {
                    "chunk_id": 1,
                    "document_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"},
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
                    text_score=doc["score"],
                    metadata=doc.get("metadata", {}),
                )
            )

        return results
