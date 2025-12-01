"""Hybrid search using MongoDB $rankFusion and $scoreFusion."""

import logging
from typing import Any, Literal

from .base import BaseRetriever, RetrievalResult, RetrievalConfig

logger = logging.getLogger(__name__)


class HybridSearcher(BaseRetriever):
    """Hybrid search combining vector and text search.

    Uses MongoDB's $rankFusion (RRF) or $scoreFusion for combining results.
    This is the primary retrieval strategy for most queries.
    """

    COLLECTION_NAME = "chunks"
    VECTOR_INDEX = "vector_index_full"
    TEXT_INDEX = "text_search_index"

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        fusion_type: Literal["rank", "score"] = "rank",
        **kwargs,
    ) -> list[RetrievalResult]:
        """Retrieve chunks using hybrid vector + text search.

        Args:
            query: The query text
            query_embedding: Query vector embedding
            top_k: Number of results to return
            fusion_type: "rank" for RRF, "score" for weighted scores

        Returns:
            List of RetrievalResult with combined scores
        """
        top_k = top_k or self.config.top_k

        if fusion_type == "rank":
            results = await self._rank_fusion_search(query, query_embedding, top_k)
        else:
            results = await self._score_fusion_search(query, query_embedding, top_k)

        results = self._apply_min_score(results)
        return results[:top_k]

    async def _rank_fusion_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Hybrid search using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank_i)) for each retriever
        MongoDB's $rankFusion implements this natively.
        """
        k = self.config.rank_fusion_k

        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            # Vector search pipeline
                            "vector": [
                                {
                                    "$vectorSearch": {
                                        "index": self.VECTOR_INDEX,
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": top_k * 10,
                                        "limit": top_k * 2,
                                    }
                                }
                            ],
                            # Text search pipeline
                            "text": [
                                {
                                    "$search": {
                                        "index": self.TEXT_INDEX,
                                        "text": {
                                            "query": query,
                                            "path": "content",
                                        },
                                    }
                                },
                                {"$limit": top_k * 2},
                            ],
                        }
                    },
                    "combination": {
                        "ranker": {
                            "rrf": {
                                "k": k,
                            }
                        }
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
                    "score": {"$meta": "rankFusionScore"},
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
                    metadata=doc.get("metadata", {}),
                )
            )

        logger.info(f"Rank fusion search returned {len(results)} results")
        return results

    async def _score_fusion_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Hybrid search using weighted score fusion.

        Combines normalized scores: score = w1 * vector_score + w2 * text_score
        MongoDB's $scoreFusion implements this with customizable weights.
        """
        vector_weight = self.config.vector_weight
        text_weight = self.config.text_weight

        pipeline = [
            {
                "$scoreFusion": {
                    "input": {
                        "pipelines": {
                            "vector": [
                                {
                                    "$vectorSearch": {
                                        "index": self.VECTOR_INDEX,
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": top_k * 10,
                                        "limit": top_k * 2,
                                    }
                                }
                            ],
                            "text": [
                                {
                                    "$search": {
                                        "index": self.TEXT_INDEX,
                                        "text": {
                                            "query": query,
                                            "path": "content",
                                        },
                                    }
                                },
                                {"$limit": top_k * 2},
                            ],
                        }
                    },
                    "combination": {
                        "weights": {
                            "vector": vector_weight,
                            "text": text_weight,
                        },
                        "normalization": "minmax",  # Normalize scores to 0-1
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
                    "score": {"$meta": "scoreFusionScore"},
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
                    metadata=doc.get("metadata", {}),
                )
            )

        logger.info(f"Score fusion search returned {len(results)} results")
        return results

    async def hybrid_with_filters(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid search with additional metadata filters.

        Args:
            query: The query text
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filters: MongoDB filter expressions (e.g., {"metadata.source": "docs"})
        """
        top_k = top_k or self.config.top_k
        filters = filters or {}

        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vector": [
                                {
                                    "$vectorSearch": {
                                        "index": self.VECTOR_INDEX,
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": top_k * 10,
                                        "limit": top_k * 3,
                                        "filter": filters,
                                    }
                                }
                            ],
                            "text": [
                                {
                                    "$search": {
                                        "index": self.TEXT_INDEX,
                                        "compound": {
                                            "must": [
                                                {
                                                    "text": {
                                                        "query": query,
                                                        "path": "content",
                                                    }
                                                }
                                            ],
                                            "filter": [
                                                {"equals": {"path": k, "value": v}}
                                                for k, v in filters.items()
                                            ] if filters else [],
                                        },
                                    }
                                },
                                {"$limit": top_k * 3},
                            ],
                        }
                    },
                    "combination": {
                        "ranker": {"rrf": {"k": self.config.rank_fusion_k}}
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
                    "score": {"$meta": "rankFusionScore"},
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
                    metadata=doc.get("metadata", {}),
                )
            )

        return results
