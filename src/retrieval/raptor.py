"""RAPTOR hierarchical retrieval for multi-level document search."""

import logging
from typing import Any

from .base import BaseRetriever, RetrievalResult, RetrievalConfig

logger = logging.getLogger(__name__)


class RAPTORRetriever(BaseRetriever):
    """RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

    Retrieves from multiple hierarchy levels:
    - Level 0: Leaf chunks (most specific)
    - Level 1: Cluster summaries (paragraph-level)
    - Level 2: Section summaries (section-level)
    - Level 3+: Document summaries (global)

    Combines results with level-based weighting for comprehensive context.
    """

    COLLECTION_NAME = "chunks"
    VECTOR_INDEX = "vector_index_full"

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        levels: list[int] | None = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        """Retrieve from multiple hierarchy levels.

        Args:
            query: The query text
            query_embedding: Query vector embedding
            top_k: Total number of results to return
            levels: Specific levels to search (default: [0, 1, 2])

        Returns:
            Combined and deduplicated results from all levels
        """
        top_k = top_k or self.config.top_k
        levels = levels or self.config.raptor_levels
        level_weights = self.config.level_weights

        all_results = []

        # Search each level
        for level in levels:
            weight = level_weights.get(level, 0.5 / (level + 1))  # Decay weight by level
            level_top_k = max(3, int(top_k * weight * 2))  # Proportional to weight

            level_results = await self._search_level(
                query_embedding, level, level_top_k
            )

            # Apply level weight to scores
            for result in level_results:
                result.level = level
                result.score *= weight
                result.metadata["raptor_level"] = level
                result.metadata["level_weight"] = weight

            all_results.extend(level_results)

        # Deduplicate and sort
        results = self._deduplicate(all_results)

        # Ensure diversity: at least one from each level if available
        results = self._ensure_level_diversity(results, levels, min_per_level=1)

        return results[:top_k]

    async def _search_level(
        self,
        query_embedding: list[float],
        level: int,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Search a specific hierarchy level."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "filter": {"level": level},
                }
            },
            {
                "$project": {
                    "chunk_id": 1,
                    "document_id": 1,
                    "content": 1,
                    "level": 1,
                    "parent_id": 1,
                    "children_ids": 1,
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
                    level=doc.get("level", 0),
                    metadata={
                        **doc.get("metadata", {}),
                        "parent_id": doc.get("parent_id"),
                        "children_ids": doc.get("children_ids", []),
                    },
                )
            )

        logger.info(f"RAPTOR level {level} search returned {len(results)} results")
        return results

    def _ensure_level_diversity(
        self,
        results: list[RetrievalResult],
        levels: list[int],
        min_per_level: int = 1,
    ) -> list[RetrievalResult]:
        """Ensure at least min_per_level results from each level."""
        # Group by level
        by_level: dict[int, list[RetrievalResult]] = {level: [] for level in levels}
        for result in results:
            level = result.level or 0
            if level in by_level:
                by_level[level].append(result)

        # Build diverse result set
        diverse = []
        added_ids = set()

        # First pass: add minimum from each level
        for level in levels:
            level_results = by_level[level]
            for result in level_results[:min_per_level]:
                if result.chunk_id not in added_ids:
                    diverse.append(result)
                    added_ids.add(result.chunk_id)

        # Second pass: add remaining by score
        for result in results:
            if result.chunk_id not in added_ids:
                diverse.append(result)
                added_ids.add(result.chunk_id)

        # Re-sort by score
        diverse.sort(key=lambda x: x.score, reverse=True)
        return diverse

    async def retrieve_with_context(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve with hierarchical context expansion.

        For each result, also returns parent and children summaries.
        Useful for understanding the broader context of a match.
        """
        top_k = top_k or self.config.top_k
        results = await self.retrieve(query, query_embedding, top_k)

        # Collect parent and children IDs
        parent_ids = set()
        children_ids = set()
        for result in results:
            if result.metadata.get("parent_id"):
                parent_ids.add(result.metadata["parent_id"])
            children_ids.update(result.metadata.get("children_ids", []))

        # Fetch parent and children chunks
        all_ids = list(parent_ids | children_ids)
        context_chunks = {}
        if all_ids:
            collection = self.mongodb.db[self.COLLECTION_NAME]
            cursor = collection.find({"chunk_id": {"$in": all_ids}})
            async for doc in cursor:
                context_chunks[doc["chunk_id"]] = doc

        # Build response with context
        results_with_context = []
        for result in results:
            context = {
                "result": result,
                "parent": None,
                "children": [],
            }

            # Add parent context
            parent_id = result.metadata.get("parent_id")
            if parent_id and parent_id in context_chunks:
                parent = context_chunks[parent_id]
                context["parent"] = {
                    "chunk_id": parent["chunk_id"],
                    "level": parent.get("level", 0),
                    "content": parent["content"][:500],  # Truncate
                }

            # Add children context
            for child_id in result.metadata.get("children_ids", []):
                if child_id in context_chunks:
                    child = context_chunks[child_id]
                    context["children"].append({
                        "chunk_id": child["chunk_id"],
                        "level": child.get("level", 0),
                        "content": child["content"][:300],  # Truncate
                    })

            results_with_context.append(context)

        return {
            "results": results_with_context,
            "levels_searched": self.config.raptor_levels,
            "total_results": len(results),
        }

    async def tree_traversal(
        self,
        root_chunk_id: str,
        direction: str = "down",
        max_depth: int = 3,
    ) -> list[RetrievalResult]:
        """Traverse the RAPTOR tree from a given node.

        Args:
            root_chunk_id: Starting chunk ID
            direction: "up" (to parents) or "down" (to children)
            max_depth: Maximum traversal depth

        Returns:
            List of chunks in traversal order
        """
        collection = self.mongodb.db[self.COLLECTION_NAME]

        # Get root node
        root = await collection.find_one({"chunk_id": root_chunk_id})
        if not root:
            return []

        traversed = [
            RetrievalResult(
                chunk_id=root["chunk_id"],
                document_id=root["document_id"],
                content=root["content"],
                score=1.0,
                level=root.get("level", 0),
                metadata=root.get("metadata", {}),
            )
        ]

        if direction == "up":
            # Traverse to parents
            current = root
            for depth in range(max_depth):
                parent_id = current.get("parent_id")
                if not parent_id:
                    break
                parent = await collection.find_one({"chunk_id": parent_id})
                if not parent:
                    break
                traversed.append(
                    RetrievalResult(
                        chunk_id=parent["chunk_id"],
                        document_id=parent["document_id"],
                        content=parent["content"],
                        score=1.0 - (depth + 1) * 0.1,
                        level=parent.get("level", 0),
                        metadata=parent.get("metadata", {}),
                    )
                )
                current = parent

        else:  # direction == "down"
            # BFS traversal to children
            queue = [(root, 0)]
            visited = {root_chunk_id}

            while queue:
                current, depth = queue.pop(0)
                if depth >= max_depth:
                    continue

                children_ids = current.get("children_ids", [])
                if not children_ids:
                    continue

                cursor = collection.find({"chunk_id": {"$in": children_ids}})
                async for child in cursor:
                    if child["chunk_id"] not in visited:
                        visited.add(child["chunk_id"])
                        traversed.append(
                            RetrievalResult(
                                chunk_id=child["chunk_id"],
                                document_id=child["document_id"],
                                content=child["content"],
                                score=1.0 - (depth + 1) * 0.1,
                                level=child.get("level", 0),
                                metadata=child.get("metadata", {}),
                            )
                        )
                        queue.append((child, depth + 1))

        return traversed
