"""GraphRAG retrieval using MongoDB $graphLookup.

December 2025 Enhancement:
- Hybrid approach: Vector search + $graphLookup + RRF fusion + Rerank
- Based on LightRAG research showing 20%+ improvement with hybrid approach
"""

import asyncio
import logging
from typing import Any

from .base import BaseRetriever, RetrievalResult, RetrievalConfig

logger = logging.getLogger(__name__)


class GraphRAGRetriever(BaseRetriever):
    """GraphRAG retrieval for global/thematic questions.

    December 2025 Enhancement: Hybrid approach with RRF fusion.

    Algorithm (LightRAG-inspired):
    1. PARALLEL: Vector search entities + $graphLookup traversal
    2. RRF fusion: Combine vector and graph results
    3. Get community summaries and source chunks
    4. Optional: Rerank with Voyage instruction-following
    """

    ENTITIES_COLLECTION = "entities"
    COMMUNITIES_COLLECTION = "communities"
    CHUNKS_COLLECTION = "chunks"
    ENTITY_VECTOR_INDEX = "entity_vector_index"
    COMMUNITY_VECTOR_INDEX = "community_vector_index"

    # December 2025: Strategy-specific rerank instruction
    RERANK_INSTRUCTION = "Focus on documents with clear entity relationships"

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        **kwargs,
    ) -> list[RetrievalResult]:
        """Retrieve using GraphRAG strategy.

        December 2025 Enhancement: Hybrid approach with parallel retrieval,
        RRF fusion, and optional reranking.

        Args:
            query: The query text
            query_embedding: Query vector embedding
            top_k: Number of results to return
            use_hybrid: Use hybrid approach (Dec 2025 enhancement)
            use_rerank: Use Voyage reranking with instructions

        Returns:
            List of RetrievalResult from entities, relationships, and communities
        """
        top_k = top_k or self.config.top_k
        graph_depth = self.config.graph_depth
        include_communities = self.config.include_communities

        if use_hybrid:
            # December 2025: Hybrid approach with parallel retrieval
            return await self._hybrid_retrieve(
                query, query_embedding, top_k, graph_depth,
                include_communities, use_rerank
            )
        else:
            # Legacy approach
            return await self._legacy_retrieve(
                query, query_embedding, top_k, graph_depth, include_communities
            )

    async def _hybrid_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        graph_depth: int,
        include_communities: bool,
        use_rerank: bool,
    ) -> list[RetrievalResult]:
        """December 2025: Hybrid retrieval with RRF fusion.

        1. Parallel: Vector search + Graph traversal
        2. RRF fusion
        3. Get communities and chunks
        4. Optional reranking
        """
        # Step 1: PARALLEL retrieval (vector + initial graph seed)
        vector_entities, seed_entities = await asyncio.gather(
            self._search_entities(query_embedding, top_k * 3),
            self._search_entities(query_embedding, top_k),  # Seed for graph
        )

        if not vector_entities and not seed_entities:
            logger.warning("No entities found, falling back to chunk search")
            return await self._fallback_chunk_search(query_embedding, top_k)

        # Step 2: Graph traversal from seed entities
        graph_entities = await self._graph_traversal(seed_entities, depth=graph_depth)

        # Step 3: RRF Fusion of vector and graph results
        fused_entities = self._reciprocal_rank_fusion(
            vector_entities, graph_entities, k=60
        )

        logger.info(f"RRF fusion: {len(vector_entities)} vector + {len(graph_entities)} graph = {len(fused_entities)} fused")

        # Step 4: Get communities if enabled
        communities = []
        if include_communities:
            community_ids = set()
            for entity in fused_entities:
                if entity.get("community_id"):
                    community_ids.add(entity["community_id"])
            if community_ids:
                communities = await self._get_communities(list(community_ids))

        # Step 5: Get source chunks
        chunk_ids = set()
        for entity in fused_entities:
            chunk_ids.update(entity.get("source_chunks", []))
        chunks = await self._get_chunks(list(chunk_ids)[:top_k * 3])

        # Step 6: Combine results
        results = self._combine_results(fused_entities, [], communities, chunks)
        results = self._deduplicate(results)

        # Step 7: Optional reranking with Voyage instruction-following
        if use_rerank and hasattr(self, 'reranker') and self.reranker:
            results = await self._rerank_with_instructions(query, results, top_k)

        return results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """December 2025: Reciprocal Rank Fusion for hybrid results.

        RRF combines rankings from multiple retrieval methods:
        score = sum(1 / (k + rank)) for each method

        Args:
            vector_results: Results from vector search
            graph_results: Results from graph traversal
            k: Ranking constant (default 60, standard for RRF)

        Returns:
            Fused results sorted by RRF score
        """
        scores: dict[str, float] = {}
        entity_map: dict[str, dict] = {}

        # Score from vector results
        for rank, entity in enumerate(vector_results):
            entity_id = entity.get("entity_id", str(entity.get("_id", "")))
            scores[entity_id] = scores.get(entity_id, 0) + 1 / (k + rank + 1)
            entity_map[entity_id] = entity

        # Score from graph results
        for rank, entity in enumerate(graph_results):
            entity_id = entity.get("entity_id", str(entity.get("_id", "")))
            scores[entity_id] = scores.get(entity_id, 0) + 1 / (k + rank + 1)
            if entity_id not in entity_map:
                entity_map[entity_id] = entity

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Add RRF score to entities
        fused = []
        for entity_id in sorted_ids:
            entity = entity_map[entity_id].copy()
            entity["rrf_score"] = scores[entity_id]
            fused.append(entity)

        return fused

    async def _rerank_with_instructions(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """December 2025: Rerank with Voyage instruction-following.

        Uses strategy-specific instruction for better relevance.
        """
        try:
            # Extract content for reranking
            documents = [r.content for r in results]

            # Rerank with instruction
            reranked = await self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=min(top_k * 2, len(results)),
                instructions=self.RERANK_INSTRUCTION,
            )

            # Map back to results
            reranked_results = []
            for idx, score in reranked:
                result = results[idx]
                result.score = score  # Update with rerank score
                result.metadata["reranked"] = True
                reranked_results.append(result)

            logger.info(f"Reranked {len(reranked_results)} results with instructions")
            return reranked_results

        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results

    async def _legacy_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        graph_depth: int,
        include_communities: bool,
    ) -> list[RetrievalResult]:
        """Legacy retrieval method (pre-December 2025)."""
        # Step 1: Find relevant entities
        entities = await self._search_entities(query_embedding, top_k * 2)

        if not entities:
            logger.warning("No entities found, falling back to chunk search")
            return await self._fallback_chunk_search(query_embedding, top_k)

        # Step 2: Graph traversal for related entities
        related_entities = await self._graph_traversal(
            entities, depth=graph_depth
        )

        # Step 3: Get community summaries if enabled
        communities = []
        if include_communities:
            community_ids = set()
            for entity in entities + related_entities:
                if entity.get("community_id"):
                    community_ids.add(entity["community_id"])

            if community_ids:
                communities = await self._get_communities(list(community_ids))

        # Step 4: Get source chunks from entities
        chunk_ids = set()
        for entity in entities + related_entities:
            chunk_ids.update(entity.get("source_chunks", []))

        chunks = await self._get_chunks(list(chunk_ids)[:top_k * 3])

        # Combine all results
        results = self._combine_results(entities, related_entities, communities, chunks)

        results = self._deduplicate(results)
        return results[:top_k]

    async def _search_entities(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Search for relevant entities using vector similarity."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.ENTITY_VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 5,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "entity_id": 1,
                    "name": 1,
                    "type": 1,
                    "description": 1,
                    "relationships": 1,
                    "source_chunks": 1,
                    "community_id": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        collection = self.mongodb.db[self.ENTITIES_COLLECTION]
        cursor = collection.aggregate(pipeline)

        entities = []
        async for doc in cursor:
            entities.append(doc)

        logger.info(f"Entity search found {len(entities)} entities")
        return entities

    async def _graph_traversal(
        self,
        seed_entities: list[dict[str, Any]],
        depth: int,
    ) -> list[dict[str, Any]]:
        """Traverse entity relationships using $graphLookup.

        $graphLookup performs recursive graph traversal in MongoDB.
        """
        if not seed_entities:
            return []

        seed_ids = [e["entity_id"] for e in seed_entities]

        pipeline = [
            {"$match": {"entity_id": {"$in": seed_ids}}},
            {
                "$graphLookup": {
                    "from": self.ENTITIES_COLLECTION,
                    "startWith": "$relationships.target_entity_id",
                    "connectFromField": "relationships.target_entity_id",
                    "connectToField": "entity_id",
                    "as": "related_entities",
                    "maxDepth": depth,
                    "depthField": "depth",
                }
            },
            {"$unwind": "$related_entities"},
            {
                "$replaceRoot": {"newRoot": "$related_entities"}
            },
            {
                "$group": {
                    "_id": "$entity_id",
                    "entity_id": {"$first": "$entity_id"},
                    "name": {"$first": "$name"},
                    "type": {"$first": "$type"},
                    "description": {"$first": "$description"},
                    "relationships": {"$first": "$relationships"},
                    "source_chunks": {"$first": "$source_chunks"},
                    "community_id": {"$first": "$community_id"},
                    "depth": {"$min": "$depth"},
                }
            },
            # Exclude seed entities
            {"$match": {"entity_id": {"$nin": seed_ids}}},
            # Sort by depth (closer entities first)
            {"$sort": {"depth": 1}},
        ]

        collection = self.mongodb.db[self.ENTITIES_COLLECTION]
        cursor = collection.aggregate(pipeline)

        related = []
        async for doc in cursor:
            related.append(doc)

        logger.info(f"Graph traversal found {len(related)} related entities")
        return related

    async def _get_communities(
        self,
        community_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get community summaries."""
        collection = self.mongodb.db[self.COMMUNITIES_COLLECTION]
        cursor = collection.find({"community_id": {"$in": community_ids}})

        communities = []
        async for doc in cursor:
            communities.append(doc)

        logger.info(f"Retrieved {len(communities)} community summaries")
        return communities

    async def _get_chunks(
        self,
        chunk_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get source chunks by ID."""
        collection = self.mongodb.db[self.CHUNKS_COLLECTION]
        cursor = collection.find({"chunk_id": {"$in": chunk_ids}})

        chunks = []
        async for doc in cursor:
            chunks.append(doc)

        return chunks

    async def _fallback_chunk_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fallback to standard chunk search if no entities found."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index_full",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
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

        collection = self.mongodb.db[self.CHUNKS_COLLECTION]
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

        return results

    def _combine_results(
        self,
        entities: list[dict[str, Any]],
        related_entities: list[dict[str, Any]],
        communities: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> list[RetrievalResult]:
        """Combine all GraphRAG sources into retrieval results."""
        results = []

        # Add entity descriptions as results (high relevance)
        for entity in entities:
            results.append(
                RetrievalResult(
                    chunk_id=f"entity_{entity['entity_id']}",
                    document_id="entities",
                    content=f"[{entity['type']}] {entity['name']}: {entity['description']}",
                    score=entity.get("score", 0.8),
                    metadata={
                        "type": "entity",
                        "entity_type": entity["type"],
                        "entity_name": entity["name"],
                    },
                )
            )

        # Add related entities (lower score based on depth)
        for entity in related_entities:
            depth = entity.get("depth", 1)
            score = max(0.5, 0.8 - (depth * 0.1))  # Decay by depth
            results.append(
                RetrievalResult(
                    chunk_id=f"entity_{entity['entity_id']}",
                    document_id="entities",
                    content=f"[{entity['type']}] {entity['name']}: {entity['description']}",
                    score=score,
                    metadata={
                        "type": "related_entity",
                        "entity_type": entity["type"],
                        "depth": depth,
                    },
                )
            )

        # Add community summaries (global context)
        for community in communities:
            results.append(
                RetrievalResult(
                    chunk_id=f"community_{community['community_id']}",
                    document_id="communities",
                    content=f"[Community: {community.get('title', 'Untitled')}] {community['summary']}",
                    score=0.75,  # Community summaries are global context
                    metadata={
                        "type": "community",
                        "community_level": community.get("level", 1),
                    },
                )
            )

        # Add source chunks
        for chunk in chunks:
            results.append(
                RetrievalResult(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    score=0.7,  # Source chunks support entities
                    metadata={
                        "type": "source_chunk",
                        **chunk.get("metadata", {}),
                    },
                )
            )

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    async def retrieve_for_global_question(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Specialized retrieval for global/thematic questions.

        Returns structured context for answer synthesis:
        - entities: Key entities with descriptions
        - relationships: How entities connect
        - communities: Thematic summaries
        - supporting_chunks: Source text
        """
        top_k = top_k or self.config.top_k

        entities = await self._search_entities(query_embedding, top_k)
        related = await self._graph_traversal(entities, self.config.graph_depth)

        # Get communities
        community_ids = set()
        for e in entities + related:
            if e.get("community_id"):
                community_ids.add(e["community_id"])
        communities = await self._get_communities(list(community_ids))

        # Get supporting chunks
        chunk_ids = set()
        for e in entities + related:
            chunk_ids.update(e.get("source_chunks", []))
        chunks = await self._get_chunks(list(chunk_ids)[:top_k * 2])

        return {
            "entities": entities,
            "related_entities": related,
            "communities": communities,
            "supporting_chunks": chunks,
            "total_context_items": len(entities) + len(related) + len(communities) + len(chunks),
        }
