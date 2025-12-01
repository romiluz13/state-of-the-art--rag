"""Community detection and summarization for GraphRAG."""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import numpy as np

from .entity_extractor import Entity

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """Community of related entities."""

    community_id: str
    level: int
    title: str
    summary: str
    entity_ids: list[str] = field(default_factory=list)
    summary_embedding: list[float] | None = None
    parent_community_id: str | None = None
    child_community_ids: list[str] = field(default_factory=list)


COMMUNITY_SUMMARY_PROMPT = """Summarize the following group of related entities into a coherent community description.

Entities:
{entities}

Provide:
1. A short title (3-5 words) for this community
2. A summary (100-200 words) describing what this community represents and the key relationships

Respond in JSON format:
{{
  "title": "...",
  "summary": "..."
}}"""


class CommunityDetector:
    """Detect and summarize communities of entities."""

    def __init__(
        self,
        summarize_function: Callable[[str], Awaitable[str]],
        embed_function: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        min_community_size: int = 2,
        max_communities: int | None = None,
    ):
        """Initialize community detector.

        Args:
            summarize_function: Async function to generate summaries (LLM)
            embed_function: Optional async function to embed summaries
            min_community_size: Minimum entities per community
            max_communities: Maximum number of communities (auto if None)
        """
        self.summarize_function = summarize_function
        self.embed_function = embed_function
        self.min_community_size = min_community_size
        self.max_communities = max_communities

    async def detect_communities(
        self,
        entities: list[Entity],
        level: int = 1,
    ) -> list[Community]:
        """Detect communities from entities using clustering.

        Uses entity embeddings if available, otherwise relationship-based clustering.

        Args:
            entities: List of entities to cluster
            level: Community level (1=first level above entities)

        Returns:
            List of Community objects
        """
        if len(entities) < self.min_community_size:
            return []

        # Get entity embeddings
        embeddings = [e.embedding for e in entities if e.embedding]

        if len(embeddings) == len(entities) and len(embeddings) >= self.min_community_size:
            # Cluster by embedding similarity
            clusters = self._cluster_by_embedding(entities, embeddings)
        else:
            # Fallback: cluster by relationships
            clusters = self._cluster_by_relationships(entities)

        # Build communities
        communities = []
        for cluster_entities in clusters:
            if len(cluster_entities) < self.min_community_size:
                continue

            community = await self._build_community(cluster_entities, level)
            if community:
                communities.append(community)

        logger.info(f"Detected {len(communities)} communities at level {level}")
        return communities

    def _cluster_by_embedding(
        self,
        entities: list[Entity],
        embeddings: list[list[float]],
    ) -> list[list[Entity]]:
        """Cluster entities by embedding similarity.

        Args:
            entities: List of entities
            embeddings: Entity embeddings

        Returns:
            List of entity clusters
        """
        emb_array = np.array(embeddings)

        # Determine number of clusters
        if self.max_communities:
            n_clusters = min(self.max_communities, len(entities) // self.min_community_size)
        else:
            n_clusters = max(2, int(np.sqrt(len(entities))))

        n_clusters = min(n_clusters, len(entities) // self.min_community_size)

        if n_clusters < 2:
            return [entities]

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(emb_array)

            clusters: dict[int, list[Entity]] = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(entities[i])

            return list(clusters.values())

        except ImportError:
            logger.warning("sklearn not installed, using relationship-based clustering")
            return self._cluster_by_relationships(entities)

    def _cluster_by_relationships(self, entities: list[Entity]) -> list[list[Entity]]:
        """Cluster entities by relationship connectivity.

        Simple connected components approach.

        Args:
            entities: List of entities

        Returns:
            List of entity clusters
        """
        # Build adjacency map
        entity_map = {e.entity_id: e for e in entities}
        visited = set()
        clusters = []

        def dfs(entity: Entity, cluster: list[Entity]):
            if entity.entity_id in visited:
                return
            visited.add(entity.entity_id)
            cluster.append(entity)

            for rel in entity.relationships:
                if rel.target_entity_id in entity_map:
                    dfs(entity_map[rel.target_entity_id], cluster)

        for entity in entities:
            if entity.entity_id not in visited:
                cluster: list[Entity] = []
                dfs(entity, cluster)
                if cluster:
                    clusters.append(cluster)

        # If all entities are in one cluster, split by type
        if len(clusters) == 1 and len(clusters[0]) > self.min_community_size * 2:
            return self._cluster_by_type(entities)

        return clusters

    def _cluster_by_type(self, entities: list[Entity]) -> list[list[Entity]]:
        """Cluster entities by their type.

        Args:
            entities: List of entities

        Returns:
            List of entity clusters grouped by type
        """
        type_clusters: dict[str, list[Entity]] = {}

        for entity in entities:
            if entity.type not in type_clusters:
                type_clusters[entity.type] = []
            type_clusters[entity.type].append(entity)

        return list(type_clusters.values())

    async def _build_community(
        self,
        entities: list[Entity],
        level: int,
    ) -> Community | None:
        """Build a community from a cluster of entities.

        Args:
            entities: Entities in the community
            level: Community level

        Returns:
            Community object or None if summarization fails
        """
        import json

        # Format entities for prompt
        entity_descriptions = []
        for e in entities:
            desc = f"- {e.name} ({e.type}): {e.description}"
            if e.relationships:
                rels = ", ".join([f"{r.relation_type} {r.target_entity_id}" for r in e.relationships[:3]])
                desc += f" [Relations: {rels}]"
            entity_descriptions.append(desc)

        entities_text = "\n".join(entity_descriptions)
        prompt = COMMUNITY_SUMMARY_PROMPT.format(entities=entities_text)

        try:
            response = await self.summarize_function(prompt)

            # Parse JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                title = data.get("title", "Community")
                summary = data.get("summary", "")
            else:
                title = f"Community of {entities[0].type}"
                summary = f"A group of {len(entities)} related entities."

            community = Community(
                community_id=f"community_{uuid.uuid4().hex[:12]}",
                level=level,
                title=title,
                summary=summary,
                entity_ids=[e.entity_id for e in entities],
            )

            # Update entities with community assignment
            for entity in entities:
                entity.community_id = community.community_id
                entity.community_level = level

            # Embed summary if function provided
            if self.embed_function:
                embeddings = await self.embed_function([summary])
                community.summary_embedding = embeddings[0]

            return community

        except Exception as e:
            logger.error(f"Failed to build community: {e}")
            return None

    async def build_hierarchy(
        self,
        entities: list[Entity],
        max_levels: int = 2,
    ) -> list[Community]:
        """Build hierarchical community structure.

        Args:
            entities: List of entities
            max_levels: Maximum hierarchy levels

        Returns:
            All communities across all levels
        """
        all_communities = []

        # Level 1: cluster entities
        level_1 = await self.detect_communities(entities, level=1)
        all_communities.extend(level_1)

        # Higher levels: cluster communities
        current_level = level_1
        for level in range(2, max_levels + 1):
            if len(current_level) < self.min_community_size:
                break

            # Create pseudo-entities from communities for clustering
            pseudo_entities = []
            for comm in current_level:
                pseudo = Entity(
                    entity_id=comm.community_id,
                    name=comm.title,
                    type="Community",
                    description=comm.summary,
                    embedding=comm.summary_embedding,
                )
                pseudo_entities.append(pseudo)

            next_level = await self.detect_communities(pseudo_entities, level=level)

            # Link parent/child communities
            for parent_comm in next_level:
                child_ids = parent_comm.entity_ids  # These are community_ids
                parent_comm.child_community_ids = child_ids
                parent_comm.entity_ids = []  # Clear since these aren't entity IDs

                for child in current_level:
                    if child.community_id in child_ids:
                        child.parent_community_id = parent_comm.community_id
                        parent_comm.entity_ids.extend(child.entity_ids)

            all_communities.extend(next_level)
            current_level = next_level

        logger.info(f"Built community hierarchy: {len(all_communities)} total communities")
        return all_communities
