"""RAPTOR hierarchical chunking - Recursive Abstractive Processing for Tree-Organized Retrieval."""

import logging
from dataclasses import dataclass, field
import numpy as np
from typing import Callable, Awaitable

from .base import BaseChunker, Chunk
from .recursive import RecursiveChunker

logger = logging.getLogger(__name__)


@dataclass
class RAPTORNode:
    """Node in the RAPTOR tree structure."""

    content: str
    level: int  # 0=leaf, 1+=summary
    embedding: list[float] | None = None
    children_indices: list[int] = field(default_factory=list)
    parent_index: int | None = None
    chunk_index: int = 0
    token_count: int = 0


class RAPTORChunker(BaseChunker):
    """RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

    Algorithm:
    1. Create leaf chunks using recursive splitting
    2. Embed all chunks
    3. Cluster similar chunks (k-means)
    4. Summarize each cluster using LLM
    5. Repeat steps 2-4 for higher levels until single root
    """

    def __init__(
        self,
        embed_function: Callable[[list[str]], Awaitable[list[list[float]]]],
        summarize_function: Callable[[list[str]], Awaitable[str]],
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        max_levels: int = 3,
        min_cluster_size: int = 3,
        target_clusters: int | None = None,
    ):
        """Initialize RAPTOR chunker.

        Args:
            embed_function: Async function to embed texts -> embeddings
            summarize_function: Async function to summarize texts -> summary
            chunk_size: Size for leaf chunks
            chunk_overlap: Overlap for leaf chunks
            max_levels: Maximum hierarchy levels (1=leaf only, 2=leaf+clusters, etc.)
            min_cluster_size: Minimum chunks to form a cluster
            target_clusters: Target number of clusters per level (auto if None)
        """
        self.embed_function = embed_function
        self.summarize_function = summarize_function
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_levels = max_levels
        self.min_cluster_size = min_cluster_size
        self.target_clusters = target_clusters

        # Use recursive chunker for leaf level
        self.leaf_chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str) -> list[Chunk]:
        """Synchronous interface - returns leaf chunks only.

        For full RAPTOR hierarchy, use build_hierarchy() instead.
        """
        return self.leaf_chunker.chunk(text)

    async def build_hierarchy(self, text: str) -> list[RAPTORNode]:
        """Build full RAPTOR hierarchy asynchronously.

        Args:
            text: Document text to process

        Returns:
            List of all nodes (leaves + summaries) in the tree
        """
        # Step 1: Create leaf chunks
        leaf_chunks = self.leaf_chunker.chunk(text)

        if not leaf_chunks:
            return []

        logger.info(f"Created {len(leaf_chunks)} leaf chunks")

        # Convert to RAPTOR nodes
        nodes: list[RAPTORNode] = []
        for i, chunk in enumerate(leaf_chunks):
            nodes.append(
                RAPTORNode(
                    content=chunk.content,
                    level=0,
                    children_indices=[],
                    chunk_index=i,
                    token_count=chunk.token_count,
                )
            )

        # Step 2: Embed all leaf chunks
        texts = [node.content for node in nodes]
        embeddings = await self.embed_function(texts)

        for i, emb in enumerate(embeddings):
            nodes[i].embedding = emb

        logger.info(f"Embedded {len(nodes)} leaf chunks")

        # Step 3-5: Build higher levels
        current_level_nodes = nodes.copy()
        current_level = 0

        while current_level < self.max_levels - 1 and len(current_level_nodes) >= self.min_cluster_size:
            current_level += 1

            # Cluster current level
            clusters = self._cluster_nodes(current_level_nodes)

            if len(clusters) <= 1:
                logger.info(f"Only {len(clusters)} cluster(s) at level {current_level}, stopping")
                break

            logger.info(f"Level {current_level}: Created {len(clusters)} clusters")

            # Summarize each cluster
            new_level_nodes = []
            for cluster_indices in clusters:
                if len(cluster_indices) < self.min_cluster_size:
                    continue

                # Get texts from cluster
                cluster_texts = [current_level_nodes[i].content for i in cluster_indices]

                # Generate summary
                summary = await self.summarize_function(cluster_texts)

                # Create summary node
                summary_node = RAPTORNode(
                    content=summary,
                    level=current_level,
                    children_indices=[
                        nodes.index(current_level_nodes[i]) for i in cluster_indices
                    ],
                    chunk_index=len(nodes),
                    token_count=self._count_tokens(summary),
                )

                # Update children's parent
                for i in cluster_indices:
                    orig_idx = nodes.index(current_level_nodes[i])
                    nodes[orig_idx].parent_index = len(nodes)

                nodes.append(summary_node)
                new_level_nodes.append(summary_node)

            # Embed new summary nodes
            if new_level_nodes:
                summary_texts = [n.content for n in new_level_nodes]
                summary_embeddings = await self.embed_function(summary_texts)

                for i, emb in enumerate(summary_embeddings):
                    new_level_nodes[i].embedding = emb

            current_level_nodes = new_level_nodes
            logger.info(f"Level {current_level}: {len(new_level_nodes)} summary nodes created")

        logger.info(f"RAPTOR tree complete: {len(nodes)} total nodes")
        return nodes

    def _cluster_nodes(self, nodes: list[RAPTORNode]) -> list[list[int]]:
        """Cluster nodes by embedding similarity using k-means.

        Args:
            nodes: Nodes to cluster

        Returns:
            List of cluster assignments (list of node indices per cluster)
        """
        if len(nodes) < self.min_cluster_size:
            return [[i for i in range(len(nodes))]]

        # Get embeddings matrix
        embeddings = np.array([n.embedding for n in nodes if n.embedding])

        if len(embeddings) < self.min_cluster_size:
            return [[i for i in range(len(nodes))]]

        # Determine number of clusters
        if self.target_clusters:
            n_clusters = min(self.target_clusters, len(nodes) // self.min_cluster_size)
        else:
            # Auto: sqrt(n) clusters
            n_clusters = max(2, int(np.sqrt(len(nodes))))

        n_clusters = min(n_clusters, len(nodes) // self.min_cluster_size)

        if n_clusters < 2:
            return [[i for i in range(len(nodes))]]

        # Simple k-means clustering
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Group by cluster
            clusters: dict[int, list[int]] = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)

            return list(clusters.values())

        except ImportError:
            logger.warning("sklearn not installed, using simple grouping")
            # Fallback: group sequentially
            clusters = []
            for i in range(0, len(nodes), self.min_cluster_size):
                clusters.append(list(range(i, min(i + self.min_cluster_size, len(nodes)))))
            return clusters

    def _count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text) // 4


def create_raptor_summarize_prompt(texts: list[str]) -> str:
    """Create prompt for summarizing cluster of chunks.

    Args:
        texts: List of chunk texts to summarize

    Returns:
        Prompt string for LLM
    """
    combined = "\n\n---\n\n".join(texts)
    return f"""Summarize the following related text chunks into a coherent summary.
The summary should capture the key information and themes across all chunks.
Keep the summary concise but comprehensive (100-200 words).

Text chunks:
{combined}

Summary:"""
