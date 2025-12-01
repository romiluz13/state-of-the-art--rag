"""ColPali embedder for multimodal document embeddings."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from src.clients.colpali import ColPaliClient, ColPaliConfig, MockColPaliClient
from src.ingestion.loaders.pdf import PageImage

logger = logging.getLogger(__name__)


@dataclass
class ColPaliPageEmbedding:
    """Embedding result for a document page."""

    document_id: str
    page_num: int
    embedding: np.ndarray  # Multi-vector: (num_patches, embedding_dim)
    image_size: tuple[int, int]
    has_text: bool
    has_images: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_list(self) -> list[list[float]]:
        """Convert embedding to list for MongoDB storage."""
        return self.embedding.tolist()

    @classmethod
    def from_list(
        cls,
        document_id: str,
        page_num: int,
        embedding_list: list[list[float]],
        image_size: tuple[int, int],
        **kwargs,
    ) -> "ColPaliPageEmbedding":
        """Create from MongoDB-stored list."""
        return cls(
            document_id=document_id,
            page_num=page_num,
            embedding=np.array(embedding_list, dtype=np.float32),
            image_size=image_size,
            **kwargs,
        )


class ColPaliEmbedder:
    """Generate ColPali embeddings for document pages.

    Integrates with the ingestion pipeline to create multimodal
    embeddings for PDF pages that can be searched visually.
    """

    def __init__(
        self,
        client: ColPaliClient | None = None,
        config: ColPaliConfig | None = None,
        use_mock: bool = False,
    ):
        """Initialize ColPali embedder.

        Args:
            client: Existing ColPali client to use
            config: ColPali configuration
            use_mock: Use mock client for testing
        """
        if client:
            self.client = client
        elif use_mock:
            self.client = MockColPaliClient(config)
        else:
            self.client = ColPaliClient(config)

    def embed_page(
        self,
        document_id: str,
        page_image: PageImage,
    ) -> ColPaliPageEmbedding:
        """Generate embedding for a single page.

        Args:
            document_id: Parent document ID
            page_image: PageImage from PDF loader

        Returns:
            ColPaliPageEmbedding with multi-vector embedding
        """
        embedding = self.client.embed_image(page_image.image)

        return ColPaliPageEmbedding(
            document_id=document_id,
            page_num=page_image.page_num,
            embedding=embedding,
            image_size=(page_image.width, page_image.height),
            has_text=page_image.has_text,
            has_images=page_image.has_images,
        )

    def embed_pages(
        self,
        document_id: str,
        page_images: list[PageImage],
    ) -> list[ColPaliPageEmbedding]:
        """Generate embeddings for multiple pages.

        Args:
            document_id: Parent document ID
            page_images: List of PageImages from PDF loader

        Returns:
            List of ColPaliPageEmbedding objects
        """
        if not page_images:
            return []

        # Extract PIL images
        images = [p.image for p in page_images]

        # Batch embed
        embeddings = self.client.embed_images(images)

        # Create result objects
        results = []
        for page_image, embedding in zip(page_images, embeddings):
            results.append(
                ColPaliPageEmbedding(
                    document_id=document_id,
                    page_num=page_image.page_num,
                    embedding=embedding,
                    image_size=(page_image.width, page_image.height),
                    has_text=page_image.has_text,
                    has_images=page_image.has_images,
                )
            )

        logger.info(f"Generated ColPali embeddings for {len(results)} pages")
        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Multi-vector query embedding
        """
        return self.client.embed_query(query)

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        page_embedding: ColPaliPageEmbedding,
    ) -> float:
        """Compute MaxSim similarity between query and page.

        Args:
            query_embedding: Query multi-vector
            page_embedding: Page embedding

        Returns:
            Similarity score
        """
        return self.client.compute_similarity(
            query_embedding, page_embedding.embedding
        )

    def rank_pages(
        self,
        query: str,
        page_embeddings: list[ColPaliPageEmbedding],
        top_k: int = 10,
    ) -> list[tuple[ColPaliPageEmbedding, float]]:
        """Rank pages by similarity to query.

        Args:
            query: Search query
            page_embeddings: List of page embeddings
            top_k: Number of results

        Returns:
            List of (page_embedding, score) tuples, sorted by score
        """
        query_embedding = self.embed_query(query)

        scored = []
        for page_emb in page_embeddings:
            score = self.compute_similarity(query_embedding, page_emb)
            scored.append((page_emb, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    async def close(self):
        """Clean up resources."""
        await self.client.close()
