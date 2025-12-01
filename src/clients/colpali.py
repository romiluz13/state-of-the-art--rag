"""ColPali client for multimodal document embeddings.

ColPali creates visual embeddings for document pages using late interaction,
enabling search of PDFs by visual content without OCR.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ColPaliConfig:
    """Configuration for ColPali client."""

    model_name: str = "vidore/colpali-v1.2"
    device: str = "cpu"  # "cpu", "cuda", "mps"
    max_image_size: int = 448  # ColPali input size
    batch_size: int = 4


@dataclass
class PageEmbedding:
    """Embedding for a document page."""

    page_num: int
    embeddings: np.ndarray  # Shape: (num_patches, embedding_dim)
    image_size: tuple[int, int]


class ColPaliClient:
    """Client for ColPali multimodal embeddings.

    ColPali uses late interaction (like ColBERT) where:
    - Each document page produces multiple patch embeddings
    - Query also produces multiple token embeddings
    - Similarity is computed via MaxSim over all patches/tokens

    This enables fine-grained visual matching without OCR.
    """

    def __init__(self, config: ColPaliConfig | None = None):
        """Initialize ColPali client.

        Args:
            config: ColPali configuration
        """
        self.config = config or ColPaliConfig()
        self._model = None
        self._processor = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of model."""
        if self._initialized:
            return

        try:
            from colpali_engine.models import ColPali, ColPaliProcessor

            logger.info(f"Loading ColPali model: {self.config.model_name}")

            self._model = ColPali.from_pretrained(
                self.config.model_name,
                torch_dtype="auto",
            )
            self._processor = ColPaliProcessor.from_pretrained(
                self.config.model_name,
            )

            # Move to device
            if self.config.device != "cpu":
                import torch
                device = torch.device(self.config.device)
                self._model = self._model.to(device)

            self._model.eval()
            self._initialized = True
            logger.info("ColPali model loaded successfully")

        except ImportError:
            logger.warning(
                "colpali-engine not installed. Install with: "
                "pip install colpali-engine torch transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load ColPali model: {e}")
            raise

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Generate embedding for a single image.

        Args:
            image: PIL Image of document page

        Returns:
            Multi-vector embedding array (num_patches, embedding_dim)
        """
        self._ensure_initialized()

        import torch

        # Preprocess image
        processed = self._processor.process_images([image])

        # Move to device
        if self.config.device != "cpu":
            device = torch.device(self.config.device)
            processed = {k: v.to(device) for k, v in processed.items()}

        # Generate embeddings
        with torch.no_grad():
            embeddings = self._model(**processed)

        # Convert to numpy
        return embeddings[0].cpu().numpy()

    def embed_images(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Generate embeddings for multiple images.

        Args:
            images: List of PIL Images

        Returns:
            List of multi-vector embeddings
        """
        self._ensure_initialized()

        import torch

        all_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Preprocess batch
            processed = self._processor.process_images(batch)

            # Move to device
            if self.config.device != "cpu":
                device = torch.device(self.config.device)
                processed = {k: v.to(device) for k, v in processed.items()}

            # Generate embeddings
            with torch.no_grad():
                embeddings = self._model(**processed)

            # Convert to numpy and collect
            for emb in embeddings:
                all_embeddings.append(emb.cpu().numpy())

        return all_embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a text query.

        Args:
            query: Search query text

        Returns:
            Multi-vector query embedding (num_tokens, embedding_dim)
        """
        self._ensure_initialized()

        import torch

        # Preprocess query
        processed = self._processor.process_queries([query])

        # Move to device
        if self.config.device != "cpu":
            device = torch.device(self.config.device)
            processed = {k: v.to(device) for k, v in processed.items()}

        # Generate embeddings
        with torch.no_grad():
            embeddings = self._model(**processed)

        return embeddings[0].cpu().numpy()

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        page_embedding: np.ndarray,
    ) -> float:
        """Compute MaxSim similarity between query and page.

        Late interaction scoring:
        - For each query token, find max similarity to any page patch
        - Sum these max similarities

        Args:
            query_embedding: Query multi-vector (num_tokens, dim)
            page_embedding: Page multi-vector (num_patches, dim)

        Returns:
            Similarity score
        """
        # Compute dot product between all query tokens and page patches
        # Shape: (num_tokens, num_patches)
        similarities = np.dot(query_embedding, page_embedding.T)

        # MaxSim: for each query token, take max over page patches
        max_similarities = np.max(similarities, axis=1)

        # Sum over query tokens
        return float(np.sum(max_similarities))

    def rank_pages(
        self,
        query: str,
        page_embeddings: list[np.ndarray],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rank pages by similarity to query.

        Args:
            query: Search query
            page_embeddings: List of page embeddings
            top_k: Number of results to return

        Returns:
            List of (page_index, score) tuples, sorted by score descending
        """
        query_embedding = self.embed_query(query)

        scores = []
        for idx, page_emb in enumerate(page_embeddings):
            score = self.compute_similarity(query_embedding, page_emb)
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    async def close(self):
        """Clean up resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._initialized = False
        logger.info("ColPali client closed")


class MockColPaliClient(ColPaliClient):
    """Mock ColPali client for testing without GPU/model."""

    def __init__(self, config: ColPaliConfig | None = None):
        """Initialize mock client."""
        self.config = config or ColPaliConfig()
        self._initialized = True
        self._embedding_dim = 128  # Mock dimension

    def _ensure_initialized(self):
        """No-op for mock."""
        pass

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Generate mock embedding for image."""
        # Generate deterministic embedding based on image size
        np.random.seed(image.size[0] * image.size[1] % 10000)
        num_patches = 256  # Mock patch count
        return np.random.randn(num_patches, self._embedding_dim).astype(np.float32)

    def embed_images(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Generate mock embeddings for images."""
        return [self.embed_image(img) for img in images]

    def embed_query(self, query: str) -> np.ndarray:
        """Generate mock embedding for query."""
        np.random.seed(hash(query) % 10000)
        num_tokens = len(query.split()) + 1
        return np.random.randn(num_tokens, self._embedding_dim).astype(np.float32)

    async def close(self):
        """No-op for mock."""
        pass
