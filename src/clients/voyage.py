"""Voyage AI client for embeddings and reranking."""

import logging
from typing import Literal
import httpx

from src.config import Settings

logger = logging.getLogger(__name__)


class VoyageClient:
    """Voyage AI client for embeddings and reranking."""

    def __init__(self, settings: Settings):
        """Initialize Voyage AI client.

        Args:
            settings: Application settings with Voyage configuration
        """
        self.settings = settings
        self.base_url = "https://api.voyageai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {settings.voyage_api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=60.0)

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        input_type: Literal["query", "document"] = "document",
        output_dtype: Literal["float", "binary"] = "float",
    ) -> dict:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to settings)
            input_type: Type of input (query or document)
            output_dtype: Output format (float or binary for quantization)

        Returns:
            Dictionary with embeddings and usage information
        """
        model = model or self.settings.voyage_embed_model

        payload = {
            "input": texts,
            "model": model,
            "input_type": input_type,
            "output_dtype": output_dtype,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return result
        except Exception as e:
            logger.error(f"Voyage embedding failed: {e}")
            raise

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str | None = None,
        top_k: int | None = None,
    ) -> dict:
        """Rerank documents for a query.

        Args:
            query: Query text
            documents: List of documents to rerank
            model: Model to use (defaults to settings)
            top_k: Number of top results to return

        Returns:
            Dictionary with reranked results
        """
        model = model or self.settings.voyage_rerank_model

        payload = {
            "query": query,
            "documents": documents,
            "model": model,
        }

        if top_k is not None:
            payload["top_k"] = top_k

        try:
            response = await self._client.post(
                f"{self.base_url}/rerank",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Reranked {len(documents)} documents")
            return result
        except Exception as e:
            logger.error(f"Voyage reranking failed: {e}")
            raise

    async def contextualized_embed(
        self,
        documents: list[list[str]],
        model: str = "voyage-context-3",
        input_type: Literal["query", "document"] = "document",
    ) -> dict:
        """Generate contextualized embeddings (late chunking).

        Each document is a list where first element is full doc, rest are chunks.
        Returns embeddings for each chunk with full document context.

        Args:
            documents: List of [full_doc, chunk1, chunk2, ...] lists
            model: Model to use (voyage-context-3 supports this)
            input_type: Type of input

        Returns:
            Dictionary with embeddings per document
        """
        payload = {
            "input": documents,
            "model": model,
            "input_type": input_type,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Generated contextualized embeddings for {len(documents)} documents")
            return result
        except Exception as e:
            logger.error(f"Contextualized embedding failed: {e}")
            raise

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()
        logger.info("Voyage client closed")
