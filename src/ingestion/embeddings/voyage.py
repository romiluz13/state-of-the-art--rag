"""Voyage AI embedder for document chunks."""

import logging
from typing import Literal

from src.clients.voyage import VoyageClient
from src.ingestion.chunking.base import Chunk

logger = logging.getLogger(__name__)


class VoyageEmbedder:
    """Generate embeddings for chunks using Voyage AI."""

    # Voyage batch limits
    MAX_BATCH_SIZE = 128
    MAX_TOKENS_PER_BATCH = 120000

    # Late chunking model
    LATE_CHUNKING_MODEL = "voyage-context-3"

    def __init__(self, voyage_client: VoyageClient):
        """Initialize embedder with Voyage client.

        Args:
            voyage_client: Configured Voyage AI client
        """
        self.client = voyage_client

    async def embed_chunks(
        self,
        chunks: list[Chunk],
        input_type: Literal["query", "document"] = "document",
        include_binary: bool = True,
    ) -> list[dict]:
        """Generate embeddings for a list of chunks.

        Args:
            chunks: List of Chunk objects to embed
            input_type: Type of input (query or document)
            include_binary: Whether to also generate binary quantized embeddings

        Returns:
            List of dicts with 'full' and optionally 'binary' embeddings
        """
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]

        # Process in batches
        all_embeddings = []
        batches = self._create_batches(texts)

        logger.info(f"Embedding {len(chunks)} chunks in {len(batches)} batches")

        for batch_idx, batch in enumerate(batches):
            # Get full precision embeddings
            result = await self.client.embed(
                texts=batch,
                input_type=input_type,
                output_dtype="float",
            )

            batch_embeddings = [
                {"full": item["embedding"], "binary": None}
                for item in result["data"]
            ]

            # Get binary embeddings if requested
            if include_binary:
                binary_result = await self.client.embed(
                    texts=batch,
                    input_type=input_type,
                    output_dtype="binary",
                )

                for i, item in enumerate(binary_result["data"]):
                    batch_embeddings[i]["binary"] = item["embedding"]

            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Completed batch {batch_idx + 1}/{len(batches)}")

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        Args:
            query: Query text

        Returns:
            Embedding vector (full precision)
        """
        result = await self.client.embed(
            texts=[query],
            input_type="query",
            output_dtype="float",
        )

        return result["data"][0]["embedding"]

    def _create_batches(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches respecting Voyage limits.

        Args:
            texts: List of texts to batch

        Returns:
            List of text batches
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            # Approximate token count (4 chars per token)
            text_tokens = len(text) // 4

            # Check if adding this text would exceed limits
            if (
                len(current_batch) >= self.MAX_BATCH_SIZE
                or current_tokens + text_tokens > self.MAX_TOKENS_PER_BATCH
            ):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def embed_with_late_chunking(
        self,
        document: str,
        chunks: list[Chunk],
        include_binary: bool = True,
    ) -> list[dict]:
        """Generate embeddings using late chunking (voyage-context-3).

        Late chunking embeds the full document first, then extracts chunk embeddings
        with full document context preserved.

        Args:
            document: Full document text
            chunks: List of chunks (with start_char and end_char)
            include_binary: Whether to include binary embeddings

        Returns:
            List of dicts with 'full' and optionally 'binary' embeddings
        """
        if not chunks:
            return []

        # For late chunking, we need to send document + chunks together
        # The API returns embeddings for each chunk with document context
        # Format: [[document, chunk1, chunk2, ...]]

        chunk_texts = [chunk.content for chunk in chunks]

        try:
            # Call contextualized embed endpoint
            result = await self.client.contextualized_embed(
                documents=[[document] + chunk_texts],
                model=self.LATE_CHUNKING_MODEL,
                input_type="document",
            )

            # Result contains embeddings for [doc, chunk1, chunk2, ...]
            # We skip the first (document) embedding and use chunk embeddings
            embeddings = result["data"][0]["embeddings"][1:]  # Skip document embedding

            all_embeddings = [{"full": emb, "binary": None} for emb in embeddings]

            # Add binary embeddings if requested
            if include_binary:
                binary_result = await self.client.embed(
                    texts=chunk_texts,
                    model=self.LATE_CHUNKING_MODEL,
                    input_type="document",
                    output_dtype="binary",
                )

                for i, item in enumerate(binary_result["data"]):
                    if i < len(all_embeddings):
                        all_embeddings[i]["binary"] = item["embedding"]

            logger.info(f"Late chunking: embedded {len(chunks)} chunks with document context")
            return all_embeddings

        except Exception as e:
            logger.warning(f"Late chunking failed, falling back to standard: {e}")
            # Fallback to standard embedding
            return await self.embed_chunks(chunks, include_binary=include_binary)

    async def embed_texts(
        self,
        texts: list[str],
        input_type: Literal["query", "document"] = "document",
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Convenience method for embedding arbitrary texts (not Chunk objects).

        Args:
            texts: List of texts to embed
            input_type: Type of input

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        batches = self._create_batches(texts)

        for batch in batches:
            result = await self.client.embed(
                texts=batch,
                input_type=input_type,
                output_dtype="float",
            )

            for item in result["data"]:
                all_embeddings.append(item["embedding"])

        return all_embeddings
