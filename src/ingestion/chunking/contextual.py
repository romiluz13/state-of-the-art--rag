"""Contextual chunking - Anthropic method for adding context to chunks."""

import logging
from typing import Callable, Awaitable

from .base import Chunk

logger = logging.getLogger(__name__)


CONTEXTUAL_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (50-100 tokens) to situate this chunk within the overall document for improving search retrieval. The context should:
1. Identify where this chunk fits in the document structure
2. Clarify any pronouns or references
3. Add relevant background that helps understand the chunk

Answer only with the context, nothing else."""


class ContextualChunker:
    """Add contextual information to chunks using LLM.

    Based on Anthropic's Contextual Retrieval technique.
    Reduces retrieval failure rate by ~35%.
    """

    def __init__(
        self,
        generate_function: Callable[[str], Awaitable[str]],
        max_document_tokens: int = 8000,
    ):
        """Initialize contextual chunker.

        Args:
            generate_function: Async function to generate text (LLM call)
            max_document_tokens: Maximum document size to include in prompt
        """
        self.generate_function = generate_function
        self.max_document_tokens = max_document_tokens

    async def add_context_to_chunk(
        self,
        document: str,
        chunk: Chunk,
    ) -> str:
        """Add contextual information to a single chunk.

        Args:
            document: Full document text
            chunk: Chunk to add context to

        Returns:
            Contextualized chunk content (context + original chunk)
        """
        # Truncate document if too long
        doc_truncated = self._truncate_document(document, chunk)

        prompt = CONTEXTUAL_PROMPT.format(
            document=doc_truncated,
            chunk=chunk.content,
        )

        try:
            context = await self.generate_function(prompt)
            context = context.strip()

            # Combine context with chunk
            contextualized = f"{context}\n\n{chunk.content}"

            logger.debug(f"Added {len(context)} char context to chunk {chunk.chunk_index}")
            return contextualized

        except Exception as e:
            logger.warning(f"Failed to generate context for chunk {chunk.chunk_index}: {e}")
            return chunk.content

    async def add_context_to_chunks(
        self,
        document: str,
        chunks: list[Chunk],
    ) -> list[str]:
        """Add contextual information to multiple chunks.

        Args:
            document: Full document text
            chunks: List of chunks to add context to

        Returns:
            List of contextualized chunk contents
        """
        contextualized = []

        for i, chunk in enumerate(chunks):
            content = await self.add_context_to_chunk(document, chunk)
            contextualized.append(content)

            if (i + 1) % 10 == 0:
                logger.info(f"Contextualized {i + 1}/{len(chunks)} chunks")

        logger.info(f"Added context to {len(chunks)} chunks")
        return contextualized

    def _truncate_document(self, document: str, chunk: Chunk) -> str:
        """Truncate document to fit in context window.

        Keeps the chunk's surrounding context when truncating.

        Args:
            document: Full document
            chunk: The chunk we're contextualizing

        Returns:
            Truncated document
        """
        # Approximate tokens (4 chars per token)
        max_chars = self.max_document_tokens * 4

        if len(document) <= max_chars:
            return document

        # Try to keep content around the chunk
        chunk_start = chunk.start_char
        chunk_end = chunk.end_char

        # Calculate how much context we can keep
        available = max_chars - (chunk_end - chunk_start)
        before_context = available // 2
        after_context = available // 2

        start = max(0, chunk_start - before_context)
        end = min(len(document), chunk_end + after_context)

        truncated = document[start:end]

        # Add markers if truncated
        if start > 0:
            truncated = "..." + truncated
        if end < len(document):
            truncated = truncated + "..."

        return truncated


class BatchContextualChunker(ContextualChunker):
    """Batch-optimized contextual chunker for cost efficiency.

    Uses prompt caching when available to reduce costs.
    """

    def __init__(
        self,
        generate_function: Callable[[str], Awaitable[str]],
        batch_generate_function: Callable[[list[str]], Awaitable[list[str]]] | None = None,
        max_document_tokens: int = 8000,
        batch_size: int = 5,
    ):
        """Initialize batch contextual chunker.

        Args:
            generate_function: Async function to generate text
            batch_generate_function: Optional batch generation function
            max_document_tokens: Maximum document size
            batch_size: Number of chunks to process together
        """
        super().__init__(generate_function, max_document_tokens)
        self.batch_generate_function = batch_generate_function
        self.batch_size = batch_size

    async def add_context_to_chunks(
        self,
        document: str,
        chunks: list[Chunk],
    ) -> list[str]:
        """Add context to chunks with batching optimization.

        Args:
            document: Full document text
            chunks: List of chunks

        Returns:
            List of contextualized chunk contents
        """
        if self.batch_generate_function is None:
            # Fall back to sequential processing
            return await super().add_context_to_chunks(document, chunks)

        # Process in batches
        contextualized = []
        doc_truncated = document[:self.max_document_tokens * 4]

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]

            # Create prompts for batch
            prompts = [
                CONTEXTUAL_PROMPT.format(
                    document=doc_truncated,
                    chunk=chunk.content,
                )
                for chunk in batch
            ]

            try:
                contexts = await self.batch_generate_function(prompts)

                for j, context in enumerate(contexts):
                    chunk = batch[j]
                    contextualized.append(f"{context.strip()}\n\n{chunk.content}")

            except Exception as e:
                logger.warning(f"Batch context generation failed: {e}")
                # Fall back to sequential for this batch
                for chunk in batch:
                    content = await self.add_context_to_chunk(document, chunk)
                    contextualized.append(content)

            logger.info(f"Contextualized {min(i + self.batch_size, len(chunks))}/{len(chunks)} chunks")

        return contextualized
