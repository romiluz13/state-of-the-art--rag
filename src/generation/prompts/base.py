"""Base prompt template for RAG generation."""

from abc import ABC, abstractmethod
from typing import Any

from src.retrieval.base import RetrievalResult


class PromptTemplate(ABC):
    """Abstract base class for RAG prompts."""

    # Citation format
    CITATION_FORMAT = "[{index}]"

    @abstractmethod
    def format(
        self,
        query: str,
        context: list[RetrievalResult],
        **kwargs,
    ) -> str:
        """Format the prompt with query and context.

        Args:
            query: User question
            context: Retrieved chunks/entities/communities
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string
        """
        pass

    def format_context(
        self,
        context: list[RetrievalResult],
        max_tokens: int = 8000,
    ) -> tuple[str, dict[str, RetrievalResult]]:
        """Format context chunks with citation markers.

        Args:
            context: List of retrieval results
            max_tokens: Maximum tokens for context

        Returns:
            Tuple of (formatted context string, citation mapping)
        """
        formatted_chunks = []
        citation_map = {}
        current_tokens = 0

        for i, result in enumerate(context):
            # Estimate tokens (rough: 4 chars per token)
            chunk_tokens = len(result.content) // 4

            if current_tokens + chunk_tokens > max_tokens:
                break

            citation_id = self.CITATION_FORMAT.format(index=i + 1)
            citation_map[citation_id] = result

            # Format chunk with citation marker
            chunk_text = f"{citation_id} {result.content}"
            formatted_chunks.append(chunk_text)
            current_tokens += chunk_tokens

        return "\n\n".join(formatted_chunks), citation_map

    def get_system_message(self) -> str:
        """Get system message for the prompt."""
        return """You are a helpful assistant that answers questions based on the provided context.

Key requirements:
1. Only use information from the provided context
2. Cite your sources using [1], [2], etc.
3. If the context doesn't contain the answer, say so clearly
4. Be concise but comprehensive"""


class SimplePrompt(PromptTemplate):
    """Simple RAG prompt for general queries."""

    TEMPLATE = """Based on the following context, answer the question.

Context:
{context}

Question: {query}

Instructions:
- Use information from the context to answer
- Cite sources using [1], [2], etc.
- If you can't answer from the context, say so

Answer:"""

    def format(
        self,
        query: str,
        context: list[RetrievalResult],
        **kwargs,
    ) -> str:
        formatted_context, _ = self.format_context(context)
        return self.TEMPLATE.format(
            context=formatted_context,
            query=query,
        )
