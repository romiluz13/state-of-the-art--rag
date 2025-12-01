"""Base chunker interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    content: str
    chunk_index: int
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Return character count of content."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Return approximate word count."""
        return len(self.content.split())


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to chunk

        Returns:
            List of Chunk objects
        """
        pass

    def _count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token average)."""
        return len(text) // 4
