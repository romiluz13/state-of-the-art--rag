"""Base loader interface for document loading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO


@dataclass
class LoadedDocument:
    """Represents a loaded document with its content and metadata."""

    content: str
    source: str
    title: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Return character count of content."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Return approximate word count."""
        return len(self.content.split())


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: str | Path | BinaryIO) -> LoadedDocument:
        """Load a document from the given source.

        Args:
            source: File path, URL, or file-like object

        Returns:
            LoadedDocument with content and metadata
        """
        pass

    @abstractmethod
    def supports(self, source: str | Path) -> bool:
        """Check if this loader supports the given source.

        Args:
            source: File path or URL to check

        Returns:
            True if this loader can handle the source
        """
        pass

    def _get_title_from_path(self, path: Path) -> str:
        """Extract title from file path."""
        return path.stem.replace("-", " ").replace("_", " ").title()
