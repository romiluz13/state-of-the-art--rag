"""Markdown file loader."""

import logging
import re
from pathlib import Path
from typing import BinaryIO

from .base import BaseLoader, LoadedDocument

logger = logging.getLogger(__name__)


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files."""

    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".mdown"}

    def load(self, source: str | Path | BinaryIO) -> LoadedDocument:
        """Load a Markdown file.

        Args:
            source: Path to Markdown file or file-like object

        Returns:
            LoadedDocument with Markdown content
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            source_name = str(path)
            title = self._extract_title(content) or self._get_title_from_path(path)
        else:
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            source_name = getattr(source, "name", "uploaded_file.md")
            title = self._extract_title(content) or Path(source_name).stem

        # Extract metadata
        metadata = {
            "format": "markdown",
            "headings": self._extract_headings(content),
        }

        logger.info(f"Loaded Markdown file: {source_name} ({len(content)} chars)")

        return LoadedDocument(
            content=content,
            source=source_name,
            title=title,
            metadata=metadata,
        )

    def supports(self, source: str | Path) -> bool:
        """Check if source is a Markdown file."""
        path = Path(source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _extract_title(self, content: str) -> str | None:
        """Extract title from first H1 heading."""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_headings(self, content: str) -> list[dict]:
        """Extract all headings from Markdown content."""
        headings = []
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({"level": level, "text": text})
        return headings
