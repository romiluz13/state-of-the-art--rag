"""Text file loader."""

import logging
from pathlib import Path
from typing import BinaryIO

from .base import BaseLoader, LoadedDocument

logger = logging.getLogger(__name__)


class TextLoader(BaseLoader):
    """Loader for plain text files."""

    SUPPORTED_EXTENSIONS = {".txt", ".text"}

    def load(self, source: str | Path | BinaryIO) -> LoadedDocument:
        """Load a text file.

        Args:
            source: Path to text file or file-like object

        Returns:
            LoadedDocument with text content
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            source_name = str(path)
            title = self._get_title_from_path(path)
        else:
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            source_name = getattr(source, "name", "uploaded_file.txt")
            title = Path(source_name).stem

        logger.info(f"Loaded text file: {source_name} ({len(content)} chars)")

        return LoadedDocument(
            content=content,
            source=source_name,
            title=title,
            metadata={"format": "text"},
        )

    def supports(self, source: str | Path) -> bool:
        """Check if source is a text file."""
        path = Path(source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
