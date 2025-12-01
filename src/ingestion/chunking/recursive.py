"""Recursive text chunker with semantic boundary preservation."""

import logging
import re

from .base import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter.

    Splits text hierarchically using multiple separators,
    preserving semantic boundaries where possible.
    """

    DEFAULT_SEPARATORS = [
        "\n\n\n",  # Multiple blank lines (major section break)
        "\n\n",  # Paragraph break
        "\n",  # Line break
        ". ",  # Sentence end
        "? ",  # Question end
        "! ",  # Exclamation end
        "; ",  # Semi-colon
        ", ",  # Comma
        " ",  # Word
        "",  # Character (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        length_function: callable = None,
    ):
        """Initialize recursive chunker.

        Args:
            chunk_size: Target size for chunks (in tokens, approx 4 chars each)
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom list of separators to use
            length_function: Custom function to measure chunk length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self._length_function = length_function or self._count_tokens

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks recursively.

        Args:
            text: Text to split

        Returns:
            List of Chunk objects with metadata
        """
        if not text or not text.strip():
            return []

        # Get raw splits
        splits = self._split_text(text, self.separators)

        # Merge splits into chunks of appropriate size
        chunks = self._merge_splits(splits, text)

        # Update total_chunks for all chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        logger.info(f"Created {len(chunks)} chunks from {len(text)} chars")
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text segments
        """
        final_chunks = []
        separator = separators[-1]  # Default to last separator
        new_separators = []

        # Find the first separator that exists in the text
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Process each split
        good_splits = []
        for split in splits:
            if self._length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Recursively split large pieces
                if good_splits:
                    merged = self._merge_small_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []

                if new_separators:
                    other_splits = self._split_text(split, new_separators)
                    final_chunks.extend(other_splits)
                else:
                    final_chunks.append(split)

        # Don't forget remaining good splits
        if good_splits:
            merged = self._merge_small_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_small_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small splits together up to chunk_size.

        Args:
            splits: List of text pieces
            separator: Separator to join with

        Returns:
            List of merged text pieces
        """
        merged = []
        current = []
        current_length = 0

        for split in splits:
            split_length = self._length_function(split)

            if current_length + split_length > self.chunk_size:
                if current:
                    merged.append(separator.join(current))
                current = [split]
                current_length = split_length
            else:
                current.append(split)
                current_length += split_length + self._length_function(separator)

        if current:
            merged.append(separator.join(current))

        return merged

    def _merge_splits(self, splits: list[str], original_text: str) -> list[Chunk]:
        """Merge splits into final chunks with overlap.

        Args:
            splits: List of text segments
            original_text: Original text for position tracking

        Returns:
            List of Chunk objects
        """
        chunks = []
        current_text = ""
        current_start = 0
        search_start = 0

        for i, split in enumerate(splits):
            potential = current_text + (" " if current_text else "") + split

            if self._length_function(potential) > self.chunk_size and current_text:
                # Create chunk from current text
                start_pos = original_text.find(current_text.strip(), search_start)
                if start_pos == -1:
                    start_pos = current_start
                end_pos = start_pos + len(current_text.strip())

                chunks.append(
                    Chunk(
                        content=current_text.strip(),
                        chunk_index=len(chunks),
                        start_char=start_pos,
                        end_char=end_pos,
                        token_count=self._length_function(current_text),
                    )
                )

                # Calculate overlap
                if self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_text)
                    current_text = overlap_text + " " + split
                    current_start = end_pos - len(overlap_text)
                else:
                    current_text = split
                    current_start = end_pos

                search_start = current_start
            else:
                current_text = potential

        # Add final chunk
        if current_text.strip():
            start_pos = original_text.find(current_text.strip(), search_start)
            if start_pos == -1:
                start_pos = current_start

            chunks.append(
                Chunk(
                    content=current_text.strip(),
                    chunk_index=len(chunks),
                    start_char=start_pos,
                    end_char=start_pos + len(current_text.strip()),
                    token_count=self._length_function(current_text),
                )
            )

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get text for overlap from end of chunk.

        Args:
            text: Full chunk text

        Returns:
            Text to use as overlap
        """
        words = text.split()
        overlap_words = []
        overlap_length = 0

        for word in reversed(words):
            if overlap_length >= self.chunk_overlap:
                break
            overlap_words.insert(0, word)
            overlap_length += len(word) + 1

        return " ".join(overlap_words)
