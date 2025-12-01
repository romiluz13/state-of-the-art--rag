"""Main ingestion pipeline orchestrating document processing."""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from src.clients.mongodb import MongoDBClient
from src.clients.voyage import VoyageClient
from src.models.chunks import ChunkDocument, EmbeddingsModel, HierarchyModel
from src.models.documents import DocumentModel

from .chunking import RecursiveChunker, Chunk
from .embeddings import VoyageEmbedder
from .loaders import (
    BaseLoader,
    LoadedDocument,
    PDFLoader,
    MarkdownLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting documents into the RAG system.

    Orchestrates: Load -> Chunk -> Embed -> Store
    """

    def __init__(
        self,
        mongodb_client: MongoDBClient,
        voyage_client: VoyageClient,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        """Initialize the ingestion pipeline.

        Args:
            mongodb_client: MongoDB client for storage
            voyage_client: Voyage client for embeddings
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in characters
        """
        self.mongodb = mongodb_client
        self.voyage = voyage_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedder = VoyageEmbedder(voyage_client)

        # Register loaders
        self.loaders: list[BaseLoader] = [
            PDFLoader(),
            MarkdownLoader(),
            TextLoader(),
        ]

    async def ingest_file(
        self,
        source: str | Path | BinaryIO,
        filename: str | None = None,
    ) -> dict:
        """Ingest a single file through the full pipeline.

        Args:
            source: File path or file-like object
            filename: Optional filename hint for file-like objects

        Returns:
            Dict with document_id, chunks_created, and status
        """
        start_time = datetime.now(timezone.utc)

        # 1. Load document
        loader = self._get_loader(source, filename)
        if not loader:
            raise ValueError(f"No loader found for source: {source}")

        loaded_doc = loader.load(source)
        logger.info(f"Loaded document: {loaded_doc.title} ({loaded_doc.char_count} chars)")

        # 2. Generate document ID
        document_id = f"doc_{uuid.uuid4().hex[:12]}"

        # 3. Chunk document
        chunks = self.chunker.chunk(loaded_doc.content)
        logger.info(f"Created {len(chunks)} chunks")

        if not chunks:
            return {
                "document_id": document_id,
                "chunks_created": 0,
                "status": "empty",
                "message": "Document had no content to chunk",
            }

        # 4. Generate embeddings
        embeddings = await self.embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embeddings)} chunks")

        # 5. Store document metadata
        doc_model = DocumentModel(
            document_id=document_id,
            title=loaded_doc.title or "Untitled",
            source=loaded_doc.source,
            content_hash=self._compute_hash(loaded_doc.content),
            total_chunks=len(chunks),
            metadata={
                **loaded_doc.metadata,
                "chunking_strategy": "recursive",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            created_at=start_time,
            updated_at=start_time,
        )

        await self._store_document(doc_model)

        # 6. Store chunks with embeddings
        chunk_docs = self._create_chunk_documents(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        await self._store_chunks(chunk_docs)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"Ingested document {document_id}: {len(chunks)} chunks in {elapsed:.2f}s"
        )

        return {
            "document_id": document_id,
            "title": loaded_doc.title,
            "source": loaded_doc.source,
            "chunks_created": len(chunks),
            "status": "success",
            "elapsed_seconds": elapsed,
        }

    async def ingest_text(
        self,
        text: str,
        title: str = "Untitled",
        source: str = "direct_input",
        metadata: dict | None = None,
    ) -> dict:
        """Ingest raw text directly.

        Args:
            text: Text content to ingest
            title: Document title
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Dict with document_id, chunks_created, and status
        """
        start_time = datetime.now(timezone.utc)
        document_id = f"doc_{uuid.uuid4().hex[:12]}"

        # Chunk
        chunks = self.chunker.chunk(text)
        if not chunks:
            return {
                "document_id": document_id,
                "chunks_created": 0,
                "status": "empty",
            }

        # Embed
        embeddings = await self.embedder.embed_chunks(chunks)

        # Store document
        doc_model = DocumentModel(
            document_id=document_id,
            title=title,
            source=source,
            content_hash=self._compute_hash(text),
            total_chunks=len(chunks),
            metadata={
                **(metadata or {}),
                "chunking_strategy": "recursive",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            created_at=start_time,
            updated_at=start_time,
        )

        await self._store_document(doc_model)

        # Store chunks
        chunk_docs = self._create_chunk_documents(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
        )
        await self._store_chunks(chunk_docs)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        return {
            "document_id": document_id,
            "title": title,
            "chunks_created": len(chunks),
            "status": "success",
            "elapsed_seconds": elapsed,
        }

    def _get_loader(
        self, source: str | Path | BinaryIO, filename: str | None
    ) -> BaseLoader | None:
        """Find appropriate loader for the source.

        Args:
            source: File source
            filename: Optional filename hint

        Returns:
            Loader instance or None if no match
        """
        # For file-like objects, use filename hint
        if hasattr(source, "read"):
            if filename:
                for loader in self.loaders:
                    if loader.supports(filename):
                        return loader
            # Default to text for unknown file-like objects
            return TextLoader()

        # For paths, check each loader
        for loader in self.loaders:
            if loader.supports(source):
                return loader

        return None

    def _create_chunk_documents(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[dict],
    ) -> list[ChunkDocument]:
        """Create ChunkDocument models from chunks and embeddings.

        Args:
            document_id: Parent document ID
            chunks: List of Chunk objects
            embeddings: List of embedding dicts

        Returns:
            List of ChunkDocument models
        """
        chunk_docs = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i:04d}"

            prev_id = f"{document_id}_chunk_{i-1:04d}" if i > 0 else None
            next_id = (
                f"{document_id}_chunk_{i+1:04d}" if i < len(chunks) - 1 else None
            )

            chunk_doc = ChunkDocument(
                chunk_id=chunk_id,
                document_id=document_id,
                content=chunk.content,
                contextual_content=None,  # Will be added with contextual enhancement
                embeddings=EmbeddingsModel(
                    full=emb["full"],
                    binary=emb.get("binary"),
                ),
                hierarchy=HierarchyModel(
                    level=0,  # Leaf level
                    parent_id=None,
                    children_ids=[],
                ),
                metadata={
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "token_count": chunk.token_count,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "prev_chunk_id": prev_id,
                    "next_chunk_id": next_id,
                },
            )
            chunk_docs.append(chunk_doc)

        return chunk_docs

    async def _store_document(self, doc: DocumentModel) -> None:
        """Store document in MongoDB.

        Args:
            doc: Document model to store
        """
        collection = self.mongodb.db["documents"]
        await collection.insert_one(doc.model_dump())
        logger.debug(f"Stored document: {doc.document_id}")

    async def _store_chunks(self, chunks: list[ChunkDocument]) -> None:
        """Store chunks in MongoDB.

        Args:
            chunks: List of ChunkDocument models
        """
        if not chunks:
            return

        collection = self.mongodb.db["chunks"]
        docs = [chunk.model_dump() for chunk in chunks]
        await collection.insert_many(docs)
        logger.debug(f"Stored {len(chunks)} chunks")

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for deduplication.

        Args:
            content: Text content

        Returns:
            Hash string
        """
        import hashlib

        return hashlib.sha256(content.encode()).hexdigest()[:16]
