"""Document model for source documents."""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class PageModel(BaseModel):
    """Page information for multimodal documents."""

    page_num: int = Field(..., description="Page number (0-indexed)")
    colpali_embedding: Optional[list[list[float]]] = Field(
        None, description="ColPali multi-vector embedding (num_patches, embedding_dim)"
    )
    image_width: int = Field(default=0, description="Page image width in pixels")
    image_height: int = Field(default=0, description="Page image height in pixels")
    has_text: bool = Field(default=True, description="Whether page contains text")
    has_images: bool = Field(default=False, description="Whether page contains images")
    has_charts: bool = Field(default=False, description="Whether page contains charts/diagrams")


class DocumentModel(BaseModel):
    """Document metadata stored in MongoDB during ingestion."""

    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    source: str = Field(..., description="Document source (file path or URL)")
    content_hash: str = Field(..., description="Hash of content for deduplication")
    total_chunks: int = Field(..., description="Number of chunks created")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    summary: Optional[str] = Field(None, description="Document-level summary")
    summary_embedding: Optional[list[float]] = Field(None, description="Summary embedding")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123",
                "title": "Architecture Guide",
                "source": "architecture.pdf",
                "content_hash": "a1b2c3d4e5f6g7h8",
                "total_chunks": 15,
                "metadata": {
                    "format": "pdf",
                    "chunking_strategy": "recursive",
                },
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        }


class DocumentDocument(BaseModel):
    """Source document with metadata and multimodal information (for ColPali)."""

    document_id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Document source (file path or URL)")
    summary: Optional[str] = Field(None, description="Document-level summary")
    summary_embedding: Optional[list[float]] = Field(None, description="Summary embedding")
    pages: list[PageModel] = Field(default_factory=list, description="Page information")
    processing: dict[str, Any] = Field(
        default_factory=dict, description="Processing metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc",
                "source": "architecture.pdf",
                "summary": "This document describes the architecture...",
                "summary_embedding": [0.1, 0.2, 0.3],
                "pages": [
                    {
                        "page_num": 0,
                        "colpali_embedding": [[0.1, 0.2], [0.3, 0.4]],
                        "image_width": 800,
                        "image_height": 600,
                        "has_text": True,
                        "has_images": False,
                        "has_charts": True,
                    }
                ],
                "processing": {
                    "chunking_strategy": "raptor",
                    "entities_extracted": 45,
                    "communities_assigned": 3,
                    "colpali_processed": True,
                },
            }
        }
