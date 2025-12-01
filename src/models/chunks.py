"""Chunk document model."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class EmbeddingsModel(BaseModel):
    """Embeddings for a chunk (full and binary quantized)."""

    full: list[float] = Field(..., description="Full precision embedding vector")
    binary: Optional[list[int]] = Field(None, description="Binary quantized embedding (list of 0/1)")
    colpali: Optional[list[list[float]]] = Field(
        None, description="ColPali late interaction vectors"
    )


class HierarchyModel(BaseModel):
    """Hierarchical information for RAPTOR."""

    level: int = Field(..., description="Hierarchy level (0=leaf, 1+=summary)")
    parent_id: Optional[str] = Field(None, description="Parent chunk ID")
    children_ids: list[str] = Field(default_factory=list, description="Child chunk IDs")


class ChunkDocument(BaseModel):
    """Chunk document stored in MongoDB."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Original chunk text")
    contextual_content: Optional[str] = Field(
        None, description="Chunk with surrounding context"
    )
    embeddings: EmbeddingsModel = Field(..., description="Embedding vectors")
    hierarchy: HierarchyModel = Field(..., description="Hierarchical structure")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_xyz",
                "document_id": "doc_abc",
                "content": "MongoDB Atlas provides vector search capabilities...",
                "contextual_content": "In the cloud database section: MongoDB Atlas provides...",
                "embeddings": {
                    "full": [0.1, 0.2, 0.3],
                    "binary": None,
                },
                "hierarchy": {"level": 0, "parent_id": None, "children_ids": []},
                "metadata": {"source": "document.pdf", "page": 5, "token_count": 245},
            }
        }
