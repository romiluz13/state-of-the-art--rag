"""Community document model for GraphRAG."""

from typing import Optional
from pydantic import BaseModel, Field


class CommunityDocument(BaseModel):
    """Community document for GraphRAG hierarchical summaries."""

    community_id: str = Field(..., description="Unique community identifier")
    level: int = Field(..., description="Hierarchical level")
    title: str = Field(..., description="Community title")
    summary: str = Field(..., description="Community summary")
    summary_embedding: list[float] = Field(..., description="Summary embedding vector")
    entity_ids: list[str] = Field(default_factory=list, description="Member entity IDs")
    parent_community_id: Optional[str] = Field(None, description="Parent community ID")
    child_community_ids: list[str] = Field(
        default_factory=list, description="Child community IDs"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "community_id": "community_5",
                "level": 1,
                "title": "Database Technologies",
                "summary": "This community contains entities related to database technologies...",
                "summary_embedding": [0.1, 0.2, 0.3],
                "entity_ids": ["entity_123", "entity_456"],
                "parent_community_id": "community_1",
                "child_community_ids": ["community_10", "community_11"],
            }
        }
