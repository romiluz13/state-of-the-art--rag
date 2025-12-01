"""Entity document model for GraphRAG."""

from typing import Optional
from pydantic import BaseModel, Field


class RelationshipModel(BaseModel):
    """Relationship to another entity."""

    target_entity_id: str = Field(..., description="Target entity ID")
    relation_type: str = Field(..., description="Relationship type (e.g., USES, PROVIDES)")
    description: str = Field(..., description="Relationship description")
    weight: float = Field(default=1.0, description="Relationship strength (0-1)")


class EntityDocument(BaseModel):
    """Entity document for knowledge graph."""

    entity_id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type (Person, Organization, Technology, etc.)")
    description: str = Field(..., description="Entity description")
    embedding: list[float] = Field(..., description="Entity embedding vector")
    relationships: list[RelationshipModel] = Field(
        default_factory=list, description="Relationships to other entities"
    )
    source_chunks: list[str] = Field(
        default_factory=list, description="Source chunk IDs"
    )
    community_id: Optional[str] = Field(None, description="Community ID")
    community_level: int = Field(default=0, description="Community hierarchy level")

    class Config:
        json_schema_extra = {
            "example": {
                "entity_id": "entity_123",
                "name": "MongoDB Atlas",
                "type": "Technology",
                "description": "Cloud database platform with vector search",
                "embedding": [0.1, 0.2, 0.3],
                "relationships": [
                    {
                        "target_entity_id": "entity_456",
                        "relation_type": "PROVIDES",
                        "description": "provides vector search capabilities",
                        "weight": 0.9,
                    }
                ],
                "source_chunks": ["chunk_1", "chunk_2"],
                "community_id": "community_5",
                "community_level": 1,
            }
        }
