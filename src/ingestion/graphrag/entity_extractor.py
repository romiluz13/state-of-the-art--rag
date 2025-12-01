"""Entity and relationship extraction for GraphRAG."""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class Relationship:
    """Relationship between entities."""

    target_entity_id: str
    relation_type: str  # USES, PROVIDES, RELATES_TO, PART_OF, etc.
    description: str
    weight: float = 1.0


@dataclass
class Entity:
    """Extracted entity from document."""

    entity_id: str
    name: str
    type: str  # Person, Organization, Technology, Concept, Location, Event
    description: str
    relationships: list[Relationship] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    community_id: str | None = None
    community_level: int = 0


ENTITY_EXTRACTION_PROMPT = """Extract all important entities and their relationships from the following text.

For each entity, provide:
- name: The entity name
- type: One of [Person, Organization, Technology, Concept, Location, Event, Document]
- description: A brief description (1-2 sentences)

For each relationship between entities, provide:
- source: Source entity name
- target: Target entity name
- relation_type: One of [USES, PROVIDES, RELATES_TO, PART_OF, CREATED_BY, CONTAINS, DEPENDS_ON]
- description: Brief description of the relationship

TEXT:
{text}

Respond in JSON format:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "relation_type": "...", "description": "..."}}
  ]
}}

Extract the most important entities and relationships. Be thorough but focused on key concepts."""


class EntityExtractor:
    """Extract entities and relationships from text using LLM."""

    ENTITY_TYPES = ["Person", "Organization", "Technology", "Concept", "Location", "Event", "Document"]
    RELATION_TYPES = ["USES", "PROVIDES", "RELATES_TO", "PART_OF", "CREATED_BY", "CONTAINS", "DEPENDS_ON"]

    def __init__(
        self,
        generate_function: Callable[[str], Awaitable[str]],
        embed_function: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    ):
        """Initialize entity extractor.

        Args:
            generate_function: Async function to generate text (LLM call)
            embed_function: Optional async function to embed entity descriptions
        """
        self.generate_function = generate_function
        self.embed_function = embed_function

    async def extract_from_text(
        self,
        text: str,
        chunk_id: str | None = None,
    ) -> tuple[list[Entity], list[dict]]:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from
            chunk_id: Optional chunk ID for source tracking

        Returns:
            Tuple of (entities, raw_relationships)
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:8000])  # Limit text size

        try:
            response = await self.generate_function(prompt)

            # Parse JSON response
            # Try to find JSON in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                logger.warning("No JSON found in response, returning empty")
                return [], []

            # Process entities
            entities: dict[str, Entity] = {}
            raw_entities = data.get("entities", [])

            for ent in raw_entities:
                name = ent.get("name", "").strip()
                if not name:
                    continue

                entity_id = f"entity_{uuid.uuid4().hex[:12]}"
                entity_type = ent.get("type", "Concept")

                # Validate type
                if entity_type not in self.ENTITY_TYPES:
                    entity_type = "Concept"

                entity = Entity(
                    entity_id=entity_id,
                    name=name,
                    type=entity_type,
                    description=ent.get("description", ""),
                    source_chunks=[chunk_id] if chunk_id else [],
                )
                entities[name] = entity

            # Process relationships
            raw_relationships = data.get("relationships", [])
            relationship_data = []

            for rel in raw_relationships:
                source_name = rel.get("source", "").strip()
                target_name = rel.get("target", "").strip()
                relation_type = rel.get("relation_type", "RELATES_TO")

                if source_name not in entities or target_name not in entities:
                    continue

                # Validate relation type
                if relation_type not in self.RELATION_TYPES:
                    relation_type = "RELATES_TO"

                source_entity = entities[source_name]
                target_entity = entities[target_name]

                relationship = Relationship(
                    target_entity_id=target_entity.entity_id,
                    relation_type=relation_type,
                    description=rel.get("description", ""),
                )

                source_entity.relationships.append(relationship)
                relationship_data.append({
                    "source_id": source_entity.entity_id,
                    "target_id": target_entity.entity_id,
                    "relation_type": relation_type,
                    "description": rel.get("description", ""),
                })

            entity_list = list(entities.values())
            logger.info(f"Extracted {len(entity_list)} entities and {len(relationship_data)} relationships")

            return entity_list, relationship_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return [], []

    async def extract_from_chunks(
        self,
        chunks: list[tuple[str, str]],  # List of (chunk_id, content)
    ) -> list[Entity]:
        """Extract entities from multiple chunks and merge.

        Args:
            chunks: List of (chunk_id, content) tuples

        Returns:
            Merged list of entities
        """
        all_entities: dict[str, Entity] = {}

        for chunk_id, content in chunks:
            entities, _ = await self.extract_from_text(content, chunk_id)

            for entity in entities:
                # Merge by name
                if entity.name in all_entities:
                    existing = all_entities[entity.name]
                    # Merge source chunks
                    existing.source_chunks.extend(entity.source_chunks)
                    # Merge relationships
                    existing.relationships.extend(entity.relationships)
                    # Keep longer description
                    if len(entity.description) > len(existing.description):
                        existing.description = entity.description
                else:
                    all_entities[entity.name] = entity

        entity_list = list(all_entities.values())

        # Optionally embed entities
        if self.embed_function and entity_list:
            descriptions = [f"{e.name}: {e.description}" for e in entity_list]
            embeddings = await self.embed_function(descriptions)

            for i, emb in enumerate(embeddings):
                entity_list[i].embedding = emb

        logger.info(f"Extracted {len(entity_list)} unique entities from {len(chunks)} chunks")
        return entity_list
