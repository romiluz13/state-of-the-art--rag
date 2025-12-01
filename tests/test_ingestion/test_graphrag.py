"""Tests for GraphRAG entity extraction and community detection."""

import pytest
from unittest.mock import AsyncMock
import json

from src.ingestion.graphrag.entity_extractor import (
    EntityExtractor,
    Entity,
    Relationship,
)
from src.ingestion.graphrag.community import (
    CommunityDetector,
    Community,
)


class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_relationship_creation(self):
        """Test basic relationship creation."""
        rel = Relationship(
            target_entity_id="entity_123",
            relation_type="USES",
            description="Uses for processing",
            weight=0.9,
        )
        assert rel.target_entity_id == "entity_123"
        assert rel.relation_type == "USES"
        assert rel.weight == 0.9


class TestEntity:
    """Tests for Entity dataclass."""

    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            entity_id="ent_1",
            name="MongoDB",
            type="Technology",
            description="A document database",
        )
        assert entity.name == "MongoDB"
        assert entity.type == "Technology"
        assert entity.relationships == []
        assert entity.embedding is None

    def test_entity_with_relationships(self):
        """Test entity with relationships."""
        rel = Relationship(
            target_entity_id="ent_2",
            relation_type="PROVIDES",
            description="Provides storage",
        )
        entity = Entity(
            entity_id="ent_1",
            name="MongoDB",
            type="Technology",
            description="Database",
            relationships=[rel],
        )
        assert len(entity.relationships) == 1
        assert entity.relationships[0].relation_type == "PROVIDES"


class TestEntityExtractor:
    """Tests for EntityExtractor."""

    @pytest.fixture
    def mock_extract_response(self):
        """Mock LLM extraction response."""
        return json.dumps({
            "entities": [
                {
                    "name": "MongoDB",
                    "type": "Technology",
                    "description": "A NoSQL document database"
                },
                {
                    "name": "Python",
                    "type": "Technology",
                    "description": "A programming language"
                }
            ],
            "relationships": [
                {
                    "source": "Python",
                    "target": "MongoDB",
                    "relation_type": "USES",
                    "description": "Python applications use MongoDB"
                }
            ]
        })

    @pytest.fixture
    def mock_generate_function(self, mock_extract_response):
        """Mock generation function."""
        async def generate(prompt):
            return mock_extract_response
        return generate

    @pytest.fixture
    def mock_embed_function(self):
        """Mock embedding function."""
        async def embed(texts):
            return [[0.1] * 1024 for _ in texts]
        return embed

    @pytest.fixture
    def extractor(self, mock_generate_function, mock_embed_function):
        """Create extractor with mocks."""
        return EntityExtractor(
            generate_function=mock_generate_function,
            embed_function=mock_embed_function,
        )

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.generate_function is not None
        assert extractor.embed_function is not None

    @pytest.mark.asyncio
    async def test_extract_from_text(self, extractor):
        """Test entity extraction from text."""
        text = "MongoDB is used by Python applications."
        entities, relationships = await extractor.extract_from_text(
            text, chunk_id="chunk_1"
        )

        assert len(entities) == 2
        assert any(e.name == "MongoDB" for e in entities)
        assert any(e.name == "Python" for e in entities)

    @pytest.mark.asyncio
    async def test_extract_from_chunks(self, extractor):
        """Test extraction from multiple chunks."""
        chunks = [
            ("chunk_1", "MongoDB stores documents."),
            ("chunk_2", "Python connects to databases."),
        ]
        entities = await extractor.extract_from_chunks(chunks)

        # Should extract entities from all chunks
        assert len(entities) > 0

    @pytest.mark.asyncio
    async def test_extraction_with_malformed_json(self):
        """Test handling of malformed JSON response."""
        async def bad_generate(prompt):
            return "This is not valid JSON"

        extractor = EntityExtractor(
            generate_function=bad_generate,
            embed_function=None,
        )

        entities, rels = await extractor.extract_from_text("Test text")
        # Should handle gracefully
        assert entities == []


class TestCommunity:
    """Tests for Community dataclass."""

    def test_community_creation(self):
        """Test basic community creation."""
        community = Community(
            community_id="comm_1",
            level=1,
            title="Technology Stack",
            summary="A group of related technologies.",
            entity_ids=["ent_1", "ent_2"],
        )
        assert community.title == "Technology Stack"
        assert community.level == 1
        assert len(community.entity_ids) == 2


class TestCommunityDetector:
    """Tests for CommunityDetector."""

    @pytest.fixture
    def mock_summarize_function(self):
        """Mock summarization function."""
        async def summarize(prompt):
            return json.dumps({
                "title": "Technology Community",
                "summary": "A group of related technology entities."
            })
        return summarize

    @pytest.fixture
    def mock_embed_function(self):
        """Mock embedding function."""
        async def embed(texts):
            return [[0.1] * 1024 for _ in texts]
        return embed

    @pytest.fixture
    def detector(self, mock_summarize_function, mock_embed_function):
        """Create detector with mocks."""
        return CommunityDetector(
            summarize_function=mock_summarize_function,
            embed_function=mock_embed_function,
            min_community_size=2,
        )

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            Entity(
                entity_id="ent_1",
                name="MongoDB",
                type="Technology",
                description="Database",
                embedding=[0.1] * 1024,
            ),
            Entity(
                entity_id="ent_2",
                name="Python",
                type="Technology",
                description="Language",
                embedding=[0.2] * 1024,
            ),
            Entity(
                entity_id="ent_3",
                name="FastAPI",
                type="Technology",
                description="Framework",
                embedding=[0.15] * 1024,
            ),
        ]

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.min_community_size == 2

    @pytest.mark.asyncio
    async def test_detect_communities_small_set(self, detector):
        """Test community detection with small entity set."""
        entity = Entity(
            entity_id="ent_1",
            name="Test",
            type="Thing",
            description="Test entity",
        )
        communities = await detector.detect_communities([entity])

        # Too few entities for community
        assert communities == []

    @pytest.mark.asyncio
    async def test_detect_communities_with_embeddings(self, detector, sample_entities):
        """Test community detection using embeddings."""
        communities = await detector.detect_communities(sample_entities, level=1)

        # Should create at least one community
        # With min_size=2 and 3 entities, at least one community possible
        assert isinstance(communities, list)

    @pytest.mark.asyncio
    async def test_build_hierarchy(self, detector, sample_entities):
        """Test building community hierarchy."""
        # Need more entities for hierarchy
        more_entities = sample_entities * 3  # 9 entities
        for i, e in enumerate(more_entities):
            e.entity_id = f"ent_{i}"
            e.embedding = [0.1 * (i % 3)] * 1024

        all_communities = await detector.build_hierarchy(
            more_entities, max_levels=2
        )

        assert isinstance(all_communities, list)

    def test_cluster_by_type(self, detector, sample_entities):
        """Test clustering by entity type."""
        # Add a different type
        sample_entities.append(Entity(
            entity_id="ent_4",
            name="John",
            type="Person",
            description="Developer",
        ))

        clusters = detector._cluster_by_type(sample_entities)

        # Should have clusters for Technology and Person
        assert len(clusters) == 2
