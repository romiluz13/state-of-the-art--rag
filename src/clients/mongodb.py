"""MongoDB Atlas client for SOTA RAG."""

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from src.config import Settings

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB Atlas client with connection pooling and collection access."""

    def __init__(self, settings: Settings):
        """Initialize MongoDB client.

        Args:
            settings: Application settings with MongoDB configuration
        """
        self.settings = settings
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongodb_uri)
        self._db = self._client[settings.database_name]

    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Get database instance."""
        return self._db

    @property
    def chunks(self) -> AsyncIOMotorCollection:
        """Get chunks collection."""
        return self._db[self.settings.chunks_collection]

    @property
    def entities(self) -> AsyncIOMotorCollection:
        """Get entities collection."""
        return self._db[self.settings.entities_collection]

    @property
    def communities(self) -> AsyncIOMotorCollection:
        """Get communities collection."""
        return self._db[self.settings.communities_collection]

    @property
    def documents(self) -> AsyncIOMotorCollection:
        """Get documents collection."""
        return self._db[self.settings.documents_collection]

    @property
    def queries(self) -> AsyncIOMotorCollection:
        """Get queries collection."""
        return self._db[self.settings.queries_collection]

    async def health_check(self) -> bool:
        """Check MongoDB connection health.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            await self._client.admin.command("ping")
            logger.info("MongoDB health check passed")
            return True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return False

    async def close(self):
        """Close MongoDB connection."""
        self._client.close()
        logger.info("MongoDB connection closed")

    async def create_indexes(self) -> dict:
        """Create indexes for all collections.

        Returns:
            Dict with index creation results
        """
        results = {}

        # Documents collection indexes
        results["documents"] = []
        results["documents"].append(
            await self.documents.create_index("document_id", unique=True)
        )
        results["documents"].append(
            await self.documents.create_index("content_hash")
        )
        results["documents"].append(
            await self.documents.create_index("created_at")
        )

        # Chunks collection indexes
        results["chunks"] = []
        results["chunks"].append(
            await self.chunks.create_index("chunk_id", unique=True)
        )
        results["chunks"].append(
            await self.chunks.create_index("document_id")
        )
        results["chunks"].append(
            await self.chunks.create_index([("document_id", 1), ("metadata.chunk_index", 1)])
        )

        # Entities collection indexes (for GraphRAG)
        results["entities"] = []
        results["entities"].append(
            await self.entities.create_index("entity_id", unique=True)
        )
        results["entities"].append(
            await self.entities.create_index("name")
        )
        results["entities"].append(
            await self.entities.create_index("type")
        )

        # Communities collection indexes (for GraphRAG)
        results["communities"] = []
        results["communities"].append(
            await self.communities.create_index("community_id", unique=True)
        )
        results["communities"].append(
            await self.communities.create_index("level")
        )

        # Queries collection indexes (for analytics)
        results["queries"] = []
        results["queries"].append(
            await self.queries.create_index("query_id", unique=True)
        )
        results["queries"].append(
            await self.queries.create_index("created_at")
        )

        logger.info(f"Created indexes: {results}")
        return results
