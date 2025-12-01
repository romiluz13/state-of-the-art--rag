"""Tests for MongoDB client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_mongodb_client_connect():
    """Test MongoDB client connection."""
    from src.clients.mongodb import MongoDBClient
    from src.config import Settings

    settings = Settings(
        mongodb_uri="mongodb://localhost:27017",
        database_name="test_db",
        voyage_api_key="test",
        gemini_api_key="test",
    )

    with patch("src.clients.mongodb.AsyncIOMotorClient") as mock_motor:
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        mock_motor.return_value = mock_client

        client = MongoDBClient(settings)
        health = await client.health_check()

        assert health is True
        mock_client.admin.command.assert_called_once_with("ping")


@pytest.mark.asyncio
async def test_mongodb_client_collections():
    """Test MongoDB client collection access."""
    from src.clients.mongodb import MongoDBClient
    from src.config import Settings

    settings = Settings(
        mongodb_uri="mongodb://localhost:27017",
        database_name="test_db",
        voyage_api_key="test",
        gemini_api_key="test",
    )

    with patch("src.clients.mongodb.AsyncIOMotorClient") as mock_motor:
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_motor.return_value = mock_client

        client = MongoDBClient(settings)

        assert client.chunks is not None
        assert client.entities is not None
        assert client.communities is not None
        assert client.documents is not None
        assert client.queries is not None


@pytest.mark.asyncio
async def test_mongodb_client_health_check_failure():
    """Test MongoDB client health check failure."""
    from src.clients.mongodb import MongoDBClient
    from src.config import Settings

    settings = Settings(
        mongodb_uri="mongodb://localhost:27017",
        database_name="test_db",
        voyage_api_key="test",
        gemini_api_key="test",
    )

    with patch("src.clients.mongodb.AsyncIOMotorClient") as mock_motor:
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(side_effect=Exception("Connection failed"))
        mock_motor.return_value = mock_client

        client = MongoDBClient(settings)
        health = await client.health_check()

        assert health is False
