"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return {
        "mongodb_uri": "mongodb://localhost:27017",
        "database_name": "test_db",
        "voyage_api_key": "test-voyage-key",
        "gemini_api_key": "test-gemini-key",
    }


@pytest.fixture
def mock_mongodb_client():
    """Mock MongoDB client."""
    client = MagicMock()
    client.admin.command = AsyncMock(return_value={"ok": 1})
    return client


@pytest.fixture
def mock_voyage_response():
    """Mock Voyage API response."""
    return {
        "embeddings": [[0.1] * 1024],
        "usage": {"total_tokens": 100},
    }


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""
    return {
        "text": "Test response",
        "usage": {"input_tokens": 50, "output_tokens": 20},
    }
