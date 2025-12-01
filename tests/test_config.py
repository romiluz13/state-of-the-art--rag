"""Tests for configuration module."""

import pytest
from pydantic import ValidationError


def test_settings_loads_from_env(monkeypatch):
    """Test that settings load from environment variables."""
    # Clear the cache
    from src.config.settings import get_settings
    get_settings.cache_clear()

    # Set environment variables
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("DATABASE_NAME", "test_db")

    settings = get_settings()

    assert settings.mongodb_uri == "mongodb://localhost:27017"
    assert settings.voyage_api_key == "test-voyage-key"
    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.database_name == "test_db"


def test_settings_defaults(monkeypatch):
    """Test that settings have correct defaults."""
    from src.config.settings import get_settings
    get_settings.cache_clear()

    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    settings = get_settings()

    assert settings.database_name == "sota_rag"
    assert settings.chunks_collection == "chunks"
    assert settings.entities_collection == "entities"
    assert settings.embedding_dimension == 1024
    assert settings.voyage_embed_model == "voyage-3.5"


def test_settings_missing_required_fields(monkeypatch, tmp_path):
    """Test that missing required fields raise validation error."""
    from src.config.settings import Settings, get_settings
    import os

    # Clear cache and env vars
    get_settings.cache_clear()
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Change to temp dir so .env isn't found
    original_dir = os.getcwd()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)  # Explicitly ignore .env file

    errors = exc_info.value.errors()
    error_fields = [e["loc"][0] for e in errors]

    assert "mongodb_uri" in error_fields
    assert "voyage_api_key" in error_fields
    assert "gemini_api_key" in error_fields
