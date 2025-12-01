"""Application settings using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # MongoDB Settings
    mongodb_uri: str = Field(..., description="MongoDB Atlas connection URI")
    database_name: str = Field(default="sota_rag", description="Database name")

    # Collection Names
    chunks_collection: str = Field(default="chunks", description="Chunks collection name")
    entities_collection: str = Field(default="entities", description="Entities collection name")
    communities_collection: str = Field(
        default="communities", description="Communities collection name"
    )
    documents_collection: str = Field(
        default="documents", description="Documents collection name"
    )
    queries_collection: str = Field(default="queries", description="Queries collection name")

    # Voyage AI Settings
    voyage_api_key: str = Field(..., description="Voyage AI API key")
    voyage_embed_model: str = Field(
        default="voyage-3.5", description="Voyage embedding model"
    )
    voyage_rerank_model: str = Field(
        default="rerank-2.5", description="Voyage reranking model"
    )
    embedding_dimension: int = Field(default=1024, description="Embedding vector dimension")

    # Google Gemini Settings
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.0-flash", description="Gemini model version"
    )

    # Application Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
