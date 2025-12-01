"""Health check endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.clients import MongoDBClient
from src.api.main import get_mongodb_client

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    mongodb: bool


@router.get("/health", response_model=HealthResponse)
async def health_check(
    mongodb_client: MongoDBClient = Depends(get_mongodb_client),
) -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status including MongoDB connectivity
    """
    mongodb_healthy = await mongodb_client.health_check()

    return HealthResponse(
        status="healthy" if mongodb_healthy else "unhealthy",
        mongodb=mongodb_healthy,
    )
