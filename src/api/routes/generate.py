"""Generate endpoint for RAG answer generation."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.generation import (
    GenerationPipeline,
    PipelineConfig,
    GenerationConfig,
)
from src.retrieval import (
    RetrievalPipeline,
    RetrievalStrategy,
    RetrievalConfig,
)
from src.clients.mongodb import MongoDBClient
from src.clients.voyage import VoyageClient
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["generate"])


class GenerateRequest(BaseModel):
    """Generation request."""

    query: str = Field(..., min_length=1, description="The user question")
    strategy: str | None = Field(
        default="auto",
        description="Retrieval strategy: vector, text, hybrid, graphrag, raptor, auto",
    )
    prompt_type: str | None = Field(
        default=None,
        description="Prompt template: factual, graphrag, raptor (auto-detected if None)",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Documents to retrieve")
    enable_crag: bool = Field(default=True, description="Enable CRAG self-reflection")
    enable_hallucination_check: bool = Field(
        default=True, description="Enable hallucination detection"
    )
    verify_citations: bool = Field(default=True, description="Verify citation accuracy")
    stream: bool = Field(default=False, description="Stream the response")


class CitationInfo(BaseModel):
    """Citation information."""

    citation_id: str
    chunk_id: str
    document_id: str
    text: str
    relevance_score: float
    verified: bool = False


class SourceInfo(BaseModel):
    """Source document info."""

    chunk_id: str
    document_id: str
    content: str
    score: float


class HallucinationInfo(BaseModel):
    """Hallucination check info."""

    has_hallucinations: bool
    faithfulness_score: float
    unsupported_claims: list[str]
    contradictions: list[str]


class CRAGInfo(BaseModel):
    """CRAG evaluation info."""

    is_relevant: bool
    confidence_score: float
    action: str
    reasoning: str


class GenerateResponse(BaseModel):
    """Generation response."""

    query: str
    answer: str
    citations: list[CitationInfo]
    sources: list[SourceInfo]
    strategy: str
    prompt_type: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    crag_evaluation: CRAGInfo | None = None
    hallucination_check: HallucinationInfo | None = None


# Global clients
_mongodb_client: MongoDBClient | None = None
_voyage_client: VoyageClient | None = None
_generation_pipeline: GenerationPipeline | None = None


def get_generation_pipeline(
    enable_crag: bool = True,
    enable_hallucination_check: bool = True,
    verify_citations: bool = True,
) -> GenerationPipeline:
    """Get or create generation pipeline."""
    global _generation_pipeline, _mongodb_client, _voyage_client

    settings = get_settings()

    if _mongodb_client is None:
        _mongodb_client = MongoDBClient(settings)

    if _voyage_client is None:
        _voyage_client = VoyageClient(settings.voyage_api_key)

    # Create retrieval pipeline
    retrieval_config = RetrievalConfig(
        top_k=10,
        use_binary_quantization=True,
        rerank=True,
    )
    retrieval_pipeline = RetrievalPipeline(
        _mongodb_client, _voyage_client, retrieval_config
    )

    # Create pipeline config
    pipeline_config = PipelineConfig(
        enable_crag=enable_crag,
        verify_citations=verify_citations,
        enable_hallucination_check=enable_hallucination_check,
        generation_config=GenerationConfig(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            temperature=0.3,
        ),
    )

    return GenerationPipeline(
        retrieval_pipeline=retrieval_pipeline,
        settings=settings,
        config=pipeline_config,
    )


def parse_strategy(strategy_str: str) -> RetrievalStrategy:
    """Parse strategy string to enum."""
    strategy_map = {
        "vector": RetrievalStrategy.VECTOR,
        "text": RetrievalStrategy.TEXT,
        "hybrid": RetrievalStrategy.HYBRID,
        "graphrag": RetrievalStrategy.GRAPHRAG,
        "raptor": RetrievalStrategy.RAPTOR,
        "auto": RetrievalStrategy.AUTO,
    }
    return strategy_map.get(strategy_str.lower(), RetrievalStrategy.AUTO)


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate an answer with full RAG pipeline.

    This endpoint:
    1. Retrieves relevant context using the specified strategy
    2. Evaluates context relevance with CRAG (if enabled)
    3. Generates an answer using Claude with appropriate prompt
    4. Extracts and verifies citations
    5. Checks for hallucinations (if enabled)

    Returns a comprehensive response with answer, citations, and quality metrics.
    """
    try:
        pipeline = get_generation_pipeline(
            enable_crag=request.enable_crag,
            enable_hallucination_check=request.enable_hallucination_check,
            verify_citations=request.verify_citations,
        )
        strategy = parse_strategy(request.strategy or "auto")

        # Run generation pipeline
        result = await pipeline.generate(
            query=request.query,
            strategy=strategy,
            prompt_type=request.prompt_type,
            top_k=request.top_k,
        )

        # Build response
        citations = [
            CitationInfo(
                citation_id=c.citation_id,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text[:300],  # Truncate for response
                relevance_score=c.relevance_score,
                verified=c.verified,
            )
            for c in result.generation_result.citations
        ]

        sources = [
            SourceInfo(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content=c.content[:500],  # Truncate for response
                score=c.score,
            )
            for c in result.context[:5]  # Top 5 sources
        ]

        # CRAG info
        crag_info = None
        if result.generation_result.crag_evaluation:
            crag = result.generation_result.crag_evaluation
            crag_info = CRAGInfo(
                is_relevant=crag.is_relevant,
                confidence_score=crag.confidence_score,
                action=crag.action,
                reasoning=crag.reasoning,
            )

        # Hallucination info
        hallucination_info = None
        if result.generation_result.hallucination_check:
            h = result.generation_result.hallucination_check
            hallucination_info = HallucinationInfo(
                has_hallucinations=h.has_hallucinations,
                faithfulness_score=h.faithfulness_score,
                unsupported_claims=h.unsupported_claims[:5],  # Limit
                contradictions=h.contradictions[:3],
            )

        return GenerateResponse(
            query=request.query,
            answer=result.generation_result.answer,
            citations=citations,
            sources=sources,
            strategy=result.retrieval_strategy,
            prompt_type=result.generation_result.prompt_type,
            metrics=result.metrics,
            crag_evaluation=crag_info,
            hallucination_check=hallucination_info,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Stream generation response.

    Returns Server-Sent Events (SSE) with:
    - context: Retrieved sources info
    - crag: CRAG evaluation result
    - generation_start: Generation starting
    - token: Individual tokens
    - generation_complete: Generation finished
    """
    import json

    async def event_generator():
        try:
            pipeline = get_generation_pipeline(
                enable_crag=request.enable_crag,
                enable_hallucination_check=False,  # Disable for streaming
                verify_citations=False,
            )
            strategy = parse_strategy(request.strategy or "auto")

            async for event in pipeline.generate_streaming(
                query=request.query,
                strategy=strategy,
                prompt_type=request.prompt_type,
                top_k=request.top_k,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/generate/prompts")
async def list_prompts() -> dict[str, Any]:
    """List available prompt templates with descriptions."""
    return {
        "prompts": [
            {
                "name": "factual",
                "description": "Strict factual prompt requiring citations for all claims",
                "best_for": "Research queries requiring verifiable answers",
                "features": ["Mandatory citations", "Source verification", "Uncertainty acknowledgment"],
            },
            {
                "name": "graphrag",
                "description": "Synthesis prompt for entity and community-based context",
                "best_for": "Global/thematic questions about topics",
                "features": ["Entity-aware", "Community summaries", "Cross-source synthesis"],
            },
            {
                "name": "raptor",
                "description": "Hierarchical prompt using summaries and details",
                "best_for": "Questions about document structure or requiring overview + detail",
                "features": ["Multi-level context", "Summary/detail separation", "Hierarchical citations"],
            },
        ],
        "auto_selection": {
            "description": "Prompt is auto-selected based on retrieval strategy if not specified",
            "mapping": {
                "graphrag strategy": "graphrag prompt",
                "raptor strategy": "raptor prompt",
                "other strategies": "factual prompt",
            },
        },
    }


@router.get("/generate/config")
async def get_config() -> dict[str, Any]:
    """Get current generation configuration."""
    settings = get_settings()
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2048,
        "temperature": 0.3,
        "features": {
            "crag_enabled": True,
            "hallucination_check_enabled": True,
            "citation_verification_enabled": True,
        },
        "crag": {
            "max_retries": 2,
            "relevance_threshold": 0.5,
            "correct_ratio_threshold": 0.3,
        },
        "hallucination": {
            "threshold": 0.7,
        },
    }
