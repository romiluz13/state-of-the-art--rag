"""Base classes for generation system."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class QualityLevel(str, Enum):
    """Quality assessment levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAILED = "failed"


@dataclass
class Citation:
    """A citation linking response text to source."""

    citation_id: str  # e.g., "[1]"
    chunk_id: str  # Source chunk ID
    document_id: str  # Source document ID
    text: str  # Cited text from chunk
    relevance_score: float = 0.0  # How relevant this citation is
    verified: bool = False  # Whether citation was verified


@dataclass
class GenerationConfig:
    """Configuration for generation."""

    # Model settings
    model: str = "gemini-1.5-flash"
    temperature: float = 0.3
    max_tokens: int = 2048

    # Citation settings
    require_citations: bool = True
    min_citations: int = 1
    max_citations: int = 10

    # Quality settings
    enable_crag: bool = True  # Enable CRAG self-reflection
    enable_hallucination_check: bool = True
    min_quality_score: float = 0.7

    # Retry settings
    max_retries: int = 2
    retry_on_low_quality: bool = True


@dataclass
class CRAGEvaluation:
    """CRAG evaluation result."""

    is_relevant: bool
    confidence_score: float  # 0-1
    action: str  # "use_context", "refine_query", "web_search", "combine"
    reasoning: str
    document_evaluations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HallucinationCheck:
    """Hallucination detection result."""

    has_hallucinations: bool
    faithfulness_score: float  # 0-1, how faithful to sources
    unsupported_claims: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result from generation pipeline."""

    # Core response
    answer: str
    query: str

    # Citations
    citations: list[Citation] = field(default_factory=list)

    # Quality metrics
    crag_evaluation: CRAGEvaluation | None = None
    hallucination_check: HallucinationCheck | None = None

    # Metadata
    model: str = "claude-sonnet-4-20250514"
    prompt_type: str = "factual"
    token_usage: dict[str, int] = field(default_factory=dict)
    generation_time_ms: float = 0.0

    @property
    def quality_score(self) -> float:
        """Overall quality score (0-1)."""
        scores = []
        if self.crag_evaluation:
            scores.append(self.crag_evaluation.confidence_score)
        if self.hallucination_check:
            scores.append(self.hallucination_check.faithfulness_score)
        if self.citations:
            # Citation coverage score
            verified = sum(1 for c in self.citations if c.verified)
            scores.append(verified / len(self.citations) if self.citations else 0)
        return sum(scores) / len(scores) if scores else 0.5

    @property
    def is_high_quality(self) -> bool:
        """Check if response meets quality threshold."""
        return self.quality_score >= 0.7
