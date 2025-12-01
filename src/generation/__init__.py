"""Generation module for SOTA RAG system.

Implements:
- RAG response generation with Claude
- Citation extraction and verification
- CRAG self-reflection
- Hallucination detection
"""

from .base import (
    GenerationResult,
    GenerationConfig,
    Citation,
    CRAGEvaluation,
    HallucinationCheck,
)
from .generator import Generator
from .prompts import PromptTemplate, FactualPrompt, GraphRAGPrompt, RAPTORPrompt
from .citations import CitationExtractor, CitationVerification
from .crag import CRAGEvaluator, CRAGAction, RelevanceGrade
from .hallucination import HallucinationDetector
from .pipeline import GenerationPipeline, PipelineConfig, PipelineResult

__all__ = [
    # Base types
    "GenerationResult",
    "GenerationConfig",
    "Citation",
    "CRAGEvaluation",
    "HallucinationCheck",
    # Generator
    "Generator",
    # Prompts
    "PromptTemplate",
    "FactualPrompt",
    "GraphRAGPrompt",
    "RAPTORPrompt",
    # Citations
    "CitationExtractor",
    "CitationVerification",
    # CRAG
    "CRAGEvaluator",
    "CRAGAction",
    "RelevanceGrade",
    # Hallucination
    "HallucinationDetector",
    # Pipeline
    "GenerationPipeline",
    "PipelineConfig",
    "PipelineResult",
]
