"""Prompt templates for RAG generation."""

from .base import PromptTemplate
from .factual import FactualPrompt
from .graphrag import GraphRAGPrompt
from .raptor import RAPTORPrompt

__all__ = [
    "PromptTemplate",
    "FactualPrompt",
    "GraphRAGPrompt",
    "RAPTORPrompt",
]
