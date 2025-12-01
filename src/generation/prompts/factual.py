"""Factual RAG prompt with strict citation requirements."""

from src.retrieval.base import RetrievalResult
from .base import PromptTemplate


class FactualPrompt(PromptTemplate):
    """Factual RAG prompt requiring citations for all claims."""

    TEMPLATE = """You are a precise research assistant. Answer the question using ONLY the provided sources.

SOURCES:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Every factual claim MUST have a citation [1], [2], etc.
2. If multiple sources support a claim, cite all of them
3. If the sources don't contain enough information, explicitly state what is missing
4. Do not make claims without source support
5. Organize your answer clearly with proper structure

ANSWER:"""

    def format(
        self,
        query: str,
        context: list[RetrievalResult],
        **kwargs,
    ) -> str:
        formatted_context, _ = self.format_context(context)
        return self.TEMPLATE.format(
            context=formatted_context,
            query=query,
        )

    def get_system_message(self) -> str:
        return """You are a meticulous research assistant focused on accuracy.

Core principles:
1. NEVER make unsupported claims - every statement needs a citation
2. Use [1], [2], [3] format for citations
3. Distinguish between facts (cite) and your analysis (explain reasoning)
4. If sources conflict, acknowledge the disagreement
5. Admit when information is incomplete or uncertain"""
