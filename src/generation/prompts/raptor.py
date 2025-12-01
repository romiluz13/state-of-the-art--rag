"""RAPTOR prompt for hierarchical multi-level context."""

from src.retrieval.base import RetrievalResult
from .base import PromptTemplate


class RAPTORPrompt(PromptTemplate):
    """RAPTOR prompt using hierarchical document structure."""

    TEMPLATE = """You have access to information at multiple levels of detail.

HIGH-LEVEL SUMMARIES (Document/Section level):
{summaries}

DETAILED CONTENT (Paragraph level):
{details}

QUESTION: {query}

INSTRUCTIONS:
1. Use high-level summaries for context and overview
2. Use detailed content for specific facts and citations
3. Cite sources using [S1], [S2] for summaries and [D1], [D2] for details
4. Start with the big picture, then provide specific details
5. If summaries and details conflict, note the discrepancy

ANSWER:"""

    def format(
        self,
        query: str,
        context: list[RetrievalResult],
        **kwargs,
    ) -> str:
        # Separate by level
        summaries = []  # Level 1+
        details = []  # Level 0

        for result in context:
            level = result.level or result.metadata.get("raptor_level", 0)
            if level > 0:
                summaries.append(result)
            else:
                details.append(result)

        # Format each section
        summaries_text = self._format_level(summaries, "S") if summaries else "No high-level summaries available."
        details_text = self._format_level(details, "D") if details else "No detailed content available."

        return self.TEMPLATE.format(
            summaries=summaries_text,
            details=details_text,
            query=query,
        )

    def _format_level(self, results: list[RetrievalResult], prefix: str) -> str:
        """Format results with level-specific citations."""
        lines = []
        for i, result in enumerate(results):
            level = result.level or 0
            citation = f"[{prefix}{i + 1}]"
            level_indicator = f"(Level {level})" if level > 0 else "(Detail)"
            lines.append(f"{citation} {level_indicator}: {result.content}")
        return "\n\n".join(lines)

    def get_system_message(self) -> str:
        return """You are an expert at understanding documents at multiple levels of abstraction.

Your approach:
1. Use summaries to understand the big picture
2. Use details to support specific claims
3. Navigate between levels as needed
4. Provide structured answers (overview then details)
5. Cite appropriately at each level"""
