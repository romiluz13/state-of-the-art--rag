"""GraphRAG prompt for global/thematic question synthesis."""

from src.retrieval.base import RetrievalResult
from .base import PromptTemplate


class GraphRAGPrompt(PromptTemplate):
    """GraphRAG prompt for synthesizing answers from entities and communities."""

    TEMPLATE = """You are synthesizing a comprehensive answer from multiple knowledge sources.

ENTITIES (Key concepts and their descriptions):
{entities}

COMMUNITY SUMMARIES (Thematic overviews):
{communities}

SUPPORTING EVIDENCE:
{chunks}

QUESTION: {query}

INSTRUCTIONS:
1. Synthesize information from ALL sources to provide a comprehensive answer
2. Structure your answer thematically, not source-by-source
3. Identify patterns and connections across sources
4. Cite specific sources when making claims [1], [2], etc.
5. Highlight any conflicting information or gaps
6. Provide a balanced, well-rounded perspective

COMPREHENSIVE ANSWER:"""

    def format(
        self,
        query: str,
        context: list[RetrievalResult],
        **kwargs,
    ) -> str:
        # Separate context by type
        entities = []
        communities = []
        chunks = []

        for result in context:
            ctx_type = result.metadata.get("type", "chunk")
            if ctx_type in ("entity", "related_entity"):
                entities.append(result)
            elif ctx_type == "community":
                communities.append(result)
            else:
                chunks.append(result)

        # Format each section
        entities_text = self._format_entities(entities) if entities else "No entities found."
        communities_text = self._format_communities(communities) if communities else "No community summaries available."
        chunks_text, _ = self.format_context(chunks) if chunks else ("No supporting chunks.", {})

        return self.TEMPLATE.format(
            entities=entities_text,
            communities=communities_text,
            chunks=chunks_text,
            query=query,
        )

    def _format_entities(self, entities: list[RetrievalResult]) -> str:
        """Format entity results."""
        lines = []
        for i, entity in enumerate(entities):
            entity_type = entity.metadata.get("entity_type", "Unknown")
            entity_name = entity.metadata.get("entity_name", "")
            citation = f"[E{i + 1}]"
            lines.append(f"{citation} [{entity_type}] {entity_name}: {entity.content}")
        return "\n".join(lines)

    def _format_communities(self, communities: list[RetrievalResult]) -> str:
        """Format community summaries."""
        lines = []
        for i, community in enumerate(communities):
            level = community.metadata.get("community_level", 1)
            citation = f"[C{i + 1}]"
            lines.append(f"{citation} (Level {level}): {community.content}")
        return "\n".join(lines)

    def get_system_message(self) -> str:
        return """You are a knowledge synthesis expert specializing in connecting information across multiple sources.

Your role:
1. Identify overarching themes and patterns
2. Connect related concepts from different sources
3. Provide comprehensive, well-structured answers
4. Balance depth with breadth
5. Acknowledge complexity and nuance"""
