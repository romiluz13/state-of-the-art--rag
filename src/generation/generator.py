"""Generator using Claude for answer generation."""

import logging
from typing import Any

import anthropic

from src.config import Settings
from src.retrieval.base import RetrievalResult
from .base import GenerationConfig, GenerationResult, Citation
from .prompts import PromptTemplate, FactualPrompt, GraphRAGPrompt, RAPTORPrompt

logger = logging.getLogger(__name__)


class Generator:
    """Generate answers using Claude with RAG context."""

    # Prompt registry by strategy
    PROMPT_REGISTRY: dict[str, type[PromptTemplate]] = {
        "factual": FactualPrompt,
        "graphrag": GraphRAGPrompt,
        "raptor": RAPTORPrompt,
    }

    def __init__(
        self,
        settings: Settings | None = None,
        config: GenerationConfig | None = None,
    ):
        """Initialize generator.

        Args:
            settings: Application settings (for API key)
            config: Generation configuration
        """
        self.settings = settings or Settings()
        self.config = config or GenerationConfig()

        # Initialize Claude client
        self.client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key,
        )

        # Initialize prompts
        self.prompts: dict[str, PromptTemplate] = {
            name: cls() for name, cls in self.PROMPT_REGISTRY.items()
        }
        self._default_prompt = FactualPrompt()

    async def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        prompt_type: str = "factual",
        **kwargs,
    ) -> GenerationResult:
        """Generate answer from query and context.

        Args:
            query: User question
            context: Retrieved chunks/entities
            prompt_type: Which prompt template to use
            **kwargs: Additional prompt variables

        Returns:
            GenerationResult with answer and citations
        """
        # Select prompt template
        prompt_template = self.prompts.get(prompt_type, self._default_prompt)

        # Format prompt
        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            **kwargs,
        )

        # Get system message
        system_message = prompt_template.get_system_message()

        # Call Claude
        logger.info(f"Generating answer for query: {query[:50]}...")

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_message,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
            )

            answer = response.content[0].text

            # Extract citations from answer
            citations = self._extract_citations(answer, context)

            # Build result
            result = GenerationResult(
                answer=answer,
                query=query,
                citations=citations,
                model=self.config.model,
                prompt_type=prompt_type,
                token_usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

            logger.info(
                f"Generated answer with {len(citations)} citations, "
                f"tokens: {response.usage.input_tokens}+{response.usage.output_tokens}"
            )

            return result

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _extract_citations(
        self,
        answer: str,
        context: list[RetrievalResult],
    ) -> list[Citation]:
        """Extract citations from generated answer.

        Args:
            answer: Generated answer text
            context: Original context chunks

        Returns:
            List of Citation objects
        """
        import re

        citations = []
        seen_ids = set()

        # Find all citation patterns: [1], [2], [S1], [D1], [E1], [C1]
        citation_pattern = r"\[([SDEC]?\d+)\]"
        matches = re.findall(citation_pattern, answer)

        for match in matches:
            if match in seen_ids:
                continue
            seen_ids.add(match)

            # Parse citation type and index
            if match[0].isalpha():
                prefix = match[0]
                index = int(match[1:]) - 1
            else:
                prefix = ""
                index = int(match) - 1

            # Find corresponding context
            chunk = self._find_context_for_citation(
                prefix, index, context
            )

            if chunk:
                citations.append(
                    Citation(
                        citation_id=f"[{match}]",
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.content[:200],  # First 200 chars
                        relevance_score=chunk.score,
                    )
                )

        return citations

    def _find_context_for_citation(
        self,
        prefix: str,
        index: int,
        context: list[RetrievalResult],
    ) -> RetrievalResult | None:
        """Find context chunk for a citation.

        Args:
            prefix: Citation prefix (S, D, E, C, or empty)
            index: 0-based index
            context: Context chunks

        Returns:
            Matching RetrievalResult or None
        """
        if prefix == "S":
            # Summary citations (RAPTOR level > 0)
            summaries = [r for r in context if (r.level or 0) > 0]
            if 0 <= index < len(summaries):
                return summaries[index]

        elif prefix == "D":
            # Detail citations (RAPTOR level 0)
            details = [r for r in context if (r.level or 0) == 0]
            if 0 <= index < len(details):
                return details[index]

        elif prefix == "E":
            # Entity citations (GraphRAG)
            entities = [
                r for r in context
                if r.metadata.get("type") in ("entity", "related_entity")
            ]
            if 0 <= index < len(entities):
                return entities[index]

        elif prefix == "C":
            # Community citations (GraphRAG)
            communities = [
                r for r in context
                if r.metadata.get("type") == "community"
            ]
            if 0 <= index < len(communities):
                return communities[index]

        else:
            # Standard numeric citation
            if 0 <= index < len(context):
                return context[index]

        return None

    async def generate_streaming(
        self,
        query: str,
        context: list[RetrievalResult],
        prompt_type: str = "factual",
        **kwargs,
    ):
        """Generate answer with streaming response.

        Args:
            query: User question
            context: Retrieved chunks
            prompt_type: Prompt template type
            **kwargs: Additional prompt variables

        Yields:
            Text chunks as they're generated
        """
        prompt_template = self.prompts.get(prompt_type, self._default_prompt)
        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            **kwargs,
        )
        system_message = prompt_template.get_system_message()

        with self.client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_message,
            messages=[{"role": "user", "content": formatted_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
