"""Generation pipeline orchestrating all generation components."""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import Settings
from src.retrieval.base import RetrievalResult
from src.retrieval.pipeline import RetrievalPipeline, RetrievalStrategy

from .base import GenerationConfig, GenerationResult
from .citations import CitationExtractor, CitationVerification
from .crag import CRAGEvaluator, CRAGAction
from .generator import Generator
from .hallucination import HallucinationDetector

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for generation pipeline."""

    # CRAG settings
    enable_crag: bool = True
    max_crag_retries: int = 2

    # Citation settings
    verify_citations: bool = True

    # Hallucination settings
    enable_hallucination_check: bool = True
    hallucination_threshold: float = 0.7

    # Generation settings
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class PipelineResult:
    """Complete pipeline result."""

    generation_result: GenerationResult
    context: list[RetrievalResult]
    citation_verifications: list[CitationVerification] | None = None
    crag_iterations: int = 0
    retrieval_strategy: str = "hybrid"
    metrics: dict[str, Any] = field(default_factory=dict)


class GenerationPipeline:
    """End-to-end pipeline: retrieve -> evaluate -> generate -> verify.

    Orchestrates:
    1. Context retrieval (via RetrievalPipeline)
    2. CRAG evaluation and potential re-retrieval
    3. Answer generation with appropriate prompt
    4. Citation extraction and verification
    5. Hallucination detection
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline | None = None,
        settings: Settings | None = None,
        config: PipelineConfig | None = None,
    ):
        """Initialize generation pipeline.

        Args:
            retrieval_pipeline: Retrieval pipeline instance
            settings: Application settings
            config: Pipeline configuration
        """
        self.settings = settings or Settings()
        self.config = config or PipelineConfig()

        # Initialize components
        self.retrieval = retrieval_pipeline or RetrievalPipeline(settings=self.settings)
        self.generator = Generator(
            settings=self.settings,
            config=self.config.generation_config,
        )
        self.crag_evaluator = CRAGEvaluator(settings=self.settings)
        self.citation_extractor = CitationExtractor()
        self.hallucination_detector = HallucinationDetector(
            settings=self.settings,
            hallucination_threshold=self.config.hallucination_threshold,
        )

    async def generate(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.AUTO,
        prompt_type: str | None = None,
        top_k: int = 10,
        **kwargs,
    ) -> PipelineResult:
        """Run full generation pipeline.

        Args:
            query: User query
            strategy: Retrieval strategy
            prompt_type: Prompt template (auto-detected if None)
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments

        Returns:
            PipelineResult with answer and metadata
        """
        import time
        start_time = time.time()

        # Step 1: Initial retrieval
        logger.info(f"Starting generation pipeline for: {query[:50]}...")

        context = await self.retrieval.retrieve(
            query=query,
            strategy=strategy,
            top_k=top_k,
        )

        actual_strategy = self._detect_strategy(strategy, context)

        # Step 2: CRAG evaluation and potential re-retrieval
        crag_iterations = 0
        current_query = query

        if self.config.enable_crag:
            context, crag_iterations, current_query = await self._crag_loop(
                query=current_query,
                context=context,
                strategy=actual_strategy,
                top_k=top_k,
            )

        # Step 3: Select prompt type based on strategy if not specified
        if prompt_type is None:
            prompt_type = self._select_prompt_type(actual_strategy)

        # Step 4: Generate answer
        generation_result = await self.generator.generate(
            query=query,  # Use original query for answer
            context=context,
            prompt_type=prompt_type,
            **kwargs,
        )

        # Step 5: Citation verification
        citation_verifications = None
        if self.config.verify_citations and generation_result.citations:
            citation_verifications = self.citation_extractor.verify_citations(
                answer=generation_result.answer,
                citations=generation_result.citations,
                context=context,
            )

        # Step 6: Hallucination check
        if self.config.enable_hallucination_check:
            hallucination_check = await self.hallucination_detector.check_hallucinations(
                answer=generation_result.answer,
                context=context,
            )
            generation_result.hallucination_check = hallucination_check

        # Compute metrics
        elapsed_time = time.time() - start_time
        metrics = {
            "total_time_seconds": elapsed_time,
            "context_count": len(context),
            "crag_iterations": crag_iterations,
            "citations_count": len(generation_result.citations),
        }

        if citation_verifications:
            verified_count = sum(1 for v in citation_verifications if v.is_valid)
            metrics["citations_verified"] = verified_count
            metrics["citation_verification_rate"] = (
                verified_count / len(citation_verifications)
                if citation_verifications else 0
            )

        if generation_result.hallucination_check:
            metrics["faithfulness_score"] = (
                generation_result.hallucination_check.faithfulness_score
            )

        logger.info(
            f"Generation complete in {elapsed_time:.2f}s, "
            f"{len(context)} context docs, {len(generation_result.citations)} citations"
        )

        return PipelineResult(
            generation_result=generation_result,
            context=context,
            citation_verifications=citation_verifications,
            crag_iterations=crag_iterations,
            retrieval_strategy=actual_strategy.value if hasattr(actual_strategy, "value") else str(actual_strategy),
            metrics=metrics,
        )

    async def _crag_loop(
        self,
        query: str,
        context: list[RetrievalResult],
        strategy: RetrievalStrategy,
        top_k: int,
    ) -> tuple[list[RetrievalResult], int, str]:
        """Run CRAG evaluation loop.

        Args:
            query: Current query
            context: Current context
            strategy: Retrieval strategy
            top_k: Number of results

        Returns:
            Tuple of (final context, iterations, final query)
        """
        iterations = 0
        current_query = query
        current_context = context

        while iterations < self.config.max_crag_retries:
            # Evaluate context
            evaluation = await self.crag_evaluator.evaluate_context(
                query=current_query,
                context=current_context,
            )

            logger.info(
                f"CRAG iteration {iterations + 1}: "
                f"action={evaluation.action}, confidence={evaluation.confidence_score:.2f}"
            )

            # Decide action
            if evaluation.action == CRAGAction.USE_CONTEXT.value:
                # Context is good, proceed
                current_context = self.crag_evaluator.filter_relevant_context(
                    current_context, evaluation
                )
                break

            elif evaluation.action == CRAGAction.REFINE_QUERY.value:
                # Refine query and re-retrieve
                current_query = await self.crag_evaluator.refine_query(
                    current_query, current_context, evaluation
                )
                current_context = await self.retrieval.retrieve(
                    query=current_query,
                    strategy=strategy,
                    top_k=top_k,
                )
                iterations += 1

            elif evaluation.action == CRAGAction.COMBINE.value:
                # Filter to relevant and continue
                current_context = self.crag_evaluator.filter_relevant_context(
                    current_context, evaluation
                )
                break

            else:
                # WEB_SEARCH or unknown - not implemented, use what we have
                logger.warning(f"Unsupported CRAG action: {evaluation.action}")
                break

        return current_context, iterations, current_query

    def _detect_strategy(
        self,
        strategy: RetrievalStrategy,
        context: list[RetrievalResult],
    ) -> RetrievalStrategy:
        """Detect actual strategy used based on context metadata.

        Args:
            strategy: Requested strategy
            context: Retrieved context

        Returns:
            Actual strategy used
        """
        if strategy != RetrievalStrategy.AUTO:
            return strategy

        # Infer from context metadata
        if context:
            first_meta = context[0].metadata
            if first_meta.get("type") in ("entity", "community"):
                return RetrievalStrategy.GRAPHRAG
            if context[0].level is not None:
                return RetrievalStrategy.RAPTOR

        return RetrievalStrategy.HYBRID

    def _select_prompt_type(self, strategy: RetrievalStrategy) -> str:
        """Select prompt type based on retrieval strategy.

        Args:
            strategy: Retrieval strategy used

        Returns:
            Prompt type name
        """
        strategy_to_prompt = {
            RetrievalStrategy.GRAPHRAG: "graphrag",
            RetrievalStrategy.RAPTOR: "raptor",
        }

        return strategy_to_prompt.get(strategy, "factual")

    async def generate_streaming(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.AUTO,
        prompt_type: str | None = None,
        top_k: int = 10,
        **kwargs,
    ):
        """Stream generation results.

        Yields context and then answer chunks.

        Args:
            query: User query
            strategy: Retrieval strategy
            prompt_type: Prompt template type
            top_k: Documents to retrieve
            **kwargs: Additional arguments

        Yields:
            Dict with type and content
        """
        # Retrieve context
        context = await self.retrieval.retrieve(
            query=query,
            strategy=strategy,
            top_k=top_k,
        )

        # Yield context info
        yield {
            "type": "context",
            "count": len(context),
            "sources": [
                {"chunk_id": c.chunk_id, "score": c.score}
                for c in context[:5]
            ],
        }

        # CRAG evaluation (non-blocking version - just filter)
        if self.config.enable_crag:
            evaluation = await self.crag_evaluator.evaluate_context(query, context)
            context = self.crag_evaluator.filter_relevant_context(context, evaluation)

            yield {
                "type": "crag",
                "action": evaluation.action,
                "confidence": evaluation.confidence_score,
            }

        # Select prompt
        actual_strategy = self._detect_strategy(strategy, context)
        if prompt_type is None:
            prompt_type = self._select_prompt_type(actual_strategy)

        # Stream generation
        yield {"type": "generation_start", "prompt_type": prompt_type}

        async for chunk in self.generator.generate_streaming(
            query=query,
            context=context,
            prompt_type=prompt_type,
            **kwargs,
        ):
            yield {"type": "token", "content": chunk}

        yield {"type": "generation_complete"}
