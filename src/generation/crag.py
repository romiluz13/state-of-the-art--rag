"""CRAG (Corrective RAG) self-reflection and evaluation."""

import logging
from dataclasses import dataclass
from enum import Enum

import anthropic

from src.config import Settings
from src.retrieval.base import RetrievalResult
from .base import CRAGEvaluation

logger = logging.getLogger(__name__)


class RelevanceGrade(str, Enum):
    """Relevance grades for retrieved documents."""

    CORRECT = "correct"  # Highly relevant, directly answers query
    AMBIGUOUS = "ambiguous"  # Partially relevant, may need more context
    INCORRECT = "incorrect"  # Not relevant to query


class CRAGAction(str, Enum):
    """Actions based on CRAG evaluation."""

    USE_CONTEXT = "use_context"  # Context is good, proceed
    REFINE_QUERY = "refine_query"  # Rephrase and re-retrieve
    WEB_SEARCH = "web_search"  # Context insufficient, search web
    COMBINE = "combine"  # Mix context with web search


@dataclass
class DocumentEvaluation:
    """Evaluation of a single document's relevance."""

    chunk_id: str
    relevance_grade: RelevanceGrade
    reasoning: str
    key_information: str | None = None
    relevance_score: float = 0.0


class CRAGEvaluator:
    """Evaluate retrieved context using CRAG methodology.

    CRAG (Corrective RAG) uses self-reflection to:
    1. Evaluate relevance of retrieved documents
    2. Decide if re-retrieval or web search is needed
    3. Filter out irrelevant documents
    4. Refine queries for better retrieval
    """

    EVALUATION_PROMPT = """You are evaluating whether a retrieved document is relevant to answering a question.

QUESTION: {query}

DOCUMENT:
{document}

Rate this document's relevance to the question:
- CORRECT: Document directly helps answer the question with specific, relevant information
- AMBIGUOUS: Document is somewhat related but may not fully answer the question
- INCORRECT: Document is not relevant to answering this question

Respond in this exact format:
GRADE: [CORRECT/AMBIGUOUS/INCORRECT]
REASONING: [Brief explanation]
KEY_INFO: [If CORRECT/AMBIGUOUS, the key relevant information; if INCORRECT, write "None"]
SCORE: [0.0-1.0 relevance score]"""

    QUERY_REFINEMENT_PROMPT = """The original query did not retrieve sufficiently relevant documents.

ORIGINAL QUERY: {query}

RETRIEVED DOCUMENTS SUMMARY:
{documents_summary}

ISSUES: {issues}

Generate an improved query that:
1. Is more specific about what information is needed
2. Uses alternative terms or phrasings
3. Breaks down complex questions if needed

Respond with ONLY the refined query, nothing else."""

    def __init__(
        self,
        settings: Settings | None = None,
        relevance_threshold: float = 0.5,
        correct_ratio_threshold: float = 0.3,
    ):
        """Initialize CRAG evaluator.

        Args:
            settings: Application settings
            relevance_threshold: Minimum score to keep document
            correct_ratio_threshold: Min ratio of CORRECT docs to proceed
        """
        self.settings = settings or Settings()
        self.relevance_threshold = relevance_threshold
        self.correct_ratio_threshold = correct_ratio_threshold

        self.client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key,
        )

    async def evaluate_context(
        self,
        query: str,
        context: list[RetrievalResult],
        max_docs_to_evaluate: int = 5,
    ) -> CRAGEvaluation:
        """Evaluate retrieved context relevance.

        Args:
            query: User query
            context: Retrieved documents
            max_docs_to_evaluate: Max docs to evaluate (for cost control)

        Returns:
            CRAGEvaluation with grades and recommended action
        """
        if not context:
            return CRAGEvaluation(
                is_relevant=False,
                confidence_score=0.0,
                action=CRAGAction.WEB_SEARCH.value,
                reasoning="No context retrieved",
            )

        # Evaluate top documents
        docs_to_eval = context[:max_docs_to_evaluate]
        evaluations = []

        for doc in docs_to_eval:
            eval_result = await self._evaluate_single_document(query, doc)
            evaluations.append(eval_result)

        # Analyze results
        correct_count = sum(
            1 for e in evaluations if e.relevance_grade == RelevanceGrade.CORRECT
        )
        ambiguous_count = sum(
            1 for e in evaluations if e.relevance_grade == RelevanceGrade.AMBIGUOUS
        )
        incorrect_count = sum(
            1 for e in evaluations if e.relevance_grade == RelevanceGrade.INCORRECT
        )

        total = len(evaluations)
        correct_ratio = correct_count / total if total > 0 else 0
        avg_score = (
            sum(e.relevance_score for e in evaluations) / total
            if total > 0 else 0
        )

        # Determine action
        if correct_ratio >= self.correct_ratio_threshold:
            action = CRAGAction.USE_CONTEXT
            is_relevant = True
            reasoning = f"Found {correct_count} highly relevant documents"
        elif correct_count > 0 or ambiguous_count > 0:
            action = CRAGAction.COMBINE
            is_relevant = True
            reasoning = f"Partial relevance ({correct_count} correct, {ambiguous_count} ambiguous). Consider supplementing."
        else:
            action = CRAGAction.REFINE_QUERY
            is_relevant = False
            reasoning = f"Low relevance ({incorrect_count}/{total} incorrect). Query refinement recommended."

        return CRAGEvaluation(
            is_relevant=is_relevant,
            confidence_score=avg_score,
            action=action.value,
            reasoning=reasoning,
            document_evaluations=[
                {
                    "chunk_id": e.chunk_id,
                    "grade": e.relevance_grade.value,
                    "score": e.relevance_score,
                    "key_info": e.key_information,
                }
                for e in evaluations
            ],
        )

    async def _evaluate_single_document(
        self,
        query: str,
        document: RetrievalResult,
    ) -> DocumentEvaluation:
        """Evaluate a single document's relevance.

        Args:
            query: User query
            document: Document to evaluate

        Returns:
            DocumentEvaluation result
        """
        prompt = self.EVALUATION_PROMPT.format(
            query=query,
            document=document.content[:2000],  # Limit length
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",  # Use smaller model for eval
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text
            return self._parse_evaluation_response(document.chunk_id, result_text)

        except Exception as e:
            logger.warning(f"Error evaluating document {document.chunk_id}: {e}")
            # Default to ambiguous on error
            return DocumentEvaluation(
                chunk_id=document.chunk_id,
                relevance_grade=RelevanceGrade.AMBIGUOUS,
                reasoning="Evaluation failed",
                relevance_score=0.5,
            )

    def _parse_evaluation_response(
        self,
        chunk_id: str,
        response: str,
    ) -> DocumentEvaluation:
        """Parse LLM evaluation response.

        Args:
            chunk_id: Document chunk ID
            response: LLM response text

        Returns:
            DocumentEvaluation
        """
        lines = response.strip().split("\n")
        grade = RelevanceGrade.AMBIGUOUS
        reasoning = ""
        key_info = None
        score = 0.5

        for line in lines:
            line = line.strip()
            if line.startswith("GRADE:"):
                grade_str = line[6:].strip().upper()
                if grade_str == "CORRECT":
                    grade = RelevanceGrade.CORRECT
                elif grade_str == "INCORRECT":
                    grade = RelevanceGrade.INCORRECT
                else:
                    grade = RelevanceGrade.AMBIGUOUS

            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()

            elif line.startswith("KEY_INFO:"):
                info = line[9:].strip()
                if info.lower() != "none":
                    key_info = info

            elif line.startswith("SCORE:"):
                try:
                    score = float(line[6:].strip())
                except ValueError:
                    pass

        return DocumentEvaluation(
            chunk_id=chunk_id,
            relevance_grade=grade,
            reasoning=reasoning,
            key_information=key_info,
            relevance_score=score,
        )

    async def refine_query(
        self,
        query: str,
        context: list[RetrievalResult],
        evaluation: CRAGEvaluation,
    ) -> str:
        """Generate refined query based on evaluation.

        Args:
            query: Original query
            context: Retrieved documents
            evaluation: CRAG evaluation result

        Returns:
            Refined query string
        """
        # Summarize documents
        docs_summary = "\n".join(
            f"- {doc.content[:200]}..."
            for doc in context[:3]
        )

        prompt = self.QUERY_REFINEMENT_PROMPT.format(
            query=query,
            documents_summary=docs_summary,
            issues=evaluation.reasoning,
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            refined = response.content[0].text.strip()
            logger.info(f"Refined query: '{query}' -> '{refined}'")
            return refined

        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return query  # Return original on failure

    def filter_relevant_context(
        self,
        context: list[RetrievalResult],
        evaluation: CRAGEvaluation,
    ) -> list[RetrievalResult]:
        """Filter context to only relevant documents.

        Args:
            context: All retrieved documents
            evaluation: CRAG evaluation with doc grades

        Returns:
            Filtered list of relevant documents
        """
        if not evaluation.document_evaluations:
            return context

        # Build set of relevant chunk IDs
        relevant_ids = set()
        for doc_eval in evaluation.document_evaluations:
            if doc_eval.get("grade") in ("correct", "ambiguous"):
                if doc_eval.get("score", 0) >= self.relevance_threshold:
                    relevant_ids.add(doc_eval["chunk_id"])

        # Filter context
        filtered = [
            doc for doc in context
            if doc.chunk_id in relevant_ids
        ]

        # Always keep at least some context
        if not filtered and context:
            return context[:2]

        return filtered
