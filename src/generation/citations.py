"""Citation extraction and verification."""

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.retrieval.base import RetrievalResult
from .base import Citation

logger = logging.getLogger(__name__)


@dataclass
class CitationVerification:
    """Result of citation verification."""

    citation: Citation
    is_valid: bool
    support_score: float  # 0-1 how well the citation supports the claim
    claim_text: str  # The text being cited
    source_text: str  # The source content
    issues: list[str]


class CitationExtractor:
    """Extract and verify citations from generated answers."""

    # Citation patterns for different prompt types
    CITATION_PATTERNS = {
        "standard": r"\[(\d+)\]",
        "summary": r"\[S(\d+)\]",
        "detail": r"\[D(\d+)\]",
        "entity": r"\[E(\d+)\]",
        "community": r"\[C(\d+)\]",
    }

    # Combined pattern for all types
    ALL_CITATIONS_PATTERN = r"\[([SDEC]?\d+)\]"

    def __init__(self, similarity_threshold: float = 0.3):
        """Initialize extractor.

        Args:
            similarity_threshold: Minimum similarity for valid citation
        """
        self.similarity_threshold = similarity_threshold

    def extract_citations(
        self,
        answer: str,
        context: list[RetrievalResult],
    ) -> list[Citation]:
        """Extract citations from answer text.

        Args:
            answer: Generated answer with citations
            context: Original context chunks

        Returns:
            List of extracted citations
        """
        citations = []
        seen_ids = set()

        # Find all citations
        matches = re.finditer(self.ALL_CITATIONS_PATTERN, answer)

        for match in matches:
            citation_id = match.group(1)
            if citation_id in seen_ids:
                continue
            seen_ids.add(citation_id)

            # Parse citation
            prefix, index = self._parse_citation_id(citation_id)

            # Find matching context
            chunk = self._find_context(prefix, index, context)

            if chunk:
                citations.append(
                    Citation(
                        citation_id=f"[{citation_id}]",
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.content[:500],
                        relevance_score=chunk.score,
                    )
                )

        return citations

    def verify_citations(
        self,
        answer: str,
        citations: list[Citation],
        context: list[RetrievalResult],
    ) -> list[CitationVerification]:
        """Verify that citations support the claims they're attached to.

        Args:
            answer: Generated answer
            citations: Extracted citations
            context: Original context

        Returns:
            List of verification results
        """
        verifications = []

        for citation in citations:
            # Find the sentence/claim containing this citation
            claim_text = self._extract_claim_for_citation(
                answer, citation.citation_id
            )

            # Find the source content
            source_text = citation.text

            # Calculate support score
            support_score = self._calculate_support_score(
                claim_text, source_text
            )

            # Identify issues
            issues = []
            if support_score < self.similarity_threshold:
                issues.append("Low semantic similarity between claim and source")

            # Check for factual misalignment
            if self._has_numerical_mismatch(claim_text, source_text):
                issues.append("Potential numerical mismatch")

            verifications.append(
                CitationVerification(
                    citation=citation,
                    is_valid=len(issues) == 0 and support_score >= self.similarity_threshold,
                    support_score=support_score,
                    claim_text=claim_text,
                    source_text=source_text[:200],
                    issues=issues,
                )
            )

            # Update citation verification status
            citation.verified = len(issues) == 0

        return verifications

    def find_uncited_claims(
        self,
        answer: str,
        context: list[RetrievalResult],
    ) -> list[str]:
        """Find factual claims in answer that lack citations.

        Args:
            answer: Generated answer
            context: Original context

        Returns:
            List of potentially uncited claims
        """
        uncited_claims = []

        # Split into sentences
        sentences = re.split(r"[.!?]\s+", answer)

        for sentence in sentences:
            # Skip short sentences or questions
            if len(sentence) < 30 or sentence.strip().endswith("?"):
                continue

            # Check if sentence has a citation
            if not re.search(self.ALL_CITATIONS_PATTERN, sentence):
                # Check if it contains factual indicators
                if self._appears_factual(sentence):
                    uncited_claims.append(sentence.strip())

        return uncited_claims

    def _parse_citation_id(self, citation_id: str) -> tuple[str, int]:
        """Parse citation ID into prefix and index.

        Args:
            citation_id: e.g., "1", "S2", "E3"

        Returns:
            Tuple of (prefix, 0-based index)
        """
        if citation_id[0].isalpha():
            return citation_id[0], int(citation_id[1:]) - 1
        return "", int(citation_id) - 1

    def _find_context(
        self,
        prefix: str,
        index: int,
        context: list[RetrievalResult],
    ) -> RetrievalResult | None:
        """Find context matching citation."""
        if prefix == "S":
            filtered = [r for r in context if (r.level or 0) > 0]
        elif prefix == "D":
            filtered = [r for r in context if (r.level or 0) == 0]
        elif prefix == "E":
            filtered = [
                r for r in context
                if r.metadata.get("type") in ("entity", "related_entity")
            ]
        elif prefix == "C":
            filtered = [
                r for r in context
                if r.metadata.get("type") == "community"
            ]
        else:
            filtered = context

        if 0 <= index < len(filtered):
            return filtered[index]
        return None

    def _extract_claim_for_citation(
        self,
        answer: str,
        citation_id: str,
    ) -> str:
        """Extract the sentence containing a citation.

        Args:
            answer: Full answer text
            citation_id: Citation to find

        Returns:
            Sentence or surrounding text
        """
        # Find citation position
        escaped_id = re.escape(citation_id)
        match = re.search(escaped_id, answer)

        if not match:
            return ""

        pos = match.start()

        # Find sentence boundaries
        # Look backward for sentence start
        start = max(0, answer.rfind(". ", 0, pos) + 2)
        if start <= 2:
            start = 0

        # Look forward for sentence end
        end = answer.find(". ", pos)
        if end == -1:
            end = len(answer)
        else:
            end += 1

        return answer[start:end].strip()

    def _calculate_support_score(
        self,
        claim: str,
        source: str,
    ) -> float:
        """Calculate how well source supports the claim.

        Uses simple sequence matching. In production, use
        embedding similarity or NLI model.

        Args:
            claim: The claim text
            source: The source text

        Returns:
            Support score 0-1
        """
        if not claim or not source:
            return 0.0

        # Normalize texts
        claim_lower = claim.lower()
        source_lower = source.lower()

        # Use sequence matcher for basic similarity
        matcher = SequenceMatcher(None, claim_lower, source_lower)
        ratio = matcher.ratio()

        # Boost if key terms overlap
        claim_words = set(claim_lower.split())
        source_words = set(source_lower.split())

        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been"}
        claim_words -= stop_words
        source_words -= stop_words

        if claim_words:
            overlap = len(claim_words & source_words) / len(claim_words)
            # Weighted average
            return 0.4 * ratio + 0.6 * overlap

        return ratio

    def _has_numerical_mismatch(self, claim: str, source: str) -> bool:
        """Check if numbers in claim match source.

        Args:
            claim: Claim text
            source: Source text

        Returns:
            True if there's a potential mismatch
        """
        # Extract numbers from both
        claim_numbers = set(re.findall(r"\d+\.?\d*", claim))
        source_numbers = set(re.findall(r"\d+\.?\d*", source))

        # If claim has numbers not in source, potential mismatch
        if claim_numbers and not claim_numbers.issubset(source_numbers):
            return True

        return False

    def _appears_factual(self, sentence: str) -> bool:
        """Check if sentence appears to make a factual claim.

        Args:
            sentence: Sentence to check

        Returns:
            True if appears factual
        """
        # Factual indicators
        factual_patterns = [
            r"\d+",  # Contains numbers
            r"\b(is|are|was|were|has|have|had)\b",  # State verbs
            r"\b(percent|percentage|million|billion|thousand)\b",
            r"\b(according to|research shows|studies indicate)\b",
            r"\b(always|never|every|all|none)\b",  # Absolutes
        ]

        for pattern in factual_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True

        return False

    def get_citation_summary(
        self,
        citations: list[Citation],
        verifications: list[CitationVerification] | None = None,
    ) -> dict:
        """Get summary statistics about citations.

        Args:
            citations: List of citations
            verifications: Optional verification results

        Returns:
            Summary dictionary
        """
        summary = {
            "total_citations": len(citations),
            "unique_sources": len(set(c.chunk_id for c in citations)),
            "unique_documents": len(set(c.document_id for c in citations)),
        }

        if verifications:
            valid_count = sum(1 for v in verifications if v.is_valid)
            summary["verified_citations"] = valid_count
            summary["verification_rate"] = valid_count / len(verifications) if verifications else 0
            summary["avg_support_score"] = (
                sum(v.support_score for v in verifications) / len(verifications)
                if verifications else 0
            )

        return summary
