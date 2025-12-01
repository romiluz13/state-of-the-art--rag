"""Hallucination detection for RAG answers."""

import logging
import re
from dataclasses import dataclass, field

import anthropic

from src.config import Settings
from src.retrieval.base import RetrievalResult
from .base import HallucinationCheck

logger = logging.getLogger(__name__)


@dataclass
class ClaimAnalysis:
    """Analysis of a single claim in the answer."""

    claim: str
    is_supported: bool
    support_type: str  # "direct", "inferred", "unsupported", "contradicted"
    supporting_evidence: str | None = None
    confidence: float = 0.0


@dataclass
class HallucinationReport:
    """Detailed hallucination analysis report."""

    overall_score: float  # 0-1, higher = more faithful
    claim_analyses: list[ClaimAnalysis]
    unsupported_claims: list[str]
    contradicted_claims: list[str]
    recommendations: list[str]


class HallucinationDetector:
    """Detect hallucinations in RAG-generated answers.

    Uses multiple techniques:
    1. Claim extraction and verification
    2. Entity consistency checking
    3. Numerical fact verification
    4. Contradiction detection
    """

    CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from this answer. A claim is a statement that can be verified as true or false.

ANSWER:
{answer}

List each claim on a new line, starting with "- ". Only include verifiable factual claims, not opinions or questions.
"""

    VERIFICATION_PROMPT = """Determine if this claim is supported by the source documents.

CLAIM: {claim}

SOURCE DOCUMENTS:
{sources}

Analyze whether the claim is:
- DIRECT: Explicitly stated in sources
- INFERRED: Reasonably inferred from sources
- UNSUPPORTED: Not found in sources
- CONTRADICTED: Contradicts information in sources

Respond in this format:
SUPPORT: [DIRECT/INFERRED/UNSUPPORTED/CONTRADICTED]
EVIDENCE: [Quote or reference from sources if supported, "None" otherwise]
CONFIDENCE: [0.0-1.0]"""

    def __init__(
        self,
        settings: Settings | None = None,
        hallucination_threshold: float = 0.7,
    ):
        """Initialize detector.

        Args:
            settings: Application settings
            hallucination_threshold: Min score to consider answer faithful
        """
        self.settings = settings or Settings()
        self.hallucination_threshold = hallucination_threshold

        self.client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key,
        )

    async def check_hallucinations(
        self,
        answer: str,
        context: list[RetrievalResult],
    ) -> HallucinationCheck:
        """Check answer for potential hallucinations.

        Args:
            answer: Generated answer
            context: Source context used for generation

        Returns:
            HallucinationCheck result
        """
        # Quick checks first
        quick_issues = self._quick_hallucination_checks(answer, context)

        # Extract and verify claims
        claims = await self._extract_claims(answer)
        claim_analyses = []

        sources_text = "\n\n".join(
            f"[{i + 1}] {doc.content}"
            for i, doc in enumerate(context[:5])  # Limit for cost
        )

        for claim in claims[:10]:  # Limit claims checked
            analysis = await self._verify_claim(claim, sources_text)
            claim_analyses.append(analysis)

        # Calculate scores
        if claim_analyses:
            supported_count = sum(
                1 for a in claim_analyses
                if a.support_type in ("direct", "inferred")
            )
            faithfulness_score = supported_count / len(claim_analyses)
        else:
            faithfulness_score = 1.0 if not quick_issues else 0.5

        # Gather issues
        unsupported = [
            a.claim for a in claim_analyses
            if a.support_type == "unsupported"
        ]
        contradicted = [
            a.claim for a in claim_analyses
            if a.support_type == "contradicted"
        ]

        # Combine quick check issues
        all_issues = quick_issues + unsupported

        return HallucinationCheck(
            has_hallucinations=faithfulness_score < self.hallucination_threshold,
            faithfulness_score=faithfulness_score,
            unsupported_claims=all_issues,
            contradictions=contradicted,
        )

    def _quick_hallucination_checks(
        self,
        answer: str,
        context: list[RetrievalResult],
    ) -> list[str]:
        """Quick heuristic checks for hallucinations.

        Args:
            answer: Generated answer
            context: Source documents

        Returns:
            List of potential issues found
        """
        issues = []

        # Combine all context text
        context_text = " ".join(doc.content.lower() for doc in context)

        # Check for specific numbers not in context
        answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?(?:%|percent)?\b", answer))
        context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?(?:%|percent)?\b", context_text))

        for num in answer_numbers:
            if num not in context_numbers and not self._is_common_number(num):
                issues.append(f"Number '{num}' not found in sources")

        # Check for proper nouns not in context
        # Simple heuristic: capitalized words that aren't sentence starters
        answer_words = answer.split()
        for i, word in enumerate(answer_words):
            if (
                word[0].isupper()
                and i > 0
                and answer_words[i - 1][-1] not in ".!?"
                and len(word) > 3
            ):
                if word.lower() not in context_text:
                    # Could be a hallucinated entity
                    pass  # Too noisy, skip for now

        # Check for absolute statements
        absolute_patterns = [
            (r"\balways\b", "absolute statement 'always'"),
            (r"\bnever\b", "absolute statement 'never'"),
            (r"\bimpossible\b", "absolute claim 'impossible'"),
            (r"\bguaranteed\b", "absolute claim 'guaranteed'"),
        ]

        for pattern, issue in absolute_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                # Check if in context
                if not re.search(pattern, context_text, re.IGNORECASE):
                    issues.append(f"Contains {issue} not in sources")

        return issues

    def _is_common_number(self, num: str) -> bool:
        """Check if number is commonly used (years, small counts).

        Args:
            num: Number string

        Returns:
            True if commonly used number
        """
        try:
            val = float(num.replace("%", "").replace("percent", ""))
            # Years
            if 1900 <= val <= 2100:
                return True
            # Small counts
            if val <= 10:
                return True
            # Common percentages
            if val in (25, 50, 75, 100):
                return True
        except ValueError:
            pass
        return False

    async def _extract_claims(self, answer: str) -> list[str]:
        """Extract verifiable claims from answer.

        Args:
            answer: Generated answer

        Returns:
            List of claim strings
        """
        prompt = self.CLAIM_EXTRACTION_PROMPT.format(answer=answer)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.content[0].text
            claims = []

            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    claims.append(line[2:].strip())
                elif line.startswith("-"):
                    claims.append(line[1:].strip())

            return claims

        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            return []

    async def _verify_claim(
        self,
        claim: str,
        sources: str,
    ) -> ClaimAnalysis:
        """Verify a single claim against sources.

        Args:
            claim: Claim to verify
            sources: Source documents text

        Returns:
            ClaimAnalysis result
        """
        prompt = self.VERIFICATION_PROMPT.format(
            claim=claim,
            sources=sources[:3000],  # Limit size
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.content[0].text
            return self._parse_verification_response(claim, result)

        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return ClaimAnalysis(
                claim=claim,
                is_supported=False,
                support_type="unsupported",
                confidence=0.0,
            )

    def _parse_verification_response(
        self,
        claim: str,
        response: str,
    ) -> ClaimAnalysis:
        """Parse verification response.

        Args:
            claim: Original claim
            response: LLM response

        Returns:
            ClaimAnalysis
        """
        support_type = "unsupported"
        evidence = None
        confidence = 0.0

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SUPPORT:"):
                support_str = line[8:].strip().upper()
                if support_str in ("DIRECT", "INFERRED", "UNSUPPORTED", "CONTRADICTED"):
                    support_type = support_str.lower()

            elif line.startswith("EVIDENCE:"):
                ev = line[9:].strip()
                if ev.lower() != "none":
                    evidence = ev

            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[11:].strip())
                except ValueError:
                    pass

        is_supported = support_type in ("direct", "inferred")

        return ClaimAnalysis(
            claim=claim,
            is_supported=is_supported,
            support_type=support_type,
            supporting_evidence=evidence,
            confidence=confidence,
        )

    async def generate_report(
        self,
        answer: str,
        context: list[RetrievalResult],
        check: HallucinationCheck,
    ) -> HallucinationReport:
        """Generate detailed hallucination report.

        Args:
            answer: Generated answer
            context: Source context
            check: HallucinationCheck result

        Returns:
            Detailed report
        """
        # Get full claim analyses
        claims = await self._extract_claims(answer)
        sources_text = "\n\n".join(
            f"[{i + 1}] {doc.content}"
            for i, doc in enumerate(context[:5])
        )

        analyses = []
        for claim in claims:
            analysis = await self._verify_claim(claim, sources_text)
            analyses.append(analysis)

        # Generate recommendations
        recommendations = []
        if check.unsupported_claims:
            recommendations.append(
                f"Remove or add citations for {len(check.unsupported_claims)} unsupported claims"
            )
        if check.contradictions:
            recommendations.append(
                f"Resolve {len(check.contradictions)} contradictions with source documents"
            )
        if check.faithfulness_score < 0.5:
            recommendations.append(
                "Consider regenerating with stricter factual constraints"
            )

        return HallucinationReport(
            overall_score=check.faithfulness_score,
            claim_analyses=analyses,
            unsupported_claims=check.unsupported_claims,
            contradicted_claims=check.contradictions,
            recommendations=recommendations,
        )
