"""Intent classification for query routing."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Query intent types mapped to retrieval strategies.

    December 2025: Added MULTI_HOP for MCTS-RAG strategy.
    """

    FACTUAL = "factual"  # Specific fact lookup -> hybrid
    GLOBAL = "global"  # Thematic/summary questions -> GraphRAG
    HIERARCHICAL = "hierarchical"  # Document structure -> LeanRAG (was RAPTOR)
    MULTIMODAL = "multimodal"  # Visual content -> ColPali/ColQwen2
    COMPARATIVE = "comparative"  # Compare multiple items -> multi-query
    MULTI_HOP = "multi_hop"  # December 2025: Complex reasoning -> MCTS-RAG


# Intent to strategy mapping (December 2025 updated)
INTENT_TO_STRATEGY = {
    QueryIntent.FACTUAL: "hybrid",
    QueryIntent.GLOBAL: "graphrag",
    QueryIntent.HIERARCHICAL: "leanrag",  # December 2025: was "raptor"
    QueryIntent.MULTIMODAL: "colpali",
    QueryIntent.COMPARATIVE: "hybrid",  # multi-query decomposition + hybrid
    QueryIntent.MULTI_HOP: "mcts",  # December 2025: MCTS-RAG for complex reasoning
}


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: QueryIntent
    confidence: float  # 0.0 to 1.0
    strategy: str
    reasoning: str
    sub_queries: list[str] | None = None  # For comparative queries


class IntentClassifier:
    """Classify query intent to determine optimal retrieval strategy.

    Uses a combination of:
    1. Keyword heuristics (fast, no API call)
    2. LLM classification (accurate, requires API call)
    """

    # Keyword patterns for each intent (December 2025: added MULTI_HOP)
    INTENT_KEYWORDS = {
        QueryIntent.GLOBAL: [
            "what are all", "list all", "summarize", "overview",
            "main themes", "key topics", "throughout", "across",
            "overall", "in general", "commonly", "typically",
            "what themes", "major points", "general summary",
        ],
        QueryIntent.HIERARCHICAL: [
            "chapter", "section", "document structure", "paper",
            "beginning", "end of", "first part", "last part",
            "first section", "last section", "first chapter", "last chapter",
            "introduction", "conclusion", "summary of",
            "table of contents", "outline", "abstract",
            "document summary", "paper summary",  # Combined patterns
        ],
        QueryIntent.MULTIMODAL: [
            "chart", "graph", "diagram", "figure", "image",
            "screenshot", "picture", "visual", "layout",
            "illustration", "infographic", "flowchart",
            "show me the", "what does it look",
            "the table", "data table", "table with",  # Tables (visual)
        ],
        QueryIntent.COMPARATIVE: [
            "compare", "difference between", "versus", "vs",
            "contrast", "how does .* differ", "similarities",
            "pros and cons", "advantages", "disadvantages",
            "which is better", "what are the differences",
        ],
        # December 2025: MULTI_HOP patterns for MCTS-RAG (more specific)
        QueryIntent.MULTI_HOP: [
            "which impacts", "which leads to", "which affects",
            "cascading effects", "chain of", "trace the relationship",
            "implications of", "consequences of",
            "step by step reasoning", "multi-step",
            "lead to", "result in",
            "what caused", "what led to", "what resulted",
            "how does .* affect", "how did .* lead",
        ],
    }

    # Special patterns that override intent classification
    SPECIAL_PATTERNS = {
        "text": ['"'],  # Quoted queries -> text search
    }

    # LLM classification prompt
    CLASSIFICATION_PROMPT = """Classify the following query into one of these intent categories:

1. FACTUAL - Looking for specific facts, definitions, or direct answers
   Examples: "What is the API rate limit?", "Who founded the company?"

2. GLOBAL - Asking about themes, summaries, or patterns across documents
   Examples: "What are the main themes?", "Summarize the key points"

3. HIERARCHICAL - Questions about document structure or specific sections
   Examples: "What does the introduction say?", "What's in chapter 3?"

4. MULTIMODAL - Questions about visual content like charts, diagrams, images
   Examples: "Show me the revenue chart", "What does the diagram show?"

5. COMPARATIVE - Comparing multiple items or asking about differences
   Examples: "Compare Python vs Java", "What are the pros and cons?"

Query: {query}

Respond with ONLY the category name (FACTUAL, GLOBAL, HIERARCHICAL, MULTIMODAL, or COMPARATIVE) and a confidence score 0-100.
Format: CATEGORY|CONFIDENCE|BRIEF_REASON

Example response: FACTUAL|85|Query asks for a specific definition"""

    def __init__(
        self,
        llm_client: Any | None = None,
        use_llm: bool = False,
        confidence_threshold: float = 0.6,
    ):
        """Initialize intent classifier.

        Args:
            llm_client: LLM client for advanced classification
            use_llm: Whether to use LLM (True) or heuristics only (False)
            confidence_threshold: Min confidence to use classified intent
        """
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None
        self.confidence_threshold = confidence_threshold

    def classify(self, query: str) -> IntentResult:
        """Classify query intent.

        Args:
            query: The user's query

        Returns:
            IntentResult with intent, confidence, and strategy
        """
        # First try keyword heuristics (fast)
        heuristic_result = self._classify_by_keywords(query)

        if heuristic_result.confidence >= self.confidence_threshold:
            logger.info(
                f"Intent classified by heuristics: {heuristic_result.intent.value} "
                f"(confidence: {heuristic_result.confidence:.2f})"
            )
            return heuristic_result

        # If LLM available and heuristics uncertain, use LLM
        if self.use_llm and heuristic_result.confidence < self.confidence_threshold:
            llm_result = self._classify_by_llm(query)
            if llm_result.confidence > heuristic_result.confidence:
                logger.info(
                    f"Intent classified by LLM: {llm_result.intent.value} "
                    f"(confidence: {llm_result.confidence:.2f})"
                )
                return llm_result

        # Default to heuristic result or FACTUAL
        if heuristic_result.confidence > 0:
            return heuristic_result

        return IntentResult(
            intent=QueryIntent.FACTUAL,
            confidence=0.5,
            strategy=INTENT_TO_STRATEGY[QueryIntent.FACTUAL],
            reasoning="Default to factual (hybrid) for general queries",
        )

    def _classify_by_keywords(self, query: str) -> IntentResult:
        """Classify using keyword matching."""
        query_lower = query.lower()

        # Check for special patterns that override intent (e.g., quoted queries)
        for strategy, patterns in self.SPECIAL_PATTERNS.items():
            for pattern in patterns:
                if pattern in query:  # Case-sensitive for quotes
                    return IntentResult(
                        intent=QueryIntent.FACTUAL,  # Intent is still factual
                        confidence=0.9,
                        strategy=strategy,  # Override strategy to text
                        reasoning=f"Special pattern matched: '{pattern}' → {strategy} search",
                    )

        # Check each intent's keywords
        best_intent = QueryIntent.FACTUAL
        best_score = 0.0
        best_matches = []

        for intent, keywords in self.INTENT_KEYWORDS.items():
            matches = []
            for kw in keywords:
                if kw in query_lower:
                    matches.append(kw)

            # Score based on number of matches and specificity
            if matches:
                # More matches = higher confidence
                score = min(0.9, 0.5 + len(matches) * 0.15)
                if score > best_score:
                    best_score = score
                    best_intent = intent
                    best_matches = matches

        # Check for comparative patterns with regex
        if best_intent != QueryIntent.COMPARATIVE:
            comparative_patterns = [
                r"compare\s+\w+\s+(and|vs|versus|with)\s+\w+",
                r"difference\s+between\s+\w+\s+and\s+\w+",
                r"\w+\s+vs\.?\s+\w+",
            ]
            for pattern in comparative_patterns:
                if re.search(pattern, query_lower):
                    best_intent = QueryIntent.COMPARATIVE
                    best_score = 0.8
                    best_matches = [pattern]
                    break

        strategy = INTENT_TO_STRATEGY[best_intent]
        reasoning = (
            f"Matched keywords: {best_matches}" if best_matches
            else "No strong keyword matches, defaulting to factual"
        )

        # Extract sub-queries for comparative
        sub_queries = None
        if best_intent == QueryIntent.COMPARATIVE:
            sub_queries = self._extract_comparison_items(query)

        return IntentResult(
            intent=best_intent,
            confidence=best_score if best_score > 0 else 0.5,
            strategy=strategy,
            reasoning=reasoning,
            sub_queries=sub_queries,
        )

    def _classify_by_llm(self, query: str) -> IntentResult:
        """Classify using LLM."""
        try:
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)

            # Call LLM (assumes sync method, adjust for your client)
            response = self.llm_client.generate(prompt)

            # Parse response: CATEGORY|CONFIDENCE|REASON
            parts = response.strip().split("|")
            if len(parts) >= 2:
                intent_str = parts[0].strip().upper()
                confidence = int(parts[1].strip()) / 100.0
                reason = parts[2].strip() if len(parts) > 2 else ""

                intent = QueryIntent(intent_str.lower())
                strategy = INTENT_TO_STRATEGY[intent]

                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    strategy=strategy,
                    reasoning=f"LLM: {reason}",
                )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        # Fallback to factual
        return IntentResult(
            intent=QueryIntent.FACTUAL,
            confidence=0.5,
            strategy=INTENT_TO_STRATEGY[QueryIntent.FACTUAL],
            reasoning="LLM classification failed, defaulting to factual",
        )

    def _extract_comparison_items(self, query: str) -> list[str]:
        """Extract items being compared from query."""
        query_lower = query.lower()

        # Pattern: "compare X and Y" or "X vs Y"
        patterns = [
            r"compare\s+(.+?)\s+(?:and|vs|versus|with)\s+(.+?)(?:\?|$|\s+in)",
            r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                items = [match.group(1).strip(), match.group(2).strip()]
                # Create sub-queries for each item
                return [f"What is {item}?" for item in items]

        return []


class MockLLMClient:
    """Mock LLM client for testing."""

    def generate(self, prompt: str) -> str:
        """Generate mock response based on prompt content."""
        prompt_lower = prompt.lower()

        if "chart" in prompt_lower or "diagram" in prompt_lower:
            return "MULTIMODAL|85|Query asks about visual content"
        elif "compare" in prompt_lower or "vs" in prompt_lower:
            return "COMPARATIVE|80|Query compares multiple items"
        elif "summarize" in prompt_lower or "overview" in prompt_lower:
            return "GLOBAL|75|Query asks for summary"
        elif "chapter" in prompt_lower or "section" in prompt_lower:
            return "HIERARCHICAL|70|Query about document structure"
        else:
            return "FACTUAL|70|General factual query"
