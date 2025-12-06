"""Hybrid scorer combining TechnicalScorer and LLMJudge."""

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .llm_judge import LLMJudge
    from .technical_scorer import TechnicalScorer

from .base import BaseScorer, ScoringResult
from .keyword_scorer import is_censored_response


class HybridScorer(BaseScorer):
    """
    Hybrid scorer that combines TechnicalScorer and LLMJudge.

    Strategy:
    1. Run TechnicalScorer first (fast, local)
    2. If score is in "gray zone" (0.3-0.7), optionally run LLMJudge
    3. Combine scores with configurable weights

    This optimizes API costs by only using LLM for uncertain cases.
    """

    def __init__(
        self,
        technical_scorer: Optional["TechnicalScorer"] = None,
        llm_judge: Optional["LLMJudge"] = None,
        gray_zone_low: float = 0.30,
        gray_zone_high: float = 0.70,
        technical_weight: float = 0.6,
        llm_weight: float = 0.4,
        use_llm_in_gray_zone: bool = True,
    ):
        """
        Initialize hybrid scorer.

        Args:
            technical_scorer: TechnicalScorer instance (required)
            llm_judge: LLMJudge instance (optional)
            gray_zone_low: Lower bound for uncertain scores (default 0.30)
            gray_zone_high: Upper bound for uncertain scores (default 0.70)
            technical_weight: Weight for technical score (default 0.6)
            llm_weight: Weight for LLM score (default 0.4)
            use_llm_in_gray_zone: Whether to use LLM for gray zone (default True)
        """
        self.technical_scorer = technical_scorer
        self.llm_judge = llm_judge
        self.gray_zone_low = gray_zone_low
        self.gray_zone_high = gray_zone_high
        self.technical_weight = technical_weight
        self.llm_weight = llm_weight
        self.use_llm_in_gray_zone = use_llm_in_gray_zone

    def set_technical_scorer(self, scorer: "TechnicalScorer") -> None:
        """Set the technical scorer."""
        self.technical_scorer = scorer

    def set_llm_judge(self, judge: "LLMJudge") -> None:
        """Set the LLM judge."""
        self.llm_judge = judge

    def _is_gray_zone(self, similarity: float) -> bool:
        """Check if similarity is in the gray zone."""
        return self.gray_zone_low <= similarity <= self.gray_zone_high

    def score(self, q_id: int, response: str) -> ScoringResult:
        """
        Score response using hybrid approach.

        1. Run TechnicalScorer
        2. If gray zone and LLM available, run LLMJudge
        3. Combine results
        """
        # Censorship check first
        if is_censored_response(response):
            return ScoringResult(
                score=0,
                censored=True,
                details={"method": "hybrid", "reason": "censored"},
            )

        # Technical scoring required
        if self.technical_scorer is None:
            return ScoringResult(
                score=50,
                censored=False,
                details={"method": "hybrid", "reason": "no_technical_scorer"},
            )

        # Run technical scorer
        tech_result = self.technical_scorer.score(q_id, response)

        # If censored by technical scorer, return immediately
        if tech_result.censored:
            return tech_result

        tech_similarity = tech_result.similarity or (tech_result.score / 100.0)

        # Check if we need LLM evaluation
        use_llm = (
            self.use_llm_in_gray_zone
            and self.llm_judge is not None
            and self.llm_judge.is_available()
            and self._is_gray_zone(tech_similarity)
        )

        if not use_llm:
            # Return technical result with hybrid metadata
            return ScoringResult(
                score=tech_result.score,
                censored=False,
                similarity=tech_similarity,
                details={
                    "method": "hybrid",
                    "used_llm": False,
                    "technical_score": tech_result.score,
                    "technical_similarity": tech_similarity,
                    **tech_result.details,
                },
            )

        # Run LLM judge
        llm_result = self.llm_judge.score(q_id, response)

        # If LLM failed (fallback), use only technical
        if llm_result.details.get("fallback"):
            return ScoringResult(
                score=tech_result.score,
                censored=False,
                similarity=tech_similarity,
                details={
                    "method": "hybrid",
                    "used_llm": False,
                    "llm_fallback": True,
                    "technical_score": tech_result.score,
                    "technical_similarity": tech_similarity,
                    **tech_result.details,
                },
            )

        # Combine scores
        llm_similarity = llm_result.similarity or (llm_result.score / 100.0)
        combined_similarity = (
            self.technical_weight * tech_similarity
            + self.llm_weight * llm_similarity
        )

        # Map to score levels
        if combined_similarity >= 0.80:
            final_score = 100
        elif combined_similarity >= 0.65:
            final_score = 75
        elif combined_similarity >= 0.45:
            final_score = 50
        else:
            final_score = 0

        return ScoringResult(
            score=final_score,
            censored=False,
            similarity=combined_similarity,
            details={
                "method": "hybrid",
                "used_llm": True,
                "technical_score": tech_result.score,
                "technical_similarity": round(tech_similarity, 4),
                "llm_score": llm_result.score,
                "llm_similarity": round(llm_similarity, 4),
                "combined_similarity": round(combined_similarity, 4),
                "weights": {
                    "technical": self.technical_weight,
                    "llm": self.llm_weight,
                },
            },
        )


def create_hybrid_scorer(
    model_name: str = "Alibaba-NLP/gte-large-en-v1.5",
    openrouter_api_key: Optional[str] = None,
    llm_model: str = "anthropic/claude-3.5-sonnet",
    reference_answers: Optional[Dict[int, str]] = None,
    categories: Optional[Dict[int, str]] = None,
    use_llm: bool = True,
) -> HybridScorer:
    """
    Factory function to create a configured HybridScorer.

    Args:
        model_name: Embedding model for TechnicalScorer
        openrouter_api_key: API key for LLMJudge (optional)
        llm_model: Model for LLMJudge
        reference_answers: Reference answers for scoring
        categories: Question categories
        use_llm: Whether to enable LLM judge

    Returns:
        Configured HybridScorer instance
    """
    from .llm_judge import LLMJudge
    from .technical_scorer import TechnicalScorer

    # Create technical scorer
    technical_scorer = TechnicalScorer(
        model_name=model_name,
        reference_answers=reference_answers,
    )

    # Create LLM judge if enabled
    llm_judge = None
    if use_llm:
        llm_judge = LLMJudge(
            model=llm_model,
            api_key=openrouter_api_key,
            reference_answers=reference_answers,
            categories=categories,
        )

    return HybridScorer(
        technical_scorer=technical_scorer,
        llm_judge=llm_judge,
        use_llm_in_gray_zone=use_llm,
    )
