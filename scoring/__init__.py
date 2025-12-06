"""Scoring system for benchmark responses."""

from .base import BaseScorer, ScoringResult
from .keyword_scorer import KeywordScorer, is_censored_response, score_response

# Conditional imports for optional dependencies
try:
    from .technical_scorer import TechnicalScorer, extract_technical_terms
    TECHNICAL_AVAILABLE = True
except ImportError:
    TechnicalScorer = None
    extract_technical_terms = None
    TECHNICAL_AVAILABLE = False

try:
    from .llm_judge import LLMJudge
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    LLMJudge = None
    LLM_JUDGE_AVAILABLE = False

try:
    from .hybrid_scorer import HybridScorer, create_hybrid_scorer
    HYBRID_AVAILABLE = True
except ImportError:
    HybridScorer = None
    create_hybrid_scorer = None
    HYBRID_AVAILABLE = False

__all__ = [
    # Base
    "BaseScorer",
    "ScoringResult",
    # Keyword scoring (always available)
    "KeywordScorer",
    "is_censored_response",
    "score_response",
    # Technical scoring (requires sentence-transformers)
    "TechnicalScorer",
    "extract_technical_terms",
    "TECHNICAL_AVAILABLE",
    # LLM Judge (requires httpx, tenacity)
    "LLMJudge",
    "LLM_JUDGE_AVAILABLE",
    # Hybrid scoring
    "HybridScorer",
    "create_hybrid_scorer",
    "HYBRID_AVAILABLE",
]
