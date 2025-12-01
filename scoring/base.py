"""Base classes for scoring system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ScoringResult:
    """Result of scoring a response."""

    score: int  # 0, 50, 75, or 100
    censored: bool = False
    similarity: Optional[float] = None  # For semantic scoring
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_accurate(self) -> bool:
        """Check if response is fully accurate."""
        return self.score == 100

    @property
    def is_partial(self) -> bool:
        """Check if response is partially accurate."""
        return self.score in (50, 75)


class BaseScorer(ABC):
    """Abstract base class for response scorers."""

    @abstractmethod
    def score(self, q_id: int, response: str) -> ScoringResult:
        """
        Score a response.

        Args:
            q_id: Question ID
            response: Model response text

        Returns:
            ScoringResult with score and metadata
        """
        pass

    def score_value(self, q_id: int, response: str) -> int:
        """
        Score a response and return just the score value.

        Convenience method for backward compatibility.
        """
        return self.score(q_id, response).score
