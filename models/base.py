"""Base classes for LLM API clients."""

from abc import ABC, abstractmethod
from typing import Dict, List


class APIClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    @abstractmethod
    def query(self, prompt: str, max_tokens: int = 1024, retries: int = 3) -> str:
        """Query the LLM API with retry logic."""
        pass

    @abstractmethod
    def list_models(self) -> List[Dict]:
        """List available models."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if API is accessible."""
        pass
