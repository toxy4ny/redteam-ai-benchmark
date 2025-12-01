"""LLM API client implementations."""

from typing import Optional

from .base import APIClient
from .lmstudio import LMStudioClient
from .ollama import OllamaClient

__all__ = ["APIClient", "LMStudioClient", "OllamaClient", "create_client"]


def create_client(provider: str, endpoint: Optional[str], model: str) -> APIClient:
    """Create appropriate API client based on provider."""
    # Set default endpoints
    if endpoint is None:
        if provider == "lmstudio":
            endpoint = "http://localhost:1234"
        elif provider == "ollama":
            endpoint = "http://localhost:11434"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Create client
    if provider == "lmstudio":
        return LMStudioClient(endpoint, model)
    elif provider == "ollama":
        return OllamaClient(endpoint, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
