"""LLM API client implementations."""

import importlib.util
from typing import Optional

from .base import APIClient
from .lmstudio import LMStudioClient
from .ollama import OllamaClient
from .openwebui import OpenWebUIClient

_httpx_available = importlib.util.find_spec("httpx") is not None
_tenacity_available = importlib.util.find_spec("tenacity") is not None

if _httpx_available and _tenacity_available:
    from .openrouter import OpenRouterClient

    OPENROUTER_AVAILABLE = True
else:
    OpenRouterClient = None  # type: ignore
    OPENROUTER_AVAILABLE = False

__all__ = [
    "APIClient",
    "LMStudioClient",
    "OllamaClient",
    "OpenWebUIClient",
    "OpenRouterClient",
    "OPENROUTER_AVAILABLE",
    "create_client",
]


def create_client(
    provider: str,
    endpoint: Optional[str],
    model: str,
    api_key: Optional[str] = None,
) -> APIClient:
    """
    Create appropriate API client based on provider.

    Args:
        provider: Provider name ("lmstudio", "ollama", "openrouter")
        endpoint: Custom endpoint URL (optional)
        model: Model name/ID
        api_key: API key for providers that require it (e.g., OpenRouter)

    Returns:
        Configured APIClient instance
    """
    # Set default endpoints
    if endpoint is None:
        if provider == "lmstudio":
            endpoint = "http://localhost:1234"
        elif provider == "ollama":
            endpoint = "http://localhost:11434"
        elif provider == "openwebui":
            endpoint = "http://localhost:3000"
        elif provider == "openrouter":
            endpoint = "https://openrouter.ai/api/v1"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Create client
    if provider == "lmstudio":
        return LMStudioClient(endpoint, model)
    elif provider == "ollama":
        return OllamaClient(endpoint, model)
    elif provider == "openwebui":
        return OpenWebUIClient(endpoint, model, api_key=api_key)
    elif provider == "openrouter":
        if not OPENROUTER_AVAILABLE:
            raise RuntimeError(
                "OpenRouter requires httpx and tenacity. "
                "Install with: pip install httpx tenacity"
            )
        return OpenRouterClient(endpoint, model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")
