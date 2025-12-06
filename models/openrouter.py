"""OpenRouter API client for accessing multiple LLM providers."""

import os
from typing import Dict, List, Optional

from .base import APIClient

# Optional httpx support
try:
    import httpx
    from tenacity import retry, stop_after_attempt, wait_exponential

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class OpenRouterClient(APIClient):
    """
    OpenRouter API client supporting 100+ LLM models.

    OpenRouter provides unified access to various LLM providers
    (OpenAI, Anthropic, Google, Meta, etc.) through a single API.
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        base_url: str = OPENROUTER_URL,
        model_name: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Initialize OpenRouter client.

        Args:
            base_url: OpenRouter API URL (default: https://openrouter.ai/api/v1)
            model_name: Model ID (e.g., "anthropic/claude-3.5-sonnet")
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds
        """
        super().__init__(base_url, model_name)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.timeout = timeout

        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx and tenacity are required for OpenRouter. "
                "Install with: pip install httpx tenacity"
            )

        if not self.api_key:
            raise RuntimeError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including auth and optional site info."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_SITE_URL", "https://github.com/redteam-ai-benchmark"
            ),
            "X-Title": os.environ.get("OPENROUTER_SITE_NAME", "RedTeam-AI-Benchmark"),
        }

    def _make_request(self, payload: Dict) -> Dict:
        """Make API request with retry logic."""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            reraise=True,
        )
        def _request():
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )

                # Handle rate limiting
                if response.status_code == 429:
                    raise RuntimeError("Rate limited by OpenRouter")

                response.raise_for_status()
                return response.json()

        return _request()

    def query(self, prompt: str, max_tokens: int = 768, retries: int = 3) -> str:
        """
        Query OpenRouter API.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens in response
            retries: Number of retries (handled by tenacity)

        Returns:
            Model response text
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }

        try:
            data = self._make_request(payload)
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"OpenRouter API error {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to OpenRouter at {self.base_url}"
            ) from e
        except KeyError as e:
            raise RuntimeError(f"Invalid API response format: {e}") from e

    def list_models(self) -> List[Dict]:
        """
        List available models from OpenRouter.

        Returns:
            List of model dictionaries with id, name, pricing, etc.
        """
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}") from e

    def test_connection(self) -> bool:
        """Test OpenRouter connection by listing models."""
        try:
            models = self.list_models()
            return len(models) > 0
        except Exception:
            return False

    def get_model_info(self) -> Optional[Dict]:
        """
        Get information about the current model.

        Returns:
            Model info dict or None if not found
        """
        try:
            models = self.list_models()
            for model in models:
                if model.get("id") == self.model_name:
                    return model
            return None
        except Exception:
            return None
