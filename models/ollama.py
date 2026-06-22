"""Ollama API client."""

import os
from typing import Dict, List, Optional

import requests

from .base import APIClient, RequestsRetryMixin


class OllamaClient(RequestsRetryMixin, APIClient):
    """Ollama API client."""

    provider_name = "Ollama"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = 150,
        api_key: Optional[str] = None,
        keep_alive: Optional[str] = None,
    ):
        super().__init__(base_url, model_name)
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self.keep_alive = keep_alive or os.environ.get("OLLAMA_KEEP_ALIVE")
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        """Return Ollama headers, including optional reverse-proxy auth."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def query(
        self,
        prompt: str,
        max_tokens: int = 768,
        retries: int = 3,
        temperature: float = 0.2,
    ) -> str:
        """Query Ollama API with retry logic."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        data = self._post_json_with_retries(
            url=url, headers=self._get_headers(), payload=payload, retries=retries
        )
        try:
            message = data["message"]
            return message.get("content") or message.get("thinking", "")
        except KeyError as e:
            raise RuntimeError(f"Invalid API response format: {e}") from e

    def list_models(self) -> List[Dict]:
        """List available models from Ollama."""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, headers=self._get_headers(), timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}") from e

    def test_connection(self) -> bool:
        """Test Ollama connection."""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, headers=self._get_headers(), timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the persistent HTTP session."""
        self.session.close()
