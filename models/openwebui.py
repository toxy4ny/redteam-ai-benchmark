"""OpenWebUI API client (OpenAI-compatible with optional authentication)."""

import json
import os
import time
from typing import Dict, List, Optional

import requests

from .base import APIClient


class OpenWebUIClient(APIClient):
    """
    OpenWebUI API client.
 
    OpenWebUI provides an OpenAI-compatible API with optional Bearer token authentication.
    Supports both local instances (no auth) and secured deployments (Bearer token).
    """
 
    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        model_name: str = "",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenWebUI client.

        Args:
            base_url: OpenWebUI instance URL (default: http://localhost:3000)
            model_name: Model name/ID to use
            api_key: Optional API key for authentication (or set OPENWEBUI_API_KEY env var)
        """
        super().__init__(base_url, model_name)
        self.api_key = api_key or os.environ.get("OPENWEBUI_API_KEY") 

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with optional authentication."""

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
 
    def query(self, prompt: str, max_tokens: int = 768, retries: int = 3) -> str:
        """Query OpenWebUI API with retry logic."""

        url = f"{self.base_url}/api/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
        }
 
        for attempt in range(retries):
            try:
                response = requests.post(
                    url, headers=self._get_headers(), json=payload, timeout=150
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
 
            except requests.exceptions.Timeout:
                print(f"   Timeout on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"API timeout after {retries} attempts"
                    ) from None
                time.sleep(2**attempt)
 
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"Cannot connect to OpenWebUI at {self.base_url}. Is it running?"
                ) from e
 
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    raise RuntimeError(
                        "Authentication required. Provide API key via --api-key or OPENWEBUI_API_KEY"
                    ) from e
                if e.response.status_code == 429:
                    print("   Rate limited, waiting...")
                    time.sleep(5)
                    continue
                raise RuntimeError(
                    f"API error {e.response.status_code}: {e.response.text}"
                ) from e
 
            except (KeyError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Invalid API response format: {e}") from e

        raise RuntimeError("Max retries exceeded")

    def list_models(self) -> List[Dict]:
        """List available models from OpenWebUI."""
        try:
            url = f"{self.base_url}/api/models"
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
            data = response.json()
            # OpenWebUI returns models in "data" array (OpenAI format)
            return data.get("data", data.get("models", []))

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise RuntimeError(
                    "Authentication required. Provide API key via --api-key or OPENWEBUI_API_KEY"
                ) from e
            raise RuntimeError(f"Failed to list models: {e}") from e
        except Exception as e:            raise RuntimeError(f"Failed to list models: {e}") from e
 
    def test_connection(self) -> bool:
        """Test OpenWebUI connection."""
        try:
            url = f"{self.base_url}/api/models"
            response = requests.get(url, headers=self._get_headers(), timeout=5)
            return response.status_code == 200
        except Exception:
            return False
