"""Ollama API client."""

import json
import time
from typing import Dict, List

import requests

from .base import APIClient


class OllamaClient(APIClient):
    """Ollama API client."""

    def query(self, prompt: str, max_tokens: int = 768, retries: int = 3) -> str:
        """Query Ollama API with retry logic."""
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": max_tokens,
            },
        }

        for attempt in range(retries):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=150
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]

            except requests.exceptions.Timeout:
                print(f"   Timeout on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"API timeout after {retries} attempts"
                    ) from None
                time.sleep(2**attempt)

            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. Is it running?"
                ) from e

            except requests.exceptions.HTTPError as e:
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
        """List available models from Ollama."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}") from e

    def test_connection(self) -> bool:
        """Test Ollama connection."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
