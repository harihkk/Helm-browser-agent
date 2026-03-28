"""
Multi-Provider AI Support
=========================
Abstraction layer supporting Groq (primary), Ollama (local/free), and Gemini (free tier + vision).
"""

import asyncio
import base64
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class AIProvider(ABC):
    """Base class for AI providers."""

    @abstractmethod
    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2000) -> str:
        pass

    @abstractmethod
    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        """Vision capability - analyze a screenshot."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        pass


class GroqProvider(AIProvider):
    """Groq API provider (primary - fast and free tier)."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2000) -> str:
        def sync_call():
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        return await asyncio.get_event_loop().run_in_executor(None, sync_call)

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        # Groq doesn't support vision natively with Llama - delegate to text
        return await self.generate(f"[Image analysis not available with Groq] {prompt}")

    def get_name(self) -> str:
        return f"Groq ({self.model})"

    def supports_vision(self) -> bool:
        return False


class OllamaProvider(AIProvider):
    """Ollama local LLM provider (100% free, runs locally)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._available = None

    async def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2000) -> str:
        if not await self._check_available():
            raise ConnectionError("Ollama is not running. Start it with: ollama serve")

        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system or "You are a web automation agent. Respond with valid JSON only.",
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": max_tokens}
            }
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get('response', '')

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        if not await self._check_available():
            raise ConnectionError("Ollama not available")

        import httpx
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "llava",  # Vision model
                "prompt": prompt,
                "images": [b64],
                "stream": False
            }
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get('response', '')

    def get_name(self) -> str:
        return f"Ollama ({self.model})"

    def supports_vision(self) -> bool:
        return True  # via llava model


class GeminiProvider(AIProvider):
    """Google Gemini provider (free tier: 15 RPM, 1M tokens/day, vision supported)."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self._available = bool(api_key)

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2000) -> str:
        if not self._available:
            raise ValueError("Gemini API key not set. Get one free at https://aistudio.google.com/app/apikey")

        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        contents = [{"parts": [{"text": prompt}]}]
        if system:
            contents.insert(0, {"parts": [{"text": system}], "role": "model"})

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "systemInstruction": {"parts": [{"text": system}]} if system else None,
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}
            })
            resp.raise_for_status()
            data = resp.json()
            return data['candidates'][0]['content']['parts'][0]['text']

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        """Use Gemini's free multimodal API to analyze screenshots."""
        if not self._available:
            raise ValueError("Gemini API key not set")

        import httpx
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64}}
                    ]
                }],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2000}
            })
            resp.raise_for_status()
            data = resp.json()
            return data['candidates'][0]['content']['parts'][0]['text']

    def get_name(self) -> str:
        return f"Gemini ({self.model})"

    def supports_vision(self) -> bool:
        return True


def create_provider(provider_type: str, **kwargs) -> AIProvider:
    """Factory to create the right provider."""
    if provider_type == "groq":
        return GroqProvider(
            api_key=kwargs.get('api_key', ''),
            model=kwargs.get('model', 'llama-3.3-70b-versatile')
        )
    elif provider_type == "ollama":
        return OllamaProvider(
            base_url=kwargs.get('base_url', 'http://localhost:11434'),
            model=kwargs.get('model', 'llama3.1')
        )
    elif provider_type == "gemini":
        return GeminiProvider(
            api_key=kwargs.get('api_key', ''),
            model=kwargs.get('model', 'gemini-1.5-flash')
        )
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
