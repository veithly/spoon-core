"""Ollama Provider implementation for the unified LLM interface.

Ollama runs locally and exposes an HTTP API (default: http://localhost:11434).
This provider supports chat, completion, and streaming.

Notes:
- Ollama does not require an API key; the configuration layer may still provide
  a placeholder api_key value for consistency.
- Tool calling is not implemented here.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from logging import getLogger
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from spoon_ai.callbacks.base import BaseCallbackHandler
from spoon_ai.schema import Message, LLMResponseChunk

from ..errors import NetworkError, ProviderError
from ..interface import LLMProviderInterface, LLMResponse, ProviderCapability, ProviderMetadata
from ..registry import register_provider

logger = getLogger(__name__)


@register_provider(
    "ollama",
    [
        ProviderCapability.CHAT,
        ProviderCapability.COMPLETION,
        ProviderCapability.STREAMING,
    ],
)
class OllamaProvider(LLMProviderInterface):
    """Local Ollama provider via HTTP."""

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.config: Dict[str, Any] = {}
        self.model: str = "llama3.2"
        self.base_url: str = "http://localhost:11434"
        self.timeout: float = 30.0
        self.max_tokens: int = 4096
        self.temperature: float = 0.3

    async def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.model = (self.config.get("model") or self.model).strip()
        self.base_url = (self.config.get("base_url") or self.base_url).rstrip("/")
        self.timeout = float(self.config.get("timeout", self.timeout))
        self.max_tokens = int(self.config.get("max_tokens", self.max_tokens))
        self.temperature = float(self.config.get("temperature", self.temperature))

        # Lazily create client
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)

        logger.info("Ollama provider initialized with model: %s", self.model)

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for m in messages:
            role = (getattr(m, "role", None) or "").strip()
            if not role:
                continue
            content = getattr(m, "content", None) or ""
            out.append({"role": role, "content": content})
        return out

    def _build_options(self, **kwargs: Any) -> Dict[str, Any]:
        """Map common parameters to Ollama options."""
        options: Dict[str, Any] = {}

        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            options["temperature"] = float(temperature)

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            # Ollama uses num_predict for max output tokens
            options["num_predict"] = int(max_tokens)

        return options

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        msg = data.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
        resp = data.get("response")
        if isinstance(resp, str):
            return resp
        return ""

    @staticmethod
    def _extract_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
        prompt = data.get("prompt_eval_count")
        completion = data.get("eval_count")
        if prompt is None and completion is None:
            return None
        prompt_i = int(prompt or 0)
        completion_i = int(completion or 0)
        return {
            "prompt_tokens": prompt_i,
            "completion_tokens": completion_i,
            "total_tokens": prompt_i + completion_i,
        }

    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        if not self.client:
            raise ProviderError("ollama", "Provider not initialized")

        model = (kwargs.get("model") or self.model).strip()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": self._build_options(**kwargs),
        }

        url = f"{self.base_url}/api/chat"
        start = asyncio.get_event_loop().time()

        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise NetworkError("ollama", "Request timeout", original_error=exc)
        except httpx.HTTPError as exc:
            raise ProviderError("ollama", f"Request failed: {exc}", original_error=exc)
        except Exception as exc:  # pragma: no cover
            raise ProviderError("ollama", f"Unexpected error: {exc}", original_error=exc)

        duration = asyncio.get_event_loop().time() - start
        content = self._extract_content(data) if isinstance(data, dict) else ""
        done_reason = (data.get("done_reason") if isinstance(data, dict) else None) or "stop"

        return LLMResponse(
            content=content,
            provider="ollama",
            model=model,
            finish_reason="stop",
            native_finish_reason=str(done_reason),
            tool_calls=[],
            usage=self._extract_usage(data) if isinstance(data, dict) else None,
            metadata={"raw": data} if isinstance(data, dict) else {},
            request_id=str(data.get("id", "")) if isinstance(data, dict) else "",
            duration=duration,
            timestamp=datetime.now(),
        )

    async def chat_stream(
        self,
        messages: List[Message],
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponseChunk]:
        if not self.client:
            raise ProviderError("ollama", "Provider not initialized")

        model = (kwargs.get("model") or self.model).strip()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": True,
            "options": self._build_options(**kwargs),
        }

        url = f"{self.base_url}/api/chat"
        full_content = ""
        chunk_index = 0

        try:
            async with self.client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except Exception:
                        continue

                    if not isinstance(data, dict):
                        continue

                    delta = self._extract_content(data)
                    if delta:
                        full_content += delta

                    done = bool(data.get("done"))
                    native_finish = data.get("done_reason") if done else None
                    finish_reason = str(native_finish) if done else None

                    usage = self._extract_usage(data) if done else None

                    yield LLMResponseChunk(
                        content=full_content,
                        delta=delta,
                        provider="ollama",
                        model=model,
                        finish_reason=finish_reason,
                        tool_calls=[],
                        tool_call_chunks=None,
                        usage=usage,
                        metadata={"done": done},
                        chunk_index=chunk_index,
                        timestamp=datetime.now().isoformat(),
                    )
                    chunk_index += 1

        except httpx.TimeoutException as exc:
            raise NetworkError("ollama", "Request timeout", original_error=exc)
        except httpx.HTTPError as exc:
            raise ProviderError("ollama", f"Request failed: {exc}", original_error=exc)

    async def completion(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.client:
            raise ProviderError("ollama", "Provider not initialized")

        model = (kwargs.get("model") or self.model).strip()
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(**kwargs),
        }

        url = f"{self.base_url}/api/generate"
        start = asyncio.get_event_loop().time()

        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise NetworkError("ollama", "Request timeout", original_error=exc)
        except httpx.HTTPError as exc:
            raise ProviderError("ollama", f"Request failed: {exc}", original_error=exc)

        duration = asyncio.get_event_loop().time() - start
        text = data.get("response", "") if isinstance(data, dict) else ""
        done_reason = (data.get("done_reason") if isinstance(data, dict) else None) or "stop"

        return LLMResponse(
            content=text,
            provider="ollama",
            model=model,
            finish_reason="stop",
            native_finish_reason=str(done_reason),
            tool_calls=[],
            usage=self._extract_usage(data) if isinstance(data, dict) else None,
            metadata={"raw": data} if isinstance(data, dict) else {},
            request_id=str(data.get("id", "")) if isinstance(data, dict) else "",
            duration=duration,
            timestamp=datetime.now(),
        )

    async def chat_with_tools(self, messages: List[Message], tools: List[Dict], **kwargs) -> LLMResponse:
        raise ProviderError("ollama", "Tool calling is not supported by the Ollama provider")

    def get_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="ollama",
            version="1.0.0",
            capabilities=[
                ProviderCapability.CHAT,
                ProviderCapability.COMPLETION,
                ProviderCapability.STREAMING,
            ],
            max_tokens=self.max_tokens,
            supports_system_messages=True,
            rate_limits={},
        )

    async def health_check(self) -> bool:
        if not self.client:
            return False

        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def cleanup(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

