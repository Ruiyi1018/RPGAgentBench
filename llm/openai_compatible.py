"""Minimal OpenAI-compatible chat client for Qwen and local gateways."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .base import GenerationConfig


class APIClientError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleClient:
    api_key: str
    base_url: str
    timeout_seconds: float = 180.0
    max_retries: int = 3
    json_mode: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    last_usage: dict[str, Any] | None = field(default=None, init=False)
    last_request_id: str | None = field(default=None, init=False)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> str:
        if not self.api_key:
            raise APIClientError("API key不能为空")
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            **self.extra_body,
        }
        if config.seed is not None:
            payload["seed"] = config.seed
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    self.last_request_id = response.headers.get("x-request-id")
                self.last_usage = body.get("usage")
                choices = body.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise APIClientError("API响应缺少choices")
                content = choices[0].get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise APIClientError("API响应缺少文本content")
                return content
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise APIClientError(
                        f"API请求失败 HTTP {error.code}: {detail}"
                    ) from error
            except urllib.error.URLError as error:
                if attempt >= self.max_retries:
                    raise APIClientError(f"API连接失败: {error.reason}") from error
            time.sleep(min(2**attempt, 8))
        raise APIClientError("API请求在重试后仍失败")
