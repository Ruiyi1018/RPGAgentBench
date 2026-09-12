"""Minimal OpenAI-compatible chat client for Qwen and local gateways."""

from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .base import GenerationConfig
from .rate_limit import model_concurrency_gate


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
    max_concurrency: int = 4
    last_usage: dict[str, Any] | None = field(default=None, init=False)
    last_request_id: str | None = field(default=None, init=False)
    last_finish_reason: str | None = field(default=None, init=False)

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
            config.capabilities.token_parameter: config.max_tokens,
            **self.extra_body,
        }
        if config.capabilities.supports_temperature:
            payload["temperature"] = config.temperature
        if config.capabilities.supports_top_p:
            payload["top_p"] = config.top_p
        if config.seed is not None and config.capabilities.supports_seed:
            payload["seed"] = config.seed
        if (
            self.json_mode
            and config.capabilities.structured_output != "prompt_only"
        ):
            payload["response_format"] = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": dict(config.response_schema),
                    },
                }
                if (
                    config.response_schema is not None
                    and config.capabilities.structured_output == "json_schema"
                )
                else {"type": "json_object"}
            )

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
            retry_delay = min(2**attempt, 8)
            try:
                with model_concurrency_gate(
                    f"{self.base_url.rstrip('/')}::{config.model}",
                    self.max_concurrency,
                ):
                    with urllib.request.urlopen(
                        request,
                        timeout=self.timeout_seconds,
                    ) as response:
                        body = json.loads(response.read().decode("utf-8"))
                        self.last_request_id = response.headers.get(
                            "x-request-id"
                        )
                self.last_usage = body.get("usage")
                choices = body.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise APIClientError("API响应缺少choices")
                content = choices[0].get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise APIClientError("API响应缺少文本content")
                self.last_finish_reason = choices[0].get("finish_reason")
                normalized_finish_reason = (
                    self.last_finish_reason or ""
                ).lower()
                if normalized_finish_reason in {
                    "length",
                    "max_tokens",
                    "max_output_tokens",
                }:
                    raise APIClientError(
                        "API输出因token上限被截断；"
                        f"finish_reason={self.last_finish_reason}"
                    )
                completion_tokens = int(
                    (self.last_usage or {}).get("completion_tokens", 0)
                )
                if completion_tokens >= config.max_tokens:
                    raise APIClientError(
                        "API输出已耗尽completion token预算，"
                        "按疑似截断处理；"
                        f"completion_tokens={completion_tokens}, "
                        f"limit={config.max_tokens}"
                    )
                return content
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise APIClientError(
                        f"API请求失败 HTTP {error.code}: {detail}"
                    ) from error
                if error.code == 429 and "insufficient_quota" in detail:
                    retry_after = error.headers.get("Retry-After")
                    try:
                        retry_after_seconds = float(retry_after)
                    except (TypeError, ValueError):
                        retry_after_seconds = 60.0
                    retry_delay = max(
                        retry_after_seconds,
                        60.0,
                    )
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                TimeoutError,
                socket.timeout,
            ) as error:
                if attempt >= self.max_retries:
                    reason = getattr(error, "reason", str(error))
                    raise APIClientError(f"API连接失败: {reason}") from error
            time.sleep(retry_delay)
        raise APIClientError("API请求在重试后仍失败")
