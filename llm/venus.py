"""Tencent Venus OpenAPI adapter for the internal DeepSeek deployment."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from openai import AsyncOpenAI

from .base import GenerationConfig
from .openai_compatible import APIClientError


LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_URL = "https://v2.open.venus.woa.com/llmproxy"
DEFAULT_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True)
class VenusResult:
    text: str
    usage: dict[str, Any] | None
    request_id: str | None


def build_venus_token(
    environ: Mapping[str, str] | None = None,
) -> str:
    env = environ or os.environ
    token = env.get("VENUS_API_KEY", "").strip()
    if token:
        return token
    secret_id = env.get("ENV_VENUS_OPENAPI_SECRET_ID", "").strip()
    if not secret_id:
        return ""
    if "@" in secret_id:
        return secret_id
    suffix = env.get("VENUS_TOKEN_SUFFIX", "").strip()
    return f"{secret_id}{suffix}" if suffix else ""


def build_venus_request(
    messages: list[dict[str, str]],
    model_name: str = DEFAULT_MODEL,
    model_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    args = dict(model_args or {})
    request: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": args.get("max_tokens", 1024),
        "temperature": args.get("temperature", 0.7),
    }
    for name in (
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
    ):
        if name in args and args[name] is not None:
            request[name] = args[name]
    return request


async def call_venus_api_result_async(
    messages: list[dict[str, str]],
    *,
    model_name: str = DEFAULT_MODEL,
    model_args: Mapping[str, Any] | None = None,
    scene: str = "dialogue",
    token: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    max_attempts: int | None = None,
    client_factory: Callable[..., Any] = AsyncOpenAI,
) -> VenusResult | None:
    if not messages:
        return None
    resolved_token = (token or build_venus_token()).strip()
    if not resolved_token:
        LOGGER.error("[Venus] missing VENUS_API_KEY scene=%s", scene)
        return None
    attempts = max(
        1,
        int(
            max_attempts
            if max_attempts is not None
            else os.getenv("VENUS_MAX_ATTEMPTS", "3")
        ),
    )
    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else os.getenv("VENUS_TIMEOUT", "300")
    )
    endpoint = (
        base_url
        or os.getenv("VENUS_BASE_URL", DEFAULT_BASE_URL)
    ).rstrip("/")
    for attempt in range(1, attempts + 1):
        client: Any | None = None
        try:
            client = client_factory(
                api_key=resolved_token,
                base_url=endpoint,
                timeout=timeout,
                max_retries=0,
            )
            completion = await client.chat.completions.create(
                **build_venus_request(messages, model_name, model_args)
            )
            if not completion.choices:
                raise APIClientError("Venus响应缺少choices")
            choice = completion.choices[0]
            content = choice.message.content
            if not isinstance(content, str) or not content.strip():
                raise APIClientError("Venus响应缺少文本content")
            usage_object = getattr(completion, "usage", None)
            usage = (
                {
                    "prompt_tokens": int(
                        getattr(usage_object, "prompt_tokens", 0)
                    ),
                    "completion_tokens": int(
                        getattr(usage_object, "completion_tokens", 0)
                    ),
                    "total_tokens": int(
                        getattr(usage_object, "total_tokens", 0)
                    ),
                }
                if usage_object is not None
                else None
            )
            request_id = getattr(completion, "_request_id", None)
            LOGGER.info(
                "[Venus] scene=%s model=%s finish_reason=%s "
                "attempt=%d prompt_tokens=%d completion_tokens=%d "
                "total_tokens=%d",
                scene,
                model_name,
                getattr(choice, "finish_reason", None),
                attempt,
                (usage or {}).get("prompt_tokens", 0),
                (usage or {}).get("completion_tokens", 0),
                (usage or {}).get("total_tokens", 0),
            )
            return VenusResult(content, usage, request_id)
        except Exception as error:
            LOGGER.warning(
                "[Venus] request failed scene=%s model=%s "
                "attempt=%d/%d error=%s",
                scene,
                model_name,
                attempt,
                attempts,
                error,
            )
            if attempt < attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 4))
        finally:
            if client is not None:
                await client.close()
    return None


async def call_venus_api_async(
    messages: list[dict[str, str]],
    model_name: str = DEFAULT_MODEL,
    model_args: Mapping[str, Any] | None = None,
    scene: str = "dialogue",
) -> str | None:
    result = await call_venus_api_result_async(
        messages,
        model_name=model_name,
        model_args=model_args,
        scene=scene,
    )
    return result.text if result is not None else None


@dataclass
class VenusClient:
    """Synchronous compatibility boundary for existing benchmark callers."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 300.0
    max_attempts: int = 3
    scene: str = "benchmark"
    last_usage: dict[str, Any] | None = field(default=None, init=False)
    last_request_id: str | None = field(default=None, init=False)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise APIClientError(
                "VenusClient同步接口不能在活动事件循环内调用"
            )
        model = config.model or DEFAULT_MODEL
        response_format: dict[str, Any] = {"type": "json_object"}
        if config.response_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": dict(config.response_schema),
                },
            }
        result = asyncio.run(
            call_venus_api_result_async(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model_name=model,
                model_args={
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_tokens": config.max_tokens,
                    "seed": config.seed,
                    "response_format": response_format,
                },
                scene=self.scene,
                token=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.max_attempts,
            )
        )
        if result is None:
            raise APIClientError("Venus请求在有限重试后失败")
        self.last_usage = result.usage
        self.last_request_id = result.request_id
        return result.text
