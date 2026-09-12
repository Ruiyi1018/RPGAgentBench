"""Tencent Venus OpenAPI adapter for internal and proxied model deployments."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from .base import GenerationConfig, ModelCapabilities
from .openai_compatible import APIClientError
from .rate_limit import model_concurrency_gate


LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_URL = "https://v2.open.venus.woa.com/llmproxy"
DEFAULT_MODEL = "deepseek-v4-pro"
COMPLETION_TOKEN_MODEL_PREFIXES = ("gpt-5", "gpt-6")


@dataclass(frozen=True)
class VenusResult:
    text: str
    usage: dict[str, Any] | None
    request_id: str | None
    finish_reason: str | None = None


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
    capabilities: ModelCapabilities | None = None,
) -> dict[str, Any]:
    args = dict(model_args or {})
    resolved_capabilities = capabilities or ModelCapabilities(
        token_parameter=(
            "max_completion_tokens"
            if model_name.lower().startswith(COMPLETION_TOKEN_MODEL_PREFIXES)
            else "max_tokens"
        ),
        supports_temperature=not model_name.lower().startswith(
            COMPLETION_TOKEN_MODEL_PREFIXES
        ),
        supports_top_p=not model_name.lower().startswith(
            COMPLETION_TOKEN_MODEL_PREFIXES
        ),
        supports_seed=not model_name.lower().startswith(
            COMPLETION_TOKEN_MODEL_PREFIXES
        ),
    )
    request: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        resolved_capabilities.token_parameter: args.get("max_tokens", 1024),
    }
    if resolved_capabilities.supports_temperature:
        request["temperature"] = args.get("temperature", 0.7)
    if resolved_capabilities.supports_top_p and args.get("top_p") is not None:
        request["top_p"] = args["top_p"]
    if resolved_capabilities.supports_seed and args.get("seed") is not None:
        request["seed"] = args["seed"]
    for name in ("presence_penalty", "frequency_penalty"):
        if name in args and args[name] is not None:
            request[name] = args[name]
    for name in (
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
    capabilities: ModelCapabilities | None = None,
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
                **build_venus_request(
                    messages,
                    model_name,
                    model_args,
                    capabilities,
                )
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
            return VenusResult(
                content,
                usage,
                request_id,
                getattr(choice, "finish_reason", None),
            )
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
            retryable = _is_retryable_error(error)
            if not retryable or attempt >= attempts:
                raise APIClientError(
                    "Venus请求失败："
                    f"{type(error).__name__}: {error}"
                ) from error
            await asyncio.sleep(
                _retry_delay_seconds(error, attempt)
            )
        finally:
            if client is not None:
                await client.close()
    return None


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(
        error,
        (RateLimitError, APIConnectionError, APITimeoutError),
    ):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in {408, 409, 429} or (
            error.status_code >= 500
        )
    return isinstance(error, (APIClientError, RuntimeError))


def _retry_delay_seconds(error: Exception, attempt: int) -> float:
    if isinstance(error, RateLimitError):
        response = getattr(error, "response", None)
        retry_after = (
            response.headers.get("retry-after")
            if response is not None
            else None
        )
        try:
            return max(float(retry_after), 1.0)
        except (TypeError, ValueError):
            pass
    return float(min(2 ** (attempt - 1), 8))


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
    max_concurrency: int = 4
    scene: str = "benchmark"
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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise APIClientError(
                "VenusClient同步接口不能在活动事件循环内调用"
            )
        model = config.model or DEFAULT_MODEL
        response_format: dict[str, Any] | None
        if config.capabilities.structured_output == "prompt_only":
            response_format = None
        elif (
            config.response_schema is not None
            and config.capabilities.structured_output == "json_schema"
        ):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": dict(config.response_schema),
                },
            }
        else:
            response_format = {"type": "json_object"}
        with model_concurrency_gate(
            f"{self.base_url.rstrip('/')}::{model}",
            self.max_concurrency,
        ):
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
                    capabilities=config.capabilities,
                )
            )
        if result is None:
            raise APIClientError("Venus请求在有限重试后失败")
        self.last_usage = result.usage
        self.last_request_id = result.request_id
        self.last_finish_reason = result.finish_reason
        normalized_finish_reason = (result.finish_reason or "").lower()
        if normalized_finish_reason in {
            "length",
            "max_tokens",
            "max_output_tokens",
        }:
            raise APIClientError(
                "Venus输出因token上限被截断；"
                f"finish_reason={result.finish_reason}"
            )
        completion_tokens = int(
            (result.usage or {}).get("completion_tokens", 0)
        )
        if completion_tokens >= config.max_tokens:
            raise APIClientError(
                "Venus输出已耗尽completion token预算，按疑似截断处理；"
                f"completion_tokens={completion_tokens}, "
                f"limit={config.max_tokens}"
            )
        return result.text
