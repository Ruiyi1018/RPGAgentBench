"""Strict provider routing for model clients."""

from __future__ import annotations

import os

from .base import LLMClient
from .config import LLMSettings
from .openai_compatible import OpenAICompatibleClient
from .venus import DEFAULT_MODEL, VenusClient, build_venus_token


VENUS_MODELS = frozenset({DEFAULT_MODEL, "deepseek-v4-flash"})


def create_llm_client(
    settings: LLMSettings,
    *,
    scene: str,
) -> LLMClient:
    provider = settings.provider.strip().lower()
    if provider == "venus":
        if settings.generation.model not in VENUS_MODELS:
            raise ValueError(
                f"Venus模型必须是{sorted(VENUS_MODELS)}之一，"
                f"收到{settings.generation.model!r}"
            )
        token = build_venus_token()
        if not token:
            raise ValueError(
                "VENUS_API_KEY未设置；也未提供"
                "ENV_VENUS_OPENAPI_SECRET_ID+VENUS_TOKEN_SUFFIX"
            )
        return VenusClient(
            api_key=token,
            base_url=(
                os.getenv("VENUS_BASE_URL", "").strip()
                or settings.base_url
            ),
            timeout_seconds=float(
                os.getenv(
                    "VENUS_TIMEOUT",
                    str(settings.timeout_seconds),
                )
            ),
            max_attempts=max(
                1,
                int(
                    os.getenv(
                        "VENUS_MAX_ATTEMPTS",
                        str(settings.max_attempts),
                    )
                ),
            ),
            scene=scene,
        )
    if provider == "dashscope":
        api_key = os.environ.get(settings.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"{settings.api_key_env}未设置")
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
            max_retries=max(settings.max_attempts - 1, 0),
            extra_body={
                "enable_thinking": settings.enable_thinking,
            },
        )
    raise ValueError(
        f"不支持的LLM provider: {settings.provider!r}；"
        "不会回退到其他付费后端"
    )
