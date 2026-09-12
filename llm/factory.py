"""Strict provider routing for model clients."""

from __future__ import annotations

import os

from .base import LLMClient
from .config import LLMSettings
from .openai_compatible import OpenAICompatibleClient
from .venus import VenusClient, build_venus_token


def create_llm_client(
    settings: LLMSettings,
    *,
    scene: str,
) -> LLMClient:
    provider = settings.provider.strip().lower()
    if provider == "venus":
        token = build_venus_token()
        if not token:
            secret = os.environ.get(settings.api_key_env, "").strip()
            if secret:
                token = (
                    secret
                    if "@" in secret
                    else f"{secret}{settings.api_key_suffix}"
                )
        if not token:
            raise ValueError(
                "Venus Token未设置；请提供VENUS_API_KEY，或配置"
                "api_key_env与api_key_suffix"
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
            max_concurrency=settings.max_concurrency,
        )
    if provider in {"dashscope", "openai_compatible"}:
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
            max_concurrency=settings.max_concurrency,
        )
    raise ValueError(
        f"不支持的LLM provider: {settings.provider!r}；"
        "不会回退到其他付费后端"
    )
