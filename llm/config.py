"""LLM experiment configuration with local .env secret loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .base import GenerationConfig, ModelCapabilities


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key_env: str
    generation: GenerationConfig
    api_key_suffix: str = ""
    enable_thinking: bool = False
    timeout_seconds: float = 300.0
    max_attempts: int = 3
    max_concurrency: int = 4


def load_llm_settings(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> LLMSettings:
    source = Path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("LLM配置根节点必须是对象")
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    env_file = root / str(raw.get("env_file", ".env"))
    load_env_file(env_file)
    generation = raw.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("LLM配置缺少generation对象")
    provider = (
        os.getenv("LLM_PROVIDER", "").strip()
        or _required_string(raw, "provider")
    ).lower()
    configured_model = _required_string(generation, "model")
    model = (
        os.getenv("VENUS_MODEL", "").strip() or configured_model
        if provider == "venus"
        else configured_model
    )
    completion_token_contract = (
        provider == "venus"
        and model.lower().startswith(("gpt-5", "gpt-6"))
    )
    return LLMSettings(
        provider=provider,
        base_url=_required_string(raw, "base_url"),
        api_key_env=_required_string(raw, "api_key_env"),
        api_key_suffix=str(raw.get("api_key_suffix", "")),
        enable_thinking=bool(raw.get("enable_thinking", False)),
        timeout_seconds=float(raw.get("timeout_seconds", 300.0)),
        max_attempts=max(1, int(raw.get("max_attempts", 3))),
        max_concurrency=max(1, int(raw.get("max_concurrency", 4))),
        generation=GenerationConfig(
            model=model,
            temperature=float(generation.get("temperature", 0.0)),
            top_p=float(generation.get("top_p", 1.0)),
            max_tokens=int(generation.get("max_tokens", 4096)),
            seed=(
                int(generation["seed"])
                if generation.get("seed") is not None
                else None
            ),
            max_format_retries=int(
                generation.get("max_format_retries", 1)
            ),
            capabilities=ModelCapabilities(
                token_parameter=(
                    "max_completion_tokens"
                    if completion_token_contract
                    else "max_tokens"
                ),
                supports_temperature=not completion_token_contract,
                supports_top_p=not completion_token_contract,
                supports_seed=not completion_token_contract,
            ),
        ),
    )


def load_env_file(path: str | Path, *, override: bool = False) -> None:
    source = Path(path)
    if not source.is_file():
        return
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"{source}:{line_number}不是KEY=VALUE格式")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"{source}:{line_number}包含无效环境变量名")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"LLM配置字段{key}必须是非空字符串")
    return value
