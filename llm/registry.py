"""Declarative backend and model registry for experiment-time resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .base import GenerationConfig, LLMClient, ModelCapabilities
from .config import LLMSettings, load_env_file
from .factory import create_llm_client


@dataclass(frozen=True)
class BackendProfile:
    name: str
    provider: str
    base_url: str
    api_key_env: str
    api_key_suffix: str = ""
    env_file: str = ".env"
    enable_thinking: bool = False
    timeout_seconds: float = 300.0
    max_attempts: int = 3
    max_concurrency: int = 4


@dataclass(frozen=True)
class ModelProfile:
    """A stable experiment name mapped to one remote deployment."""

    name: str
    backend: str
    remote_model: str
    generation: GenerationConfig
    max_concurrency: int | None = None
    timeout_seconds: float | None = None
    max_attempts: int | None = None


class ModelRegistry:
    def __init__(
        self,
        *,
        backends: Mapping[str, BackendProfile],
        models: Mapping[str, ModelProfile],
    ) -> None:
        self.backends = dict(backends)
        self.models = dict(models)
        if not self.backends:
            raise ValueError("模型注册表至少需要一个backend")
        if not self.models:
            raise ValueError("模型注册表至少需要一个model")
        unknown = {
            model.backend
            for model in self.models.values()
            if model.backend not in self.backends
        }
        if unknown:
            raise ValueError(f"模型引用未知backend: {sorted(unknown)}")

    def model(self, name: str) -> ModelProfile:
        try:
            return self.models[name]
        except KeyError as error:
            raise ValueError(
                f"未注册模型{name!r}；可用模型={sorted(self.models)}"
            ) from error

    def settings(self, name: str) -> LLMSettings:
        model = self.model(name)
        backend = self.backends[model.backend]
        return LLMSettings(
            provider=backend.provider,
            base_url=backend.base_url,
            api_key_env=backend.api_key_env,
            api_key_suffix=backend.api_key_suffix,
            generation=model.generation,
            enable_thinking=backend.enable_thinking,
            timeout_seconds=(
                model.timeout_seconds
                if model.timeout_seconds is not None
                else backend.timeout_seconds
            ),
            max_attempts=(
                model.max_attempts
                if model.max_attempts is not None
                else backend.max_attempts
            ),
            max_concurrency=(
                model.max_concurrency
                if model.max_concurrency is not None
                else backend.max_concurrency
            ),
        )

    def generation_config(
        self,
        name: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        max_format_retries: int | None = None,
    ) -> GenerationConfig:
        base = self.model(name).generation
        return replace(
            base,
            model=model if model is not None else base.model,
            temperature=(
                temperature if temperature is not None else base.temperature
            ),
            top_p=top_p if top_p is not None else base.top_p,
            max_tokens=max_tokens if max_tokens is not None else base.max_tokens,
            seed=seed if seed is not None else base.seed,
            max_format_retries=(
                max_format_retries
                if max_format_retries is not None
                else base.max_format_retries
            ),
        )

    def create_client(self, name: str, *, scene: str) -> LLMClient:
        return create_llm_client(self.settings(name), scene=scene)


@dataclass(frozen=True)
class RoleModelAssignment:
    candidate: str
    player: str
    evaluator: str

    def names(self) -> set[str]:
        return {self.candidate, self.player, self.evaluator}


def load_model_registry(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ModelRegistry:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("模型注册表根节点必须是对象")
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    backend_values = _required_mapping(raw, "backends")
    model_values = _required_mapping(raw, "models")
    backends: dict[str, BackendProfile] = {}
    for name, value in backend_values.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise ValueError("每个backend必须是命名对象")
        backend = BackendProfile(
            name=name,
            provider=_required_string(value, "type").lower(),
            base_url=_required_string(value, "base_url"),
            api_key_env=_required_string(value, "api_key_env"),
            api_key_suffix=str(value.get("api_key_suffix", "")),
            env_file=str(value.get("env_file", ".env")),
            enable_thinking=bool(value.get("enable_thinking", False)),
            timeout_seconds=float(value.get("timeout_seconds", 300.0)),
            max_attempts=max(1, int(value.get("max_attempts", 3))),
            max_concurrency=max(1, int(value.get("max_concurrency", 4))),
        )
        load_env_file(root / backend.env_file)
        backends[name] = backend
    models: dict[str, ModelProfile] = {}
    for name, value in model_values.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise ValueError("每个model必须是命名对象")
        remote_model = _required_string(value, "remote_model")
        defaults = value.get("defaults", {})
        capabilities = value.get("capabilities", {})
        if not isinstance(defaults, Mapping):
            raise ValueError(f"模型{name}.defaults必须是对象")
        if not isinstance(capabilities, Mapping):
            raise ValueError(f"模型{name}.capabilities必须是对象")
        models[name] = ModelProfile(
            name=name,
            backend=_required_string(value, "backend"),
            remote_model=remote_model,
            max_concurrency=(
                max(1, int(value["max_concurrency"]))
                if value.get("max_concurrency") is not None
                else None
            ),
            timeout_seconds=(
                float(value["timeout_seconds"])
                if value.get("timeout_seconds") is not None
                else None
            ),
            max_attempts=(
                max(1, int(value["max_attempts"]))
                if value.get("max_attempts") is not None
                else None
            ),
            generation=GenerationConfig(
                model=remote_model,
                temperature=float(defaults.get("temperature", 0.0)),
                top_p=float(defaults.get("top_p", 1.0)),
                max_tokens=int(defaults.get("max_tokens", 4096)),
                seed=(
                    int(defaults["seed"])
                    if defaults.get("seed") is not None
                    else None
                ),
                max_format_retries=int(
                    defaults.get("max_format_retries", 2)
                ),
                capabilities=ModelCapabilities(
                    token_parameter=str(
                        capabilities.get("token_parameter", "max_tokens")
                    ),
                    supports_temperature=bool(
                        capabilities.get("temperature", True)
                    ),
                    supports_top_p=bool(capabilities.get("top_p", True)),
                    supports_seed=bool(capabilities.get("seed", True)),
                    structured_output=str(
                        capabilities.get(
                            "structured_output",
                            "json_schema",
                        )
                    ),
                ),
            ),
        )
    return ModelRegistry(backends=backends, models=models)


def load_role_assignment(
    path: str | Path,
    registry: ModelRegistry,
) -> RoleModelAssignment:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("实验配置根节点必须是对象")
    roles = _required_mapping(raw, "roles")
    assignment = RoleModelAssignment(
        candidate=_required_string(roles, "candidate"),
        player=_required_string(roles, "player"),
        evaluator=_required_string(roles, "evaluator"),
    )
    for name in assignment.names():
        registry.model(name)
    return assignment


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{key}必须是非空对象")
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key}必须是非空字符串")
    return value.strip()
