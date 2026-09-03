"""Provider-neutral LLM interface and strict JSON parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from agents.prompt_builder import PromptBundle


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 2048
    seed: int | None = None
    max_format_retries: int = 1


class LLMClient(Protocol):
    last_usage: dict[str, Any] | None
    last_request_id: str | None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Return raw model text without parsing or state mutation."""


class StaticLLMClient:
    """Deterministic response queue used by tests and offline prompt review."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.last_usage: dict[str, Any] | None = None
        self.last_request_id: str | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> str:
        self.requests.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
            }
        )
        if not self.responses:
            raise RuntimeError("StaticLLMClient没有剩余响应")
        return self.responses.pop(0)


class StructuredOutputError(ValueError):
    """Raised only for malformed structured model output."""


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract the first complete JSON object from a model response."""

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise StructuredOutputError("Markdown代码块未闭合")
        text = "\n".join(lines[1:-1]).strip()
    object_start = text.find("{")
    if object_start < 0:
        raise StructuredOutputError("输出中没有JSON对象")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[object_start:])
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"输出不是合法JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise StructuredOutputError("输出根节点必须是JSON对象")
    return value


def generate_structured(
    client: LLMClient,
    bundle: "PromptBundle",
    config: GenerationConfig,
    validator: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Retry malformed output only; never retry a GameCore semantic failure."""

    user_prompt = bundle.user_prompt
    last_error: StructuredOutputError | None = None
    for attempt in range(config.max_format_retries + 1):
        raw = client.generate(
            system_prompt=bundle.system_prompt,
            user_prompt=user_prompt,
            config=config,
        )
        try:
            parsed = parse_json_object(raw)
            validator(parsed)
            return parsed
        except StructuredOutputError as error:
            last_error = error
        except (TypeError, ValueError) as error:
            last_error = StructuredOutputError(str(error))
        if attempt < config.max_format_retries:
            user_prompt = (
                f"{bundle.user_prompt}\n\n"
                "上一次输出未满足JSON格式契约。请仅修正输出格式和字段结构，"
                f"不要改变原决定。格式错误：{last_error}"
            )
    raise last_error or StructuredOutputError("结构化输出生成失败")
