"""History-adaptive Normal and Pressure player agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from gamecore import GameContext

from llm import GenerationConfig, LLMClient, generate_structured
from .prompt_builder import PromptBuilder, PromptBundle


class PlayerMode(str, Enum):
    NORMAL = "normal"
    PRESSURE = "pressure"


@dataclass
class PlayerAgent:
    client: LLMClient
    config: GenerationConfig
    prompt_builder: PromptBuilder
    mode: PlayerMode
    player_profile: Mapping[str, Any] | None = None

    def generate(
        self,
        context: GameContext,
        scenario: Mapping[str, Any],
        *,
        correction: str | None = None,
    ) -> dict[str, str]:
        bundle = self.prompt_builder.build_player(
            context,
            scenario,
            mode=self.mode.value,
            player_profile=self.player_profile,
        )
        if correction:
            bundle = PromptBundle(
                system_prompt=bundle.system_prompt,
                user_prompt=(
                    f"{bundle.user_prompt}\n\n"
                    "上一次本轮输出未通过JSON结构校验。只修正当前Player"
                    "输出的格式和字段，不改变请求意图；只输出原契约要求的"
                    f"JSON对象。\n校验错误：{correction}"
                ),
                response_schema=bundle.response_schema,
            )
        output = generate_structured(
            self.client,
            bundle,
            self.config,
            self._validate_output,
        )
        return {"query": output["query"]}

    @staticmethod
    def _validate_output(output: dict[str, Any]) -> None:
        if set(output) != {"query"}:
            raise ValueError("Player输出只能包含query")
        if not isinstance(output["query"], str) or not output["query"].strip():
            raise ValueError("query必须是非空字符串")
