"""History-adaptive Normal and Pressure player agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from gamecore import GameContext

from llm import GenerationConfig, LLMClient, generate_structured
from .prompt_builder import PromptBuilder


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
    ) -> dict[str, str]:
        bundle = self.prompt_builder.build_player(
            context,
            scenario,
            mode=self.mode.value,
            player_profile=self.player_profile,
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
