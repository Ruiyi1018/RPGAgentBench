"""NPC agent under evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gamecore import ActionValidationError, GameContext

from llm import GenerationConfig, LLMClient, generate_structured
from .prompt_builder import PromptBuilder, StateAccess


@dataclass
class NPCAgent:
    client: LLMClient
    config: GenerationConfig
    prompt_builder: PromptBuilder
    state_access: StateAccess = StateAccess.HISTORY_ONLY

    def generate(
        self,
        context: GameContext,
        scenario: Mapping[str, Any],
        *,
        available_actions: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        selected_actions = list(
            available_actions or scenario.get("available_actions", [])
        )
        if not selected_actions:
            raise ValueError("NPC至少需要一个场景可用Action")
        bundle = self.prompt_builder.build_npc(
            context,
            scenario,
            available_actions=selected_actions,
            state_access=self.state_access,
        )
        valid_arguments = self.prompt_builder.npc_valid_action_arguments(
            context,
            selected_actions,
        )

        def validate(output: dict[str, Any]) -> None:
            self._validate_output_shape(output)
            self._validate_runtime_arguments(
                output,
                selected_actions,
                valid_arguments,
            )

        return generate_structured(
            self.client,
            bundle,
            self.config,
            validate,
        )

    @staticmethod
    def _validate_output_shape(output: dict[str, Any]) -> None:
        if set(output) != {"utterance", "actions"}:
            raise ValueError("NPC输出只能包含utterance和actions")
        if not isinstance(output["utterance"], str) or not output[
            "utterance"
        ].strip():
            raise ValueError("utterance必须是非空字符串")
        actions = output["actions"]
        if not isinstance(actions, list) or len(actions) > 2:
            raise ValueError("actions必须是最多包含两项的数组")
        for action in actions:
            if not isinstance(action, dict) or set(action) != {
                "name",
                "parameters",
            }:
                raise ValueError("每个Action只能包含name和parameters")
            if not isinstance(action["name"], str) or not isinstance(
                action["parameters"], dict
            ):
                raise ValueError("Action字段类型无效")

    def _validate_runtime_arguments(
        self,
        output: dict[str, Any],
        available_actions: Sequence[str],
        valid_arguments: Mapping[str, Sequence[str]],
    ) -> None:
        for action in output["actions"]:
            name = action["name"]
            if name not in available_actions:
                raise ValueError(f"本场景未提供Action: {name}")
            try:
                self.prompt_builder.registry.validate_call(action)
            except ActionValidationError as error:
                raise ValueError(str(error)) from error
            for parameter, value in action["parameters"].items():
                key = f"{name}.{parameter}"
                allowed = valid_arguments.get(key)
                if allowed is not None and value not in allowed:
                    raise ValueError(
                        f"{key}必须复制valid_action_arguments中的值；"
                        f"收到{value!r}"
                    )
