"""NPC agent under evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gamecore import ActionValidationError, GameContext

from llm import GenerationConfig, LLMClient, generate_structured
from .prompt_builder import PromptBuilder, PromptBundle, StateAccess


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
        correction: str | None = None,
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
        if correction:
            bundle = PromptBundle(
                system_prompt=bundle.system_prompt,
                user_prompt=(
                    f"{bundle.user_prompt}\n\n"
                    "上一次本轮输出未通过结构或Action校验。只修正本轮输出，"
                    "不要改变角色决定；仍然只输出原契约要求的JSON对象。\n"
                    f"校验错误：{correction}"
                ),
                response_schema=bundle.response_schema,
            )
        def validate(output: dict[str, Any]) -> None:
            self._validate_output_shape(output)
            for action in output["actions"]:
                if action["name"] not in selected_actions:
                    raise ValueError(
                        f"Action不在本轮可用列表中: {action['name']}"
                    )
                try:
                    self.prompt_builder.registry.validate_call(action)
                except ActionValidationError as error:
                    raise ValueError(str(error)) from error

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
