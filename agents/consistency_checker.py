"""Windowed role-contract conflict checker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from gamecore import GameContext
from llm import GenerationConfig, LLMClient, generate_structured
from .prompt_builder import PromptBuilder


@dataclass
class ConsistencyChecker:
    client: LLMClient
    config: GenerationConfig
    prompt_builder: PromptBuilder

    GENERAL_RULES: ClassVar[tuple[dict[str, str], ...]] = (
        {
            "id": "words_actions_state_alignment",
            "statement": (
                "NPC台词表达的本轮决定或已经发生的效果，必须与提交的"
                "Action类型、参数及GameCore实际状态变化一致。"
            ),
        },
    )

    def check_batch(
        self,
        context: GameContext,
        role_contract: Mapping[str, Any],
        turns: list[int],
        *,
        prior_judgment: list[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not turns or len(turns) != len(set(turns)):
            raise ValueError("Checker批次轮次必须是非空无重复数组")
        effective_contract = self._effective_contract(role_contract)
        rule_ids = self._rule_ids(effective_contract)
        available_turns = {
            int(entry["turn"])
            for entry in context.history
            if entry["speaker"] == "npc"
        }
        missing = set(turns) - available_turns
        if missing:
            raise ValueError(f"轨迹中不存在NPC轮次: {sorted(missing)}")
        bundle = self.prompt_builder.build_checker(
            context,
            effective_contract,
            turns=turns,
            prior_judgment=prior_judgment,
        )
        result = generate_structured(
            self.client,
            bundle,
            self.config,
            lambda output: self._validate_output(
                output,
                rule_ids,
                turns,
            ),
        )
        failures = {
            item["turn"]: item for item in result["failures"]
        }
        return [
            failures.get(
                turn,
                {
                    "turn": turn,
                    "failure": False,
                    "violated_rule": None,
                    "evidence": [],
                    "cannot_determine": False,
                },
            )
            for turn in turns
        ]

    @classmethod
    def _effective_contract(
        cls,
        role_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        del role_contract
        return {"rules": [dict(rule) for rule in cls.GENERAL_RULES]}

    @staticmethod
    def _rule_ids(role_contract: Mapping[str, Any]) -> set[str]:
        rules = role_contract.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError("role_contract.rules必须是非空数组")
        rule_ids = {
            str(rule["id"])
            for rule in rules
            if isinstance(rule, Mapping) and rule.get("id")
        }
        if len(rule_ids) != len(rules):
            raise ValueError("每条角色规则必须包含唯一id")
        return rule_ids

    @staticmethod
    def _validate_output(
        output: dict[str, Any],
        rule_ids: set[str],
        expected_turns: list[int],
    ) -> None:
        if set(output) != {"failures"} or not isinstance(
            output["failures"], list
        ):
            raise ValueError("Checker输出必须且只能包含failures数组")
        required = {
            "turn",
            "failure",
            "violated_rule",
            "evidence",
            "cannot_determine",
        }
        actual_turns: list[int] = []
        for item in output["failures"]:
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("Checker失败条目字段不符合单一冲突契约")
            if not isinstance(item["turn"], int):
                raise ValueError("Checker turn必须是整数")
            actual_turns.append(item["turn"])
            if not isinstance(item["failure"], bool) or not isinstance(
                item["cannot_determine"], bool
            ):
                raise ValueError("failure和cannot_determine必须是布尔值")
            if not isinstance(item["evidence"], list) or not all(
                isinstance(evidence, str) and evidence.strip()
                for evidence in item["evidence"]
            ):
                raise ValueError("evidence必须是非空字符串数组")
            if not item["failure"]:
                raise ValueError(
                    "failures数组只能包含failure=true的条目"
                )
            if item["violated_rule"] not in rule_ids:
                raise ValueError("failure=true时必须引用有效角色规则id")
            if not item["evidence"]:
                raise ValueError("failure=true时必须提供具体证据")
            if item["cannot_determine"]:
                raise ValueError("证据不足时不能判定failure=true")
        if len(actual_turns) != len(set(actual_turns)):
            raise ValueError("Checker失败轮次不得重复")
        if not set(actual_turns).issubset(expected_turns):
            raise ValueError("Checker只能返回指定轮次中的失败")
