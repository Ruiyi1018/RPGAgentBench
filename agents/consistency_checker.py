"""Episode-level frozen consistency checker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from gamecore import GameContext

from llm import GenerationConfig, LLMClient, generate_structured
from .prompt_builder import PromptBuilder


_FAILURE_TYPES = {
    "knowledge",
    "commitment",
    "relationship",
    "goal",
    "secret",
    "authority",
    "resource",
    "format",
    "utterance_action",
    "grounding",
    "utility",
}


@dataclass
class ConsistencyChecker:
    client: LLMClient
    config: GenerationConfig
    prompt_builder: PromptBuilder

    def check(
        self,
        context: GameContext,
        evaluation_spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_turns = sorted(
            {
                int(entry["turn"])
                for entry in context.history
                if entry["speaker"] == "npc"
            }
        )
        bundle = self.prompt_builder.build_checker(context, evaluation_spec)
        result = generate_structured(
            self.client,
            bundle,
            self.config,
            lambda output: self._validate_output(output, expected_turns),
        )
        self._merge_gamecore_violations(result, context)
        return result

    @staticmethod
    def _validate_output(
        output: dict[str, Any],
        expected_turns: list[int],
    ) -> None:
        if set(output) != {"turns", "episode"} or not isinstance(
            output["turns"], list
        ):
            raise ValueError("Checker输出必须包含turns数组和episode对象")
        episode = output["episode"]
        if not isinstance(episode, dict) or set(episode) != {
            "interaction_value_pass",
            "evidence",
        }:
            raise ValueError("Checker episode字段不完整")
        if not isinstance(episode["interaction_value_pass"], bool):
            raise ValueError("interaction_value_pass必须是布尔值")
        if not isinstance(episode["evidence"], list) or not all(
            isinstance(item, str) and item.strip()
            for item in episode["evidence"]
        ):
            raise ValueError("episode.evidence必须是字符串数组")
        actual_turns: list[int] = []
        required = {
            "turn",
            "verbal_violation",
            "decision_violation",
            "utterance_action_mismatch",
            "grounding_violation",
            "utility_failure",
            "failure_types",
            "evidence",
            "cannot_determine",
        }
        for item in output["turns"]:
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("Checker轮次条目字段不完整")
            if not isinstance(item["turn"], int) or item["turn"] < 1:
                raise ValueError("Checker turn必须是正整数")
            actual_turns.append(item["turn"])
            for field in (
                "verbal_violation",
                "decision_violation",
                "utterance_action_mismatch",
                "grounding_violation",
                "utility_failure",
                "cannot_determine",
            ):
                if not isinstance(item[field], bool):
                    raise ValueError(f"{field}必须是布尔值")
            if (
                not isinstance(item["failure_types"], list)
                or not set(item["failure_types"]) <= _FAILURE_TYPES
                or len(item["failure_types"]) != len(set(item["failure_types"]))
            ):
                raise ValueError("failure_types包含未知或重复标签")
            if not isinstance(item["evidence"], list) or not all(
                isinstance(evidence, str) and evidence.strip()
                for evidence in item["evidence"]
            ):
                raise ValueError("evidence必须是非空字符串数组")
            if (
                item["verbal_violation"]
                or item["decision_violation"]
                or item["utterance_action_mismatch"]
                or item["grounding_violation"]
                or item["utility_failure"]
            ) and not item["evidence"]:
                raise ValueError("违规判断必须提供证据")
        if sorted(actual_turns) != expected_turns:
            raise ValueError("Checker必须且只能覆盖每个NPC轮次一次")

    @staticmethod
    def _merge_gamecore_violations(
        output: dict[str, Any],
        context: GameContext,
    ) -> None:
        by_turn = {item["turn"]: item for item in output["turns"]}
        for entry in context.history:
            if entry["speaker"] != "npc":
                continue
            for event in entry.get("events", []):
                if (
                    not isinstance(event, dict)
                    or event.get("type") != "decision_violation"
                ):
                    continue
                item = by_turn[int(entry["turn"])]
                code = str(event.get("code", "unknown"))
                item["decision_violation"] = True
                failure_type = ConsistencyChecker._gamecore_failure_type(code)
                if failure_type not in item["failure_types"]:
                    item["failure_types"].append(failure_type)
                evidence = f"第{entry['turn']}轮GameCore拒绝Action：{code}"
                if evidence not in item["evidence"]:
                    item["evidence"].append(evidence)

    @staticmethod
    def _gamecore_failure_type(code: str) -> str:
        if any(
            token in code
            for token in (
                "shape",
                "parameter",
                "unknown_action",
                "action_not_available",
                "too_many",
                "utterance",
            )
        ):
            return "format"
        if "access" in code or "authorized" in code:
            return "authority"
        return "resource"
