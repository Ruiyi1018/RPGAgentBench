"""Deterministic transitions for reviewed offline history fixtures."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .context import GameContext, WorldDefinition
from .errors import ActionValidationError


@dataclass(frozen=True)
class FixtureTransitionResult:
    context: GameContext
    event_id: str
    operations: tuple[dict[str, Any], ...]
    context_delta: dict[str, Any]


class FixtureEngine:
    """Apply human-reviewed exogenous events without using an LLM."""

    SUPPORTED_OPERATIONS = {
        "add_knowledge",
        "supersede_knowledge",
        "set_claim",
        "set_relationship",
        "create_commitment",
        "resolve_commitment",
        "set_goal",
        "set_inventory",
        "set_location",
        "set_access",
        "set_dialogue_status",
    }

    def __init__(self, world: WorldDefinition) -> None:
        self.world = world

    def apply(
        self,
        context: GameContext,
        *,
        event_id: str,
        operations: Sequence[Mapping[str, Any]],
        description: str,
        history_turn: int | None = None,
    ) -> FixtureTransitionResult:
        if not event_id:
            raise ActionValidationError(
                "invalid_fixture_event", "fixture event_id不能为空"
            )
        if context.event_exists(event_id):
            raise ActionValidationError(
                "duplicate_fixture_event", f"事件已存在: {event_id}"
            )
        working = context.clone()
        before = working.to_dict()
        normalized: list[dict[str, Any]] = []
        for operation in operations:
            item = copy.deepcopy(dict(operation))
            name = item.pop("op", None)
            if name not in self.SUPPORTED_OPERATIONS:
                raise ActionValidationError(
                    "unknown_fixture_operation",
                    f"不支持的fixture operation: {name}",
                )
            getattr(self, f"_op_{name}")(working, item, event_id)
            normalized.append({"op": name, **item})
        turn = working.next_turn if history_turn is None else history_turn
        if turn < 1:
            raise ActionValidationError(
                "invalid_fixture_turn", "fixture history_turn必须为正整数"
            )
        working.append_history(
            {
                "turn": turn,
                "speaker": "gamecore",
                "utterance": description,
                "events": [
                    {
                        "id": event_id,
                        "type": "fixture_event",
                        "operations": copy.deepcopy(normalized),
                    }
                ],
            }
        )
        working.validate()
        return FixtureTransitionResult(
            context=working,
            event_id=event_id,
            operations=tuple(normalized),
            context_delta=self._state_delta(before, working.data),
        )

    @staticmethod
    def _op_add_knowledge(
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        fact_id = FixtureEngine._required_string(item, "fact_id")
        content = FixtureEngine._required_string(item, "content")
        knowledge = context.runtime_state["knowledge"]
        if fact_id in knowledge and knowledge[fact_id].get("status") == "active":
            raise ActionValidationError(
                "duplicate_fact", f"有效事实已存在: {fact_id}"
            )
        knowledge[fact_id] = {
            "content": content,
            "status": "active",
            "source_event": event_id,
            "visibility": copy.deepcopy(item.get("visibility", ["npc"])),
        }

    @staticmethod
    def _op_supersede_knowledge(
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        fact_id = FixtureEngine._required_string(item, "fact_id")
        knowledge = context.runtime_state["knowledge"]
        if fact_id not in knowledge or knowledge[fact_id].get("status") != "active":
            raise ActionValidationError(
                "unknown_active_fact", f"没有可替换的有效事实: {fact_id}"
            )
        knowledge[fact_id]["status"] = "superseded"
        knowledge[fact_id]["superseded_by"] = event_id

    @staticmethod
    def _op_set_claim(
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        claim_id = FixtureEngine._required_string(item, "claim_id")
        decision = FixtureEngine._required_string(item, "decision")
        if decision not in {"accept", "reject", "uncertain"}:
            raise ActionValidationError(
                "invalid_claim_decision", f"无效claim decision: {decision}"
            )
        context.runtime_state["claims"][claim_id] = {
            "content": FixtureEngine._required_string(item, "content"),
            "decision": decision,
            "evidence_event": event_id,
        }

    @staticmethod
    def _op_set_relationship(
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        target = FixtureEngine._required_string(item, "target")
        level = FixtureEngine._required_string(item, "level")
        if level not in {
            "hostile",
            "distrustful",
            "neutral",
            "trusting",
            "loyal",
        }:
            raise ActionValidationError(
                "invalid_relationship_level", f"无效关系等级: {level}"
            )
        context.runtime_state["relationships"][target] = {
            "level": level,
            "basis_event": event_id,
        }

    @staticmethod
    def _op_create_commitment(
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        commitment_id = FixtureEngine._required_string(item, "commitment_id")
        commitments = context.runtime_state["commitments"]
        if any(entry.get("id") == commitment_id for entry in commitments):
            raise ActionValidationError(
                "duplicate_commitment", f"承诺已存在: {commitment_id}"
            )
        commitments.append(
            {
                "id": commitment_id,
                "target": FixtureEngine._required_string(item, "target"),
                "content": FixtureEngine._required_string(item, "content"),
                "condition": item.get("condition"),
                "status": "active",
                "source_event": event_id,
            }
        )

    @staticmethod
    def _op_resolve_commitment(
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        commitment_id = FixtureEngine._required_string(item, "commitment_id")
        status = FixtureEngine._required_string(item, "status")
        if status not in {"fulfilled", "cancelled", "violated"}:
            raise ActionValidationError(
                "invalid_commitment_status", f"无效承诺状态: {status}"
            )
        commitment = next(
            (
                entry
                for entry in context.runtime_state["commitments"]
                if entry.get("id") == commitment_id
            ),
            None,
        )
        if commitment is None or commitment.get("status") != "active":
            raise ActionValidationError(
                "unknown_active_commitment",
                f"没有可结束的有效承诺: {commitment_id}",
            )
        commitment.update(
            {"status": status, "resolution_event": event_id}
        )

    @staticmethod
    def _op_set_goal(
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        goal_id = FixtureEngine._required_string(item, "goal_id")
        status = FixtureEngine._required_string(item, "status")
        if status not in {"active", "fulfilled", "cancelled", "failed"}:
            raise ActionValidationError(
                "invalid_goal_status", f"无效目标状态: {status}"
            )
        goals = context.runtime_state["goals"]
        goal = next((entry for entry in goals if entry.get("id") == goal_id), None)
        if goal is None:
            goal = {
                "id": goal_id,
                "content": FixtureEngine._required_string(item, "content"),
            }
            goals.append(goal)
        goal.update({"status": status, "basis_event": event_id})

    def _op_set_inventory(
        self,
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        del event_id
        holder = self._required_string(item, "holder")
        item_id = self._required_string(item, "item")
        if item_id not in self.world.items:
            raise ActionValidationError(
                "unknown_item", f"未知物品: {item_id}"
            )
        inventory = context.environment["inventories"].setdefault(holder, [])
        present = item.get("present")
        if not isinstance(present, bool):
            raise ActionValidationError(
                "invalid_fixture_value", "present必须是布尔值"
            )
        if present and item_id not in inventory:
            inventory.append(item_id)
        if not present and item_id in inventory:
            inventory.remove(item_id)

    def _op_set_location(
        self,
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        del event_id
        entity = self._required_string(item, "entity")
        location = self._required_string(item, "location")
        if location not in self.world.locations:
            raise ActionValidationError(
                "unknown_location", f"未知地点: {location}"
            )
        context.environment["locations"][entity] = location

    @staticmethod
    def _op_set_access(
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        del event_id
        resource = FixtureEngine._required_string(item, "resource")
        subject = FixtureEngine._required_string(item, "subject")
        decision = FixtureEngine._required_string(item, "decision")
        if decision not in {"granted", "denied", "revoked"}:
            raise ActionValidationError(
                "invalid_access_decision", f"无效访问状态: {decision}"
            )
        record = context.environment["access"].setdefault(
            resource,
            {"controllers": copy.deepcopy(item.get("controllers", [])), "subjects": {}},
        )
        record.setdefault("subjects", {})[subject] = decision

    @staticmethod
    def _op_set_dialogue_status(
        context: GameContext,
        item: dict[str, Any],
        event_id: str,
    ) -> None:
        del event_id
        status = FixtureEngine._required_string(item, "status")
        if status not in {"active", "ended"}:
            raise ActionValidationError(
                "invalid_dialogue_status", f"无效对话状态: {status}"
            )
        context.environment["dialogue_status"] = status

    @staticmethod
    def _required_string(item: Mapping[str, Any], key: str) -> str:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise ActionValidationError(
                "invalid_fixture_value", f"{key}必须是非空字符串"
            )
        return value

    @staticmethod
    def _state_delta(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        for section in ("runtime_state", "environment"):
            before_section = before.get(section, {})
            after_section = after.get(section, {})
            changed = {
                key: {
                    "before": copy.deepcopy(before_section.get(key)),
                    "after": copy.deepcopy(after_section.get(key)),
                }
                for key in sorted(set(before_section) | set(after_section))
                if before_section.get(key) != after_section.get(key)
            }
            if changed:
                delta[section] = changed
        return delta
