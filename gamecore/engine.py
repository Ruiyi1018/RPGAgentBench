"""Per-Action execution and auditable transition logging."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping

from .action_registry import ActionRegistry, ActionSpec
from .context import GameContext, WorldDefinition
from .errors import ActionValidationError
from .handlers import HANDLERS, Handler


@dataclass(frozen=True)
class TransitionResult:
    context: GameContext
    accepted: bool
    events: tuple[dict[str, Any], ...]
    violations: tuple[dict[str, str], ...]
    public_observation: str
    preflight_failed: bool = False
    """True when the whole turn was discarded before any Action ran."""


class GameCoreEngine:
    """Apply at most two NPC actions, each as its own deterministic step."""

    def __init__(
        self,
        registry: ActionRegistry,
        world: WorldDefinition,
        *,
        handlers: Mapping[str, Handler] | None = None,
    ) -> None:
        self.registry = registry
        self.world = world
        self.handlers = dict(handlers or HANDLERS)
        missing = {
            self.registry.get(name).handler
            for name in self.registry.names
            if self.registry.get(name).handler not in self.handlers
        }
        if missing:
            raise ActionValidationError(
                "missing_handler",
                f"缺少Action handler: {', '.join(sorted(missing))}",
            )

    @classmethod
    def from_project(
        cls,
        environment_path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> "GameCoreEngine":
        root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        registry = ActionRegistry.load_directory(root / "gamecore" / "actions")
        world = WorldDefinition.load_yaml(environment_path)
        return cls(registry, world)

    def append_player_query(
        self,
        context: GameContext,
        query: str,
        *,
        claims: Mapping[str, str] | None = None,
    ) -> GameContext:
        """Append player text; optional claim metadata comes from scenario logic."""

        if not isinstance(query, str) or not query.strip():
            raise ActionValidationError(
                "invalid_player_query", "Player query必须是非空字符串"
            )
        updated = context.clone()
        if updated.environment["dialogue_status"] != "active":
            raise ActionValidationError("dialogue_already_ended", "对话已经结束")
        turn = updated.next_turn
        events: list[dict[str, Any]] = []
        for claim_id, content in (claims or {}).items():
            if claim_id in updated.runtime_state["claims"]:
                raise ActionValidationError(
                    "duplicate_claim", f"claim已存在: {claim_id}"
                )
            updated.runtime_state["claims"][claim_id] = {
                "content": content,
                "source": "player",
                "introduced_turn": turn,
                "decision": "unassessed",
            }
            events.append(
                {
                    "id": f"claim_event_{turn}_{len(events) + 1}",
                    "type": "claim_introduced",
                    "claim_id": claim_id,
                    "turn": turn,
                }
            )
        updated.append_history(
            {
                "turn": turn,
                "speaker": "player",
                "utterance": query,
                "events": events,
            }
        )
        updated.validate()
        return updated

    def step(
        self,
        context: GameContext,
        npc_output: Mapping[str, Any],
        *,
        allowed_actions: Collection[str] | None = None,
    ) -> TransitionResult:
        """Apply each Action independently; only malformed output is rejected."""

        turn = self._current_turn(context)
        try:
            utterance, actions = self._validate_npc_output(npc_output)
            if context.environment["dialogue_status"] != "active":
                raise ActionValidationError(
                    "dialogue_already_ended", "对话已经结束"
                )
            specs = self._preflight(actions, allowed_actions)
        except ActionValidationError as error:
            return self._rejected_result(
                context,
                turn,
                str(npc_output.get("utterance", "")),
                list(npc_output.get("actions", []))
                if isinstance(npc_output.get("actions"), list)
                else [],
                error,
            )

        working = context.clone()
        before = working.to_dict()
        events: list[dict[str, Any]] = []
        violations: list[dict[str, str]] = []
        entry_events: list[dict[str, Any]] = []
        for index, (action, spec) in enumerate(
            zip(actions, specs), start=1
        ):
            candidate = working.clone()
            try:
                event = self.handlers[spec.handler](
                    candidate,
                    self.world,
                    action["parameters"],
                    f"event_{turn}_{index}",
                    turn,
                )
            except ActionValidationError as error:
                violations.append(
                    {
                        "code": error.code,
                        "message": error.message,
                        "type": "decision_violation",
                        "action": str(action["name"]),
                    }
                )
                entry_events.append(
                    {
                        "id": f"violation_{turn}_{index}",
                        "type": "decision_violation",
                        "code": error.code,
                        "action": str(action["name"]),
                    }
                )
                continue
            working = candidate
            events.append(event)
            entry_events.append(copy.deepcopy(event))

        delta = self._state_delta(before, working.data)
        observation = self._observation(len(events), len(violations))
        working.append_history(
            {
                "turn": turn,
                "speaker": "npc",
                "utterance": utterance,
                "actions": copy.deepcopy(actions),
                "events": entry_events,
                "context_delta": delta,
            }
        )
        working.append_history(
            {
                "turn": turn,
                "speaker": "gamecore",
                "utterance": observation,
                "events": [
                    {
                        "id": f"transition_{turn}",
                        "type": "transition_accepted"
                        if not violations
                        else "transition_partial",
                        "action_count": len(events),
                        "rejected_action_count": len(violations),
                    }
                ],
            }
        )
        working.validate()
        return TransitionResult(
            context=working,
            accepted=not violations,
            events=tuple(events),
            violations=tuple(violations),
            public_observation=observation,
        )

    def _preflight(
        self,
        actions: list[dict[str, Any]],
        allowed_actions: Collection[str] | None,
    ) -> list[ActionSpec]:
        """Validate Action shape and availability before touching state."""

        specs: list[ActionSpec] = []
        for index, action in enumerate(actions, start=1):
            if (
                allowed_actions is not None
                and action.get("name") not in allowed_actions
            ):
                raise ActionValidationError(
                    "action_not_available",
                    f"本场景未提供Action: {action.get('name')}",
                )
            if (
                action.get("name") == "end_dialogue"
                and index != len(actions)
            ):
                raise ActionValidationError(
                    "end_dialogue_not_final",
                    "end_dialogue必须是本轮最后一个Action",
                )
            specs.append(self.registry.validate_call(action))
        return specs

    @staticmethod
    def _observation(applied: int, rejected: int) -> str:
        if rejected and applied:
            return "部分动作已执行，其余未生效。"
        if rejected:
            return "动作未生效。"
        return "动作已执行。" if applied else "本轮无状态动作。"

    @staticmethod
    def _validate_npc_output(
        output: Mapping[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        if set(output) != {"utterance", "actions"}:
            raise ActionValidationError(
                "invalid_npc_output_shape",
                "NPC输出只能包含utterance和actions",
            )
        utterance = output.get("utterance")
        actions = output.get("actions")
        if not isinstance(utterance, str) or not utterance.strip():
            raise ActionValidationError(
                "invalid_utterance", "utterance必须是非空字符串"
            )
        if not isinstance(actions, list):
            raise ActionValidationError(
                "invalid_actions", "actions必须是数组"
            )
        if len(actions) > 2:
            raise ActionValidationError(
                "too_many_actions", "每轮最多提交两个Action"
            )
        if not all(isinstance(action, dict) for action in actions):
            raise ActionValidationError(
                "invalid_actions", "每个Action必须是对象"
            )
        return utterance, copy.deepcopy(actions)

    @staticmethod
    def _current_turn(context: GameContext) -> int:
        if context.history and context.history[-1]["speaker"] == "player":
            return int(context.history[-1]["turn"])
        return context.next_turn

    @staticmethod
    def _state_delta(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        for section in ("runtime_state", "environment"):
            before_section = before.get(section, {})
            after_section = after.get(section, {})
            section_delta: dict[str, Any] = {}
            for key in sorted(set(before_section) | set(after_section)):
                if before_section.get(key) != after_section.get(key):
                    section_delta[key] = {
                        "before": copy.deepcopy(before_section.get(key)),
                        "after": copy.deepcopy(after_section.get(key)),
                    }
            if section_delta:
                delta[section] = section_delta
        return delta

    @staticmethod
    def _rejected_result(
        context: GameContext,
        turn: int,
        utterance: str,
        actions: list[Any],
        error: ActionValidationError,
    ) -> TransitionResult:
        """Keep runtime state unchanged but retain the failed decision attempt."""

        rejected = context.clone()
        violation = {
            "code": error.code,
            "message": error.message,
            "type": "decision_violation",
        }
        rejected.append_history(
            {
                "turn": turn,
                "speaker": "npc",
                "utterance": utterance or "[invalid utterance]",
                "actions": copy.deepcopy(actions),
                "events": [
                    {
                        "id": f"violation_{turn}",
                        "type": "decision_violation",
                        "code": error.code,
                    }
                ],
                "context_delta": {},
            }
        )
        rejected.append_history(
            {
                "turn": turn,
                "speaker": "gamecore",
                "utterance": "动作未生效。",
                "events": [
                    {
                        "id": f"transition_{turn}",
                        "type": "transition_rejected",
                        "code": error.code,
                    }
                ],
            }
        )
        rejected.validate()
        return TransitionResult(
            context=rejected,
            accepted=False,
            events=(),
            violations=(violation,),
            public_observation="动作未生效。",
            preflight_failed=True,
        )
