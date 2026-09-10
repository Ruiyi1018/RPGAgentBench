"""Generic executable contracts for Stage 3 action trajectories."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

from gamecore import GameContext


FAILURE_TYPES = {
    "illegal_transition",
    "missing_transition",
    "trajectory_conflict",
}
TRANSITION_KINDS = {"forbidden", "required"}
TRACE_KINDS = {"invariant", "precedence", "must_follow"}


@dataclass(frozen=True)
class ContractStep:
    runtime: dict[str, Any]
    violations: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]


class ContractSpecError(ValueError):
    """Raised when a Stage 3 contract is not executable."""


class Stage3ContractEngine:
    """Evaluate action transitions and temporal constraints deterministically."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        action_names: Collection[str],
    ) -> None:
        self.spec = copy.deepcopy(dict(spec))
        validate_contract_spec(self.spec, action_names=action_names)
        self._contracts = {
            str(item["id"]): item
            for section in ("transition_contracts", "trace_contracts")
            for item in self.spec.get(section, [])
        }

    def initial_runtime(self) -> dict[str, Any]:
        return {
            "contracts": {
                contract_id: {
                    "status": "inactive",
                    "activated_turn": None,
                    "opportunities": 0,
                }
                for contract_id in self._contracts
            }
        }

    def evaluate(
        self,
        before: GameContext,
        after: GameContext,
        *,
        turn: int,
        mode: str,
        active_challenge_id: str | None,
        runtime: Mapping[str, Any],
    ) -> ContractStep:
        updated = copy.deepcopy(dict(runtime))
        states = updated.setdefault("contracts", {})
        violations: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        current_events = _npc_action_events(after, turn)

        for contract in self.spec.get("transition_contracts", []):
            if not _applies_to_mode(contract, mode):
                continue
            state = states.setdefault(
                str(contract["id"]),
                {
                    "status": "inactive",
                    "activated_turn": None,
                    "opportunities": 0,
                },
            )
            kind = str(contract["kind"])
            if kind == "forbidden":
                active = _predicate_matches(
                    before,
                    contract.get("when"),
                    turn=turn,
                    current_events=current_events,
                )
                if active:
                    state["status"] = "active"
                    state["activated_turn"] = (
                        state.get("activated_turn") or turn
                    )
                matched = _matching_event(
                    current_events,
                    contract["action"],
                )
                if active and matched is not None:
                    state["status"] = "violated"
                    violations.append(
                        _violation(
                            turn,
                            "illegal_transition",
                            contract,
                            (
                                f"Action {matched.get('action')}在当前状态下"
                                "不属于允许转移"
                            ),
                        )
                    )
                continue

            trigger = _predicate_matches(
                before,
                contract["activate_when"],
                turn=turn,
                current_events=current_events,
            )
            if state["status"] == "inactive" and trigger:
                state.update(
                    {
                        "status": "active",
                        "activated_turn": turn,
                        "opportunities": 0,
                    }
                )
            if state["status"] != "active":
                continue
            if _required_target_reached(
                after,
                contract,
                turn=turn,
                current_events=current_events,
            ):
                state["status"] = "satisfied"
                state["satisfied_turn"] = turn
                continue
            if _is_opportunity(contract, active_challenge_id):
                state["opportunities"] = int(state["opportunities"]) + 1
            if int(state["opportunities"]) >= int(
                contract["opportunity_limit"]
            ):
                state["status"] = "violated"
                violations.append(
                    _violation(
                        turn,
                        "missing_transition",
                        contract,
                        (
                            "触发条件已经成立，但在"
                            f"{state['opportunities']}次相关决策机会后"
                            "仍未到达要求的后继状态"
                        ),
                    )
                )

        for contract in self.spec.get("trace_contracts", []):
            if not _applies_to_mode(contract, mode):
                continue
            state = states.setdefault(
                str(contract["id"]),
                {
                    "status": "inactive",
                    "activated_turn": None,
                    "opportunities": 0,
                },
            )
            kind = str(contract["kind"])
            if kind == "invariant":
                if state["status"] == "inactive" and _predicate_matches(
                    before,
                    contract["activate_when"],
                    turn=turn,
                    current_events=current_events,
                ):
                    state.update(
                        {
                            "status": "active",
                            "activated_turn": turn,
                            "opportunities": 0,
                        }
                    )
                if state["status"] != "active":
                    continue
                if _predicate_matches(
                    after,
                    contract.get("resolve_when"),
                    turn=turn,
                    current_events=current_events,
                ):
                    state["status"] = "satisfied"
                    state["satisfied_turn"] = turn
                    continue
                if not _predicate_matches(
                    after,
                    contract["must_hold"],
                    turn=turn,
                    current_events=current_events,
                ):
                    state["status"] = "violated"
                    violations.append(
                        _violation(
                            turn,
                            "trajectory_conflict",
                            contract,
                            "轨迹破坏了激活期间必须保持的状态不变量",
                        )
                    )
                continue

            if kind == "precedence":
                matched_index = next(
                    (
                        index
                        for index, event in enumerate(current_events)
                        if _matching_event(
                            [event],
                            contract["action"],
                        )
                        is not None
                    ),
                    None,
                )
                if matched_index is not None:
                    state["activated_turn"] = turn
                    if not _predicate_matches(
                        before,
                        contract["required_before"],
                        turn=turn,
                        current_events=current_events[:matched_index],
                    ):
                        state["status"] = "violated"
                        violations.append(
                            _violation(
                                turn,
                                "trajectory_conflict",
                                contract,
                                "Action在要求的前置轨迹完成前发生",
                            )
                        )
                    else:
                        state["status"] = "satisfied"
                        state["satisfied_turn"] = turn
                continue

            if state["status"] == "inactive" and _matching_event(
                current_events,
                contract["trigger_action"],
            ) is not None:
                state.update(
                    {
                        "status": "active",
                        "activated_turn": turn,
                        "opportunities": 0,
                    }
                )
            if state["status"] != "active":
                continue
            if _matching_event(
                current_events,
                contract["follow_action"],
            ) is not None:
                state["status"] = "satisfied"
                state["satisfied_turn"] = turn
                continue
            if _is_opportunity(contract, active_challenge_id):
                state["opportunities"] = int(state["opportunities"]) + 1
            if int(state["opportunities"]) >= int(
                contract["opportunity_limit"]
            ):
                state["status"] = "violated"
                violations.append(
                    _violation(
                        turn,
                        "trajectory_conflict",
                        contract,
                        "前序Action发生后未在规定机会内完成后续Action",
                    )
                )

        overdue = next(
            (
                item
                for item in after.runtime_state["commitments"]
                if isinstance(item, Mapping)
                and item.get("status", "active") == "active"
                and isinstance(item.get("due_turn"), int)
                and int(item["due_turn"]) <= turn
            ),
            None,
        )
        if overdue is not None:
            violations.append(
                {
                    "turn": turn,
                    "failure": True,
                    "failure_type": "trajectory_conflict",
                    "violated_rule": "commitment_lifecycle",
                    "evidence": [
                        f"承诺{overdue.get('id')}到期后仍处于active"
                    ],
                    "cannot_determine": False,
                    "source": "contract_engine",
                }
            )

        return ContractStep(
            runtime=updated,
            violations=tuple(violations),
            diagnostics=tuple(diagnostics),
        )


def validate_contract_spec(
    spec: Mapping[str, Any],
    *,
    action_names: Collection[str],
) -> None:
    allowed_actions = set(action_names)
    known_ids: set[str] = set()
    for section, kinds in (
        ("transition_contracts", TRANSITION_KINDS),
        ("trace_contracts", TRACE_KINDS),
    ):
        entries = spec.get(section, [])
        if not isinstance(entries, list):
            raise ContractSpecError(f"{section}必须是数组")
        for item in entries:
            if not isinstance(item, Mapping):
                raise ContractSpecError(f"{section}条目必须是对象")
            contract_id = item.get("id")
            kind = item.get("kind")
            if not isinstance(contract_id, str) or not contract_id:
                raise ContractSpecError(f"{section}条目缺少有效id")
            if contract_id in known_ids:
                raise ContractSpecError(f"契约id重复: {contract_id}")
            known_ids.add(contract_id)
            if kind not in kinds:
                raise ContractSpecError(
                    f"{contract_id}.kind必须属于{sorted(kinds)}"
                )
            if section == "transition_contracts":
                if kind == "forbidden":
                    _validate_action_pattern(
                        item.get("action"),
                        allowed_actions,
                        contract_id,
                    )
                else:
                    if not isinstance(item.get("activate_when"), Mapping):
                        raise ContractSpecError(
                            f"{contract_id}.activate_when必须是对象"
                        )
                    if not isinstance(
                        item.get("opportunity_limit"), int
                    ) or int(item["opportunity_limit"]) < 1:
                        raise ContractSpecError(
                            f"{contract_id}.opportunity_limit必须为正整数"
                        )
                    if "action" not in item and "target_state" not in item:
                        raise ContractSpecError(
                            f"{contract_id}必须包含action或target_state"
                        )
                    if "action" in item:
                        _validate_action_pattern(
                            item["action"],
                            allowed_actions,
                            contract_id,
                        )
            elif kind == "precedence":
                _validate_action_pattern(
                    item.get("action"),
                    allowed_actions,
                    contract_id,
                )
                if not isinstance(item.get("required_before"), Mapping):
                    raise ContractSpecError(
                        f"{contract_id}.required_before必须是对象"
                    )
            elif kind == "must_follow":
                _validate_action_pattern(
                    item.get("trigger_action"),
                    allowed_actions,
                    contract_id,
                )
                _validate_action_pattern(
                    item.get("follow_action"),
                    allowed_actions,
                    contract_id,
                )
                if not isinstance(
                    item.get("opportunity_limit"), int
                ) or int(item["opportunity_limit"]) < 1:
                    raise ContractSpecError(
                        f"{contract_id}.opportunity_limit必须为正整数"
                    )
            else:
                for key in ("activate_when", "must_hold"):
                    if not isinstance(item.get(key), Mapping):
                        raise ContractSpecError(
                            f"{contract_id}.{key}必须是对象"
                        )


def _validate_action_pattern(
    value: Any,
    allowed_actions: set[str],
    contract_id: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("name") not in allowed_actions
        or not isinstance(value.get("parameters", {}), Mapping)
    ):
        raise ContractSpecError(
            f"{contract_id}包含无效Action匹配模式"
        )


def _applies_to_mode(contract: Mapping[str, Any], mode: str) -> bool:
    return mode in contract.get("modes", ["normal", "pressure"])


def _is_opportunity(
    contract: Mapping[str, Any],
    active_challenge_id: str | None,
) -> bool:
    challenge_ids = contract.get("challenge_ids", [])
    return (
        not challenge_ids
        or active_challenge_id is not None
        and active_challenge_id in challenge_ids
    )


def _required_target_reached(
    context: GameContext,
    contract: Mapping[str, Any],
    *,
    turn: int,
    current_events: Sequence[Mapping[str, Any]],
) -> bool:
    action_ok = (
        "action" not in contract
        or _matching_event(current_events, contract["action"]) is not None
    )
    state_ok = (
        "target_state" not in contract
        or _predicate_matches(
            context,
            contract["target_state"],
            turn=turn,
            current_events=current_events,
        )
    )
    return action_ok and state_ok


def _predicate_matches(
    context: GameContext,
    predicate: Any,
    *,
    turn: int,
    current_events: Sequence[Mapping[str, Any]],
) -> bool:
    if predicate is None:
        return False
    if not isinstance(predicate, Mapping):
        raise ContractSpecError("状态谓词必须是对象")
    if "all" in predicate:
        return all(
            _predicate_matches(
                context,
                item,
                turn=turn,
                current_events=current_events,
            )
            for item in predicate["all"]
        )
    if "any" in predicate:
        return any(
            _predicate_matches(
                context,
                item,
                turn=turn,
                current_events=current_events,
            )
            for item in predicate["any"]
        )
    if "not" in predicate:
        return not _predicate_matches(
            context,
            predicate["not"],
            turn=turn,
            current_events=current_events,
        )
    if "event" in predicate:
        return context.event_exists(str(predicate["event"]))
    if "action" in predicate:
        scope = predicate.get("scope", "history")
        events = (
            current_events
            if scope == "current"
            else _historical_action_events(
                context,
                turn=turn,
                current_events=current_events,
            )
        )
        return _matching_event(events, predicate["action"]) is not None
    if "state" in predicate:
        actual = _state_value(context, str(predicate["state"]))
        if "equals" in predicate:
            return actual == predicate["equals"]
        if "in" in predicate:
            return actual in predicate["in"]
        if "not_equals" in predicate:
            return actual != predicate["not_equals"]
        raise ContractSpecError("state谓词缺少equals、in或not_equals")
    if "turn_at_least" in predicate:
        return turn >= int(predicate["turn_at_least"])
    raise ContractSpecError(f"未知状态谓词: {sorted(predicate)}")


def _state_value(context: GameContext, path: str) -> Any:
    current: Any = context.data
    for segment in path.split("."):
        if isinstance(current, Mapping):
            if segment not in current:
                return None
            current = current[segment]
            continue
        if isinstance(current, list):
            current = next(
                (
                    item
                    for item in current
                    if isinstance(item, Mapping)
                    and str(item.get("id")) == segment
                ),
                None,
            )
            if current is None:
                return None
            continue
        return None
    return current


def _matching_event(
    events: Sequence[Mapping[str, Any]],
    pattern: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for event in events:
        if event.get("action") != pattern.get("name"):
            continue
        if all(
            _value_matches(event.get(key), expected)
            for key, expected in pattern.get("parameters", {}).items()
        ):
            return event
    return None


def _value_matches(actual: Any, expected: Any) -> bool:
    return actual in expected if isinstance(expected, list) else actual == expected


def _npc_action_events(
    context: GameContext,
    turn: int,
) -> list[Mapping[str, Any]]:
    return [
        event
        for entry in context.history
        if entry.get("speaker") == "npc"
        and int(entry.get("turn", 0)) == turn
        for event in entry.get("events", [])
        if isinstance(event, Mapping)
        and event.get("type") == "action_applied"
    ]


def _historical_action_events(
    context: GameContext,
    *,
    turn: int,
    current_events: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    prior = [
        event
        for entry in context.history
        if entry.get("speaker") == "npc"
        and int(entry.get("turn", 0)) < turn
        for event in entry.get("events", [])
        if isinstance(event, Mapping)
        and event.get("type") == "action_applied"
    ]
    return [*prior, *current_events]


def _violation(
    turn: int,
    failure_type: str,
    contract: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    if failure_type not in FAILURE_TYPES:
        raise ValueError(f"未知Stage3失败类型: {failure_type}")
    return {
        "turn": turn,
        "failure": True,
        "failure_type": failure_type,
        "violated_rule": str(contract["id"]),
        "evidence": [reason],
        "cannot_determine": False,
        "source": "contract_engine",
    }
