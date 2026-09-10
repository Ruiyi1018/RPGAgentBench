"""Deterministic handlers for the benchmark actions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .context import GameContext, WorldDefinition
from .errors import ActionValidationError

Handler = Callable[
    [GameContext, WorldDefinition, Mapping[str, Any], str, int],
    dict[str, Any],
]


def _event(
    event_id: str,
    action: str,
    turn: int,
    **details: Any,
) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "action_applied",
        "action": action,
        "turn": turn,
        **details,
    }


def _require_event(context: GameContext, event_id: str) -> None:
    if _find_event(context, event_id) is None:
        raise ActionValidationError(
            "unknown_reason_event", f"历史中不存在事件: {event_id}"
        )


def _find_event(
    context: GameContext,
    event_id: str,
) -> Mapping[str, Any] | None:
    for entry in context.history:
        for event in entry.get("events", []):
            if isinstance(event, Mapping) and event.get("id") == event_id:
                return event
    return None


def _find_event_turn(context: GameContext, event_id: str) -> int | None:
    for entry in context.history:
        if any(
            isinstance(event, Mapping) and event.get("id") == event_id
            for event in entry.get("events", [])
        ):
            return int(entry["turn"])
    return None


def _inventory(context: GameContext, holder: str) -> list[str]:
    inventories = context.environment["inventories"]
    if holder not in inventories:
        raise ActionValidationError(
            "unknown_inventory_holder", f"未知物品持有者: {holder}"
        )
    inventory = inventories[holder]
    if not isinstance(inventory, list):
        raise ActionValidationError(
            "invalid_inventory", f"{holder}的inventory必须是数组"
        )
    return inventory


def _move_item(
    context: GameContext,
    item: str,
    source: str,
    destination: str,
) -> None:
    source_inventory = _inventory(context, source)
    if item not in source_inventory:
        raise ActionValidationError(
            "item_not_held", f"{source}未持有物品: {item}"
        )
    destination_inventory = _inventory(context, destination)
    source_inventory.remove(item)
    destination_inventory.append(item)
    artifact = context.environment["artifacts"].get(item)
    if isinstance(artifact, dict):
        artifact["holder"] = destination


def create_commitment(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    expected_action = parameters["expected_action"]
    if expected_action not in HANDLERS or expected_action in {
        "create_commitment",
        "resolve_commitment",
    }:
        raise ActionValidationError(
            "invalid_expected_action",
            f"承诺引用了不可履行的Action: {expected_action}",
        )
    if int(parameters["due_turn"]) <= turn:
        raise ActionValidationError(
            "invalid_commitment_deadline",
            "承诺due_turn必须晚于当前轮次",
        )
    interaction_deadline = context.environment.get("interaction_deadline")
    if (
        isinstance(interaction_deadline, int)
        and int(parameters["due_turn"]) > interaction_deadline
    ):
        raise ActionValidationError(
            "commitment_beyond_interaction",
            "承诺due_turn不能超过本次互动截止轮次",
        )
    expected_parameters = parameters["expected_parameters"]
    if expected_action == "update_task":
        task = next(
            (
                item
                for item in context.runtime_state["goals"]
                if isinstance(item, Mapping)
                and item.get("id") == expected_parameters.get("task_id")
            ),
            None,
        )
        if (
            task is not None
            and task.get("status", "active")
            == expected_parameters.get("status")
        ):
            raise ActionValidationError(
                "commitment_has_no_effect",
                "承诺要求的update_task不会产生状态变化",
            )
    commitments = context.runtime_state["commitments"]
    for commitment in commitments:
        if (
            commitment.get("target") == parameters["target"]
            and commitment.get("expected_action") == expected_action
            and commitment.get("expected_parameters")
            == parameters["expected_parameters"]
            and commitment.get("status", "active") == "active"
        ):
            raise ActionValidationError(
                "duplicate_commitment", "相同的有效行动承诺已存在"
            )
    commitment_id = f"commitment_{len(commitments) + 1:04d}"
    commitment = {
        "id": commitment_id,
        "target": parameters["target"],
        "content": parameters["content"],
        "expected_action": expected_action,
        "expected_parameters": dict(parameters["expected_parameters"]),
        "due_turn": int(parameters["due_turn"]),
        "status": "active",
        "created_turn": turn,
        "source_event": event_id,
    }
    commitments.append(commitment)
    return _event(
        event_id,
        "create_commitment",
        turn,
        commitment_id=commitment_id,
        target=parameters["target"],
    )


def resolve_commitment(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    reason_event = _find_event(context, parameters["reason_event"])
    if reason_event is None:
        raise ActionValidationError(
            "unknown_reason_event",
            f"历史中不存在事件: {parameters['reason_event']}",
        )
    commitment = next(
        (
            item
            for item in context.runtime_state["commitments"]
            if item.get("id") == parameters["commitment_id"]
        ),
        None,
    )
    if commitment is None:
        raise ActionValidationError(
            "unknown_commitment", "要处理的承诺不存在"
        )
    if commitment.get("status", "active") != "active":
        raise ActionValidationError(
            "commitment_already_resolved", "该承诺已经结束"
        )
    reason_turn = _find_event_turn(context, parameters["reason_event"])
    if (
        reason_turn is None
        or reason_turn < int(commitment.get("created_turn", 0))
    ):
        raise ActionValidationError(
            "commitment_reason_too_old",
            "收束承诺必须引用建立承诺之后发生的事件",
        )
    if (
        parameters["resolution"] == "fulfilled"
        and (
            reason_event.get("action") != commitment.get("expected_action")
            or any(
                reason_event.get(key) != value
                for key, value in commitment.get(
                    "expected_parameters",
                    {},
                ).items()
            )
        )
    ):
        raise ActionValidationError(
            "commitment_effect_mismatch",
            "履行事件的Action与承诺要求不一致",
        )
    commitment.update(
        {
            "status": parameters["resolution"],
            "resolved_turn": turn,
            "reason_event": parameters["reason_event"],
        }
    )
    return _event(
        event_id,
        "resolve_commitment",
        turn,
        commitment_id=parameters["commitment_id"],
        resolution=parameters["resolution"],
    )


_RELATIONSHIP_LEVELS = [
    "hostile",
    "distrustful",
    "neutral",
    "trusting",
    "loyal",
]


def update_relationship(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    _require_event(context, parameters["reason_event"])
    relationships = context.runtime_state["relationships"]
    target = parameters["target"]
    current = relationships.get(target, {"level": "neutral"})
    if isinstance(current, str):
        current = {"level": current}
    elif isinstance(current, int):
        current = {"level": current, "min": -2, "max": 2}
    elif not isinstance(current, dict):
        raise ActionValidationError(
            "invalid_relationship", f"{target}的关系状态无效"
        )
    reason_turn = _find_event_turn(context, parameters["reason_event"])
    if int(current.get("updated_turn", 0)) >= int(reason_turn or 0):
        raise ActionValidationError(
            "relationship_reason_not_new",
            "关系变化必须引用上次变化之后的新事件",
        )

    level = current.get("level", "neutral")
    direction = parameters["direction"]
    if isinstance(level, int):
        minimum = int(current.get("min", -2))
        maximum = int(current.get("max", 2))
        delta = {"increase": 1, "decrease": -1}[direction]
        new_level: int | str = level + delta
        if not minimum <= new_level <= maximum:
            raise ActionValidationError(
                "relationship_out_of_range", "关系状态已到允许边界"
            )
    elif level in _RELATIONSHIP_LEVELS:
        index = _RELATIONSHIP_LEVELS.index(level)
        delta = {"increase": 1, "decrease": -1}[direction]
        new_index = index + delta
        if not 0 <= new_index < len(_RELATIONSHIP_LEVELS):
            raise ActionValidationError(
                "relationship_out_of_range", "关系状态已到允许边界"
            )
        new_level = _RELATIONSHIP_LEVELS[new_index]
    else:
        raise ActionValidationError(
            "invalid_relationship_level", f"未知关系等级: {level}"
        )

    current.update(
        {
            "level": new_level,
            "last_reason_event": parameters["reason_event"],
            "updated_turn": turn,
        }
    )
    relationships[target] = current
    return _event(
        event_id,
        "update_relationship",
        turn,
        target=target,
        previous_level=level,
        new_level=new_level,
    )


def decide_claim(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    claims = context.runtime_state["claims"]
    claim = claims.get(parameters["claim_id"])
    if claim is None:
        raise ActionValidationError("unknown_claim", "要判断的claim不存在")
    if not isinstance(claim, dict):
        claim = {"content": claim}
        claims[parameters["claim_id"]] = claim
    evidence = parameters.get("evidence_event")
    if evidence is not None:
        _require_event(context, evidence)
    previous = claim.get("decision", "unassessed")
    previous_evidence = claim.get("evidence_event")
    if (
        previous == parameters["decision"]
        and (
            evidence is None
            or evidence == previous_evidence
            or int(_find_event_turn(context, evidence) or 0)
            <= int(claim.get("decided_turn", 0))
        )
    ):
        raise ActionValidationError(
            "claim_decision_unchanged",
            "相同主张判断必须引用不同于上次的新证据事件",
        )
    if (
        previous not in {"unassessed", parameters["decision"]}
        and evidence is None
    ):
        raise ActionValidationError(
            "claim_change_without_evidence",
            "改变既有主张判断时必须引用新证据事件",
        )
    claim["decision"] = parameters["decision"]
    if evidence is not None:
        claim["evidence_event"] = evidence
    claim["decided_turn"] = turn
    return _event(
        event_id,
        "decide_claim",
        turn,
        claim_id=parameters["claim_id"],
        decision=parameters["decision"],
        evidence_event=evidence,
    )


def reveal_fact(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    knowledge = context.runtime_state["knowledge"]
    fact_id = parameters["fact_id"]
    known = fact_id in knowledge
    if not known:
        raise ActionValidationError(
            "fact_not_known", f"NPC当前不知道事实: {fact_id}"
        )
    disclosures = context.runtime_state["disclosures"]
    disclosure = {
        "fact_id": fact_id,
        "recipient": parameters["recipient"],
        "turn": turn,
        "event_id": event_id,
    }
    disclosures.append(disclosure)
    return _event(
        event_id,
        "reveal_fact",
        turn,
        fact_id=fact_id,
        recipient=parameters["recipient"],
    )


def move(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    actor = context.character_id
    locations = context.environment["locations"]
    source = locations.get(actor)
    destination = parameters["destination"]
    if source is None:
        raise ActionValidationError("location_unknown", "NPC当前位置未知")
    if destination not in world.locations:
        raise ActionValidationError(
            "unknown_destination", f"目标地点不存在: {destination}"
        )
    if not world.connected(str(source), destination):
        raise ActionValidationError(
            "location_not_connected", f"{source}与{destination}不直接相连"
        )
    locations[actor] = destination
    return _event(
        event_id,
        "move",
        turn,
        actor=actor,
        source=source,
        destination=destination,
    )


def transfer_item(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    item = parameters["item"]
    if parameters["source"] != context.character_id:
        raise ActionValidationError(
            "item_source_not_actor", "NPC只能转移自己持有的物品"
        )
    if (
        item not in world.items
        and item not in context.environment["artifacts"]
    ):
        raise ActionValidationError("unknown_item", f"物品不存在: {item}")
    _move_item(context, item, parameters["source"], parameters["destination"])
    return _event(
        event_id,
        "transfer_item",
        turn,
        item=item,
        source=parameters["source"],
        destination=parameters["destination"],
    )


def create_artifact(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    artifact_id = parameters["artifact_id"].strip()
    content = parameters["content"].strip()
    if not artifact_id or not content:
        raise ActionValidationError(
            "invalid_artifact", "artifact_id和content必须是非空字符串"
        )
    artifacts = context.environment["artifacts"]
    if artifact_id in world.items or artifact_id in artifacts:
        raise ActionValidationError(
            "duplicate_artifact", f"物品或记录已存在: {artifact_id}"
        )
    creator = context.character_id
    artifacts[artifact_id] = {
        "id": artifact_id,
        "kind": parameters["kind"],
        "content": content,
        "creator": creator,
        "holder": creator,
        "created_turn": turn,
    }
    _inventory(context, creator).append(artifact_id)
    return _event(
        event_id,
        "create_artifact",
        turn,
        artifact_id=artifact_id,
        kind=parameters["kind"],
        creator=creator,
    )


def update_task(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    reason_event = parameters["reason_event"]
    _require_event(context, reason_event)
    task = next(
        (
            item
            for item in context.runtime_state["goals"]
            if isinstance(item, dict)
            and item.get("id") == parameters["task_id"]
        ),
        None,
    )
    if task is None:
        raise ActionValidationError(
            "unknown_task",
            f"目标或任务不存在: {parameters['task_id']}",
        )
    previous_status = task.get("status", "active")
    if previous_status == parameters["status"]:
        raise ActionValidationError(
            "task_status_unchanged",
            "update_task必须产生实际状态变化",
        )
    if int(task.get("updated_turn", 0)) >= int(
        _find_event_turn(context, reason_event) or 0
    ):
        raise ActionValidationError(
            "task_evidence_not_new",
            "任务变化必须引用上次变化之后的新事件",
        )
    task.update(
        {
            "status": parameters["status"],
            "reason_event": reason_event,
            "updated_turn": turn,
        }
    )
    return _event(
        event_id,
        "update_task",
        turn,
        task_id=parameters["task_id"],
        previous_status=previous_status,
        status=parameters["status"],
        reason_event=reason_event,
    )


def use_item(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    actor = context.character_id
    item = parameters["item"]
    if item not in _inventory(context, actor):
        raise ActionValidationError("item_not_held", "NPC未持有该物品")
    definition = world.items.get(item)
    if definition is None:
        raise ActionValidationError("unknown_item", f"物品不存在: {item}")
    effect = definition.get("use")
    if not isinstance(effect, Mapping):
        raise ActionValidationError(
            "unsupported_item_use", f"{item}没有确定性use规则"
        )
    target = parameters["target"]
    health_delta = effect.get("health_delta", 0)
    if not isinstance(health_delta, int):
        raise ActionValidationError(
            "invalid_item_rule", f"{item}.use.health_delta必须是整数"
        )
    if health_delta:
        health = context.environment["health"]
        if target not in health:
            raise ActionValidationError(
                "unknown_health_target", f"目标没有health状态: {target}"
            )
        health[target] = max(0, int(health[target]) + health_delta)
    if effect.get("consume", False):
        _inventory(context, actor).remove(item)
    return _event(
        event_id,
        "use_item",
        turn,
        actor=actor,
        item=item,
        target=target,
        health_delta=health_delta,
        consumed=bool(effect.get("consume", False)),
    )


def decide_access(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    _require_event(context, parameters["reason_event"])
    actor = context.character_id
    access = context.environment["access"]
    resource = parameters["resource"]
    record = access.get(resource)
    if not isinstance(record, dict):
        raise ActionValidationError(
            "unknown_access_resource", f"未配置访问资源: {resource}"
        )
    controllers = record.get("controllers", [])
    if actor not in controllers:
        raise ActionValidationError(
            "access_not_authorized", f"NPC无权决定资源访问: {resource}"
        )
    subjects = record.setdefault("subjects", {})
    subject = parameters["subject"]
    decision = parameters["decision"]
    if decision == "revoke":
        subjects.pop(subject, None)
    else:
        subjects[subject] = "granted" if decision == "grant" else "denied"
    return _event(
        event_id,
        "decide_access",
        turn,
        subject=subject,
        resource=resource,
        decision=decision,
        reason_event=parameters["reason_event"],
    )


def attack(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    actor = context.character_id
    target = parameters["target"]
    method = parameters["method"]
    health = context.environment["health"]
    if target not in health:
        raise ActionValidationError(
            "unknown_health_target", f"目标没有health状态: {target}"
        )
    if int(health[target]) <= 0:
        raise ActionValidationError("target_incapacitated", "目标已失去行动能力")
    allowed_methods = world.combat.get("allowed_methods")
    if allowed_methods and method not in allowed_methods:
        raise ActionValidationError(
            "attack_method_not_allowed", f"不允许的攻击方式: {method}"
        )
    required_items = world.combat.get("requires_item", {})
    required_item = required_items.get(method)
    if required_item and required_item not in _inventory(context, actor):
        raise ActionValidationError(
            "required_item_missing", f"攻击方式需要物品: {required_item}"
        )
    damage = world.attack_damage(method)
    previous_health = int(health[target])
    health[target] = max(0, previous_health - damage)
    return _event(
        event_id,
        "attack",
        turn,
        actor=actor,
        target=target,
        method=method,
        damage=damage,
        previous_health=previous_health,
        new_health=health[target],
    )


def end_dialogue(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    if context.environment["dialogue_status"] != "active":
        raise ActionValidationError("dialogue_already_ended", "对话已经结束")
    context.environment["dialogue_status"] = "ended"
    return _event(
        event_id,
        "end_dialogue",
        turn,
        reason=parameters["reason"],
    )


HANDLERS: dict[str, Handler] = {
    "create_commitment": create_commitment,
    "resolve_commitment": resolve_commitment,
    "update_relationship": update_relationship,
    "decide_claim": decide_claim,
    "reveal_fact": reveal_fact,
    "move": move,
    "transfer_item": transfer_item,
    "create_artifact": create_artifact,
    "update_task": update_task,
    "use_item": use_item,
    "decide_access": decide_access,
    "attack": attack,
    "end_dialogue": end_dialogue,
}
