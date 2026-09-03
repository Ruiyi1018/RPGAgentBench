"""Deterministic handlers for the 13 benchmark actions."""

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
    if not context.event_exists(event_id):
        raise ActionValidationError(
            "unknown_reason_event", f"历史中不存在事件: {event_id}"
        )


def _inventory(context: GameContext, holder: str) -> list[str]:
    inventories = context.environment["inventories"]
    inventory = inventories.setdefault(holder, [])
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


def create_commitment(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    commitments = context.runtime_state["commitments"]
    for commitment in commitments:
        if (
            commitment.get("target") == parameters["target"]
            and commitment.get("content") == parameters["content"]
            and commitment.get("status", "active") == "active"
        ):
            raise ActionValidationError(
                "duplicate_commitment", "相同的有效承诺已存在"
            )
    commitment_id = f"commitment_{len(commitments) + 1:04d}"
    commitment = {
        "id": commitment_id,
        "target": parameters["target"],
        "content": parameters["content"],
        "condition": parameters.get("condition"),
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
    _require_event(context, parameters["reason_event"])
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

    level = current.get("level", "neutral")
    direction = parameters["direction"]
    if isinstance(level, int):
        minimum = int(current.get("min", -2))
        maximum = int(current.get("max", 2))
        delta = {"increase": 1, "decrease": -1, "maintain": 0}[direction]
        new_level: int | str = level + delta
        if not minimum <= new_level <= maximum:
            raise ActionValidationError(
                "relationship_out_of_range", "关系状态已到允许边界"
            )
    elif level in _RELATIONSHIP_LEVELS:
        index = _RELATIONSHIP_LEVELS.index(level)
        delta = {"increase": 1, "decrease": -1, "maintain": 0}[direction]
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


def accept_claim(
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
    claim.update(
        {
            "decision": parameters["decision"],
            "evidence_event": evidence,
            "decided_turn": turn,
        }
    )
    return _event(
        event_id,
        "accept_claim",
        turn,
        claim_id=parameters["claim_id"],
        decision=parameters["decision"],
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
    if item not in world.items:
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


def create_offer(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    actor = context.character_id
    offered_items = list(parameters["offered_items"])
    requested_items = list(parameters.get("requested_items", []))
    if not offered_items and not requested_items:
        raise ActionValidationError(
            "empty_offer", "Offer必须包含给予或请求的物品"
        )
    unknown = (set(offered_items) | set(requested_items)) - set(world.items)
    if unknown:
        raise ActionValidationError(
            "unknown_item", f"Offer包含未知物品: {', '.join(sorted(unknown))}"
        )
    actor_inventory = _inventory(context, actor)
    missing = set(offered_items) - set(actor_inventory)
    if missing:
        raise ActionValidationError(
            "item_not_held", f"NPC未持有: {', '.join(sorted(missing))}"
        )
    offers = context.environment["offers"]
    offer_id = f"offer_{len(offers) + 1:04d}"
    offers.append(
        {
            "id": offer_id,
            "proposer": actor,
            "recipient": parameters["recipient"],
            "offered_items": offered_items,
            "requested_items": requested_items,
            "status": "pending",
            "created_turn": turn,
        }
    )
    return _event(
        event_id,
        "create_offer",
        turn,
        offer_id=offer_id,
        recipient=parameters["recipient"],
    )


def respond_offer(
    context: GameContext,
    world: WorldDefinition,
    parameters: Mapping[str, Any],
    event_id: str,
    turn: int,
) -> dict[str, Any]:
    del world
    actor = context.character_id
    offer = next(
        (
            item
            for item in context.environment["offers"]
            if item.get("id") == parameters["offer_id"]
        ),
        None,
    )
    if offer is None:
        raise ActionValidationError("unknown_offer", "Offer不存在")
    if offer.get("status") != "pending":
        raise ActionValidationError("offer_closed", "Offer已经结束")
    if offer.get("recipient") != actor:
        raise ActionValidationError(
            "not_offer_recipient", "NPC不是该Offer的接收方"
        )

    decision = parameters["decision"]
    if decision == "accept":
        proposer = str(offer["proposer"])
        offered_items = list(offer.get("offered_items", []))
        requested_items = list(offer.get("requested_items", []))
        proposer_inventory = _inventory(context, proposer)
        actor_inventory = _inventory(context, actor)
        if not set(offered_items) <= set(proposer_inventory):
            raise ActionValidationError(
                "offer_items_unavailable", "提议方已不再持有全部物品"
            )
        if not set(requested_items) <= set(actor_inventory):
            raise ActionValidationError(
                "requested_items_unavailable", "NPC不再持有全部交换物品"
            )
        for item in offered_items:
            _move_item(context, item, proposer, actor)
        for item in requested_items:
            _move_item(context, item, actor, proposer)
        offer["status"] = "accepted"
    else:
        offer["status"] = "rejected"
    offer["resolved_turn"] = turn
    return _event(
        event_id,
        "respond_offer",
        turn,
        offer_id=parameters["offer_id"],
        decision=decision,
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
    "accept_claim": accept_claim,
    "reveal_fact": reveal_fact,
    "move": move,
    "transfer_item": transfer_item,
    "create_offer": create_offer,
    "respond_offer": respond_offer,
    "use_item": use_item,
    "decide_access": decide_access,
    "attack": attack,
    "end_dialogue": end_dialogue,
}
