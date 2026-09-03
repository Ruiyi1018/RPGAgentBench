from __future__ import annotations

from pathlib import Path

import pytest

from gamecore import ActionRegistry, GameContext, GameCoreEngine, WorldDefinition


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry.load_directory(ROOT / "gamecore" / "actions")


@pytest.fixture
def world() -> WorldDefinition:
    return WorldDefinition.from_mapping(
        {
            "world_id": "test_world",
            "locations": [
                {"id": "office", "connected_to": ["hall"]},
                {"id": "hall", "connected_to": ["office"]},
            ],
            "items": [
                {"id": "letter", "type": "document"},
                {
                    "id": "medicine",
                    "type": "consumable",
                    "use": {"health_delta": 20, "consume": True},
                },
                {"id": "pistol", "type": "weapon"},
            ],
            "combat": {
                "default_damage": 10,
                "allowed_methods": ["pistol"],
                "requires_item": {"pistol": "pistol"},
            },
        }
    )


@pytest.fixture
def context() -> GameContext:
    return GameContext(
        {
            "character_card": {"character_id": "npc"},
            "runtime_state": {
                "claims": {
                    "claim_1": {
                        "content": "门已经打开",
                        "decision": "unassessed",
                    }
                },
                "commitments": [],
                "relationships": {"player": {"level": "neutral"}},
                "goals": [],
                "disclosures": [],
                "knowledge": {"secret_1": {"content": "测试秘密"}},
            },
            "environment": {
                "health": {"npc": 80, "player": 100},
                "inventories": {
                    "npc": ["letter", "medicine", "pistol"],
                    "player": [],
                },
                "locations": {"npc": "office", "player": "hall"},
                "access": {
                    "archive": {
                        "controllers": ["npc"],
                        "subjects": {},
                    }
                },
                "offers": [],
                "task_status": "active",
                "dialogue_status": "active",
            },
            "history": [
                {
                    "turn": 1,
                    "speaker": "player",
                    "utterance": "请作出决定。",
                    "events": [{"id": "seed_event", "type": "scenario_event"}],
                }
            ],
        }
    )


@pytest.fixture
def engine(
    registry: ActionRegistry,
    world: WorldDefinition,
) -> GameCoreEngine:
    return GameCoreEngine(registry, world)


def run_action(
    engine: GameCoreEngine,
    context: GameContext,
    name: str,
    parameters: dict,
):
    return engine.step(
        context,
        {
            "utterance": "执行这个决定。",
            "actions": [{"name": name, "parameters": parameters}],
        },
    )


def test_registry_loads_all_thirteen_actions(registry: ActionRegistry) -> None:
    assert len(registry.names) == 13
    assert "create_commitment" in registry.names
    assert "end_dialogue" in registry.names


def test_commitment_lifecycle(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    created = run_action(
        engine,
        context,
        "create_commitment",
        {"target": "player", "content": "归还信件"},
    )
    assert created.accepted
    commitment = created.context.runtime_state["commitments"][0]
    assert commitment["status"] == "active"

    second_turn = engine.append_player_query(created.context, "事情已经办完。")
    resolved = run_action(
        engine,
        second_turn,
        "resolve_commitment",
        {
            "commitment_id": commitment["id"],
            "resolution": "fulfilled",
            "reason_event": "event_1_1",
        },
    )
    assert resolved.accepted
    assert resolved.context.runtime_state["commitments"][0]["status"] == "fulfilled"


def test_role_state_actions(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    relationship = run_action(
        engine,
        context,
        "update_relationship",
        {
            "target": "player",
            "direction": "increase",
            "reason_event": "seed_event",
        },
    )
    assert relationship.context.runtime_state["relationships"]["player"]["level"] == "trusting"

    claim = run_action(
        engine,
        context,
        "accept_claim",
        {
            "claim_id": "claim_1",
            "decision": "uncertain",
            "evidence_event": "seed_event",
        },
    )
    assert claim.context.runtime_state["claims"]["claim_1"]["decision"] == "uncertain"

    disclosure = run_action(
        engine,
        context,
        "reveal_fact",
        {"fact_id": "secret_1", "recipient": "player"},
    )
    assert disclosure.context.runtime_state["disclosures"][0]["fact_id"] == "secret_1"


def test_world_state_actions(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    moved = run_action(engine, context, "move", {"destination": "hall"})
    assert moved.context.environment["locations"]["npc"] == "hall"

    transferred = run_action(
        engine,
        context,
        "transfer_item",
        {"item": "letter", "source": "npc", "destination": "player"},
    )
    assert "letter" in transferred.context.environment["inventories"]["player"]

    used = run_action(
        engine,
        context,
        "use_item",
        {"item": "medicine", "target": "npc"},
    )
    assert used.context.environment["health"]["npc"] == 100
    assert "medicine" not in used.context.environment["inventories"]["npc"]

    access = run_action(
        engine,
        context,
        "decide_access",
        {"subject": "player", "resource": "archive", "decision": "grant"},
    )
    assert (
        access.context.environment["access"]["archive"]["subjects"]["player"]
        == "granted"
    )

    attacked = run_action(
        engine,
        context,
        "attack",
        {"target": "player", "method": "pistol"},
    )
    assert attacked.context.environment["health"]["player"] == 90

    ended = run_action(
        engine,
        context,
        "end_dialogue",
        {"reason": "会面已经结束"},
    )
    assert ended.context.environment["dialogue_status"] == "ended"


def test_offer_actions(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    created = run_action(
        engine,
        context,
        "create_offer",
        {
            "recipient": "player",
            "offered_items": ["letter"],
            "requested_items": [],
        },
    )
    assert created.accepted
    assert created.context.environment["offers"][0]["status"] == "pending"
    assert "letter" in created.context.environment["inventories"]["npc"]

    offered_to_npc = context.clone()
    offered_to_npc.environment["inventories"]["player"].append("letter")
    offered_to_npc.environment["inventories"]["npc"].remove("letter")
    offered_to_npc.environment["offers"].append(
        {
            "id": "offer_external",
            "proposer": "player",
            "recipient": "npc",
            "offered_items": ["letter"],
            "requested_items": [],
            "status": "pending",
            "created_turn": 1,
        }
    )
    accepted = run_action(
        engine,
        offered_to_npc,
        "respond_offer",
        {"offer_id": "offer_external", "decision": "accept"},
    )
    assert accepted.accepted
    assert "letter" in accepted.context.environment["inventories"]["npc"]
    assert accepted.context.environment["offers"][0]["status"] == "accepted"


def test_invalid_action_is_failure_and_transaction_is_atomic(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    result = engine.step(
        context,
        {
            "utterance": "我先离开，再交出不存在的东西。",
            "actions": [
                {"name": "move", "parameters": {"destination": "hall"}},
                {
                    "name": "transfer_item",
                    "parameters": {
                        "item": "missing",
                        "source": "npc",
                        "destination": "player",
                    },
                },
            ],
        },
    )
    assert not result.accepted
    assert result.violations[0]["type"] == "decision_violation"
    assert result.context.environment["locations"]["npc"] == "office"
    assert result.context.history[-1]["utterance"] == "动作未生效。"


def test_no_action_turn_is_valid(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    result = engine.step(
        context,
        {"utterance": "我需要先核实你的说法。", "actions": []},
    )
    assert result.accepted
    assert not result.events
