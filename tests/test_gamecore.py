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
                "goals": [
                    {
                        "id": "verify_request",
                        "content": "核验请求",
                        "status": "active",
                    }
                ],
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


def test_registry_loads_thirteen_non_overlapping_actions(
    registry: ActionRegistry,
) -> None:
    assert len(registry.names) == 13
    assert "decide_claim" in registry.names
    assert "create_offer" not in registry.names
    assert "respond_offer" not in registry.names
    assert "create_commitment" in registry.names
    assert "create_artifact" in registry.names
    assert "update_task" in registry.names
    assert "end_dialogue" in registry.names


def test_end_dialogue_must_be_final_action(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    result = engine.step(
        context,
        {
            "utterance": "先结束，再补任务。",
            "actions": [
                {
                    "name": "end_dialogue",
                    "parameters": {"reason": "done"},
                },
                {
                    "name": "update_task",
                    "parameters": {
                        "task_id": "verify_request",
                        "status": "completed",
                        "reason_event": "seed_event",
                    },
                },
            ],
        },
    )

    assert result.accepted is False
    assert result.violations[0]["code"] == "end_dialogue_not_final"


def test_commitment_lifecycle(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    created = run_action(
        engine,
        context,
        "create_commitment",
        {
            "target": "player",
            "content": "前往大厅",
            "expected_action": "move",
            "expected_parameters": {"destination": "hall"},
            "due_turn": 4,
        },
    )
    assert created.accepted
    commitment = created.context.runtime_state["commitments"][0]
    assert commitment["status"] == "active"

    second_turn = engine.append_player_query(created.context, "现在去吧。")
    performed = run_action(
        engine,
        second_turn,
        "move",
        {"destination": "hall"},
    )
    third_turn = engine.append_player_query(performed.context, "事情已经办完。")
    resolved = run_action(
        engine,
        third_turn,
        "resolve_commitment",
        {
            "commitment_id": commitment["id"],
            "resolution": "fulfilled",
            "reason_event": "event_2_1",
        },
    )
    assert resolved.accepted
    assert resolved.context.runtime_state["commitments"][0]["status"] == "fulfilled"


def test_commitment_cannot_be_fulfilled_by_unrelated_action(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    created = run_action(
        engine,
        context,
        "create_commitment",
        {
            "target": "player",
            "content": "前往大厅",
            "expected_action": "move",
            "expected_parameters": {"destination": "hall"},
            "due_turn": 5,
        },
    )
    second_turn = engine.append_player_query(created.context, "先作判断。")
    decided = run_action(
        engine,
        second_turn,
        "decide_claim",
        {
            "claim_id": "claim_1",
            "decision": "uncertain",
            "evidence_event": "seed_event",
        },
    )
    third_turn = engine.append_player_query(decided.context, "承诺完成了吗？")
    rejected = run_action(
        engine,
        third_turn,
        "resolve_commitment",
        {
            "commitment_id": "commitment_0001",
            "resolution": "fulfilled",
            "reason_event": "event_2_1",
        },
    )

    assert not rejected.accepted
    assert (
        rejected.violations[0]["code"]
        == "commitment_effect_mismatch"
    )


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
        "decide_claim",
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


def test_decide_claim_rejects_same_decision_without_new_evidence(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    first = run_action(
        engine,
        context,
        "decide_claim",
        {
            "claim_id": "claim_1",
            "decision": "uncertain",
            "evidence_event": "seed_event",
        },
    )
    second_turn = engine.append_player_query(
        first.context,
        "没有新证据，请再判断一次。",
    )
    repeated = run_action(
        engine,
        second_turn,
        "decide_claim",
        {
            "claim_id": "claim_1",
            "decision": "uncertain",
        },
    )

    assert not repeated.accepted
    assert repeated.violations[0]["code"] == "claim_decision_unchanged"
    assert (
        first.context.runtime_state["claims"]["claim_1"]["evidence_event"]
        == "seed_event"
    )


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

    artifact = run_action(
        engine,
        context,
        "create_artifact",
        {
            "artifact_id": "public_note",
            "kind": "note",
            "content": "公开日期存在差异",
        },
    )
    assert artifact.accepted
    assert "public_note" in artifact.context.environment["inventories"]["npc"]
    assert (
        artifact.context.environment["artifacts"]["public_note"]["content"]
        == "公开日期存在差异"
    )

    task = run_action(
        engine,
        context,
        "update_task",
        {
            "task_id": "verify_request",
            "status": "completed",
            "reason_event": "seed_event",
        },
    )
    assert task.accepted
    assert task.context.runtime_state["goals"][0]["status"] == "completed"

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
        {
            "subject": "player",
            "resource": "archive",
            "decision": "grant",
            "reason_event": "seed_event",
        },
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


def test_failed_action_does_not_roll_back_the_whole_turn(
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
    assert [violation["code"] for violation in result.violations] == [
        "unknown_item"
    ]
    assert result.context.environment["locations"]["npc"] == "hall"
    assert len(result.events) == 1
    assert result.events[0]["action"] == "move"
    npc_entry = result.context.history[-2]
    assert [event["type"] for event in npc_entry["events"]] == [
        "action_applied",
        "decision_violation",
    ]
    assert result.context.history[-1]["utterance"] == (
        "部分动作已执行，其余未生效。"
    )
    assert result.context.history[-1]["events"][0]["type"] == (
        "transition_partial"
    )


def test_malformed_action_still_rejects_whole_turn(
    engine: GameCoreEngine,
    context: GameContext,
) -> None:
    result = engine.step(
        context,
        {
            "utterance": "我先离开，再用一个不存在的Action。",
            "actions": [
                {"name": "move", "parameters": {"destination": "hall"}},
                {"name": "teleport", "parameters": {}},
            ],
        },
    )
    assert not result.accepted
    assert result.violations[0]["code"] == "unknown_action"
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
