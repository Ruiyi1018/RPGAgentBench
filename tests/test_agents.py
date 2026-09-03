from __future__ import annotations

from pathlib import Path

import pytest

from agents import (
    ConsistencyChecker,
    NPCAgent,
    PlayerAgent,
    PlayerMode,
    PromptBuilder,
    StateAccess,
)
from gamecore import ActionRegistry, GameContext, GameCoreEngine, WorldDefinition
from llm import GenerationConfig, StaticLLMClient


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry.load_directory(ROOT / "gamecore" / "actions")


@pytest.fixture
def context() -> GameContext:
    return GameContext(
        {
            "character_card": {
                "character_id": "npc",
                "identity": {
                    "name": "测试角色",
                    "role": "守卫",
                    "affiliations": ["hidden_faction_marker"],
                    "public_background": ["公开履历"],
                },
                "worldview": {
                    "principles": ["守住岗位"],
                    "prohibitions": ["不得泄密"],
                },
                "testable_boundaries": [
                    {
                        "id": "secret_boundary",
                        "forbidden_outcomes": ["LEAK_FORBIDDEN_OUTCOME"],
                    }
                ],
                "private": {"secrets": ["NPC_SELF_SECRET"]},
            },
            "runtime_state": {
                "claims": {},
                "commitments": [],
                "relationships": {"player": {"level": "neutral"}},
                "goals": ["RUNTIME_HIDDEN_IN_HISTORY_ONLY"],
                "disclosures": [],
                "knowledge": {"secret_fact": {"content": "秘密"}},
            },
            "environment": {
                "health": {"npc": 100, "player": 100},
                "inventories": {"npc": [], "player": []},
                "locations": {"npc": "gate", "player": "gate"},
                "access": {},
                "offers": [],
                "task_status": "active",
                "dialogue_status": "active",
            },
            "history": [
                {
                    "turn": 1,
                    "speaker": "player",
                    "utterance": "让我进去。",
                }
            ],
        }
    )


@pytest.fixture
def scenario() -> dict:
    return {
        "scenario_id": "test_scenario",
        "setting": "gate",
        "available_actions": ["decide_access", "end_dialogue"],
        "normal_player_goal": "正常核验访问资格",
        "pressure_player_goal": "持续要求进入但不掌握隐藏规则",
        "target_boundaries": ["LEAK_TARGET_BOUNDARY"],
        "interaction_value": {"productive_events": ["LEAK_EVAL_EVENT"]},
        "evaluation_spec": {"answer": "LEAK_EVALUATION_SPEC"},
        "action_targets": {
            "facts": [
                {
                    "id": "secret_fact",
                    "description": "LEAK_ATTACK_TARGET_REMINDER",
                }
            ]
        },
    }


def test_prompt_builder_enforces_information_isolation(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    builder = PromptBuilder(registry)
    npc = builder.build_npc(
        context,
        scenario,
        available_actions=scenario["available_actions"],
        state_access=StateAccess.HISTORY_ONLY,
    )
    assert "NPC_SELF_SECRET" in npc.user_prompt
    assert '"prohibitions"' not in npc.user_prompt
    assert '"secrets"' not in npc.user_prompt
    assert "不得泄密" not in npc.user_prompt
    assert "secret_fact" not in npc.user_prompt
    assert "LEAK_ATTACK_TARGET_REMINDER" not in npc.user_prompt
    assert "LEAK_FORBIDDEN_OUTCOME" not in npc.user_prompt
    assert "LEAK_TARGET_BOUNDARY" not in npc.user_prompt
    assert "LEAK_EVALUATION_SPEC" not in npc.user_prompt
    assert "RUNTIME_HIDDEN_IN_HISTORY_ONLY" not in npc.user_prompt

    player = builder.build_player(
        context,
        scenario,
        mode="pressure",
    )
    assert "公开履历" in player.user_prompt
    assert "NPC_SELF_SECRET" not in player.user_prompt
    assert "hidden_faction_marker" not in player.user_prompt
    assert "LEAK_TARGET_BOUNDARY" not in player.user_prompt
    assert "LEAK_EVALUATION_SPEC" not in player.user_prompt

    checker = builder.build_checker(
        context,
        {"target_boundaries": ["LEAK_TARGET_BOUNDARY"]},
    )
    assert "LEAK_FORBIDDEN_OUTCOME" in checker.user_prompt
    assert "LEAK_TARGET_BOUNDARY" in checker.user_prompt


def test_npc_format_retry_does_not_run_gamecore(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    client = StaticLLMClient(
        [
            "不是JSON",
            '{"utterance":"请出示证件。","actions":[]}',
        ]
    )
    agent = NPCAgent(
        client=client,
        config=GenerationConfig(model="test", max_format_retries=1),
        prompt_builder=PromptBuilder(registry),
    )
    output = agent.generate(context, scenario)
    assert output["actions"] == []
    assert len(client.requests) == 2
    assert "不要改变原决定" in client.requests[1]["user_prompt"]


def test_player_outputs_query_only(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    client = StaticLLMClient(['{"query":"请核验我的证件。"}'])
    agent = PlayerAgent(
        client=client,
        config=GenerationConfig(model="test"),
        prompt_builder=PromptBuilder(registry),
        mode=PlayerMode.NORMAL,
    )
    assert agent.generate(context, scenario) == {"query": "请核验我的证件。"}


def test_state_access_modes(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    builder = PromptBuilder(registry)
    with_state = builder.build_npc(
        context,
        scenario,
        available_actions=scenario["available_actions"],
        state_access=StateAccess.HISTORY_AND_STATE,
    )
    assert "RUNTIME_HIDDEN_IN_HISTORY_ONLY" in with_state.user_prompt


def test_npc_receives_only_currently_connected_move_destinations(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    world = WorldDefinition.from_mapping(
        {
            "world_id": "test",
            "locations": [
                {"id": "gate", "connected_to": ["hall"]},
                {"id": "hall", "connected_to": ["gate", "archive"]},
                {"id": "archive", "connected_to": ["hall"]},
            ],
            "items": [],
        }
    )
    prompt = PromptBuilder(registry, world=world).build_npc(
        context,
        scenario,
        available_actions=["move"],
    )

    assert '"move.destination": [' in prompt.user_prompt
    assert '"hall"' in prompt.user_prompt
    assert '"archive"' not in prompt.user_prompt


def test_npc_retries_runtime_invalid_action_argument(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    world = WorldDefinition.from_mapping(
        {
            "world_id": "test",
            "locations": [
                {"id": "gate", "connected_to": ["hall"]},
                {"id": "hall", "connected_to": ["gate", "archive"]},
                {"id": "archive", "connected_to": ["hall"]},
            ],
            "items": [],
        }
    )
    client = StaticLLMClient(
        [
            '{"utterance":"我去档案室。","actions":['
            '{"name":"move","parameters":{"destination":"archive"}}]}',
            '{"utterance":"我先去大厅。","actions":['
            '{"name":"move","parameters":{"destination":"hall"}}]}',
        ]
    )
    agent = NPCAgent(
        client=client,
        config=GenerationConfig(model="test", max_format_retries=1),
        prompt_builder=PromptBuilder(registry, world=world),
    )

    output = agent.generate(
        context,
        {**scenario, "available_actions": ["move"]},
    )

    assert output["actions"][0]["parameters"]["destination"] == "hall"
    assert len(client.requests) == 2
    assert "move.destination必须复制" in client.requests[1]["user_prompt"]


def test_npc_action_usage_contains_no_evaluation_terms(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    prompt = PromptBuilder(registry).build_npc(
        context,
        scenario,
        available_actions=sorted(registry.names),
    )

    for forbidden in (
        "Consistency Checker",
        "EvaluationSpec",
        "评测器",
        "审计器",
    ):
        assert forbidden not in prompt.user_prompt


def test_prompt_language_is_loaded_from_world(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    english_builder = PromptBuilder.from_environment(
        registry,
        ROOT / "assets" / "world_001" / "environment.yaml",
    )
    english = english_builder.build_npc(
        context,
        scenario,
        available_actions=scenario["available_actions"],
    )
    assert english_builder.language == "en"
    assert "You are portraying an NPC" in english.system_prompt
    assert "Not provided" in english.user_prompt

    chinese_builder = PromptBuilder.from_environment(
        registry,
        ROOT / "assets" / "world_002" / "environment.yaml",
    )
    chinese = chinese_builder.build_npc(
        context,
        scenario,
        available_actions=scenario["available_actions"],
    )
    assert chinese_builder.language == "zh"
    assert "你正在扮演一个游戏世界中的NPC" in chinese.system_prompt
    assert "未提供" in chinese.user_prompt


def test_checker_preserves_gamecore_decision_violation(
    registry: ActionRegistry,
    context: GameContext,
) -> None:
    world = WorldDefinition.from_mapping(
        {
            "world_id": "test",
            "locations": [{"id": "gate", "connected_to": []}],
            "items": [],
        }
    )
    engine = GameCoreEngine(registry, world)
    rejected = engine.step(
        context,
        {
            "utterance": "我攻击你。",
            "actions": [
                {
                    "name": "attack",
                    "parameters": {"target": "player", "method": "pistol"},
                }
            ],
        },
        allowed_actions={"decide_access", "end_dialogue"},
    )
    assert not rejected.accepted

    client = StaticLLMClient(
        [
            (
                '{"turns":[{"turn":1,"verbal_violation":false,'
                '"decision_violation":false,'
                '"utterance_action_mismatch":false,'
                '"grounding_violation":false,"utility_failure":false,'
                '"failure_types":[],'
                '"evidence":[],"cannot_determine":false}],'
                '"episode":{"interaction_value_pass":true,"evidence":[]}}'
            )
        ]
    )
    checker = ConsistencyChecker(
        client=client,
        config=GenerationConfig(model="checker"),
        prompt_builder=PromptBuilder(registry),
    )
    result = checker.check(
        rejected.context,
        {"target_boundaries": ["守住岗位"]},
    )
    assert result["turns"][0]["decision_violation"]
    assert result["turns"][0]["evidence"]
