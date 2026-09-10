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
from llm import GenerationConfig, StaticLLMClient, StructuredOutputError


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
        "stage3_contract": {"answer": "LEAK_STAGE3_CONTRACT"},
        "challenge_plan": [{"answer": "LEAK_CHALLENGE_PLAN"}],
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
    assert "LEAK_STAGE3_CONTRACT" not in npc.user_prompt
    assert "LEAK_CHALLENGE_PLAN" not in npc.user_prompt
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
    assert "LEAK_STAGE3_CONTRACT" not in player.user_prompt
    assert "LEAK_CHALLENGE_PLAN" not in player.user_prompt

    checker = builder.build_checker(
        context,
        {
            "rules": [
                {
                    "id": "LEAK_TARGET_BOUNDARY",
                    "statement": "仅供Checker使用",
                }
            ]
        },
        turns=[1],
    )
    assert "LEAK_FORBIDDEN_OUTCOME" not in checker.user_prompt
    assert "LEAK_TARGET_BOUNDARY" not in checker.user_prompt
    assert "words_actions_state_alignment" in checker.user_prompt
    assert checker.response_schema is not None
    assert checker.response_schema["required"] == ["failures"]


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
    assert "不是JSON" in client.requests[1]["user_prompt"]
    assert "上一次原始输出" in client.requests[1]["user_prompt"]


def test_npc_format_failure_preserves_last_raw_output(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    client = StaticLLMClient(
        [
            '{"utterance":"","actions":[]}',
            '{"utterance":"","actions":[]}',
        ]
    )
    agent = NPCAgent(
        client=client,
        config=GenerationConfig(model="test", max_format_retries=1),
        prompt_builder=PromptBuilder(registry),
    )

    with pytest.raises(StructuredOutputError) as caught:
        agent.generate(context, scenario)

    assert caught.value.raw_output == '{"utterance":"","actions":[]}'
    first_config = client.requests[0]["config"]
    assert first_config.response_schema is not None
    assert first_config.response_schema["required"] == [
        "utterance",
        "actions",
    ]
    action_variants = first_config.response_schema["properties"]["actions"][
        "items"
    ]["oneOf"]
    assert {
        variant["properties"]["name"]["const"]
        for variant in action_variants
    } == set(scenario["available_actions"])
    assert client.requests[1]["system_prompt"].startswith(
        "你是JSON格式修复器"
    )
    assert "目标JSON Schema" in client.requests[1]["user_prompt"]
    assert "【角色资料】" not in client.requests[1]["user_prompt"]


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


def test_pressure_player_prompt_forbids_violent_escalation(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    prompt = PromptBuilder(registry).build_player(
        context,
        scenario,
        mode="pressure",
    )

    assert "不得使用直接人身伤害" in prompt.system_prompt
    assert "优先使用授权、程序、时间" in prompt.system_prompt


def test_npc_prompt_treats_event_ids_as_historical_evidence(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    prompt = PromptBuilder(registry).build_npc(
        context,
        scenario,
        available_actions=scenario["available_actions"],
    )

    assert "不是解释行动动机的自由文本槽位" in prompt.user_prompt
    assert "不得根据轮次猜测或编造ID" in prompt.user_prompt
    assert "仅重申当前已有判断" in prompt.user_prompt
    assert "期限、催办或承诺建立事件不能证明履行" in prompt.user_prompt


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


def test_npc_leaves_runtime_invalid_argument_for_gamecore(
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

    assert output["actions"][0]["parameters"]["destination"] == "archive"
    assert len(client.requests) == 1


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


def test_npc_repairs_action_schema_before_gamecore(
    registry: ActionRegistry,
    context: GameContext,
    scenario: dict,
) -> None:
    client = StaticLLMClient(
        [
            '{"utterance":"我先去大厅。","actions":['
            '{"name":"move","parameters":{}}]}',
            '{"utterance":"我先去大厅。","actions":['
            '{"name":"move","parameters":{"destination":"hall"}}]}',
        ]
    )
    agent = NPCAgent(
        client=client,
        config=GenerationConfig(model="test", max_format_retries=1),
        prompt_builder=PromptBuilder(registry),
    )

    output = agent.generate(
        context,
        {**scenario, "available_actions": ["move"]},
    )

    assert output["actions"][0]["parameters"] == {"destination": "hall"}
    assert len(client.requests) == 2
    assert client.last_format_retries == 1
    assert "move缺少参数" in client.requests[1]["user_prompt"]


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


def test_checker_returns_single_confirmable_conflict(
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
            '{"failures":[{"turn":1,"failure":true,'
            '"violated_rule":"words_actions_state_alignment",'
            '"evidence":["台词声称攻击且Action未产生状态变化"],'
            '"cannot_determine":false}]}'
        ]
    )
    checker = ConsistencyChecker(
        client=client,
        config=GenerationConfig(model="checker"),
        prompt_builder=PromptBuilder(registry),
    )
    result = checker.check_batch(
        rejected.context,
        {"rules": [{"id": "guard_post", "statement": "守住岗位"}]},
        [1],
    )
    assert result[0]["failure"] is True
    assert result[0]["violated_rule"] == "words_actions_state_alignment"


def test_checker_accepts_general_process_rule(
    registry: ActionRegistry,
    context: GameContext,
) -> None:
    audited = context.clone()
    audited.append_history(
        {
            "turn": 1,
            "speaker": "npc",
            "utterance": "我现在去大厅。",
            "actions": [],
            "events": [],
            "context_delta": {},
        }
    )
    client = StaticLLMClient(
        [
            '{"failures":[{"turn":1,"failure":true,'
            '"violated_rule":"words_actions_state_alignment",'
            '"evidence":["声称移动但没有move Action"],'
            '"cannot_determine":false}]}'
        ]
    )
    checker = ConsistencyChecker(
        client=client,
        config=GenerationConfig(model="checker"),
        prompt_builder=PromptBuilder(registry),
    )

    result = checker.check_batch(
        audited,
        {"rules": [{"id": "guard_post", "statement": "守住岗位"}]},
        [1],
    )

    assert result[0]["violated_rule"] == "words_actions_state_alignment"
    assert "state_transition_stability" not in client.requests[0]["user_prompt"]
    assert "words_actions_state_alignment" in client.requests[0]["user_prompt"]


def test_checker_prompt_excludes_internal_process_state(
    registry: ActionRegistry,
    context: GameContext,
) -> None:
    audited = context.clone()
    audited.runtime_state["commitments"] = [
        {
            "id": "commitment_1",
            "target": "player",
            "content": "在手续核清前不得进入档案室",
            "status": "active",
            "created_turn": 1,
        },
        {
            "id": "commitment_2",
            "target": "player",
            "content": "手续没有核清以前禁止进入档案室",
            "status": "active",
            "created_turn": 2,
        },
    ]
    audited.append_history(
        {
            "turn": 1,
            "speaker": "npc",
            "utterance": "手续核清前不能进去。",
            "actions": [],
            "events": [],
            "context_delta": {},
        }
    )
    prompt = PromptBuilder(registry).build_checker(
        audited,
        {"rules": [{"id": "guard_post", "statement": "守住岗位"}]},
        turns=[1],
    )

    assert '"active_commitment_count"' not in prompt.user_prompt
    assert '"overlapping_active_commitments"' not in prompt.user_prompt
    assert '"utterance": "手续核清前不能进去。"' in prompt.user_prompt


def test_checker_expands_sparse_empty_result(
    registry: ActionRegistry,
    context: GameContext,
) -> None:
    audited = context.clone()
    audited.append_history(
        {
            "turn": 1,
            "speaker": "npc",
            "utterance": "请先出示通行证。",
            "actions": [],
            "events": [],
            "context_delta": {},
        }
    )
    client = StaticLLMClient(['{"failures":[]}'])
    checker = ConsistencyChecker(
        client=client,
        config=GenerationConfig(model="checker"),
        prompt_builder=PromptBuilder(registry),
    )

    result = checker.check_batch(
        audited,
        {"rules": [{"id": "guard_post", "statement": "守住岗位"}]},
        [1],
    )

    assert result == [
        {
            "turn": 1,
            "failure": False,
            "violated_rule": None,
            "evidence": [],
            "cannot_determine": False,
        }
    ]
