import json
import re
import threading
from pathlib import Path
from typing import Any

from datagen.generation.anchor_draft import (
    _validate_anchor_outline,
    _validate_anchor_output,
)
from datagen.generation.catalog_foundation import _cached_or_generate
from datagen.audit.benchmark_assets import _rubric_contract_error
from datagen.shared.io import load_yaml
from datagen.generation.stage1_tests import _profile_questions
from datagen.generation.stage1_history import (
    _generate_history_structured,
    _hidden_event_overlap,
    _validate_plan,
    audit_surface_quality,
    build_session_blueprints,
    generate_llm_history,
)
from agents.prompt_builder import PromptBundle
from datagen.generation.foundation import scaffold_world
from datagen.audit.source_assets import audit_anchor_plan, audit_foundation
from gamecore import ActionRegistry
from llm import GenerationConfig


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "assets" / "world_002"


def test_hidden_event_overlap_detects_player_leak() -> None:
    forbidden = [
        {
            "anchor_id": "qf_15_li_ya_exchanged_for_qiu",
            "event": "左蓝以被捕的李涯交换秋掌柜，换俘已经完成",
        }
    ]

    assert _hidden_event_overlap(
        "院里传来消息，说换俘已经完成。",
        forbidden,
    ) == "qf_15_li_ya_exchanged_for_qiu"
    assert _hidden_event_overlap("院里有人送来一份登记。", forbidden) is None


def test_plan_validator_normalizes_harmless_extra_fields() -> None:
    plan = {
        "session_id": "session_01",
        "local_event": "核对来件",
        "setting": "行动队办公室",
        "stakes": "需要确认来源",
        "extra": "ignored",
        "acts": [
            {
                "act": 1,
                "turn_range": [1, 5],
                "development": "逐项核对",
                "player_intent": "询问来源",
                "npc_response_strategy": "要求提供证据",
                "extra": "ignored",
            }
        ],
        "memory_seeds": [
            {
                "content": "来件右下角有水印",
                "introduced_turn": 2,
                "recall_turn": 4,
                "introduced_by": "player",
                "extra": "ignored",
            },
            {
                "memory_id": "second",
                "content": "经手时间是下午三点",
                "introduced_turn": 3,
                "recall_turn": None,
                "introduced_by": "npc",
            },
        ],
    }

    _validate_plan(plan, "session_01", [[1, 5]], [], [])

    assert set(plan) == {
        "session_id",
        "local_event",
        "setting",
        "stakes",
        "acts",
        "memory_seeds",
    }
    assert set(plan["acts"][0]) == {
        "act",
        "turn_range",
        "development",
        "player_intent",
        "npc_response_strategy",
    }
    assert set(plan["memory_seeds"][0]) == {
        "memory_id",
        "content",
        "introduced_turn",
        "recall_turn",
        "introduced_by",
    }


class _FreshRetryClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> str:
        del system_prompt, user_prompt, config
        self.calls += 1
        return json.dumps({"valid": self.calls > 1})


def test_history_generation_freshly_retries_invalid_content() -> None:
    client = _FreshRetryClient()

    def validate(value: dict[str, Any]) -> None:
        if not value.get("valid"):
            raise ValueError("content invalid")

    result = _generate_history_structured(
        client,
        PromptBundle(system_prompt="test", user_prompt="test"),
        GenerationConfig(model="fake", max_format_retries=0),
        validate,
    )

    assert result == {"valid": True}
    assert client.calls == 2
    assert client.last_fresh_retries == 1


def test_rubric_contract_rejects_missing_required_parameter() -> None:
    registry = ActionRegistry.load_directory(ROOT / "gamecore" / "actions")

    error = _rubric_contract_error(
        {
            "action": "decide_access",
            "parameters": {
                "subject": "player",
                "resource": "archive_room",
                "decision": "deny",
            },
        },
        registry,
    )

    assert error is not None
    assert "reason_event" in error


def test_review_gates_require_reapproval_when_rules_expand() -> None:
    foundation = audit_foundation(WORLD, ["yu_zecheng"])
    anchors = audit_anchor_plan(WORLD, "yu_zecheng")

    assert foundation["machine_passed"]
    assert not foundation["human_approved"]
    assert not foundation["ready_for_anchor_generation"]
    assert anchors["machine_passed"]
    assert not anchors["human_approved"]
    assert not anchors["ready_for_history_generation"]
    assert set(anchors["metrics"]["state_dimension_counts"]) == {
        "knowledge",
        "commitment_goal",
        "relationship",
        "resource",
    }
    assert anchors["metrics"]["turning_points"] >= 8


def test_reviewed_yu_assets_satisfy_generic_anchor_draft_contract() -> None:
    selected = load_yaml(WORLD / "frozen/yu_zecheng/anchors.yaml")
    transitions = load_yaml(
        WORLD / "frozen/yu_zecheng/transitions.yaml"
    )
    environment = load_yaml(WORLD / "environment.yaml")
    pool = load_yaml(WORLD / "anchor_pool.yaml")
    eligible = [
        event
        for event in pool["events"]
        if "yu_zecheng" in event.get("anchor_for", [])
    ]
    probe_plan = [
        {
            "anchor_id": transition["anchor_id"],
            "type": (
                "sensitivity"
                if "decision_probe" in transition
                else "invariance"
                if "invariance_probe" in transition
                else "none"
            ),
        }
        for transition in transitions["transitions"]
    ]

    _validate_anchor_outline(
        {
            "canon_anchors": selected["canon_anchors"],
            "generated_anchors": selected["generated_anchors"],
            "probe_plan": probe_plan,
        },
        eligible,
        30,
        10,
    )

    _validate_anchor_output(
        {
            "canon_anchors": selected["canon_anchors"],
            "generated_anchors": selected["generated_anchors"],
            "transitions": transitions["transitions"],
        },
        "yu_zecheng",
        environment,
        transitions["initial_context"],
        eligible,
        30,
        10,
    )


def test_stage1_profile_questions_follow_world_language() -> None:
    card = load_yaml(WORLD / "characters/yu_zecheng.yaml")

    questions = _profile_questions("yu_zecheng", card, "en")

    assert questions[0]["question"] == "What is the character's public role?"


def test_new_world_scaffold_starts_unapproved(
    tmp_path: Path,
) -> None:
    world = tmp_path / "world_003"

    report = scaffold_world(
        world,
        world_id="world_003",
        name="test_world",
        language="en",
        character_ids=["npc_one"],
    )
    review = load_yaml(
        world / "frozen/npc_one/source_review.yaml"
    )

    assert report["approved"] is False
    assert review["foundation"]["approved"] is False
    assert review["anchors"]["approved"] is False


def test_session_blueprints_preserve_reviewed_anchor_state() -> None:
    first = build_session_blueprints(WORLD, "yu_zecheng", rounds=600)
    second = build_session_blueprints(WORLD, "yu_zecheng", rounds=600)

    assert first == second
    assert len(first) == 30
    assert sum(item["session_size"] for item in first) == 600
    assert len({item["session_size"] for item in first}) >= 4
    assert all(
        sum(end - start + 1 for start, end in item["act_ranges"])
        == item["session_size"]
        for item in first
    )
    assert {
        end - start + 1
        for item in first
        for start, end in item["act_ranges"]
    } >= {4, 5, 6}
    anchor_counts = [len(item["anchors"]) for item in first]
    assert sum(anchor_counts) == 30
    assert 0 in anchor_counts
    assert max(anchor_counts) >= 2
    assert len(
        {
            round(anchor["turn"] / item["session_size"], 1)
            for item in first
            for anchor in item["anchors"]
        }
    ) >= 5
    assert any(
        anchor["effect_start_session"] > item["session_index"]
        for item in first
        for anchor in item["anchors"]
    )
    assert all(
        1 <= anchor["turn"] <= item["session_size"]
        for item in first
        for anchor in item["anchors"]
    )
    assert any(
        item["pre_state"] != item["post_state"]
        for item in first
        if item["anchors"]
    )
    assert (
        first[-1]["post_state"]["runtime_state"]["claims"][
            "repeated_identity_pressure"
        ]["decision"]
        == "reject"
    )


def test_surface_audit_detects_template_and_schema_leaks() -> None:
    history = [
        {
            "round": 1,
            "session_id": "s1",
            "messages": [
                {"speaker": "player", "utterance": "目前目前只有一条线索。"},
                {
                    "speaker": "npc",
                    "utterance": "这件事没有形成新的权限或义务。",
                },
            ],
            "generation_meta": {"recalled_memories": []},
        },
        {
            "round": 2,
            "session_id": "s1",
            "messages": [
                {"speaker": "player", "utterance": "我再核对一次记录。"},
                {
                    "speaker": "npc",
                    "name": "王翠平",
                    "utterance": "王翠平",
                },
            ],
            "generation_meta": {"recalled_memories": []},
        },
    ]

    report = audit_surface_quality(history)

    assert not report["passed"]
    assert report["repeated_word_lines"]
    assert report["schema_leaks"]
    assert report["bad_phrases"]
    assert report["invalid_dialogue"]


class _HistoryClient:
    def __init__(self, requests: list[dict[str, str]], lock: threading.Lock):
        self.requests = requests
        self.lock = lock
        self.calls = 0
        self.last_usage: dict[str, int] | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> str:
        del config
        self.calls += 1
        self.last_usage = {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
        with self.lock:
            self.requests.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
        if "session规划器" in system_prompt:
            payload = json.loads(
                user_prompt.split("规划约束：", 1)[1].split("\n\n", 1)[0]
            )
            session_id = re.search(
                r'"session_id": "([^"]+)"',
                user_prompt,
            ).group(1)
            turns = int(
                re.search(r'"turns": ([0-9]+)', user_prompt).group(1)
            )
            anchor_turn = min(
                (
                    int(anchor["turn"])
                    for anchor in payload["anchors"]
                ),
                default=turns // 2,
            )
            return json.dumps(
                {
                    "session_id": session_id,
                    "local_event": "一份值班记录的签字时间前后不一致",
                    "setting": "天津站机要室",
                    "stakes": "需要在不惊动经手人的情况下核清来源",
                    "acts": [
                        {
                            "act": act,
                            "turn_range": turn_range,
                            "development": f"出现第{act}组可核对记录",
                            "player_intent": f"追问第{act}组细节",
                            "npc_response_strategy": f"安排核查第{act}组记录",
                        }
                        for act, turn_range in enumerate(
                            payload["required_act_ranges"],
                            start=1,
                        )
                    ],
                    "memory_seeds": [
                        {
                            "memory_id": "first",
                            "content": "值班表由陈科长在九点经手",
                            "introduced_turn": 3,
                            "recall_turn": min(13, turns),
                            "introduced_by": "player",
                        },
                        {
                            "memory_id": "second",
                            "content": "登记页右下角缺少蓝色印章",
                            "introduced_turn": min(anchor_turn, turns - 2),
                            "recall_turn": None,
                            "introduced_by": "npc",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        if "长对话编剧" in system_prompt:
            payload = json.loads(user_prompt.split("\n\n", 1)[0])
            start, end = [
                int(value)
                for value in re.search(
                    r'"turn_range": \[([0-9]+), ([0-9]+)\]',
                    user_prompt,
                ).groups()
            ]
            return json.dumps(
                {
                    "turns": [
                        {
                            "turn_in_session": turn,
                            "player": f"我接着说明第{turn}项线索，请你核对。",
                            "npc": (
                                f"这项记录先留着，我查第{turn}处签字。"
                                + "；".join(
                                    str(memory["content"])
                                    for memory in (
                                        payload[
                                            "details_to_introduce_naturally"
                                        ]
                                        + payload[
                                            "past_details_to_recall_naturally"
                                        ]
                                    )
                                    if memory.get("introduced_turn") == turn
                                    or memory.get("recall_turn") == turn
                                )
                            ),
                            "memory_refs": [
                                str(memory["memory_id"])
                                for memory in (
                                    payload["details_to_introduce_naturally"]
                                    + payload[
                                        "past_details_to_recall_naturally"
                                    ]
                                )
                                if memory.get("introduced_turn") == turn
                                or memory.get("recall_turn") == turn
                            ],
                            "anchor_effect_refs": [
                                str(effect["anchor_id"])
                                for effect in payload["anchor_effects_to_show"]
                                if int(effect["turn"]) == turn
                            ],
                        }
                        for turn in range(start, end + 1)
                    ]
                },
                ensure_ascii=False,
            )
        if "数据抽样质检员" in system_prompt:
            return json.dumps(
                {
                    "continuity_pass": True,
                    "anchor_effect_pass": True,
                    "cross_event_pass": True,
                    "history_value_pass": True,
                    "player_agency_pass": True,
                    "schema_safe_pass": True,
                    "role_consistency_pass": True,
                    "pattern_diversity_pass": True,
                    "issues": [],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"utterance": f"这项记录先留在这里，我查第{self.calls}处签字。"},
            ensure_ascii=False,
        )


def test_llm_history_generator_runs_interactive_sessions(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    requests: list[dict[str, str]] = []
    lock = threading.Lock()
    monkeypatch.setattr(
        "datagen.generation.stage1_history.require_generation_ready",
        lambda *_args, **_kwargs: None,
    )

    def factory() -> _HistoryClient:
        return _HistoryClient(requests, lock)

    history, report = generate_llm_history(
        WORLD,
        "yu_zecheng",
        client_factory=factory,
        config=GenerationConfig(model="fake", max_format_retries=0),
        rounds=450,
        workers=4,
        work_root=tmp_path / "work",
        show_progress=False,
    )

    assert len(history) == 450
    assert sum(record["is_anchor"] for record in history) == 30
    assert report["logical_api_calls"] == 121
    assert report["semantic_audit_calls"] == 1
    assert len({record["session_id"] for record in history}) == 30
    assert any(
        '"previous_dialogue": [{"speaker": "player"'
        in request["user_prompt"]
        for request in requests
        if "长对话编剧" in request["system_prompt"]
    )
    history_requests = [
        request
        for request in requests
        if "session规划器" in request["system_prompt"]
        or "长对话编剧" in request["system_prompt"]
    ]
    assert history_requests
    assert all(
        '"player_forbidden_knowledge"' in request["user_prompt"]
        for request in history_requests
    )
    assert all(
        '"public_background"' not in request["user_prompt"]
        and '"personal_history"' not in request["user_prompt"]
        for request in history_requests
    )


def test_catalog_cache_regenerates_invalid_checkpoint(tmp_path: Path) -> None:
    cache_path = tmp_path / "profile.json"
    cache_path.write_text('{"valid": false}\n', encoding="utf-8")
    generation_calls = 0

    def validate(value: dict[str, Any]) -> None:
        if value.get("valid") is not True:
            raise ValueError("invalid checkpoint")

    def generate() -> dict[str, Any]:
        nonlocal generation_calls
        generation_calls += 1
        return {"valid": True}

    result = _cached_or_generate(
        cache_path,
        cache_valid=True,
        validator=validate,
        generator=generate,
    )

    assert result == {"valid": True}
    assert generation_calls == 1
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"valid": True}
