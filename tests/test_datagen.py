import json
import re
import threading
from pathlib import Path
from typing import Any

from datagen.llm_history import (
    audit_surface_quality,
    build_session_blueprints,
    generate_llm_history,
)
from llm import GenerationConfig


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "assets" / "world_002"


def test_session_blueprints_preserve_reviewed_anchor_state() -> None:
    first = build_session_blueprints(WORLD, "yu_zecheng", rounds=600)
    second = build_session_blueprints(WORLD, "yu_zecheng", rounds=600)

    assert first == second
    assert len(first) == 30
    assert all(item["session_size"] == 20 for item in first)
    assert all(8 <= item["anchor_turn"] <= 14 for item in first)
    assert first[0]["pre_state"] != first[0]["post_state"]
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
        }
    ]

    report = audit_surface_quality(history)

    assert not report["passed"]
    assert report["repeated_word_lines"]
    assert report["schema_leaks"]
    assert report["bad_phrases"]


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
            session_id = re.search(
                r'"session_id": "([^"]+)"',
                user_prompt,
            ).group(1)
            turns = int(
                re.search(r'"turns": ([0-9]+)', user_prompt).group(1)
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
                            "turn_range": [
                                (act - 1) * 5 + 1,
                                act * 5,
                            ],
                            "development": f"出现第{act}组可核对记录",
                            "player_intent": f"追问第{act}组细节",
                            "npc_response_strategy": f"安排核查第{act}组记录",
                        }
                        for act in range(1, turns // 5 + 1)
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
                            "introduced_turn": 7,
                            "recall_turn": min(12, turns),
                            "introduced_by": "npc",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        if "长对话编剧" in system_prompt:
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
                            "npc": f"这项记录先留着，我查第{turn}处签字。",
                        }
                        for turn in range(start, end + 1)
                    ]
                },
                ensure_ascii=False,
            )
        if "数据质检员" in system_prompt:
            return json.dumps(
                {
                    "continuity_pass": True,
                    "player_naturalness_pass": True,
                    "npc_style_pass": True,
                    "state_effect_visible": True,
                    "memory_value_pass": True,
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
) -> None:
    requests: list[dict[str, str]] = []
    lock = threading.Lock()

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
    assert report["logical_api_calls"] == 150
    assert len({record["session_id"] for record in history}) == 30
    assert any(
        '"previous_dialogue": [{"speaker": "player"'
        in request["user_prompt"]
        for request in requests
        if "长对话编剧" in request["system_prompt"]
    )
