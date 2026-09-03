from pathlib import Path
from typing import Any

from datagen.executable_branches import write_executable_pairs
from datagen.executable_qa import build_executable_qa
from datagen.llm_history import build_session_blueprints
from datagen.projection import project_frozen_history


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "assets" / "world_002"


def _structural_history() -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for blueprint in build_session_blueprints(
        WORLD,
        "yu_zecheng",
        rounds=600,
    ):
        for turn in range(1, blueprint["session_size"] + 1):
            is_anchor = turn == blueprint["anchor_turn"]
            round_number = blueprint["start_round"] + turn - 1
            history.append(
                {
                    "world_id": "world_002",
                    "character_id": "yu_zecheng",
                    "round": round_number,
                    "session_id": blueprint["session_id"],
                    "turn_in_session": turn,
                    "messages": [
                        {
                            "speaker": "player",
                            "utterance": f"继续处理本次事件的第{turn}步。",
                        },
                        {
                            "speaker": "npc",
                            "name": "余则成",
                            "utterance": f"先核对第{turn}步的具体记录。",
                        },
                    ],
                    "observation": (
                        {
                            "event_id": blueprint["event_id"],
                            "content": blueprint["anchor"]["event"],
                            "visible_to": (
                                ["npc", "player"]
                                if blueprint["transition"]["player_visible"]
                                else ["npc"]
                            ),
                        }
                        if is_anchor
                        else None
                    ),
                    "is_anchor": is_anchor,
                    "anchor_id": (
                        blueprint["anchor"]["id"] if is_anchor else None
                    ),
                    "source_anchor_id": (
                        blueprint["anchor"]["id"] if is_anchor else None
                    ),
                    "state_transition": (
                        {
                            "pre_state": blueprint["pre_state"],
                            "operations": blueprint["transition"][
                                "operations"
                            ],
                            "context_delta": blueprint["context_delta"],
                            "post_state": blueprint["post_state"],
                        }
                        if is_anchor
                        else None
                    ),
                    "generation_meta": {
                        "introduced_memories": [],
                        "recalled_memories": [],
                    },
                }
            )
    return history


def test_structural_history_is_visibility_safe() -> None:
    history = _structural_history()

    assert len(history) == 600
    assert sum(record["is_anchor"] for record in history) == 30
    assert all(
        record["state_transition"]["context_delta"]
        for record in history
        if record["is_anchor"]
    )

    player_projection = project_frozen_history(history, audience="player")
    npc_projection = project_frozen_history(history, audience="npc")
    assert "anchor_qf_03_lv_killed" not in player_projection
    assert "anchor_qf_03_lv_killed" in npc_projection
    assert "state_transition" not in npc_projection


def test_qa_uses_state_answers() -> None:
    qa = build_executable_qa(WORLD, "yu_zecheng", _structural_history())

    assert len(qa) == 50
    assert len({item["question"] for item in qa}) == 50
    assert sum("state_path" in item for item in qa) == 38
    assert {
        layer: sum(item["evaluation_layer"] == layer for item in qa)
        for layer in {
            "profile",
            "temporal",
            "local_state",
            "checkpoint_state",
            "long_range_final",
            "multi_hop",
        }
    } == {
        "profile": 6,
        "temporal": 6,
        "local_state": 10,
        "checkpoint_state": 10,
        "long_range_final": 12,
        "multi_hop": 6,
    }


def test_pairs_append_one_minimal_probe(tmp_path: Path) -> None:
    pairs = write_executable_pairs(
        WORLD,
        "yu_zecheng",
        _structural_history(),
        output_root=tmp_path / "branches",
    )

    assert len(pairs) == 10
    assert sum(pair["pair_type"] == "sensitivity" for pair in pairs) == 7
    assert all(
        pair["intervention"]["visible_changed_rounds"] == [601]
        for pair in pairs
    )
    assert all(
        pair["intervention"].get("supersedes_prior_record")
        for pair in pairs[:7]
    )
