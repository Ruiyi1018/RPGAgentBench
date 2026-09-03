"""Build minimal Stage2 probes on top of an LLM-generated base history."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_jsonl
from .history_sources import load_reviewed_anchors, load_transition_plan
from .projection import project_frozen_history


def write_executable_pairs(
    world_dir: str | Path,
    character_id: str,
    base_history: list[dict[str, Any]] | None = None,
    *,
    output_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    world = Path(world_dir)
    history = base_history or read_jsonl(
        world / "frozen" / character_id / "history.jsonl"
    )
    plan = load_transition_plan(world, character_id)
    anchors = {
        anchor["id"]: anchor
        for anchor in load_reviewed_anchors(world, character_id)
    }
    output_path = (
        Path(output_root)
        if output_root is not None
        else world / "branches" / character_id
    )
    descriptors: list[dict[str, Any]] = []
    probe_round = int(history[-1]["round"]) + 1

    sensitivity = [
        transition
        for transition in plan["transitions"]
        if "decision_probe" in transition
    ]
    if len(sensitivity) != 7:
        raise ValueError("敏感性pair必须恰好有7个")
    for index, transition in enumerate(sensitivity, start=1):
        pair_id = f"{character_id}_pair_{index:02d}"
        branch_a = _with_probe(
            history,
            pair_id,
            probe_round,
            transition["anchor_id"],
            _source_event(history, transition["anchor_id"]),
            transition["operations"],
        )
        branch_b = _with_probe(
            history,
            pair_id,
            probe_round,
            transition["anchor_id"],
            str(anchors[transition["anchor_id"]]["cf_edit"]),
            transition["counterfactual_operations"],
        )
        _write_pair_histories(
            output_path,
            pair_id,
            branch_a,
            branch_b,
        )
        assert_projection_minimality(branch_a, branch_b)
        probe = transition["decision_probe"]
        descriptors.append(
            {
                "pair_id": pair_id,
                "pair_type": "sensitivity",
                **_branch_paths(character_id, pair_id),
                "intervention": {
                    "source_anchor_id": transition["anchor_id"],
                    "round": probe_round,
                    "visible_changed_rounds": [probe_round],
                    "state_changed_rounds": [probe_round],
                    "supersedes_prior_record": True,
                },
                "final_query": probe["query_variants"][index % 2],
                "decision_target": _decision_target(probe),
                "decision_rubric": {
                    "branch_a": probe["branch_a"],
                    "branch_b": probe["branch_b"],
                },
            }
        )

    invariance = [
        transition
        for transition in plan["transitions"]
        if "invariance_probe" in transition
    ]
    if len(invariance) != 3:
        raise ValueError("不变性pair必须恰好有3个")
    for index, transition in enumerate(invariance, start=8):
        pair_id = f"{character_id}_pair_{index:02d}"
        original = _source_event(history, transition["anchor_id"])
        paraphrase = transition["invariance_probe"]["paraphrase"]
        branch_a = _with_probe(
            history,
            pair_id,
            probe_round,
            transition["anchor_id"],
            original,
            None,
        )
        branch_b = _with_probe(
            history,
            pair_id,
            probe_round,
            transition["anchor_id"],
            paraphrase,
            None,
        )
        _write_pair_histories(
            output_path,
            pair_id,
            branch_a,
            branch_b,
        )
        assert_projection_minimality(branch_a, branch_b)
        probe = transition["invariance_probe"]
        descriptors.append(
            {
                "pair_id": pair_id,
                "pair_type": "invariance",
                **_branch_paths(character_id, pair_id),
                "intervention": {
                    "source_anchor_id": transition["anchor_id"],
                    "round": probe_round,
                    "visible_changed_rounds": [probe_round],
                    "state_changed_rounds": [],
                    "supersedes_prior_record": False,
                },
                "final_query": probe["query"],
                "decision_target": {
                    "state_path": probe["expected_state"]["path"]
                },
                "decision_rubric": {
                    "expected_equivalent": True,
                    "expected_state": probe["expected_state"],
                },
            }
        )
    write_jsonl(output_path / "pairs.jsonl", descriptors)
    return descriptors


def assert_projection_minimality(
    branch_a: list[dict[str, Any]],
    branch_b: list[dict[str, Any]],
) -> None:
    left = project_frozen_history(branch_a).splitlines()
    right = project_frozen_history(branch_b).splitlines()
    if len(left) != len(right):
        raise ValueError("pair投影行数不一致")
    changed = [
        index
        for index, (a, b) in enumerate(zip(left, right, strict=True))
        if a != b
    ]
    if len(changed) != 1:
        raise ValueError(f"pair投影必须只有一行不同，实际为{len(changed)}")


def _with_probe(
    history: list[dict[str, Any]],
    pair_id: str,
    round_number: int,
    source_anchor_id: str,
    content: str,
    operations: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    branch = copy.deepcopy(history)
    branch.append(
        {
            "world_id": history[-1]["world_id"],
            "character_id": history[-1]["character_id"],
            "round": round_number,
            "session_id": f"{pair_id}_probe",
            "turn_in_session": 1,
            "messages": [
                {
                    "speaker": "player",
                    "utterance": "刚收到一项经过复核的更新，请以它为准处理。",
                },
                {
                    "speaker": "npc",
                    "name": history[-1]["messages"][1].get("name"),
                    "utterance": "我已收到这项更新，会按最新情况判断。",
                },
            ],
            "observation": {
                "event_id": f"{pair_id}_update",
                "content": content,
                "visible_to": ["npc", "player"],
                "supersedes": f"anchor_{source_anchor_id}",
            },
            "is_anchor": True,
            "anchor_id": f"{source_anchor_id}_probe",
            "source_anchor_id": source_anchor_id,
            "state_transition": (
                {"probe_only": True, "operations": copy.deepcopy(operations)}
                if operations is not None
                else None
            ),
            "generation_meta": {
                "probe_only": True,
                "introduced_memories": [],
                "recalled_memories": [],
            },
        }
    )
    return branch


def _source_event(
    history: list[dict[str, Any]],
    anchor_id: str,
) -> str:
    target = next(
        record
        for record in history
        if record.get("source_anchor_id") == anchor_id
    )
    return str(target["observation"]["content"])

def _decision_target(probe: dict[str, Any]) -> dict[str, Any]:
    decisions = (
        probe["branch_a"]["admissible_decisions"]
        + probe["branch_b"]["admissible_decisions"]
    )
    parameter_maps = [item.get("parameters", {}) for item in decisions]
    common_parameters: dict[str, Any] = {}
    if parameter_maps:
        common_keys = set(parameter_maps[0]).intersection(
            *(set(parameters) for parameters in parameter_maps[1:])
        )
        common_parameters = {
            key: parameter_maps[0][key]
            for key in sorted(common_keys)
            if all(
                parameters[key] == parameter_maps[0][key]
                for parameters in parameter_maps[1:]
            )
            and key not in {"decision", "direction", "resolution", "reason"}
        }
    return {
        "candidate_actions": sorted(
            {item["action"] for item in decisions}
        ),
        "fixed_parameters": common_parameters,
    }


def _branch_paths(
    character_id: str,
    pair_id: str,
) -> dict[str, str]:
    root = Path("branches") / character_id / pair_id
    return {
        "branch_a": str(root / "branch_a.jsonl"),
        "branch_b": str(root / "branch_b.jsonl"),
    }


def _write_pair_histories(
    output_root: Path,
    pair_id: str,
    branch_a: list[dict[str, Any]],
    branch_b: list[dict[str, Any]],
) -> None:
    pair_dir = output_root / pair_id
    write_jsonl(pair_dir / "branch_a.jsonl", branch_a)
    write_jsonl(pair_dir / "branch_b.jsonl", branch_b)
