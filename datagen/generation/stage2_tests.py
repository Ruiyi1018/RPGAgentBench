"""Generate Stage 2 pairs.jsonl and minimal counterfactual branch histories."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from gamecore import FixtureEngine, GameContext, WorldDefinition

from ..shared.io import load_yaml, read_jsonl, write_jsonl
from ..shared.reviewed_sources import load_reviewed_anchors, load_transition_plan
from ..shared.history_projection import project_frozen_history


def write_executable_pairs(
    world_dir: str | Path,
    character_id: str,
    base_history: list[dict[str, Any]] | None = None,
    *,
    output_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    world = Path(world_dir)
    language = load_yaml(world / "environment.yaml").get(
        "language",
        "zh",
    )
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
        _assert_persistent_probe_difference(
            world,
            character_id,
            history,
            plan,
            transition,
        )
        pair_id = f"{character_id}_pair_{index:02d}"
        branch_a = _with_probe(
            history,
            pair_id,
            probe_round,
            transition["anchor_id"],
            _source_event(history, transition["anchor_id"]),
            transition["operations"],
            language,
        )
        branch_b = _with_probe(
            history,
            pair_id,
            probe_round,
            transition["anchor_id"],
            str(anchors[transition["anchor_id"]]["cf_edit"]),
            transition["counterfactual_operations"],
            language,
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
            language,
        )
        branch_b = _with_probe(
            history,
            pair_id,
            probe_round,
            transition["anchor_id"],
            paraphrase,
            None,
            language,
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
                    "admissible_decisions": [
                        {"action": "respond_only", "parameters": {}}
                    ],
                },
            }
        )
    write_jsonl(output_path / "pairs.jsonl", descriptors)
    return descriptors


def _replay_history_context(
    world: Path,
    character_id: str,
    history: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    replacement_anchor_id: str | None = None,
    replacement_operations: list[dict[str, Any]] | None = None,
) -> GameContext:
    definition = WorldDefinition.load_yaml(world / "environment.yaml")
    fixture = FixtureEngine(definition)
    context = GameContext(
        {
            "character_card": load_yaml(
                world / "characters" / f"{character_id}.yaml"
            ),
            **copy.deepcopy(plan["initial_context"]),
            "history": [],
        }
    )
    for record in history:
        transition = record.get("state_transition")
        if not isinstance(transition, dict):
            continue
        operations = transition.get("operations")
        if not operations:
            continue
        if record.get("source_anchor_id") == replacement_anchor_id:
            operations = replacement_operations
        context = fixture.apply(
            context,
            event_id=f"stage2_replay_{record['round']}",
            operations=operations,
            description=str(record["observation"]["content"]),
            history_turn=int(record["round"]),
        ).context
    return context


def _assert_persistent_probe_difference(
    world: Path,
    character_id: str,
    history: list[dict[str, Any]],
    plan: dict[str, Any],
    transition: dict[str, Any],
) -> None:
    anchor_id = str(transition["anchor_id"])
    original = _replay_history_context(
        world,
        character_id,
        history,
        plan,
    )
    counterfactual = _replay_history_context(
        world,
        character_id,
        history,
        plan,
        replacement_anchor_id=anchor_id,
        replacement_operations=transition["counterfactual_operations"],
    )
    if (
        original.runtime_state == counterfactual.runtime_state
        and original.environment == counterfactual.environment
    ):
        raise ValueError(
            f"{anchor_id}在600轮最终上下文中未产生分支状态差异"
        )


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
    language: str,
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
                    "utterance": (
                        "A verified update has just arrived. "
                        "Use it as the basis for your decision."
                        if language == "en"
                        else "刚收到一项经过复核的更新，请以它为准处理。"
                    ),
                },
                {
                    "speaker": "npc",
                    "name": history[-1]["messages"][1].get("name"),
                    "utterance": (
                        "I have received the update and will decide "
                        "from the latest information."
                        if language == "en"
                        else "我已收到这项更新，会按最新情况判断。"
                    ),
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
