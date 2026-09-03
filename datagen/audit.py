"""Quantitative readiness audit for executable character fixtures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from gamecore import ActionRegistry

from .common import load_yaml, read_jsonl
from .llm_history import audit_surface_quality
from .projection import project_frozen_history


def audit_executable_character(
    world_dir: str | Path,
    character_id: str,
    *,
    expected_rounds: int = 600,
) -> dict[str, Any]:
    world = Path(world_dir)
    history = read_jsonl(
        world / "frozen" / character_id / "history.jsonl"
    )
    qa = read_jsonl(world / "frozen" / character_id / "qa.jsonl")
    pairs = read_jsonl(world / "branches" / character_id / "pairs.jsonl")
    anchors = [record for record in history if record["is_anchor"]]
    connectors = [record for record in history if not record["is_anchor"]]

    npc_counts = Counter(
        record["messages"][1]["utterance"] for record in connectors
    )
    player_projection = project_frozen_history(history, audience="player")
    npc_projection = project_frozen_history(history, audience="npc")
    hidden = [
        record
        for record in anchors
        if "player" not in record["observation"]["visible_to"]
    ]
    hidden_leaks = [
        record["source_anchor_id"]
        for record in hidden
        if record["observation"]["content"]
        in record["messages"][0]["utterance"]
        or record["observation"]["event_id"] in player_projection
    ]

    qa_categories = Counter(item["category"] for item in qa)
    qa_layers = Counter(
        item.get("evaluation_layer", item["category"]) for item in qa
    )
    npc_utterances = {
        record["messages"][1]["utterance"] for record in history
    }
    answer_surface_collisions = [
        item["qa_id"]
        for item in qa
        if "state_path" in item and str(item["answer"]) in npc_utterances
    ]

    sensitivity = [
        pair for pair in pairs if pair["pair_type"] == "sensitivity"
    ]
    invariance = [
        pair for pair in pairs if pair["pair_type"] == "invariance"
    ]
    pair_errors: list[str] = []
    surface_quality = audit_surface_quality(history)
    registry = ActionRegistry.load_directory(
        Path(__file__).resolve().parents[1] / "gamecore" / "actions"
    )
    for pair in pairs:
        branch_a = read_jsonl(world / pair["branch_a"])
        branch_b = read_jsonl(world / pair["branch_b"])
        changed = _visible_changed_rounds(branch_a, branch_b)
        if changed != pair["intervention"]["visible_changed_rounds"]:
            pair_errors.append(f"{pair['pair_id']}: visible diff不一致")
        if len(changed) != 1:
            pair_errors.append(f"{pair['pair_id']}: 不止一轮可见差异")
        if pair["pair_type"] == "sensitivity":
            left = pair["decision_rubric"]["branch_a"][
                "admissible_decisions"
            ]
            right = pair["decision_rubric"]["branch_b"][
                "admissible_decisions"
            ]
            if json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
                right, sort_keys=True, ensure_ascii=False
            ):
                pair_errors.append(f"{pair['pair_id']}: 两分支决策相同")
            if not pair["intervention"]["state_changed_rounds"]:
                pair_errors.append(f"{pair['pair_id']}: 状态未传播")
            for branch_name in ("branch_a", "branch_b"):
                rubric = pair["decision_rubric"][branch_name]
                for decision in (
                    rubric["admissible_decisions"]
                    + rubric["forbidden_decisions"]
                ):
                    error = _rubric_contract_error(decision, registry)
                    if error:
                        pair_errors.append(
                            f"{pair['pair_id']}:{branch_name}: {error}"
                        )
        elif _state_changed_rounds(branch_a, branch_b):
            pair_errors.append(f"{pair['pair_id']}: 不变性pair改变了状态")

    checks = {
        "structure": (
            len(history) == expected_rounds
            and len(anchors) == 30
            and len(qa) == 50
            and len(pairs) == 10
        ),
        "gamecore_executable": all(
            record["state_transition"]["operations"]
            and record["state_transition"]["context_delta"]
            for record in anchors
        ),
        "visibility_safe": not hidden_leaks,
        "connector_diversity": (
            len(npc_counts) / len(connectors) >= 0.35
            and max(npc_counts.values(), default=0) <= 20
        ),
        "natural_language_surface": surface_quality["passed"],
        "qa_measurable": (
            len({item["question"] for item in qa}) == 50
            and qa_layers
            == {
                "profile": 6,
                "temporal": 6,
                "local_state": 10,
                "checkpoint_state": 10,
                "long_range_final": 12,
                "multi_hop": 6,
            }
            and not answer_surface_collisions
        ),
        "pairs_minimal_and_causal": (
            len(sensitivity) == 7
            and len(invariance) == 3
            and not pair_errors
        ),
    }
    canon = load_yaml(world / "canon.yaml")
    missing_transitions = [
        npc
        for npc in canon["evaluated_npcs"]
        if not (world / "frozen" / npc / "transitions.yaml").is_file()
    ]
    remaining = [
        "使用目标Qwen tokenizer或首个API usage验证精确输入token数",
        "人工抽样复核LLM session质检未覆盖的细微角色风格问题",
    ]
    if missing_transitions:
        remaining.append(
            f"补齐其他角色transitions.yaml: {', '.join(missing_transitions)}"
        )
    api_ready_checks = {
        key: value
        for key, value in checks.items()
        if key != "natural_language_surface"
    }
    return {
        "character_id": character_id,
        "ready_for_api_smoke_test": all(api_ready_checks.values()),
        "formal_benchmark_ready": all(checks.values()),
        "checks": checks,
        "metrics": {
            "rounds": len(history),
            "anchors": len(anchors),
            "hidden_npc_only_events": len(hidden),
            "hidden_leaks": hidden_leaks,
            "unique_connector_npc": len(npc_counts),
            "connector_npc_unique_ratio": round(
                len(npc_counts) / len(connectors), 4
            ),
            "max_exact_connector_npc_repeat": max(
                npc_counts.values(), default=0
            ),
            "npc_projection_characters": len(npc_projection),
            "player_projection_characters": len(player_projection),
            "qa_categories": dict(qa_categories),
            "qa_layers": dict(qa_layers),
            "answer_surface_collisions": answer_surface_collisions,
            "sensitivity_pairs": len(sensitivity),
            "invariance_pairs": len(invariance),
            "pair_errors": pair_errors,
            "surface_quality": surface_quality,
        },
        "remaining_before_formal_benchmark": remaining,
    }


def _visible_changed_rounds(
    branch_a: list[dict[str, Any]],
    branch_b: list[dict[str, Any]],
) -> list[int]:
    return [
        int(left["round"])
        for left, right in zip(branch_a, branch_b, strict=True)
        if {
            "messages": left["messages"],
            "observation": left["observation"],
        }
        != {
            "messages": right["messages"],
            "observation": right["observation"],
        }
    ]


def _state_changed_rounds(
    branch_a: list[dict[str, Any]],
    branch_b: list[dict[str, Any]],
) -> list[int]:
    return [
        int(left["round"])
        for left, right in zip(branch_a, branch_b, strict=True)
        if left["state_transition"] != right["state_transition"]
    ]


def _rubric_contract_error(
    decision: dict[str, Any],
    registry: ActionRegistry,
) -> str | None:
    action = decision.get("action")
    parameters = decision.get("parameters")
    if not isinstance(action, str) or not isinstance(parameters, dict):
        return "decision必须包含action字符串和parameters对象"
    if action == "respond_only":
        return None if parameters == {} else "respond_only不得携带参数"
    try:
        spec = registry.get(action)
    except Exception:
        return f"未知Action: {action}"
    unknown = set(parameters) - set(spec.parameters)
    if unknown:
        return f"{action}包含未知参数: {sorted(unknown)}"
    python_types: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "string": str,
        "array": list,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
    }
    for name, value in parameters.items():
        parameter = spec.parameters[name]
        if not isinstance(value, python_types[parameter.type]):
            return f"{action}.{name}类型错误"
        if parameter.enum and value not in parameter.enum:
            return f"{action}.{name}不在enum中"
    return None
