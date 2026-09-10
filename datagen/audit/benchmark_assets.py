"""Quantitative readiness audit for executable character fixtures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from gamecore import ActionRegistry

from ..shared.io import load_yaml, read_jsonl
from ..generation.stage1_history import audit_surface_quality
from ..shared.history_projection import project_frozen_history


OUTPUT_MANUAL_RULES = {
    "history_memory_quality": "600轮历史连续、角色稳定，普通记忆与状态变化可追踪。",
    "qa_answerable": "50道QA均有唯一可判定答案，证据引用正确且未泄漏答案。",
    "pair_causal": "Sensitivity因关键状态改变而改变决定，Invariance的状态和决定均保持正确且一致。",
    "role_visibility_safe": "抽样确认角色行为一致，Player与NPC均未泄漏不可见事实。",
}


def write_output_review_template(
    world_dir: str | Path,
    character_id: str,
) -> Path:
    path = Path(world_dir) / "frozen" / character_id / "output_review.yaml"
    payload = {
        "schema_version": 1,
        "character_id": character_id,
        "reviewer": "",
        "reviewed_at": "",
        "approved": False,
        "checks": {rule_id: False for rule_id in OUTPUT_MANUAL_RULES},
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


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
        Path(__file__).resolve().parents[2] / "gamecore" / "actions"
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
        else:
            if _state_changed_rounds(branch_a, branch_b):
                pair_errors.append(f"{pair['pair_id']}: 不变性pair改变了状态")
            admissible = pair.get("decision_rubric", {}).get(
                "admissible_decisions",
                [],
            )
            if not admissible:
                pair_errors.append(
                    f"{pair['pair_id']}: 不变性pair缺少共同可接受决定"
                )
            for decision in admissible:
                error = _rubric_contract_error(decision, registry)
                if error:
                    pair_errors.append(f"{pair['pair_id']}: {error}")

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
                "temporal": 4,
                "episodic": 6,
                "local_state": 8,
                "checkpoint_state": 8,
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
    output_review = _output_review_status(world, character_id)
    if not output_review["approved"]:
        remaining.append("完成人工output_review.yaml并将全部检查项批准")
    if output_review["blocking_findings"]:
        remaining.append("关闭review_findings.yaml中的blocker和major问题")
    return {
        "character_id": character_id,
        "ready_for_api_smoke_test": all(api_ready_checks.values()),
        "formal_benchmark_ready": (
            all(checks.values())
            and output_review["approved"]
            and not output_review["blocking_findings"]
        ),
        "output_review": output_review,
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


def _output_review_status(
    world: Path,
    character_id: str,
) -> dict[str, Any]:
    path = world / "frozen" / character_id / "output_review.yaml"
    approved = False
    if path.is_file():
        review = load_yaml(path)
        checks = review.get("checks")
        approved = bool(
            review.get("character_id") == character_id
            and review.get("approved") is True
            and isinstance(checks, dict)
            and all(checks.get(rule_id) is True for rule_id in OUTPUT_MANUAL_RULES)
        )
    findings_path = world / "frozen" / character_id / "review_findings.yaml"
    blocking = _blocking_findings(findings_path)
    return {
        "approved": approved,
        "blocking_findings": blocking,
        "manual_rules": OUTPUT_MANUAL_RULES,
    }


def _blocking_findings(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        payload = load_yaml(path)
    except Exception:
        return ["invalid_review_findings"]
    if isinstance(payload, dict):
        values = payload.get("findings", [payload])
    else:
        values = payload
    if not isinstance(values, list):
        return ["invalid_review_findings"]
    return [
        str(item.get("item_id", "unknown"))
        for item in values
        if isinstance(item, dict)
        and item.get("status") == "open"
        and item.get("severity") in {"blocker", "major"}
    ]


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
    if action == "accept_claim":
        action = "decide_claim"
    try:
        registry.validate_call({"name": action, "parameters": parameters})
    except Exception as exc:
        return str(exc)
    return None
