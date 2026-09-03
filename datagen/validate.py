"""Structural and minimality validation for generated Stage 1/2 assets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .audit import audit_executable_character
from .common import load_yaml, read_jsonl


EXPECTED_EXECUTABLE_QA_LAYERS = {
    "profile": 6,
    "temporal": 6,
    "local_state": 10,
    "checkpoint_state": 10,
    "long_range_final": 12,
    "multi_hop": 6,
}


def validate_world_outputs(
    world_dir: str | Path,
    *,
    rounds: int = 600,
    qa_count: int = 50,
    pair_count: int = 10,
) -> dict[str, Any]:
    world = Path(world_dir)
    canon = load_yaml(world / "canon.yaml")
    reports = [
        validate_character_outputs(
            world,
            character_id,
            rounds=rounds,
            qa_count=qa_count,
            pair_count=pair_count,
        )
        for character_id in canon["evaluated_npcs"]
    ]
    return {
        "world_id": load_yaml(world / "environment.yaml")["world_id"],
        "character_count": len(reports),
        "rounds": sum(report["rounds"] for report in reports),
        "messages": sum(report["messages"] for report in reports),
        "anchors": sum(report["anchors"] for report in reports),
        "qa": sum(report["qa"] for report in reports),
        "open_tasks": sum(report["open_tasks"] for report in reports),
        "pairs": sum(report["pairs"] for report in reports),
        "sensitivity_pairs": sum(
            report["sensitivity_pairs"] for report in reports
        ),
        "invariance_pairs": sum(
            report["invariance_pairs"] for report in reports
        ),
    }


def validate_character_outputs(
    world_dir: str | Path,
    character_id: str,
    *,
    rounds: int = 600,
    qa_count: int = 50,
    pair_count: int = 10,
) -> dict[str, Any]:
    world = Path(world_dir)
    frozen_dir = world / "frozen" / character_id
    if not (frozen_dir / "transitions.yaml").is_file():
        raise ValueError(f"{character_id}缺少人工审核的transitions.yaml")
    history = read_jsonl(frozen_dir / "history.jsonl")
    qa = read_jsonl(frozen_dir / "qa.jsonl")
    open_tasks = read_jsonl(frozen_dir / "open_tasks.jsonl")
    pairs = read_jsonl(world / "branches" / character_id / "pairs.jsonl")

    if len(history) != rounds:
        raise ValueError(f"{character_id}历史轮数错误: {len(history)}")
    if [record["round"] for record in history] != list(range(1, rounds + 1)):
        raise ValueError(f"{character_id}round不连续")
    if any(len(record.get("messages", [])) != 2 for record in history):
        raise ValueError(f"{character_id}每轮必须包含两条消息")
    for record in history:
        speakers = [message.get("speaker") for message in record["messages"]]
        if speakers != ["player", "npc"]:
            raise ValueError(f"{character_id}消息顺序错误")
        if any(
            not isinstance(message.get("utterance"), str)
            or not message["utterance"].strip()
            for message in record["messages"]
        ):
            raise ValueError(f"{character_id}包含空消息")
    anchor_records = [record for record in history if record["is_anchor"]]
    if len(anchor_records) != 30:
        raise ValueError(f"{character_id}必须包含30个锚点")
    anchor_ids = {record["anchor_id"] for record in anchor_records}
    if len(anchor_ids) != 30:
        raise ValueError(f"{character_id}锚点ID重复")
    evidence_ids = {
        record["observation"]["event_id"] for record in anchor_records
    }

    if len(qa) != qa_count:
        raise ValueError(f"{character_id}QA数量错误")
    if len({item["qa_id"] for item in qa}) != qa_count:
        raise ValueError(f"{character_id}QA ID重复")
    if len({item["question"] for item in qa}) != qa_count:
        raise ValueError(f"{character_id}存在重复QA问题")
    distribution_key = "evaluation_layer"
    expected_distribution = EXPECTED_EXECUTABLE_QA_LAYERS
    category_counts = Counter(item[distribution_key] for item in qa)
    if dict(category_counts) != expected_distribution:
        raise ValueError(
            f"{character_id}QA类别配比错误: {dict(category_counts)}"
        )
    for item in qa:
        for evidence in item["evidence"]:
            if evidence.startswith("character_card:"):
                continue
            if evidence not in evidence_ids:
                raise ValueError(
                    f"{character_id}QA引用未知证据: {evidence}"
                )
    if len(open_tasks) != 2:
        raise ValueError(f"{character_id}开放任务数量错误")

    if len(pairs) != pair_count:
        raise ValueError(f"{character_id}Pair数量错误")
    pair_types = Counter(pair["pair_type"] for pair in pairs)
    expected_sensitivity = round(pair_count * 0.7)
    if pair_types != Counter(
        {
            "sensitivity": expected_sensitivity,
            "invariance": pair_count - expected_sensitivity,
        }
    ):
        raise ValueError(f"{character_id}Pair类型配比错误")
    audit = audit_executable_character(
        world,
        character_id,
        expected_rounds=rounds,
    )
    if not audit["ready_for_api_smoke_test"]:
        raise ValueError(
            f"{character_id}可执行数据审计未通过: {audit['checks']}"
        )

    return {
        "character_id": character_id,
        "rounds": len(history),
        "messages": len(history) * 2,
        "anchors": len(anchor_records),
        "qa": len(qa),
        "open_tasks": len(open_tasks),
        "pairs": len(pairs),
        "sensitivity_pairs": pair_types["sensitivity"],
        "invariance_pairs": pair_types["invariance"],
    }
