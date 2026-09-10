"""Small Stage 1/2 experiment runner for executable world data."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agents.prompt_builder import PromptBundle, project_npc_character_card
from datagen.audit.benchmark_assets import audit_executable_character
from datagen.shared.io import load_yaml, read_jsonl, write_jsonl
from datagen.shared.history_projection import project_frozen_history
from gamecore import ActionRegistry, ActionValidationError
from llm import (
    GenerationConfig,
    create_llm_client,
    generate_structured,
    load_llm_settings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "models" / "venus_deepseek.yaml"
)
_NON_EXACT_PARAMETERS = {"reason"}


def build_qa_bundle(
    world_dir: str | Path,
    character_id: str,
    qa_items: list[dict[str, Any]],
) -> PromptBundle:
    world = Path(world_dir)
    card = project_npc_character_card(
        load_yaml(world / "characters" / f"{character_id}.yaml")
    )
    environment = load_yaml(world / "environment.yaml")
    history = read_jsonl(
        world / "frozen" / character_id / "history.jsonl"
    )
    language = environment.get("language", "zh")
    if language == "en":
        system = (
            "You are taking the RPG-AgentBench frozen-history diagnostic. "
            "Answer only from the character card and history, without outside "
            "story knowledge. For each question, give a short answer, supporting "
            "Event IDs, and confidence from 0 to 1. Character-profile answers "
            "may cite exact character_card paths. Copy labels, entity IDs, "
            "Event IDs, and paths exactly. Output one JSON object only."
        )
    else:
        system = (
            "你正在接受RPG-AgentBench冻结历史诊断。只能依据角色卡和历史回答，"
            "不得使用外部剧情知识。每题给出简短答案、支持答案的Event ID和0到1的"
            "置信度；角色设定题可引用给定的character_card路径。标签、实体ID、"
            "Event ID和character_card路径必须原样复制。只输出一个JSON对象。"
        )
    questions = [
        {
            "qa_id": item["qa_id"],
            "question": item["question"],
            "answer_type": item.get("answer_type", "string"),
            "allowed_answers": _allowed_qa_answers(item, environment),
            "answer_schema": item.get("answer_schema"),
        }
        for item in qa_items
    ]
    projected_history = project_frozen_history(history, audience="npc")
    entities = {
        "locations": environment["locations"],
        "items": environment["items"],
    }
    if language == "en":
        user = (
            f"Character card:\n{json.dumps(card, ensure_ascii=False)}\n\n"
            f"World entity ID map:\n{json.dumps(entities, ensure_ascii=False)}\n\n"
            f"Complete {len(history)}-round history:\n{projected_history}\n\n"
            f"Questions:\n{json.dumps(questions, ensure_ascii=False)}\n\n"
            "When allowed_answers is non-empty, copy one value exactly without "
            "paraphrasing. Booleans must be JSON true/false. evidence may contain "
            "only Event IDs from the history, or exact character_card paths for "
            "profile questions; never add round numbers.\n\n"
            'Output: {"answers":[{"qa_id":"...","answer":"...",'
            '"evidence":["anchor_..."],"confidence":0.0}]}'
        )
    else:
        user = (
            f"角色卡：\n{json.dumps(card, ensure_ascii=False)}\n\n"
            f"世界实体ID映射：\n{json.dumps(entities, ensure_ascii=False)}\n\n"
            f"完整{len(history)}轮历史：\n{projected_history}\n\n"
            f"问题：\n{json.dumps(questions, ensure_ascii=False)}\n\n"
            "若allowed_answers非空，answer必须严格复制其中一个值，不得改写为同义"
            "中文；boolean必须输出JSON true/false。evidence只能填写历史中的Event "
            "ID，角色卡问题则填写精确的character_card字段路径，禁止加入Round编号。\n\n"
            "输出格式："
            '{"answers":[{"qa_id":"...","answer":"...",'
            '"evidence":["anchor_..."],"confidence":0.0}]}'
        )
    return PromptBundle(system_prompt=system, user_prompt=user)


def build_branch_bundle(
    world_dir: str | Path,
    character_id: str,
    pair: Mapping[str, Any],
    branch: str,
    registry: ActionRegistry,
) -> PromptBundle:
    if branch not in {"a", "b"}:
        raise ValueError("branch必须为a或b")
    world = Path(world_dir)
    language = load_yaml(world / "environment.yaml").get(
        "language",
        "zh",
    )
    card = project_npc_character_card(
        load_yaml(world / "characters" / f"{character_id}.yaml")
    )
    history = read_jsonl(world / pair[f"branch_{branch}"])
    action_specs = []
    for name in registry.names:
        spec = registry.get(name)
        action_specs.append(
            {
                "type": name,
                "description": (
                    f"Execute the {name} state action."
                    if language == "en"
                    else spec.description
                ),
                "parameters": {
                    key: {
                        "type": value.type,
                        "required": value.required,
                        "enum": list(value.enum),
                    }
                    for key, value in spec.parameters.items()
                },
            }
        )
    action_specs.append(
        {
            "type": "respond_only",
            "description": (
                "Reply without changing GameContext state."
                if language == "en"
                else "只回复，不修改任何GameContext状态。"
            ),
            "parameters": {},
        }
    )
    projected_history = project_frozen_history(history, audience="npc")
    decision_target = copy.deepcopy(pair.get("decision_target", {}))
    if isinstance(decision_target, dict):
        candidates = decision_target.get("candidate_actions")
        if isinstance(candidates, list):
            decision_target["candidate_actions"] = [
                "decide_claim" if value == "accept_claim" else value
                for value in candidates
            ]
    if language == "en":
        system = (
            "You are the evaluated NPC. Using only the character card and "
            "complete history, make an in-character, executable decision on the "
            "final request. Do not inspect or infer scoring rules, and do not "
            "output reasoning. state_basis may cite only visible Event IDs. "
            "When the request requires a claim, access, relationship, or other "
            "state decision, return the corresponding structured Action rather "
            "than expressing it only through respond_only. Output one JSON object."
        )
        user = (
            f"Character card:\n{json.dumps(card, ensure_ascii=False)}\n\n"
            f"Available decisions:\n{json.dumps(action_specs, ensure_ascii=False)}\n\n"
            f"Complete {len(history)}-round history:\n{projected_history}\n\n"
            "Structured decision target:\n"
            f"{json.dumps(decision_target, ensure_ascii=False)}\n"
            f"Player final request: {pair['final_query']}\n\n"
            "For a state_path query, use respond_only and put the current state "
            "in state_answer. For an Action decision, set state_answer to null.\n\n"
            'Output: {"decision":{"type":"action_type","parameters":{}},'
            '"state_answer":null,"state_basis":["anchor_..."],'
            '"utterance":"in-character response"}'
        )
    else:
        system = (
            "你是被测NPC。依据角色卡和完整历史，对最后请求作出角色一致且可执行的"
            "决定。不得查看或猜测评分规则，不得输出思维过程。state_basis只能引用"
            "历史中可见的Event ID。若请求要求明确判断、授权、关系变化或其他状态"
            "决定，必须输出对应结构化Action，不能只用respond_only在语言中表态。"
            "只输出一个JSON对象。"
        )
        user = (
            f"角色卡：\n{json.dumps(card, ensure_ascii=False)}\n\n"
            f"可用决定：\n{json.dumps(action_specs, ensure_ascii=False)}\n\n"
            f"完整{len(history)}轮历史：\n{projected_history}\n\n"
            f"本题结构化决定目标：{json.dumps(decision_target, ensure_ascii=False)}\n"
            f"Player最终请求：{pair['final_query']}\n\n"
            "如果目标是state_path查询，decision使用respond_only，并在state_answer"
            "填写当前状态；如果目标要求Action决定，state_answer填写null。\n\n"
            "输出格式："
            '{"decision":{"type":"action_type","parameters":{}},'
            '"state_answer":null,"state_basis":["anchor_..."],'
            '"utterance":"角色化回复"}'
        )
    return PromptBundle(system_prompt=system, user_prompt=user)


def run_pilot(
    world_dir: str | Path,
    character_id: str,
    *,
    dry_run: bool,
    config_path: str | Path = DEFAULT_CONFIG,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    qa_limit: int = 10,
    pair_limit: int = 2,
    output_root: str | Path = "runs",
) -> dict[str, Any]:
    world = Path(world_dir)
    settings = load_llm_settings(
        config_path,
        project_root=PROJECT_ROOT,
    )
    model = model or settings.generation.model
    base_url = base_url or settings.base_url
    api_key_env = api_key_env or settings.api_key_env
    audit = audit_executable_character(world, character_id)
    if not audit["ready_for_api_smoke_test"]:
        raise ValueError(f"数据未通过API前审计: {audit['checks']}")
    qa = _select_qa(
        read_jsonl(world / "frozen" / character_id / "qa.jsonl"),
        qa_limit,
    )
    pairs = read_jsonl(world / "branches" / character_id / "pairs.jsonl")[
        :pair_limit
    ]
    registry = ActionRegistry.load_directory(
        PROJECT_ROOT / "gamecore" / "actions"
    )
    qa_bundle = build_qa_bundle(world, character_id, qa)
    branch_bundles = [
        (
            pair,
            branch,
            build_branch_bundle(world, character_id, pair, branch, registry),
        )
        for pair in pairs
        for branch in ("a", "b")
    ]
    prompt_sizes = {
        "qa_characters": _bundle_characters(qa_bundle),
        "branch_characters": [
            {
                "pair_id": pair["pair_id"],
                "branch": branch,
                "characters": _bundle_characters(bundle),
            }
            for pair, branch, bundle in branch_bundles
        ],
    }
    max_characters = max(
        [prompt_sizes["qa_characters"]]
        + [item["characters"] for item in prompt_sizes["branch_characters"]]
    )
    preflight = {
        "model": model,
        "data_ready": True,
        "api_calls": 1 + len(branch_bundles),
        "max_prompt_characters": max_characters,
        "conservative_token_upper_estimate": math.ceil(max_characters * 1.5),
        "provider": settings.provider,
        "configured_context_safety_limit": 997_952,
        "within_documented_context": math.ceil(max_characters * 1.5)
        < 997_952,
        "prompt_sizes": prompt_sizes,
    }
    if dry_run:
        return {"dry_run": True, "preflight": preflight, "audit": audit}

    configured = settings.generation
    config = GenerationConfig(
        model=model,
        temperature=configured.temperature,
        top_p=configured.top_p,
        max_tokens=configured.max_tokens,
        seed=configured.seed,
        max_format_retries=configured.max_format_retries,
    )
    effective_settings = replace(
        settings,
        base_url=base_url,
        api_key_env=api_key_env,
        generation=config,
    )
    client = create_llm_client(effective_settings, scene="pilot")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / f"{timestamp}_{character_id}_{model}"
    run_dir.mkdir(parents=True, exist_ok=False)

    qa_output = generate_structured(
        client,
        qa_bundle,
        config,
        lambda value: _validate_qa_output(value, qa),
    )
    qa_records = _score_qa(qa, qa_output, client.last_usage)
    write_jsonl(run_dir / "stage1_qa.jsonl", qa_records)

    branch_records: list[dict[str, Any]] = []
    for pair, branch, bundle in branch_bundles:
        output = generate_structured(
            client,
            bundle,
            config,
            lambda value: _validate_branch_output(value, registry),
        )
        branch_records.append(
            _score_branch(
                pair,
                branch,
                output,
                client.last_usage,
            )
        )
    write_jsonl(run_dir / "stage2_branches.jsonl", branch_records)
    summary = {
        "character_id": character_id,
        "model": model,
        "base_url": base_url,
        "preflight": preflight,
        "stage1": {
            "count": len(qa_records),
            "answer_accuracy": _mean(
                record["answer_correct"] for record in qa_records
            ),
            "evidence_f1": _mean(
                record["evidence_f1"] for record in qa_records
            ),
        },
        "stage2": {
            "branch_count": len(branch_records),
            "branch_action_accuracy": _mean(
                record["action_correct"] for record in branch_records
            ),
            "pairwise_joint_accuracy": _pairwise_joint(branch_records),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"dry_run": False, "run_dir": str(run_dir), "summary": summary}


def _validate_qa_output(
    value: dict[str, Any],
    expected: list[dict[str, Any]],
) -> None:
    _validate_qa_partial_output(value, expected)
    expected_ids = {item["qa_id"] for item in expected}
    actual_ids = {answer["qa_id"] for answer in value["answers"]}
    if actual_ids != expected_ids or len(value["answers"]) != len(expected):
        missing = sorted(expected_ids - actual_ids)
        raise ValueError(
            "QA输出未恰好覆盖请求的问题；"
            f"缺少qa_id={missing}"
        )


def _validate_qa_partial_output(
    value: dict[str, Any],
    expected: list[dict[str, Any]],
) -> None:
    if set(value) != {"answers"} or not isinstance(value["answers"], list):
        raise ValueError("QA输出必须只包含answers数组")
    if not value["answers"]:
        raise ValueError("QA输出的answers不能为空")
    expected_ids = {item["qa_id"] for item in expected}
    actual_ids: set[str] = set()
    for answer in value["answers"]:
        if not isinstance(answer, dict) or set(answer) != {
            "qa_id",
            "answer",
            "evidence",
            "confidence",
        }:
            raise ValueError("QA答案字段不符合契约")
        if not isinstance(answer["qa_id"], str):
            raise ValueError("qa_id必须是字符串")
        if answer["qa_id"] not in expected_ids:
            raise ValueError(
                f"QA输出包含未请求的qa_id: {answer['qa_id']}"
            )
        if answer["qa_id"] in actual_ids:
            raise ValueError(
                f"QA输出包含重复qa_id: {answer['qa_id']}"
            )
        if not isinstance(answer["evidence"], list) or not all(
            isinstance(item, str) for item in answer["evidence"]
        ):
            raise ValueError("evidence必须是字符串数组")
        confidence = answer["confidence"]
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence必须在0到1之间")
        actual_ids.add(answer["qa_id"])


def _validate_branch_output(
    value: dict[str, Any],
    registry: ActionRegistry,
) -> None:
    if set(value) != {
        "decision",
        "state_answer",
        "state_basis",
        "utterance",
    }:
        raise ValueError("Stage2输出字段不符合契约")
    decision = value["decision"]
    if not isinstance(decision, dict) or set(decision) != {
        "type",
        "parameters",
    }:
        raise ValueError("decision字段不符合契约")
    if not isinstance(value["state_basis"], list) or not all(
        isinstance(item, str) for item in value["state_basis"]
    ):
        raise ValueError("state_basis必须是字符串数组")
    if not isinstance(value["utterance"], str) or not value["utterance"].strip():
        raise ValueError("utterance不能为空")
    if decision["type"] == "respond_only":
        if decision["parameters"] != {}:
            raise ValueError("respond_only不得携带参数")
        return
    aliases = {
        "reject_claim": (
            "reject_claim不是正式Action；必须使用type=decide_claim，并在"
            "parameters中填写claim_id和decision=reject"
        ),
        "grant_access": (
            "grant_access不是正式Action；必须使用type=decide_access，并在"
            "parameters中填写subject、resource和decision=grant"
        ),
        "deny_access": (
            "deny_access不是正式Action；必须使用type=decide_access，并在"
            "parameters中填写subject、resource和decision=deny"
        ),
    }
    if decision["type"] in aliases:
        raise ValueError(aliases[decision["type"]])
    try:
        registry.validate_call(
            {"name": decision["type"], "parameters": decision["parameters"]}
        )
    except ActionValidationError as error:
        raise ValueError(str(error)) from error


def _score_qa(
    expected: list[dict[str, Any]],
    output: dict[str, Any],
    usage: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    by_id = {item["qa_id"]: item for item in output["answers"]}
    records: list[dict[str, Any]] = []
    for item in expected:
        predicted = by_id[item["qa_id"]]
        expected_evidence = {
            _normalize_evidence_id(value) for value in item["evidence"]
        }
        predicted_evidence = {
            _normalize_evidence_id(value) for value in predicted["evidence"]
        }
        overlap = len(expected_evidence & predicted_evidence)
        precision = overlap / len(predicted_evidence) if predicted_evidence else 0.0
        recall = overlap / len(expected_evidence) if expected_evidence else 1.0
        evidence_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        records.append(
            {
                "qa_id": item["qa_id"],
                "category": item["category"],
                "evaluation_layer": item.get(
                    "evaluation_layer", item["category"]
                ),
                "expected_answer": item["answer"],
                "predicted_answer": predicted["answer"],
                "answer_correct": _normalize_answer(item["answer"])
                == _normalize_answer(predicted["answer"]),
                "expected_evidence": item["evidence"],
                "predicted_evidence": predicted["evidence"],
                "evidence_f1": round(evidence_f1, 6),
                "confidence": predicted["confidence"],
                "usage": usage,
            }
        )
    return records


def _score_branch(
    pair: Mapping[str, Any],
    branch: str,
    output: dict[str, Any],
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    if pair["pair_type"] == "sensitivity":
        admissible = pair["decision_rubric"][f"branch_{branch}"][
            "admissible_decisions"
        ]
        correct: bool | None = any(
            _decision_matches(output["decision"], expected)
            for expected in admissible
        )
    else:
        admissible = None
        correct = None
    expected_state = (
        pair["decision_rubric"].get("expected_state")
        if pair["pair_type"] == "invariance"
        else None
    )
    state_correct = (
        _normalize_answer(output["state_answer"])
        == _normalize_answer(expected_state["value"])
        if expected_state is not None
        else None
    )
    return {
        "pair_id": pair["pair_id"],
        "pair_type": pair["pair_type"],
        "branch": branch,
        "decision": output["decision"],
        "state_answer": output["state_answer"],
        "state_correct": state_correct,
        "state_basis": output["state_basis"],
        "utterance": output["utterance"],
        "admissible_decisions": admissible,
        "action_correct": correct,
        "usage": usage,
    }


def _decision_matches(
    predicted: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    expected_action = expected.get("action")
    if expected_action == "accept_claim":
        expected_action = "decide_claim"
    if predicted.get("type") != expected_action:
        return False
    predicted_parameters = predicted.get("parameters", {})
    return all(
        key in _NON_EXACT_PARAMETERS
        or predicted_parameters.get(key) == value
        for key, value in expected.get("parameters", {}).items()
    )


def _normalize_answer(value: Any) -> str:
    if isinstance(value, Mapping):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).strip().lower()
    if isinstance(value, list):
        return json.dumps(
            sorted(str(item).strip().lower() for item in value),
            ensure_ascii=False,
        )
    return str(value).strip().lower()


def _normalize_evidence_id(value: str) -> str:
    text = value.strip()
    if not text.startswith("character_card"):
        return text
    text = text.replace("character_card.", "character_card:", 1)
    text = text.replace("[", ".").replace("]", "")
    return text


def _select_qa(
    items: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("qa_limit不能为负数")
    categories = [
        "profile",
        "temporal",
        "knowledge",
        "commitment_goal",
        "relationship",
        "resource",
    ]
    grouped = {
        category: [
            item for item in items if item["category"] == category
        ]
        for category in categories
    }
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < min(limit, len(items)):
        added = False
        for category in categories:
            values = grouped[category]
            if index < len(values) and len(selected) < limit:
                selected.append(values[index])
                added = True
        if not added:
            break
        index += 1
    return selected


def _allowed_qa_answers(
    item: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> list[Any]:
    answer_type = item.get("answer_type")
    state_path = str(item.get("state_path", ""))
    if answer_type == "boolean":
        return [True, False]
    if answer_type == "event_id":
        return []
    if ".claims." in state_path:
        return ["accept", "reject", "uncertain"]
    if ".relationships." in state_path:
        return ["hostile", "distrustful", "neutral", "trusting", "loyal"]
    if ".goals." in state_path:
        return ["active", "fulfilled", "cancelled", "failed"]
    if ".commitments." in state_path:
        return ["active", "fulfilled", "cancelled", "violated"]
    if ".knowledge." in state_path:
        return ["active", "superseded"]
    if ".access." in state_path:
        return ["granted", "denied", "revoked"]
    if state_path == "environment.dialogue_status":
        return ["active", "ended"]
    if state_path.startswith("environment.locations."):
        return [location["id"] for location in environment["locations"]]
    return []


def _pairwise_joint(records: list[dict[str, Any]]) -> float | None:
    grouped: dict[str, list[bool]] = {}
    for record in records:
        if record["action_correct"] is None:
            continue
        grouped.setdefault(record["pair_id"], []).append(
            record["action_correct"]
        )
    if not grouped:
        return None
    return _mean(len(values) == 2 and all(values) for values in grouped.values())


def _mean(values: Any) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(float(item) for item in items) / len(items)


def _bundle_characters(bundle: PromptBundle) -> int:
    return len(bundle.system_prompt) + len(bundle.user_prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--character", default="yu_zecheng")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--qa-limit", type=int, default=10)
    parser.add_argument("--pair-limit", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_pilot(
        args.world,
        args.character,
        dry_run=args.dry_run,
        config_path=args.config,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        qa_limit=args.qa_limit,
        pair_limit=args.pair_limit,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
