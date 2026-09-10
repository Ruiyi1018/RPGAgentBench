"""Generate review-only Anchor and Transition drafts from approved sources."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from agents.prompt_builder import PromptBundle
from gamecore import FixtureEngine, GameContext, WorldDefinition
from llm import GenerationConfig, LLMClient, generate_structured

from ..shared.io import load_yaml
from ..audit.source_assets import audit_foundation


def generate_anchor_draft(
    world_dir: str | Path,
    character_id: str,
    *,
    client: LLMClient,
    config: GenerationConfig,
    total_anchors: int = 30,
    canon_anchors: int = 10,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate drafts only; never overwrite reviewed benchmark assets."""

    world = Path(world_dir)
    foundation = audit_foundation(world, [character_id])
    if not foundation["ready_for_anchor_generation"]:
        raise ValueError("世界与角色画像尚未通过机器审查和人工门禁")
    if total_anchors < 4 or not 0 <= canon_anchors <= total_anchors:
        raise ValueError("Anchor数量配置无效")
    destination = (
        Path(output_dir)
        if output_dir is not None
        else world / "drafts" / character_id
    )
    existing = [
        destination / name
        for name in (
            "anchors.yaml",
            "transitions.yaml",
            "generation_report.yaml",
        )
        if (destination / name).exists()
    ]
    if existing:
        raise FileExistsError(f"Anchor草案已存在: {existing}")

    card = load_yaml(world / "characters" / f"{character_id}.yaml")
    environment = load_yaml(world / "environment.yaml")
    pool = load_yaml(world / "anchor_pool.yaml")
    eligible = [
        event
        for event in pool["events"]
        if character_id in event.get("anchor_for", [])
    ]
    if len(eligible) < canon_anchors:
        raise ValueError(
            f"{character_id}只有{len(eligible)}个可用正典Anchor，"
            f"不足{canon_anchors}个"
        )
    initial_context = _load_initial_context(world, character_id)
    bundle = _anchor_outline_prompt(
        card,
        environment,
        eligible,
        total_anchors,
        canon_anchors,
    )
    outline = generate_structured(
        client,
        bundle,
        config,
        lambda value: _validate_anchor_outline(
            value,
            eligible,
            total_anchors,
            canon_anchors,
        ),
    )
    canon_by_id = {event["id"]: event for event in eligible}
    generated_by_id = {
        anchor["id"]: anchor for anchor in outline["generated_anchors"]
    }
    ordered_ids = list(outline["canon_anchors"]) + list(generated_by_id)
    probe_by_id = {
        item["anchor_id"]: item["type"]
        for item in outline["probe_plan"]
    }
    definition = WorldDefinition.from_mapping(environment)
    context = GameContext(
        {
            "character_card": card,
            **copy.deepcopy(dict(initial_context)),
            "history": [],
        }
    )
    fixture = FixtureEngine(definition)
    transitions: list[dict[str, Any]] = []
    for batch_start in range(0, len(ordered_ids), 5):
        batch_ids = ordered_ids[batch_start : batch_start + 5]
        batch_anchors = [
            (
                {
                    "id": anchor_id,
                    "source": "controlled",
                    **generated_by_id[anchor_id],
                }
                if anchor_id in generated_by_id
                else {
                    "id": anchor_id,
                    "source": "canon",
                    "event": canon_by_id[anchor_id]["fact"],
                    "state_change": canon_by_id[anchor_id]["state_change"],
                    "cf_edit": canon_by_id[anchor_id]["cf_edit"],
                }
            )
            for anchor_id in batch_ids
        ]
        transition_bundle = _transition_batch_prompt(
            card,
            environment,
            context,
            batch_anchors,
            {
                anchor_id: probe_by_id[anchor_id]
                for anchor_id in batch_ids
            },
        )
        batch_output = generate_structured(
            client,
            transition_bundle,
            config,
            lambda value: _validate_transition_batch(
                value,
                context,
                definition,
                batch_ids,
                probe_by_id,
            ),
        )
        for transition in batch_output["transitions"]:
            result = fixture.apply(
                context,
                event_id=f"draft_{transition['anchor_id']}",
                operations=transition["operations"],
                description=transition["npc_surface"],
            )
            context = result.context
            transitions.append(transition)
    output = {
        "canon_anchors": outline["canon_anchors"],
        "generated_anchors": outline["generated_anchors"],
        "transitions": transitions,
    }
    _validate_anchor_output(
        output,
        character_id,
        environment,
        initial_context,
        eligible,
        total_anchors,
        canon_anchors,
    )
    anchors_payload = {
        "character_id": character_id,
        "anchor_count": total_anchors,
        "canon_anchors": output["canon_anchors"],
        "generated_anchors": output["generated_anchors"],
    }
    transitions_payload = {
        "character_id": character_id,
        "initial_context": initial_context,
        "transitions": output["transitions"],
    }
    destination.mkdir(parents=True, exist_ok=True)
    _write_yaml(destination / "anchors.yaml", anchors_payload)
    _write_yaml(destination / "transitions.yaml", transitions_payload)
    report = {
        "schema_version": 1,
        "character_id": character_id,
        "draft_only": True,
        "anchors": total_anchors,
        "canon_anchors": canon_anchors,
        "controlled_anchors": total_anchors - canon_anchors,
        "logical_api_calls": 1 + (total_anchors + 4) // 5,
        "files": {
            "anchors": str(destination / "anchors.yaml"),
            "transitions": str(destination / "transitions.yaml"),
        },
        "next_gate": "人工审查后才可复制到frozen/<npc>/",
    }
    _write_yaml(destination / "generation_report.yaml", report)
    return report


def _load_initial_context(
    world: Path,
    character_id: str,
) -> dict[str, Any]:
    source = world / "frozen" / character_id / "initial_context.yaml"
    if source.is_file():
        return load_yaml(source)
    transition_source = (
        world / "frozen" / character_id / "transitions.yaml"
    )
    if transition_source.is_file():
        value = load_yaml(transition_source).get("initial_context")
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
    raise ValueError(
        f"{character_id}缺少initial_context.yaml或transitions.yaml中的"
        "initial_context"
    )


def _anchor_outline_prompt(
    card: Mapping[str, Any],
    environment: Mapping[str, Any],
    eligible: list[dict[str, Any]],
    total_anchors: int,
    canon_anchors: int,
) -> PromptBundle:
    payload = {
        "character": card,
        "world": {
            "world_id": environment.get("world_id"),
            "name": environment.get("name"),
            "language": environment.get("language"),
            "locations": environment.get("locations", []),
            "items": environment.get("items", []),
        },
        "eligible_canon_events": eligible,
        "required_counts": {
            "total": total_anchors,
            "canon": canon_anchors,
            "controlled": total_anchors - canon_anchors,
        },
    }
    return PromptBundle(
        system_prompt=(
            "你是跨世界角色评测的Anchor大纲生成器。只根据给定来源、世界和"
            "角色画像生成结构化草案。每个Anchor必须能够产生可执行状态"
            "变化，并在后续选择、关系、目标、承诺、知识或资源使用中留下持续"
            "影响。不能只写情绪变化、语气变化、更加谨慎或保持不变。"
            "controlled Anchor必须是原创可控事件，不得伪装成正典。"
            "cf_edit只改变一个关键原因，并应导致不同状态或决定。"
            "整组必须覆盖knowledge、commitment/goal、relationship和resource。"
            "至少8个Anchor必须撤销、履行、失败、替代、纠正或覆盖一个先前状态，"
            "不能全部只是新增互不相关的信息。"
            "从全部Anchor中选择恰好7个Sensitivity和3个Invariance Probe，"
            "其余标为none。"
            "只输出JSON对象，不输出解释、推理过程或Markdown。"
        ),
        user_prompt=(
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "输出字段：canon_anchors（正典事件ID数组）；generated_anchors"
            "（id、event、state_change、cf_edit）；probe_plan（anchor_id、type，"
            "type只能是sensitivity、invariance或none）。probe_plan必须按"
            "canon_anchors后接generated_anchors的顺序覆盖全部Anchor。"
        ),
    )


def _validate_anchor_outline(
    output: dict[str, Any],
    eligible: list[dict[str, Any]],
    total_anchors: int,
    canon_anchors: int,
) -> None:
    if set(output) != {
        "canon_anchors",
        "generated_anchors",
        "probe_plan",
    }:
        raise ValueError("Anchor大纲字段不完整")
    selected = output["canon_anchors"]
    generated = output["generated_anchors"]
    probe_plan = output["probe_plan"]
    if not isinstance(selected, list) or len(selected) != canon_anchors:
        raise ValueError(f"canon_anchors必须恰好包含{canon_anchors}项")
    eligible_ids = {event["id"] for event in eligible}
    if len(set(selected)) != len(selected) or not set(selected) <= eligible_ids:
        raise ValueError("canon_anchors包含重复或不可用事件")
    controlled_count = total_anchors - canon_anchors
    if not isinstance(generated, list) or len(generated) != controlled_count:
        raise ValueError(
            f"generated_anchors必须恰好包含{controlled_count}项"
        )
    generated_ids: list[str] = []
    for item in generated:
        if not isinstance(item, Mapping) or set(item) != {
            "id",
            "event",
            "state_change",
            "cf_edit",
        }:
            raise ValueError("generated_anchor字段不完整")
        if not all(
            isinstance(item[key], str) and item[key].strip()
            for key in ("id", "event", "state_change", "cf_edit")
        ):
            raise ValueError("generated_anchor字段必须是非空字符串")
        generated_ids.append(str(item["id"]))
    if (
        len(generated_ids) != len(set(generated_ids))
        or set(generated_ids) & eligible_ids
    ):
        raise ValueError("generated_anchor ID重复或与正典事件冲突")
    expected_ids = list(selected) + generated_ids
    if not isinstance(probe_plan, list) or [
        item.get("anchor_id")
        for item in probe_plan
        if isinstance(item, Mapping)
    ] != expected_ids:
        raise ValueError("probe_plan必须按顺序覆盖全部Anchor")
    probe_types = [
        item.get("type")
        for item in probe_plan
        if isinstance(item, Mapping)
    ]
    if not set(probe_types) <= {"sensitivity", "invariance", "none"}:
        raise ValueError("probe_plan包含未知type")
    if (
        probe_types.count("sensitivity") != 7
        or probe_types.count("invariance") != 3
    ):
        raise ValueError("probe_plan必须包含7个Sensitivity和3个Invariance")


def _transition_batch_prompt(
    card: Mapping[str, Any],
    environment: Mapping[str, Any],
    context: GameContext,
    anchors: list[Mapping[str, Any]],
    probe_plan: Mapping[str, str],
) -> PromptBundle:
    payload = {
        "character": card,
        "world": {
            "world_id": environment.get("world_id"),
            "name": environment.get("name"),
            "language": environment.get("language"),
            "locations": environment.get("locations", []),
            "items": environment.get("items", []),
        },
        "current_context": context.to_dict(),
        "anchors_in_order": anchors,
        "probe_type_by_anchor": probe_plan,
        "allowed_operations": _operation_contract(),
    }
    return PromptBundle(
        system_prompt=(
            "你是跨世界角色评测的Transition草案生成器。按给定顺序把每个"
            "Anchor落实为FixtureEngine可执行的状态变化。operations必须与"
            "state_change含义一致并基于current_context顺序执行。npc_surface"
            "必须描述Anchor后可观察的决定变化，不能只写情绪或语气。"
            "Sensitivity必须同时给出counterfactual_operations和decision_probe；"
            "其最终问题必须依赖被改变的目标状态，不能脱离历史仅凭常识或角色"
            "画像作答，两个分支必须要求不同的可观察决定；"
            "Invariance必须给出invariance_probe；none不得添加Probe字段。"
            "只输出JSON，不输出解释、推理过程或Markdown。"
        ),
        user_prompt=(
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "输出：transitions数组。基础字段为anchor_id、player_visible、"
            "npc_surface、operations。decision_probe包含query_variants及"
            "branch_a/branch_b，每个branch包含admissible_decisions和"
            "forbidden_decisions。invariance_probe包含paraphrase、query和"
            "expected_state。"
        ),
    )


def _validate_transition_batch(
    output: dict[str, Any],
    context: GameContext,
    definition: WorldDefinition,
    expected_ids: list[str],
    probe_by_id: Mapping[str, str],
) -> None:
    if set(output) != {"transitions"} or not isinstance(
        output["transitions"],
        list,
    ):
        raise ValueError("Transition批次只能包含transitions数组")
    transitions = output["transitions"]
    if [
        item.get("anchor_id")
        for item in transitions
        if isinstance(item, Mapping)
    ] != expected_ids:
        raise ValueError("Transition批次未按顺序覆盖指定Anchor")
    fixture = FixtureEngine(definition)
    working = context
    for transition in transitions:
        required = {
            "anchor_id",
            "player_visible",
            "npc_surface",
            "operations",
        }
        if not required <= set(transition):
            raise ValueError("Transition缺少基础字段")
        probe_type = probe_by_id[transition["anchor_id"]]
        optional = set(transition) - required
        expected_optional = {
            "sensitivity": {
                "counterfactual_operations",
                "decision_probe",
            },
            "invariance": {"invariance_probe"},
            "none": set(),
        }[probe_type]
        if optional != expected_optional:
            raise ValueError(
                f"{transition['anchor_id']}的Probe字段与probe_plan不一致"
            )
        if not isinstance(transition["player_visible"], bool):
            raise ValueError("player_visible必须是布尔值")
        if not isinstance(transition["npc_surface"], str) or not transition[
            "npc_surface"
        ].strip():
            raise ValueError("npc_surface必须是非空字符串")
        operations = transition["operations"]
        if not isinstance(operations, list) or not operations:
            raise ValueError("operations必须是非空数组")
        pre_context = working
        result = fixture.apply(
            working,
            event_id=f"draft_validate_{transition['anchor_id']}",
            operations=operations,
            description=transition["npc_surface"],
        )
        working = result.context
        if probe_type == "sensitivity":
            counterfactual = fixture.apply(
                pre_context,
                event_id=f"draft_validate_cf_{transition['anchor_id']}",
                operations=transition["counterfactual_operations"],
                description="counterfactual",
            )
            if (
                counterfactual.context.runtime_state
                == result.context.runtime_state
                and counterfactual.context.environment
                == result.context.environment
            ):
                raise ValueError("Sensitivity反事实没有产生状态差异")


def _operation_contract() -> dict[str, list[str]]:
    return {
        "add_knowledge": ["fact_id", "content", "visibility"],
        "supersede_knowledge": ["fact_id"],
        "set_claim": ["claim_id", "content", "decision"],
        "set_relationship": ["target", "level"],
        "create_commitment": ["commitment_id", "target", "content"],
        "resolve_commitment": ["commitment_id", "status"],
        "set_goal": ["goal_id", "content", "status"],
        "set_inventory": ["holder", "item", "present"],
        "set_location": ["entity", "location"],
        "set_access": [
            "resource",
            "subject",
            "decision",
            "controllers",
        ],
        "set_dialogue_status": ["status"],
    }


def _validate_anchor_output(
    output: dict[str, Any],
    character_id: str,
    environment: Mapping[str, Any],
    initial_context: Mapping[str, Any],
    eligible: list[dict[str, Any]],
    total_anchors: int,
    canon_anchors: int,
) -> None:
    if set(output) != {
        "canon_anchors",
        "generated_anchors",
        "transitions",
    }:
        raise ValueError("Anchor草案字段不完整")
    selected = output["canon_anchors"]
    generated = output["generated_anchors"]
    transitions = output["transitions"]
    if not isinstance(selected, list) or len(selected) != canon_anchors:
        raise ValueError(f"canon_anchors必须恰好包含{canon_anchors}项")
    eligible_ids = {event["id"] for event in eligible}
    if (
        len(set(selected)) != len(selected)
        or not set(selected) <= eligible_ids
    ):
        raise ValueError("canon_anchors包含重复或不可用事件")
    controlled_count = total_anchors - canon_anchors
    if not isinstance(generated, list) or len(generated) != controlled_count:
        raise ValueError(
            f"generated_anchors必须恰好包含{controlled_count}项"
        )
    generated_ids: list[str] = []
    for item in generated:
        if not isinstance(item, Mapping) or set(item) != {
            "id",
            "event",
            "state_change",
            "cf_edit",
        }:
            raise ValueError("generated_anchor字段不完整")
        if not all(
            isinstance(item[key], str) and item[key].strip()
            for key in ("id", "event", "state_change", "cf_edit")
        ):
            raise ValueError("generated_anchor字段必须是非空字符串")
        generated_ids.append(str(item["id"]))
    if len(generated_ids) != len(set(generated_ids)):
        raise ValueError("generated_anchor ID重复")
    if set(generated_ids) & eligible_ids:
        raise ValueError("generated_anchor ID与正典事件冲突")
    expected_order = list(selected) + generated_ids
    if not isinstance(transitions, list) or [
        item.get("anchor_id") for item in transitions
    ] != expected_order:
        raise ValueError("transitions未按Anchor顺序逐一覆盖")

    definition = WorldDefinition.from_mapping(environment)
    context = GameContext(
        {
            "character_card": {"character_id": character_id},
            **copy.deepcopy(dict(initial_context)),
            "history": [],
        }
    )
    fixture = FixtureEngine(definition)
    dimensions: set[str] = set()
    sensitivity_count = 0
    invariance_count = 0
    for index, transition in enumerate(transitions):
        required_fields = {
            "anchor_id",
            "player_visible",
            "npc_surface",
            "operations",
        }
        allowed_fields = required_fields | {
            "counterfactual_operations",
            "decision_probe",
            "invariance_probe",
        }
        if (
            not isinstance(transition, Mapping)
            or not required_fields <= set(transition)
            or set(transition) - allowed_fields
        ):
            raise ValueError("transition字段不完整")
        has_decision_probe = "decision_probe" in transition
        has_counterfactual = "counterfactual_operations" in transition
        has_invariance = "invariance_probe" in transition
        if has_decision_probe != has_counterfactual:
            raise ValueError(
                "Sensitivity Transition必须同时包含"
                "counterfactual_operations和decision_probe"
            )
        if has_decision_probe and has_invariance:
            raise ValueError("同一Transition不能同时用于两类Pair")
        sensitivity_count += int(has_decision_probe)
        invariance_count += int(has_invariance)
        if not isinstance(transition["player_visible"], bool):
            raise ValueError("player_visible必须是布尔值")
        if not isinstance(transition["npc_surface"], str) or not transition[
            "npc_surface"
        ].strip():
            raise ValueError("npc_surface必须是非空字符串")
        operations = transition["operations"]
        if not isinstance(operations, list) or not operations:
            raise ValueError("每个transition必须包含状态操作")
        for operation in operations:
            if not isinstance(operation, Mapping):
                raise ValueError("operation必须是对象")
            dimension = _operation_dimension(str(operation.get("op", "")))
            if dimension:
                dimensions.add(dimension)
        result = fixture.apply(
            context,
            event_id=f"draft_anchor_{index + 1:02d}",
            operations=operations,
            description=transition["npc_surface"],
            history_turn=index + 1,
        )
        if not result.context_delta:
            raise ValueError("Anchor没有产生实际状态变化")
        context = result.context
    required = {"knowledge", "commitment_goal", "relationship", "resource"}
    if not required <= dimensions:
        raise ValueError(f"Anchor状态维度不足: {sorted(required - dimensions)}")
    if sensitivity_count != 7 or invariance_count != 3:
        raise ValueError(
            "Stage2 Probe配比必须是7个Sensitivity和3个Invariance"
        )


def _operation_dimension(operation: str) -> str | None:
    if operation in {"add_knowledge", "supersede_knowledge", "set_claim"}:
        return "knowledge"
    if operation in {
        "create_commitment",
        "resolve_commitment",
        "set_goal",
    }:
        return "commitment_goal"
    if operation == "set_relationship":
        return "relationship"
    if operation in {
        "set_inventory",
        "set_location",
        "set_access",
        "set_dialogue_status",
    }:
        return "resource"
    return None


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
