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


ANCHOR_BATCH_SIZE = 10
ANCHOR_PROBE_QUOTAS = ((3, 1), (2, 1), (2, 1))


def generate_anchor_draft(
    world_dir: str | Path,
    character_id: str,
    *,
    client: LLMClient,
    config: GenerationConfig,
    total_anchors: int = 30,
    canon_anchors: int = 0,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate drafts only; never overwrite reviewed benchmark assets."""

    world = Path(world_dir)
    foundation = audit_foundation(world, [character_id])
    if not foundation["ready_for_anchor_generation"]:
        raise ValueError("世界与角色画像尚未通过机器审查和人工门禁")
    if total_anchors != 30 or canon_anchors != 0:
        raise ValueError("当前Stage 1/2协议固定为30个快照后受控Anchor")
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
    canon = load_yaml(world / "canon.yaml")
    environment = load_yaml(world / "environment.yaml")
    pool = load_yaml(world / "anchor_pool.yaml")
    historical_context = [
        event
        for event in pool["events"]
        if _event_relevant_to_character(event, character_id)
    ]
    if historical_context and all(
        event.get("chapter_order") for event in historical_context
    ):
        historical_context.sort(
            key=lambda event: tuple(
                int(part)
                for part in str(event["chapter_order"]).split(".")
            )
        )
    initial_context = _load_initial_context(world, character_id)
    foundation_references = _foundation_references(
        canon,
        card,
        environment,
        historical_context,
    )
    generated_anchors: list[dict[str, Any]] = []
    probe_plan: list[dict[str, Any]] = []
    for batch_index, (sensitivity_count, invariance_count) in enumerate(
        ANCHOR_PROBE_QUOTAS,
        start=1,
    ):
        batch_start = (batch_index - 1) * ANCHOR_BATCH_SIZE
        bundle = _anchor_outline_prompt(
            canon,
            card,
            environment,
            initial_context,
            historical_context,
            foundation_references,
            previous_anchors=generated_anchors,
            batch_index=batch_index,
            batch_start=batch_start,
            batch_size=ANCHOR_BATCH_SIZE,
            sensitivity_count=sensitivity_count,
            invariance_count=invariance_count,
        )
        batch_outline = generate_structured(
            client,
            bundle,
            config,
            lambda value, sensitivity_count=sensitivity_count,
            invariance_count=invariance_count: _validate_anchor_outline(
                value,
                [],
                ANCHOR_BATCH_SIZE,
                0,
                allowed_foundation_refs=set(foundation_references),
                sensitivity_count=sensitivity_count,
                invariance_count=invariance_count,
            ),
        )
        generated_anchors.extend(batch_outline["generated_anchors"])
        probe_plan.extend(batch_outline["probe_plan"])
    outline = {
        "canon_anchors": [],
        "generated_anchors": generated_anchors,
        "probe_plan": probe_plan,
    }
    _validate_anchor_outline(
        {
            "generated_anchors": generated_anchors,
            "probe_plan": probe_plan,
        },
        [],
        total_anchors,
        canon_anchors,
        allowed_foundation_refs=set(foundation_references),
    )
    generated_by_id = {
        anchor["id"]: anchor for anchor in outline["generated_anchors"]
    }
    ordered_ids = list(generated_by_id)
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
            {
                "id": anchor_id,
                "source": "controlled",
                **generated_by_id[anchor_id],
            }
            for anchor_id in batch_ids
        ]
        transition_bundle = _transition_batch_prompt(
            canon,
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
                character_id,
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
        [],
        total_anchors,
        canon_anchors,
        allowed_foundation_refs=set(foundation_references),
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
        "schema_version": 2,
        "character_id": character_id,
        "draft_only": True,
        "protocol": "post_snapshot_controlled_v1",
        "anchors": total_anchors,
        "canon_anchors": canon_anchors,
        "controlled_anchors": total_anchors - canon_anchors,
        "anchor_api_calls": len(ANCHOR_PROBE_QUOTAS),
        "transition_api_calls": (total_anchors + 4) // 5,
        "logical_api_calls": (
            len(ANCHOR_PROBE_QUOTAS) + (total_anchors + 4) // 5
        ),
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
    canon: Mapping[str, Any],
    card: Mapping[str, Any],
    environment: Mapping[str, Any],
    initial_context: Mapping[str, Any],
    historical_context: list[dict[str, Any]],
    foundation_references: Mapping[str, Any],
    *,
    previous_anchors: list[dict[str, Any]],
    batch_index: int,
    batch_start: int,
    batch_size: int,
    sensitivity_count: int,
    invariance_count: int,
) -> PromptBundle:
    payload = {
        "canon_policy": canon.get("canon_policy", {}),
        "character": card,
        "world": {
            "world_id": environment.get("world_id"),
            "name": environment.get("name"),
            "language": environment.get("language"),
            "locations": environment.get("locations", []),
            "items": environment.get("items", []),
        },
        "initial_context": initial_context,
        "historical_context_before_snapshot": historical_context,
        "foundation_reference_catalog": foundation_references,
        "previous_controlled_anchors": previous_anchors,
        "batch": {
            "index": batch_index,
            "count": len(ANCHOR_PROBE_QUOTAS),
            "ordinal_start": batch_start + 1,
            "ordinal_end": batch_start + batch_size,
        },
        "required_counts": {
            "total": batch_size,
            "canon": 0,
            "controlled": batch_size,
            "sensitivity": sensitivity_count,
            "invariance": invariance_count,
            "none": batch_size - sensitivity_count - invariance_count,
        },
    }
    return PromptBundle(
        system_prompt=(
            "你是跨世界角色评测的Anchor大纲生成器。只根据给定来源、世界和"
            "角色画像生成结构化草案。每个Anchor必须能够产生可执行状态"
            "变化，并在后续选择、关系、目标、承诺、知识或资源使用中留下持续"
            "影响。不能只写情绪变化、语气变化、更加谨慎或保持不变。"
            "全部Anchor都是共享快照之后发生的原创可控Player-NPC互动，不得"
            "重演historical_context中的正典事件，不得泄漏排除版本或快照后的"
            "正典剧情。NPC之间没有共享模拟：事件只能让Player与当前NPC行动，"
            "其他NPC只能作为既有背景或证据来源，不能在事件中行动或更新状态。"
            "每个Anchor必须从initial_context向后演进，并给出至少两个有效"
            "foundation_refs，其中必须包含canon:shared_snapshot和至少一个"
            "profile引用。Sensitivity的cf_edit只改变一个关键原因并导致不同"
            "状态或决定；Invariance的cf_edit必须是语义等价改写。"
            "本批不得重复previous_controlled_anchors中的事件、ID或状态变化。"
            "跨批次整组必须覆盖knowledge、commitment/goal、relationship和"
            "resource，并至少8次撤销、履行、失败、替代、纠正或覆盖先前状态；"
            "后续批次应优先闭合或覆盖前批建立的状态。严格遵守required_counts"
            "中的本批Probe配额，其余标为none。"
            "只输出JSON对象，不输出解释、推理过程或Markdown。"
        ),
        user_prompt=(
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "只输出两个字段：generated_anchors（id、event、state_change、"
            "cf_edit、foundation_refs）和probe_plan（anchor_id、type，type"
            "只能是sensitivity、invariance或none）。probe_plan必须按"
            "generated_anchors顺序覆盖全部Anchor。"
        ),
        response_schema=_anchor_outline_schema(
            batch_size,
            list(foundation_references),
        ),
    )


def _anchor_outline_schema(
    total_anchors: int,
    foundation_refs: list[str],
) -> dict[str, Any]:
    anchor_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "event",
            "state_change",
            "cf_edit",
            "foundation_refs",
        ],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "event": {"type": "string", "minLength": 1},
            "state_change": {"type": "string", "minLength": 1},
            "cf_edit": {"type": "string", "minLength": 1},
            "foundation_refs": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "string", "enum": foundation_refs},
            },
        },
    }
    probe_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["anchor_id", "type"],
        "properties": {
            "anchor_id": {"type": "string", "minLength": 1},
            "type": {
                "type": "string",
                "enum": ["sensitivity", "invariance", "none"],
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["generated_anchors", "probe_plan"],
        "properties": {
            "generated_anchors": {
                "type": "array",
                "minItems": total_anchors,
                "maxItems": total_anchors,
                "items": anchor_schema,
            },
            "probe_plan": {
                "type": "array",
                "minItems": total_anchors,
                "maxItems": total_anchors,
                "items": probe_schema,
            },
        },
    }


def _validate_anchor_outline(
    output: dict[str, Any],
    eligible: list[dict[str, Any]],
    total_anchors: int,
    canon_anchors: int,
    *,
    allowed_foundation_refs: set[str] | None = None,
    sensitivity_count: int = 7,
    invariance_count: int = 3,
) -> None:
    expected_output_fields = {"generated_anchors", "probe_plan"}
    if canon_anchors:
        expected_output_fields.add("canon_anchors")
    if set(output) != expected_output_fields:
        raise ValueError("Anchor大纲字段不完整")
    selected = output.get("canon_anchors", [])
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
    expected_fields = {
            "id",
            "event",
            "state_change",
            "cf_edit",
    }
    if canon_anchors == 0:
        expected_fields.add("foundation_refs")
    for item in generated:
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            raise ValueError("generated_anchor字段不完整")
        if not all(
            isinstance(item[key], str) and item[key].strip()
            for key in ("id", "event", "state_change", "cf_edit")
        ):
            raise ValueError("generated_anchor字段必须是非空字符串")
        if canon_anchors == 0:
            _validate_foundation_refs(
                item["foundation_refs"],
                allowed_foundation_refs or set(),
            )
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
        probe_types.count("sensitivity") != sensitivity_count
        or probe_types.count("invariance") != invariance_count
    ):
        raise ValueError(
            "probe_plan必须包含"
            f"{sensitivity_count}个Sensitivity和"
            f"{invariance_count}个Invariance"
        )


def _transition_batch_prompt(
    canon: Mapping[str, Any],
    card: Mapping[str, Any],
    environment: Mapping[str, Any],
    context: GameContext,
    anchors: list[Mapping[str, Any]],
    probe_plan: Mapping[str, str],
) -> PromptBundle:
    payload = {
        "canon_policy": canon.get("canon_policy", {}),
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
            "只能更新当前NPC或Player的状态；其他NPC不能行动，也不能成为"
            "relationship、commitment、inventory、location或access的更新对象。"
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
    character_id: str,
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
        for operation in operations:
            _validate_operation_scope(operation, character_id)
        pre_context = working
        result = fixture.apply(
            working,
            event_id=f"draft_validate_{transition['anchor_id']}",
            operations=operations,
            description=transition["npc_surface"],
        )
        working = result.context
        if probe_type == "sensitivity":
            for operation in transition["counterfactual_operations"]:
                _validate_operation_scope(operation, character_id)
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
    *,
    allowed_foundation_refs: set[str] | None = None,
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
    expected_fields = {
            "id",
            "event",
            "state_change",
            "cf_edit",
    }
    if canon_anchors == 0:
        expected_fields.add("foundation_refs")
    for item in generated:
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            raise ValueError("generated_anchor字段不完整")
        if not all(
            isinstance(item[key], str) and item[key].strip()
            for key in ("id", "event", "state_change", "cf_edit")
        ):
            raise ValueError("generated_anchor字段必须是非空字符串")
        if canon_anchors == 0:
            _validate_foundation_refs(
                item["foundation_refs"],
                allowed_foundation_refs or set(),
            )
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
            if canon_anchors == 0:
                _validate_operation_scope(operation, character_id)
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


def _event_relevant_to_character(
    event: Mapping[str, Any],
    character_id: str,
) -> bool:
    fields = (
        "anchor_for",
        "known_by",
        "believed_by",
        "suspected_by",
        "outcome_known_by",
        "later_known_by",
    )
    return any(character_id in (event.get(field) or []) for field in fields)


def _foundation_references(
    canon: Mapping[str, Any],
    card: Mapping[str, Any],
    environment: Mapping[str, Any],
    historical_context: list[dict[str, Any]],
) -> dict[str, Any]:
    policy = canon.get("canon_policy", {})
    references: dict[str, Any] = {
        "canon:shared_snapshot": (
            policy.get("shared_snapshot")
            if isinstance(policy, Mapping)
            else ""
        )
    }
    for boundary in card.get("testable_boundaries", []):
        if isinstance(boundary, Mapping) and boundary.get("id"):
            references[f"profile:boundary:{boundary['id']}"] = dict(boundary)
    worldview = card.get("worldview", {})
    if isinstance(worldview, Mapping):
        for field in ("principles", "stances", "core_goals", "prohibitions"):
            for index, value in enumerate(worldview.get(field, []), start=1):
                references[f"profile:{field}:{index}"] = value
    capability = card.get("capability", {})
    if isinstance(capability, Mapping):
        for field in ("authority", "limitations"):
            for index, value in enumerate(capability.get(field, []), start=1):
                references[f"profile:{field}:{index}"] = value
    social = card.get("social", {})
    if isinstance(social, Mapping):
        for index, value in enumerate(
            social.get("relationship_rules", []),
            start=1,
        ):
            references[f"profile:relationship_rules:{index}"] = value
    for location in environment.get("locations", []):
        if isinstance(location, Mapping) and location.get("id"):
            references[f"environment:location:{location['id']}"] = dict(location)
    for item in environment.get("items", []):
        if isinstance(item, Mapping) and item.get("id"):
            references[f"environment:item:{item['id']}"] = dict(item)
    for event in historical_context:
        if event.get("id"):
            references[f"history:{event['id']}"] = event.get("fact", "")
    return references


def _validate_foundation_refs(
    value: Any,
    allowed: set[str],
) -> None:
    if (
        not isinstance(value, list)
        or len(value) < 2
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("foundation_refs必须包含至少两个不重复的引用")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"foundation_refs包含未知引用: {sorted(unknown)}")
    if "canon:shared_snapshot" not in value:
        raise ValueError("foundation_refs必须包含canon:shared_snapshot")
    if not any(item.startswith("profile:") for item in value):
        raise ValueError("foundation_refs必须至少包含一个profile引用")


def _validate_operation_scope(
    operation: Any,
    character_id: str,
) -> None:
    if not isinstance(operation, Mapping):
        raise ValueError("operation必须是对象")
    operation_name = str(operation.get("op", ""))
    scoped_fields = {
        "set_relationship": "target",
        "create_commitment": "target",
        "set_inventory": "holder",
        "set_location": "entity",
        "set_access": "subject",
    }
    field = scoped_fields.get(operation_name)
    if field is None:
        return
    if operation.get(field) not in {character_id, "player"}:
        raise ValueError(
            f"{operation_name}.{field}只能引用当前NPC或player"
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
