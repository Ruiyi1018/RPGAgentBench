"""Reusable machine checks and human review gates for source assets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from gamecore import ActionRegistry, FixtureEngine, GameContext, WorldDefinition

from ..shared.io import load_yaml
from ..shared.reviewed_sources import load_reviewed_anchors, load_transition_plan


FOUNDATION_MANUAL_RULES = {
    "source_scope_resolved": "正典来源、版本、时间快照和排除范围明确。",
    "snapshot_consistent": "角色画像只包含共享时间快照内已经成立的信息。",
    "snapshot_role_consistent": "角色职务、阵营、存活、位置和权限与时间快照及首个Anchor一致。",
    "future_knowledge_excluded": "未来经历、未来关系和角色尚未知晓的事件未倒灌进Profile。",
    "character_source_grounded": "身份、经历、立场、能力和关系均可由来源支持。",
    "public_private_separated": "逐项区分公开事实、NPC私密知识、传闻和未知信息。",
    "player_role_bounded": "Player的身份、能力、可接触材料和临时授权边界明确且前后一致。",
    "boundaries_behavioral": "角色边界描述为可观察行为，而非抽象人格形容词。",
    "relationships_directional": "关系按角色视角分别描述，不把单向信任自动写成双向关系。",
    "style_source_grounded": "语言风格来自角色稳定表达习惯，不使用剧情流程或固定办事模板代替风格。",
    "world_affordances_sufficient": "地点、物品和权限足以承载计划测试的行为。",
}

ANCHOR_MANUAL_RULES = {
    "chronology_total_order": "Anchor按正典顺序排列，且与Profile时间快照和初始状态形成无矛盾的总时间线。",
    "event_boundary_exact": "Anchor描述的事件、结论和状态改变在该Anchor才成立，不能提前或滞后到其他轮次。",
    "event_causal": "每个Anchor都能因果性地改变至少一项角色或世界状态。",
    "evidence_supports_state": "知识、主张和关系变化不超出当时已有证据，不把怀疑提前写成确认。",
    "state_change_observable": "改变能在后续选择、信息流、关系或任务中被观察。",
    "post_event_window_sufficient": "Anchor后保留足够观察窗口；最终Anchor不能与历史结束或对话结束占同一轮。",
    "counterfactual_minimal": "cf_edit只改变一个关键原因，并会导致不同状态或决定。",
    "visibility_correct": "分别审核事件存在、参与者、原因、结果和来源的可见性。",
    "hidden_event_not_disclosed": "隐藏Anchor不能被Player说出，也不能由NPC直接解释；只能表现允许公开的间接后果。",
    "memory_consequence_persistent": "Anchor影响会延续到后续互动，而非只出现一次。",
    "coverage_balanced": "整组Anchor覆盖知识、目标/承诺、关系和资源状态。",
    "turning_points_sufficient": "至少8个Anchor会撤销、履行、失败、替代、纠正或覆盖先前状态。",
    "state_operation_aligned": "state_change中的知识、关系、目标、承诺或资源变化均有对应Operation。",
    "probe_decision_relevant": "每个Stage 2 Probe的最终问题必须依赖目标状态，不能脱离历史仅凭常识或Profile作答。",
    "operation_preserves_later_state": "操作不会复活已完成目标、恢复已撤销权限或覆盖后续已形成的关系状态。",
    "commitment_lifecycle_closed": "需要结束的承诺和目标在后续Anchor中有明确状态收束。",
    "resource_transfer_conserved": "物品转移同时记录来源移除和目标获得，或明确说明消耗。",
    "dialogue_lifecycle_valid": "结束对话的操作只出现在不再需要后续对话或分支测试的位置。",
}

_CARD_LIST_FIELDS = {
    "identity": ("affiliations", "public_background"),
    "worldview": (
        "principles",
        "stances",
        "core_goals",
        "prohibitions",
    ),
    "capability": ("skills", "authority", "limitations"),
}

_OPERATION_DIMENSIONS = {
    "add_knowledge": "knowledge",
    "supersede_knowledge": "knowledge",
    "set_claim": "knowledge",
    "create_commitment": "commitment_goal",
    "resolve_commitment": "commitment_goal",
    "set_goal": "commitment_goal",
    "set_relationship": "relationship",
    "set_inventory": "resource",
    "set_location": "resource",
    "set_access": "resource",
    "set_dialogue_status": "resource",
}


def review_path(world_dir: str | Path, character_id: str) -> Path:
    return Path(world_dir) / "frozen" / character_id / "source_review.yaml"


def write_review_template(
    world_dir: str | Path,
    character_id: str,
) -> Path:
    path = review_path(world_dir, character_id)
    if path.exists():
        payload = load_yaml(path)
        changed = payload.get("schema_version") != 2
        payload["schema_version"] = 2
        for key in ("reviewer", "reviewed_at"):
            if key not in payload:
                payload[key] = ""
                changed = True
        for section_name, rules in (
            ("foundation", FOUNDATION_MANUAL_RULES),
            ("anchors", ANCHOR_MANUAL_RULES),
        ):
            section = payload.setdefault(
                section_name,
                {"approved": False, "checks": {}},
            )
            checks = section.setdefault("checks", {})
            missing = set(rules) - set(checks)
            if missing:
                checks.update({rule_id: False for rule_id in missing})
                section["approved"] = False
                changed = True
        if changed:
            path.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "character_id": character_id,
        "reviewer": "",
        "reviewed_at": "",
        "foundation": {
            "approved": False,
            "checks": {
                rule_id: False for rule_id in FOUNDATION_MANUAL_RULES
            },
        },
        "anchors": {
            "approved": False,
            "checks": {
                rule_id: False for rule_id in ANCHOR_MANUAL_RULES
            },
        },
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def audit_foundation(
    world_dir: str | Path,
    character_ids: list[str] | None = None,
) -> dict[str, Any]:
    world = Path(world_dir)
    errors: list[dict[str, str]] = []
    canon = _load_required(world / "canon.yaml", errors)
    environment = _load_required(world / "environment.yaml", errors)
    anchor_pool = _load_required(world / "anchor_pool.yaml", errors)
    if errors:
        return _foundation_report(world, [], errors, {})

    world_ids = {
        str(canon.get("world_id", "")),
        str(environment.get("world_id", "")),
        str(anchor_pool.get("world_id", "")),
    }
    if len(world_ids) != 1 or "" in world_ids:
        _error(
            errors,
            "world_id_mismatch",
            "<world>",
            "canon、environment和anchor_pool的world_id必须一致且非空",
        )
    evaluated = canon.get("evaluated_npcs")
    if not _nonempty_unique_strings(evaluated):
        _error(
            errors,
            "invalid_evaluated_npcs",
            "canon.yaml:evaluated_npcs",
            "evaluated_npcs必须是非空且无重复的字符串数组",
        )
        evaluated = []
    selected = character_ids or list(evaluated)
    unknown = set(selected) - set(evaluated)
    if unknown:
        _error(
            errors,
            "unknown_character",
            "canon.yaml:evaluated_npcs",
            f"未登记角色: {sorted(unknown)}",
        )

    policy = canon.get("canon_policy")
    if not isinstance(policy, Mapping) or not all(
        policy.get(key)
        for key in ("primary_source", "source_type", "quotation_policy")
    ):
        _error(
            errors,
            "incomplete_canon_policy",
            "canon.yaml:canon_policy",
            "必须声明primary_source、source_type和quotation_policy",
        )
    sources = canon.get("sources")
    if not isinstance(sources, list) or not any(
        isinstance(source, Mapping)
        and source.get("authority") == "primary"
        for source in sources
    ):
        _error(
            errors,
            "missing_primary_source",
            "canon.yaml:sources",
            "至少需要一个authority=primary的来源",
        )
    if not _nonempty_strings(canon.get("data_constraints")):
        _error(
            errors,
            "missing_data_constraints",
            "canon.yaml:data_constraints",
            "必须提供正典与知识边界约束",
        )

    _audit_environment(environment, errors)
    events = anchor_pool.get("events")
    if not isinstance(events, list) or not events:
        _error(
            errors,
            "missing_anchor_pool",
            "anchor_pool.yaml:events",
            "Anchor池必须包含事件",
        )
        events = []
    event_ids = [
        str(event.get("id", ""))
        for event in events
        if isinstance(event, Mapping)
    ]
    if len(event_ids) != len(set(event_ids)) or "" in event_ids:
        _error(
            errors,
            "duplicate_anchor_event",
            "anchor_pool.yaml:events",
            "Anchor事件ID必须非空且唯一",
        )
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            _error(
                errors,
                "invalid_anchor_event",
                f"anchor_pool.yaml:events.{index}",
                "事件必须是对象",
            )
            continue
        if not all(event.get(key) for key in ("fact", "state_change", "cf_edit")):
            _error(
                errors,
                "incomplete_anchor_event",
                f"anchor_pool.yaml:events.{index}",
                "事件必须包含fact、state_change和cf_edit",
            )
        anchor_for = event.get("anchor_for", [])
        if not isinstance(anchor_for, list) or set(anchor_for) - set(evaluated):
            _error(
                errors,
                "invalid_anchor_targets",
                f"anchor_pool.yaml:events.{index}.anchor_for",
                "anchor_for只能引用evaluated_npcs中的角色",
            )

    approvals: dict[str, bool] = {}
    for character_id in selected:
        _audit_character_card(world, character_id, errors)
        approvals[character_id] = _review_section_passed(
            world,
            character_id,
            "foundation",
            FOUNDATION_MANUAL_RULES,
        )
    return _foundation_report(world, selected, errors, approvals)


def audit_anchor_plan(
    world_dir: str | Path,
    character_id: str,
) -> dict[str, Any]:
    world = Path(world_dir)
    errors: list[dict[str, str]] = []
    operation_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    state_paths: set[str] = set()
    turning_point_count = 0
    try:
        anchors = load_reviewed_anchors(world, character_id)
        plan = load_transition_plan(world, character_id)
        definition = WorldDefinition.load_yaml(world / "environment.yaml")
        card = load_yaml(world / "characters" / f"{character_id}.yaml")
        context = GameContext(
            {
                "character_card": card,
                **plan["initial_context"],
                "history": [],
            }
        )
    except Exception as error:
        _error(
            errors,
            "anchor_source_load_failed",
            f"frozen/{character_id}",
            str(error),
        )
        return _anchor_report(
            world,
            character_id,
            errors,
            operation_counts,
            dimension_counts,
            0,
            0,
        )

    transitions = plan.get("transitions")
    if not isinstance(transitions, list):
        transitions = []
        _error(
            errors,
            "invalid_transitions",
            f"frozen/{character_id}/transitions.yaml",
            "transitions必须是数组",
        )
    anchor_ids = [str(anchor["id"]) for anchor in anchors]
    source_counts = Counter(str(anchor.get("source", "")) for anchor in anchors)
    if len(anchors) != 30 or source_counts != Counter(
        {"canon": 10, "controlled": 20}
    ):
        _error(
            errors,
            "invalid_anchor_source_distribution",
            f"frozen/{character_id}/anchors.yaml",
            "必须恰好包含30个Anchor（10个canon、20个controlled）",
        )
    transition_ids = [
        str(transition.get("anchor_id", ""))
        for transition in transitions
        if isinstance(transition, Mapping)
    ]
    if transition_ids != anchor_ids:
        _error(
            errors,
            "anchor_transition_mismatch",
            f"frozen/{character_id}/transitions.yaml",
            "transitions必须按anchors.yaml顺序逐一覆盖Anchor",
        )

    fixture = FixtureEngine(definition)
    registry = ActionRegistry.load_directory(
        Path(__file__).resolve().parents[2] / "gamecore" / "actions"
    )
    sensitivity_count = 0
    invariance_count = 0
    for index, transition in enumerate(transitions):
        path = f"frozen/{character_id}/transitions.yaml:transitions.{index}"
        if not isinstance(transition, Mapping):
            _error(errors, "invalid_transition", path, "Transition必须是对象")
            continue
        operations = transition.get("operations")
        if not isinstance(operations, list) or not operations:
            _error(
                errors,
                "empty_anchor_transition",
                path,
                "每个Anchor必须包含至少一个状态操作",
            )
            continue
        prior_state_paths = set(state_paths)
        transition_paths: set[str] = set()
        closes_prior_state = False
        surface = transition.get("npc_surface")
        if not isinstance(surface, str) or not surface.strip():
            _error(
                errors,
                "missing_behavioral_consequence",
                path,
                "npc_surface必须描述Anchor后的可观察行为变化",
            )
        for operation in operations:
            if isinstance(operation, Mapping):
                name = str(operation.get("op", ""))
                operation_counts[name] += 1
                dimension = _OPERATION_DIMENSIONS.get(name)
                if dimension:
                    dimension_counts[dimension] += 1
                state_path = _operation_state_path(operation)
                if state_path:
                    transition_paths.add(state_path)
                    state_paths.add(state_path)
                if name in {"supersede_knowledge", "resolve_commitment"}:
                    closes_prior_state = True
        if closes_prior_state or transition_paths & prior_state_paths:
            turning_point_count += 1
        has_sensitivity = "decision_probe" in transition
        has_counterfactual = "counterfactual_operations" in transition
        has_invariance = "invariance_probe" in transition
        sensitivity_count += int(has_sensitivity)
        invariance_count += int(has_invariance)
        if has_sensitivity != has_counterfactual:
            _error(
                errors,
                "incomplete_sensitivity_probe",
                path,
                "decision_probe与counterfactual_operations必须同时存在",
            )
        if has_sensitivity and has_invariance:
            _error(
                errors,
                "overlapping_stage2_probe",
                path,
                "同一Transition不能同时用于两类Pair",
            )
        if has_sensitivity:
            _audit_decision_probe(
                transition["decision_probe"],
                registry,
                errors,
                path,
            )
        if has_invariance:
            _audit_invariance_probe(
                transition["invariance_probe"],
                errors,
                path,
            )
        pre_context = context
        main_result = None
        try:
            result = fixture.apply(
                context,
                event_id=f"anchor_review_{transition['anchor_id']}",
                operations=operations,
                description=str(surface or transition["anchor_id"]),
                history_turn=index + 1,
            )
            main_result = result
            if not result.context_delta:
                _error(
                    errors,
                    "anchor_without_state_delta",
                    path,
                    "Anchor操作没有产生实际状态变化",
                )
            context = result.context
        except Exception as error:
            _error(
                errors,
                "transition_not_executable",
                path,
                str(error),
            )
        if has_counterfactual:
            try:
                counterfactual_result = fixture.apply(
                    pre_context,
                    event_id=(
                        f"anchor_review_cf_{transition['anchor_id']}"
                    ),
                    operations=transition["counterfactual_operations"],
                    description="counterfactual",
                    history_turn=index + 1,
                )
                if (
                    main_result is not None
                    and counterfactual_result.context.runtime_state
                    == main_result.context.runtime_state
                    and counterfactual_result.context.environment
                    == main_result.context.environment
                ):
                    _error(
                        errors,
                        "counterfactual_without_state_diff",
                        path,
                        "反事实操作没有产生不同的角色或世界状态",
                    )
            except Exception as error:
                _error(
                    errors,
                    "counterfactual_not_executable",
                    path,
                    str(error),
                )

    required_dimensions = {
        "knowledge",
        "commitment_goal",
        "relationship",
        "resource",
    }
    missing_dimensions = required_dimensions - set(dimension_counts)
    if missing_dimensions:
        _error(
            errors,
            "insufficient_state_coverage",
            f"frozen/{character_id}/transitions.yaml",
            f"缺少状态维度: {sorted(missing_dimensions)}",
        )
    if len(state_paths) < 12:
        _error(
            errors,
            "insufficient_qa_state_paths",
            f"frozen/{character_id}/transitions.yaml",
            f"Stage1至少需要12个不同状态路径，当前为{len(state_paths)}",
        )
    if turning_point_count < 8:
        _error(
            errors,
            "insufficient_turning_points",
            f"frozen/{character_id}/transitions.yaml",
            f"Stage1至少需要8个覆盖或收束先前状态的转折，当前为{turning_point_count}",
        )
    if sensitivity_count != 7 or invariance_count != 3:
        _error(
            errors,
            "invalid_stage2_probe_distribution",
            f"frozen/{character_id}/transitions.yaml",
            "必须包含7个Sensitivity和3个Invariance Probe",
        )
    return _anchor_report(
        world,
        character_id,
        errors,
        operation_counts,
        dimension_counts,
        len(anchors),
        turning_point_count,
    )


def require_generation_ready(
    world_dir: str | Path,
    character_id: str,
) -> None:
    foundation = audit_foundation(world_dir, [character_id])
    anchors = audit_anchor_plan(world_dir, character_id)
    failures = []
    if not foundation["ready_for_anchor_generation"]:
        failures.append("foundation")
    if not anchors["ready_for_history_generation"]:
        failures.append("anchors")
    if failures:
        raise ValueError(
            f"{character_id}未通过生成门禁: {', '.join(failures)}"
        )


def _audit_environment(
    environment: Mapping[str, Any],
    errors: list[dict[str, str]],
) -> None:
    if environment.get("language") not in {"zh", "en"}:
        _error(
            errors,
            "unsupported_world_language",
            "environment.yaml:language",
            "language必须是zh或en",
        )
    locations = environment.get("locations")
    items = environment.get("items")
    if not isinstance(locations, list) or not locations:
        _error(
            errors,
            "missing_locations",
            "environment.yaml:locations",
            "至少需要一个地点",
        )
        locations = []
    if not isinstance(items, list):
        _error(
            errors,
            "invalid_items",
            "environment.yaml:items",
            "items必须是数组",
        )
        items = []
    location_ids = [
        str(item.get("id", ""))
        for item in locations
        if isinstance(item, Mapping)
    ]
    item_ids = [
        str(item.get("id", ""))
        for item in items
        if isinstance(item, Mapping)
    ]
    for name, values in (("locations", location_ids), ("items", item_ids)):
        if "" in values or len(values) != len(set(values)):
            _error(
                errors,
                f"invalid_{name}_ids",
                f"environment.yaml:{name}",
                "ID必须非空且唯一",
            )
    valid_locations = set(location_ids)
    for index, location in enumerate(locations):
        if not isinstance(location, Mapping):
            continue
        unknown = set(location.get("connected_to", [])) - valid_locations
        if unknown:
            _error(
                errors,
                "unknown_location_connection",
                f"environment.yaml:locations.{index}.connected_to",
                f"引用未知地点: {sorted(unknown)}",
            )


def _audit_character_card(
    world: Path,
    character_id: str,
    errors: list[dict[str, str]],
) -> None:
    path = world / "characters" / f"{character_id}.yaml"
    card = _load_required(path, errors)
    if not card:
        return
    if card.get("character_id") != character_id:
        _error(
            errors,
            "character_id_mismatch",
            str(path),
            "角色文件中的character_id与文件名不一致",
        )
    identity = card.get("identity")
    if not isinstance(identity, Mapping) or not all(
        identity.get(key) for key in ("name", "role")
    ):
        _error(
            errors,
            "incomplete_identity",
            str(path),
            "identity必须包含name和role",
        )
    for section, fields in _CARD_LIST_FIELDS.items():
        value = card.get(section)
        if not isinstance(value, Mapping):
            _error(
                errors,
                "missing_character_section",
                f"{path}:{section}",
                f"缺少{section}",
            )
            continue
        for field in fields:
            if not _nonempty_strings(value.get(field)):
                _error(
                    errors,
                    "empty_character_dimension",
                    f"{path}:{section}.{field}",
                    "字段必须是非空字符串数组",
                )
    boundaries = card.get("testable_boundaries")
    if not isinstance(boundaries, list) or len(boundaries) < 3:
        _error(
            errors,
            "insufficient_testable_boundaries",
            f"{path}:testable_boundaries",
            "至少需要三条可测试角色边界",
        )
    else:
        ids: list[str] = []
        for index, boundary in enumerate(boundaries):
            if not isinstance(boundary, Mapping) or not all(
                boundary.get(key)
                for key in (
                    "id",
                    "position",
                    "challenge_space",
                    "forbidden_outcomes",
                    "allowed_change",
                )
            ):
                _error(
                    errors,
                    "incomplete_testable_boundary",
                    f"{path}:testable_boundaries.{index}",
                    "边界字段不完整",
                )
                continue
            ids.append(str(boundary["id"]))
        if len(ids) != len(set(ids)):
            _error(
                errors,
                "duplicate_boundary_id",
                f"{path}:testable_boundaries",
                "边界ID必须唯一",
            )


def _foundation_report(
    world: Path,
    characters: list[str],
    errors: list[dict[str, str]],
    approvals: Mapping[str, bool],
) -> dict[str, Any]:
    machine_passed = not errors
    human_approved = bool(characters) and all(
        approvals.get(character_id, False)
        for character_id in characters
    )
    return {
        "world": str(world),
        "characters": characters,
        "machine_passed": machine_passed,
        "human_approved": human_approved,
        "ready_for_anchor_generation": machine_passed and human_approved,
        "manual_rules": FOUNDATION_MANUAL_RULES,
        "errors": errors,
    }


def _anchor_report(
    world: Path,
    character_id: str,
    errors: list[dict[str, str]],
    operation_counts: Counter[str],
    dimension_counts: Counter[str],
    anchor_count: int,
    turning_point_count: int,
) -> dict[str, Any]:
    machine_passed = not errors
    human_approved = _review_section_passed(
        world,
        character_id,
        "anchors",
        ANCHOR_MANUAL_RULES,
    )
    return {
        "world": str(world),
        "character_id": character_id,
        "machine_passed": machine_passed,
        "human_approved": human_approved,
        "ready_for_history_generation": machine_passed and human_approved,
        "manual_rules": ANCHOR_MANUAL_RULES,
        "metrics": {
            "anchors": anchor_count,
            "operation_counts": dict(operation_counts),
            "state_dimension_counts": dict(dimension_counts),
            "turning_points": turning_point_count,
        },
        "errors": errors,
    }


def _review_section_passed(
    world: Path,
    character_id: str,
    section: str,
    rules: Mapping[str, str],
) -> bool:
    path = review_path(world, character_id)
    if not path.is_file():
        return False
    if any(
        _has_blocking_review_findings(candidate)
        for candidate in (
            path.parent / "review_findings.yaml",
            world / "drafts" / character_id / "review_findings.yaml",
        )
    ):
        return False
    review = load_yaml(path)
    if review.get("character_id") != character_id:
        return False
    value = review.get(section)
    if not isinstance(value, Mapping) or value.get("approved") is not True:
        return False
    checks = value.get("checks")
    return isinstance(checks, Mapping) and all(
        checks.get(rule_id) is True for rule_id in rules
    )


def _has_blocking_review_findings(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_yaml(path)
    except Exception:
        return True
    if isinstance(payload, Mapping):
        findings = payload.get("findings", [payload])
    else:
        findings = payload
    if not isinstance(findings, list):
        return True
    return any(
        isinstance(item, Mapping)
        and item.get("status") == "open"
        and item.get("severity") in {"blocker", "major"}
        for item in findings
    )


def _load_required(
    path: Path,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    if not path.is_file():
        _error(errors, "missing_file", str(path), "必需文件不存在")
        return {}
    try:
        value = load_yaml(path)
    except Exception as error:
        _error(errors, "invalid_yaml", str(path), str(error))
        return {}
    return value


def _nonempty_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _operation_state_path(operation: Mapping[str, Any]) -> str | None:
    name = operation.get("op")
    if name in {"add_knowledge", "supersede_knowledge"}:
        return f"knowledge.{operation.get('fact_id')}"
    if name == "set_claim":
        return f"claims.{operation.get('claim_id')}"
    if name in {"create_commitment", "resolve_commitment"}:
        return f"commitments.{operation.get('commitment_id')}"
    if name == "set_goal":
        return f"goals.{operation.get('goal_id')}"
    if name == "set_relationship":
        return f"relationships.{operation.get('target')}"
    if name == "set_inventory":
        return (
            f"inventories.{operation.get('holder')}."
            f"{operation.get('item')}"
        )
    if name == "set_location":
        return f"locations.{operation.get('entity')}"
    if name == "set_access":
        return (
            f"access.{operation.get('resource')}."
            f"{operation.get('subject')}"
        )
    if name == "set_dialogue_status":
        return "dialogue_status"
    return None


def _audit_decision_probe(
    probe: Any,
    registry: ActionRegistry,
    errors: list[dict[str, str]],
    path: str,
) -> None:
    if not isinstance(probe, Mapping):
        _error(errors, "invalid_decision_probe", path, "decision_probe必须是对象")
        return
    variants = probe.get("query_variants")
    if not isinstance(variants, list) or len(variants) < 2 or not all(
        isinstance(item, str) and item.strip() for item in variants
    ):
        _error(
            errors,
            "invalid_probe_queries",
            path,
            "Sensitivity Probe至少需要两个非空query_variants",
        )
    branches: dict[str, Mapping[str, Any]] = {}
    for branch_name in ("branch_a", "branch_b"):
        branch = probe.get(branch_name)
        if not isinstance(branch, Mapping):
            _error(
                errors,
                "invalid_probe_branch",
                path,
                f"{branch_name}必须是对象",
            )
            continue
        branches[branch_name] = branch
        for decision_kind in (
            "admissible_decisions",
            "forbidden_decisions",
        ):
            decisions = branch.get(decision_kind)
            if not isinstance(decisions, list) or not decisions:
                _error(
                    errors,
                    "empty_probe_decisions",
                    path,
                    f"{branch_name}.{decision_kind}必须是非空数组",
                )
                continue
            for decision in decisions:
                _validate_probe_decision(
                    decision,
                    registry,
                    errors,
                    path,
                )
    if set(branches) == {"branch_a", "branch_b"}:
        left = branches["branch_a"].get("admissible_decisions")
        right = branches["branch_b"].get("admissible_decisions")
        if json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
            right,
            sort_keys=True,
            ensure_ascii=False,
        ):
            _error(
                errors,
                "symmetric_probe_decisions",
                path,
                "Sensitivity两分支的可接受决定不能完全相同",
            )


def _validate_probe_decision(
    decision: Any,
    registry: ActionRegistry,
    errors: list[dict[str, str]],
    path: str,
) -> None:
    if not isinstance(decision, Mapping):
        _error(errors, "invalid_probe_decision", path, "决定必须是对象")
        return
    action = decision.get("action")
    parameters = decision.get("parameters")
    if action == "respond_only":
        if parameters != {}:
            _error(
                errors,
                "invalid_respond_only",
                path,
                "respond_only的parameters必须为空对象",
            )
        return
    if action == "accept_claim":
        action = "decide_claim"
    try:
        registry.validate_call(
            {"name": action, "parameters": parameters}
        )
    except Exception as error:
        _error(
            errors,
            "invalid_probe_action",
            path,
            str(error),
        )


def _audit_invariance_probe(
    probe: Any,
    errors: list[dict[str, str]],
    path: str,
) -> None:
    if not isinstance(probe, Mapping) or not all(
        probe.get(key) for key in ("paraphrase", "query", "expected_state")
    ):
        _error(
            errors,
            "invalid_invariance_probe",
            path,
            "invariance_probe必须包含paraphrase、query和expected_state",
        )
        return
    expected = probe["expected_state"]
    if not isinstance(expected, Mapping) or not isinstance(
        expected.get("path"),
        str,
    ):
        _error(
            errors,
            "invalid_invariance_state",
            path,
            "expected_state必须包含path",
        )


def _nonempty_unique_strings(value: Any) -> bool:
    return _nonempty_strings(value) and len(value) == len(set(value))


def _error(
    errors: list[dict[str, str]],
    code: str,
    path: str,
    message: str,
) -> None:
    errors.append({"code": code, "path": path, "message": message})
