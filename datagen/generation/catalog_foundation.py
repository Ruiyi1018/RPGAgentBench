"""Generate review-gated Foundation and Profile drafts from a world catalog."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from agents.prompt_builder import PromptBundle
from llm import GenerationConfig, LLMClient, generate_structured

from ..audit.source_assets import audit_foundation
from ..shared.io import load_yaml
from .foundation import scaffold_world


ClientFactory = Callable[[], LLMClient]
RELATIONSHIP_LEVELS = {"hostile", "distrustful", "neutral", "trusting", "loyal"}
FOUNDATION_PROMPT_VERSION = 2


def generate_catalog_foundations(
    catalog_path: str | Path,
    *,
    assets_root: str | Path,
    client_factory: ClientFactory,
    config: GenerationConfig,
    workers: int = 4,
    world_ids: list[str] | None = None,
    replace_existing_drafts: bool = False,
) -> dict[str, Any]:
    """Generate planned worlds; existing worlds are skipped unless replaced."""

    catalog = load_yaml(catalog_path)
    worlds = _validate_catalog(catalog)
    selected_ids = set(world_ids or [])
    unknown = selected_ids - {str(world["world_id"]) for world in worlds}
    if unknown:
        raise ValueError(f"catalog中不存在world_id: {sorted(unknown)}")
    planned = [
        world
        for world in worlds
        if (not selected_ids or str(world["world_id"]) in selected_ids)
        and world.get("status") != "existing"
    ]
    if not planned:
        raise ValueError("没有符合条件的待生成世界")
    if workers < 1:
        raise ValueError("workers必须至少为1")
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(planned))) as executor:
        futures = {
            executor.submit(
                generate_foundation_world,
                world,
                assets_root=assets_root,
                client_factory=client_factory,
                config=config,
                replace_existing_draft=replace_existing_drafts,
            ): str(world["world_id"])
            for world in planned
        }
        for future in as_completed(futures):
            world_id = futures[future]
            try:
                results[world_id] = future.result()
                status = results[world_id].get("status", "completed")
                print(f"[foundation] {world_id}: {status}", flush=True)
            except Exception as error:
                results[world_id] = {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"[foundation] {world_id}: failed: {error}", flush=True)
    existing = [
        str(world["world_id"])
        for world in worlds
        if world.get("status") == "existing"
    ]
    return {
        "catalog": str(catalog_path),
        "assets_root": str(assets_root),
        "generated": results,
        "skipped_existing": existing,
        "summary": {
            "requested": len(planned),
            "generated": sum(
                item.get("status") == "generated" for item in results.values()
            ),
            "failed": sum(
                item.get("status") == "failed" for item in results.values()
            ),
        },
    }


def generate_foundation_world(
    spec: Mapping[str, Any],
    *,
    assets_root: str | Path,
    client_factory: ClientFactory,
    config: GenerationConfig,
    replace_existing_draft: bool = False,
) -> dict[str, Any]:
    world_id = str(spec["world_id"])
    world_dir = Path(assets_root) / world_id
    foundation_exists = (world_dir / "canon.yaml").exists()
    if foundation_exists and not replace_existing_draft:
        return {"status": "skipped", "reason": "foundation already exists"}
    if foundation_exists:
        report_path = world_dir / "foundation_generation_report.yaml"
        if not report_path.is_file():
            raise ValueError(f"{world_id}不是可覆盖的批量Foundation草案")
        if load_yaml(report_path).get("human_approved") is not False:
            raise ValueError(f"{world_id}已人工批准，禁止自动覆盖")
    work_dir = (
        Path(assets_root).parent
        / "generation_work"
        / "catalog_foundation"
        / world_id
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_marker = work_dir / "prompt_version.txt"
    cache_valid = (
        cache_marker.is_file()
        and cache_marker.read_text(encoding="utf-8").strip()
        == str(FOUNDATION_PROMPT_VERSION)
    )
    cache_marker.write_text(
        f"{FOUNDATION_PROMPT_VERSION}\n",
        encoding="utf-8",
    )
    generation_config = replace(config, max_tokens=max(config.max_tokens, 8192))
    world_assets = _cached_or_generate(
        work_dir / "world.json",
        cache_valid=cache_valid,
        validator=lambda value: _validate_world_output(value, spec),
        generator=lambda: generate_structured(
            client_factory(),
            _world_prompt(spec),
            generation_config,
            lambda value: _validate_world_output(value, spec),
        ),
    )
    profiles: list[dict[str, Any]] = []
    profile_errors: dict[str, str] = {}
    with ThreadPoolExecutor(
        max_workers=min(3, len(spec["characters"]))
    ) as executor:
        profile_futures = {
            executor.submit(
                _generate_profile,
                spec,
                character,
                world_assets,
                client_factory(),
                config,
                work_dir / "profiles" / f"{character['id']}.json",
                cache_valid,
            ): str(character["id"])
            for character in spec["characters"]
        }
        for future in as_completed(profile_futures):
            character_id = profile_futures[future]
            try:
                profiles.append(future.result())
                print(
                    f"[foundation] {world_id}/{character_id}: profile generated",
                    flush=True,
                )
            except Exception as error:
                profile_errors[character_id] = f"{type(error).__name__}: {error}"
    if profile_errors:
        raise ValueError(f"Profile生成失败: {profile_errors}")
    order = {
        str(character["id"]): index
        for index, character in enumerate(spec["characters"])
    }
    profiles.sort(key=lambda item: order[str(item["card"]["character_id"])])
    profile_assets = {"characters": profiles}
    character_ids = [str(item["id"]) for item in spec["characters"]]
    if not foundation_exists:
        scaffold_world(
            world_dir,
            world_id=world_id,
            name=str(spec["title"]),
            language=str(spec["language"]),
            character_ids=character_ids,
        )
    _write_foundation_assets(world_dir, spec, world_assets, profile_assets)
    report = audit_foundation(world_dir, character_ids)
    generation_report = {
        "schema_version": 1,
        "world_id": world_id,
        "status": "generated",
        "human_approved": False,
        "machine_audit": report,
        "next_step": (
            "按datagen/audit/HUMAN_REVIEW_RULES.md审核，"
            "再更新各角色source_review.yaml"
        ),
    }
    _write_yaml(world_dir / "foundation_generation_report.yaml", generation_report)
    return {
        "status": "generated",
        "world": str(world_dir),
        "characters": character_ids,
        "machine_passed": report["machine_passed"],
        "human_approved": False,
    }


def _world_prompt(spec: Mapping[str, Any]) -> PromptBundle:
    roster = [
        {"id": item["id"], "name": item["name"]} for item in spec["characters"]
    ]
    language = str(spec["language"])
    return PromptBundle(
        system_prompt=(
            "你是RPG benchmark的世界基础资产起草器。输出只能是JSON对象。"
            "所有事实必须限定在给定作品版本和共享时间快照，不得使用快照后的剧情。"
            "只做事实级转述，不复制原作台词。无法可靠确认的细节宁可省略，"
            "不得伪造URL、章节号或集数。Anchor池事件必须按时间排序；"
            "Catalog角色不要求全部进入Anchor池；只有后续入选Longitudinal层的"
            "角色才需要满足每角色10个正典Anchor。"
            "地点和物品只能来自当前来源范围及共享快照，不能混入其他章节或版本。"
            "每个cf_edit必须给出只改变关键原因的真实替代情形，禁止写“无”。"
            if language == "zh"
            else
            "You draft foundation assets for an RPG benchmark. Output one JSON "
            "object only. Constrain every fact to the specified source and shared "
            "snapshot; never use later plot knowledge. Paraphrase facts and do not "
            "copy dialogue. Omit uncertain details; never invent URLs, chapter "
            "numbers, or episode numbers. Sort anchor-pool events chronologically "
            "Catalog characters do not all need anchor-pool coverage. Only later "
            "Longitudinal-core characters require ten canonical anchors each. "
            "Locations and items must occur within this source scope and snapshot, "
            "not another chapter or adaptation. Every cf_edit must be a real minimal "
            "alternative; never write none or N/A."
        ),
        user_prompt=(
            f"World specification:\n{json.dumps(dict(spec), ensure_ascii=False)}\n\n"
            f"Roster:\n{json.dumps(roster, ensure_ascii=False)}\n\n"
            "Return exactly these top-level fields: canon_policy, event_entities, "
            "sources, data_constraints, environment, anchor_events. "
            "canon_policy must contain primary_source, source_type, "
            "excluded_versions, shared_snapshot, generated_timeline, "
            "quotation_policy. sources must include at least one primary source "
            "with id,title,creator,scope,authority. data_constraints must be an "
            "array of non-empty rule strings. environment must contain name, "
            "locations and items; every location has id,name,connected_to and all "
            "connections reference listed IDs; every item has id,name,type,portable. "
            "Every place used by the roster or anchor_events as an active scene, "
            "movement destination, meeting place, workplace, or residence must have "
            "a matching environment location. Places mentioned only as remote "
            "background or past origin do not need map nodes. Direct one-step "
            "connections are allowed regardless of real-world distance. "
            "Create 18-24 canonical anchor_events. Every event has id,fact,known_by,"
            "state_change,cf_edit,anchor_for. known_by and anchor_for use roster IDs."
        ),
    )


def _generate_profile(
    spec: Mapping[str, Any],
    character: Mapping[str, Any],
    world_assets: Mapping[str, Any],
    client: LLMClient,
    config: GenerationConfig,
    cache_path: Path,
    cache_valid: bool,
) -> dict[str, Any]:
    return _cached_or_generate(
        cache_path,
        cache_valid=cache_valid,
        validator=lambda output: _validate_profile_output(
            output, character, world_assets
        ),
        generator=lambda: generate_structured(
            client,
            _profile_prompt(spec, character, world_assets),
            config,
            lambda output: _validate_profile_output(
                output,
                character,
                world_assets,
            ),
        ),
    )


def _profile_prompt(
    spec: Mapping[str, Any],
    character: Mapping[str, Any],
    world_assets: Mapping[str, Any],
) -> PromptBundle:
    language = str(spec["language"])
    compact_world = {
        "canon_policy": world_assets["canon_policy"],
        "environment": world_assets["environment"],
        "anchor_events": world_assets["anchor_events"],
    }
    return PromptBundle(
        system_prompt=(
            "你是RPG benchmark的角色Profile起草器。只输出JSON。"
            "每个角色严格使用同一共享时间快照；角色只知道其在场、被告知或能合理"
            "确认的事实。Profile不能写入未来职务、未来关系、未来死亡或观众全知信息。"
            "角色边界必须是可观察决定，语言风格不能用固定剧情流程代替。"
            "即使角色资料较少，affiliations和public_background也不能留空；"
            "只描述该角色在当前来源场景中可观察到的身份，不得编造额外经历。"
            if language == "zh"
            else
            "You draft character profiles for an RPG benchmark. Output JSON only. "
            "Use one shared snapshot for every character. A character knows only "
            "facts they witnessed, were told, or could verify. Never add future "
            "roles, relationships, deaths, or audience-only knowledge. Boundaries "
            "must define observable decisions; style cannot be a plot template. "
            "Even for a sparse character, affiliations and public_background cannot "
            "be empty: describe only the observed role in this source scene without "
            "inventing additional history."
        ),
        user_prompt=(
            f"World specification:\n"
            f"{json.dumps(dict(spec), ensure_ascii=False)}\n\n"
            f"Target character:\n"
            f"{json.dumps(dict(character), ensure_ascii=False)}\n\n"
            f"Approved-for-drafting world context:\n"
            f"{json.dumps(compact_world, ensure_ascii=False)}\n\n"
            'Return exactly {"card": {...}, "initial_state": {...}} for the '
            "target character. "
            "card fields: character_id,"
            "snapshot,identity{name,role,affiliations,public_background},worldview"
            "{principles,stances,core_goals,prohibitions},testable_boundaries,"
            "capability{skills,authority,limitations},private{secrets,"
            "private_background},social{relationship_rules},style{register,"
            "verbal_habits,forbidden_styles}. Provide at least three boundaries; "
            "All plural profile fields named above must be arrays of non-empty "
            "strings, never scalar strings. "
            "each has id,position,challenge_space,forbidden_outcomes,allowed_change. "
            "initial_state fields: location_id, relationships (map roster/player IDs "
            "to hostile|distrustful|neutral|trusting|loyal), goals (list of "
            "{id,content,status}), knowledge (list of {id,content,visibility}). "
            "Any place described in the card as a current scene, movement "
            "destination, meeting place, workplace, or residence must match one of "
            "the provided environment locations; unlisted places may only be remote "
            "background or past history."
        ),
    )


def _validate_catalog(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    worlds = catalog.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise ValueError("catalog缺少worlds")
    ids = [str(item.get("world_id", "")) for item in worlds]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError("catalog world_id必须非空且唯一")
    for world in worlds:
        if world.get("language") not in {"zh", "en"}:
            raise ValueError(f"{world.get('world_id')}语言无效")
        characters = world.get("characters")
        if not isinstance(characters, list) or not characters:
            raise ValueError(f"{world.get('world_id')}缺少characters")
        character_ids = [str(item.get("id", "")) for item in characters]
        if "" in character_ids or len(character_ids) != len(set(character_ids)):
            raise ValueError(f"{world.get('world_id')}角色ID无效")
    return worlds


def _validate_world_output(
    value: dict[str, Any],
    spec: Mapping[str, Any],
) -> None:
    required = {
        "canon_policy",
        "event_entities",
        "sources",
        "data_constraints",
        "environment",
        "anchor_events",
    }
    if set(value) != required:
        raise ValueError(f"世界草案字段错误: {sorted(value)}")
    policy = value["canon_policy"]
    if not isinstance(policy, Mapping) or not all(
        policy.get(key)
        for key in (
            "primary_source",
            "source_type",
            "shared_snapshot",
            "generated_timeline",
            "quotation_policy",
        )
    ):
        raise ValueError("canon_policy字段不完整")
    sources = value["sources"]
    if not isinstance(sources, list) or not any(
        isinstance(item, Mapping) and item.get("authority") == "primary"
        for item in sources
    ):
        raise ValueError("sources缺少primary来源")
    if not _nonempty_string_list(value["data_constraints"]):
        raise ValueError("data_constraints必须是非空字符串数组")
    environment = value["environment"]
    locations = environment.get("locations") if isinstance(environment, Mapping) else None
    items = environment.get("items") if isinstance(environment, Mapping) else None
    if not isinstance(locations, list) or len(locations) < 4:
        raise ValueError("至少生成4个地点")
    if not isinstance(items, list) or len(items) < 3:
        raise ValueError("至少生成3个物品")
    location_ids = {str(item.get("id", "")) for item in locations}
    if "" in location_ids or len(location_ids) != len(locations):
        raise ValueError("地点ID必须非空且唯一")
    if any(
        set(item.get("connected_to", [])) - location_ids for item in locations
    ):
        raise ValueError("地点连接引用未知ID")
    roster = {str(item["id"]) for item in spec["characters"]}
    event_entities = {
        str(item)
        for item in value["event_entities"]
        if isinstance(item, str)
    }
    allowed_known_entities = roster | event_entities
    events = value["anchor_events"]
    if not isinstance(events, list) or not 12 <= len(events) <= 30:
        raise ValueError("anchor_events必须为12至30项")
    event_ids: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping) or not all(
            event.get(key)
            for key in ("id", "fact", "state_change", "cf_edit", "anchor_for")
        ):
            raise ValueError("anchor event字段不完整")
        if str(event["cf_edit"]).strip().lower() in {
            "无",
            "none",
            "n/a",
            "na",
            "not applicable",
        }:
            raise ValueError(f"{event['id']}缺少真实最小反事实")
        event_id = str(event["id"])
        if event_id in event_ids:
            raise ValueError("anchor event ID重复")
        event_ids.add(event_id)
        if set(event.get("known_by", [])) - allowed_known_entities:
            raise ValueError("known_by引用未知角色")
        targets = set(event["anchor_for"])
        if not targets or targets - roster:
            raise ValueError("anchor_for引用未知角色")


def _validate_profile_output(
    value: dict[str, Any],
    character: Mapping[str, Any],
    world_assets: Mapping[str, Any],
) -> None:
    if set(value) != {"card", "initial_state"}:
        raise ValueError("单角色草案必须只包含card和initial_state")
    card = value["card"]
    if isinstance(card, dict):
        for section, fields in {
            "identity": ("affiliations", "public_background"),
            "worldview": (
                "principles",
                "stances",
                "core_goals",
                "prohibitions",
            ),
            "capability": ("skills", "authority", "limitations"),
            "private": ("secrets", "private_background"),
            "social": ("relationship_rules",),
            "style": ("verbal_habits", "forbidden_styles"),
        }.items():
            section_value = card.get(section)
            if not isinstance(section_value, dict):
                continue
            for field in fields:
                field_value = section_value.get(field)
                if isinstance(field_value, str) and field_value.strip():
                    section_value[field] = [field_value]
        boundaries = card.get("testable_boundaries")
        if isinstance(boundaries, list):
            for boundary in boundaries:
                if not isinstance(boundary, dict):
                    continue
                for field in ("challenge_space", "forbidden_outcomes"):
                    field_value = boundary.get(field)
                    if isinstance(field_value, str) and field_value.strip():
                        boundary[field] = [field_value]
    _validate_profiles_output(
        {"characters": [value]},
        {"characters": [dict(character)]},
        world_assets,
    )


def _validate_profiles_output(
    value: dict[str, Any],
    spec: Mapping[str, Any],
    world_assets: Mapping[str, Any],
) -> None:
    if set(value) != {"characters"} or not isinstance(value["characters"], list):
        raise ValueError("角色草案必须只包含characters数组")
    expected = {str(item["id"]): str(item["name"]) for item in spec["characters"]}
    actual: dict[str, Mapping[str, Any]] = {}
    locations = {
        str(item["id"]) for item in world_assets["environment"]["locations"]
    }
    for item in value["characters"]:
        if not isinstance(item, Mapping) or set(item) != {"card", "initial_state"}:
            raise ValueError("角色项必须包含card和initial_state")
        card = item["card"]
        character_id = str(card.get("character_id", ""))
        if character_id in actual:
            raise ValueError("角色Profile重复")
        actual[character_id] = item
        if card.get("identity", {}).get("name") != expected.get(character_id):
            raise ValueError(f"{character_id}角色姓名不匹配")
        for section, fields in {
            "identity": ("affiliations", "public_background"),
            "worldview": ("principles", "stances", "core_goals", "prohibitions"),
            "capability": ("skills", "authority", "limitations"),
        }.items():
            if not all(
                _nonempty_string_list(card.get(section, {}).get(field))
                for field in fields
            ):
                raise ValueError(f"{character_id}.{section}不完整")
        for section, fields in {
            "private": ("secrets", "private_background"),
            "social": ("relationship_rules",),
            "style": ("verbal_habits", "forbidden_styles"),
        }.items():
            for field in fields:
                field_value = card.get(section, {}).get(field)
                if not isinstance(field_value, list) or not all(
                    isinstance(entry, str) for entry in field_value
                ):
                    raise ValueError(
                        f"{character_id}.{section}.{field}必须是字符串数组"
                    )
        if not isinstance(card.get("style", {}).get("register", ""), str):
            raise ValueError(f"{character_id}.style.register必须是字符串")
        boundaries = card.get("testable_boundaries")
        if not isinstance(boundaries, list) or len(boundaries) < 3:
            raise ValueError(f"{character_id}至少需要3条边界")
        for boundary in boundaries:
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
                raise ValueError(f"{character_id}边界字段不完整")
        state = item["initial_state"]
        if state.get("location_id") not in locations:
            raise ValueError(f"{character_id}初始地点不存在")
        if set(state.get("relationships", {}).values()) - RELATIONSHIP_LEVELS:
            raise ValueError(f"{character_id}关系等级无效")
    if set(actual) != set(expected):
        raise ValueError("角色Profile未完整覆盖catalog")


def _write_foundation_assets(
    world: Path,
    spec: Mapping[str, Any],
    world_assets: Mapping[str, Any],
    profile_assets: Mapping[str, Any],
) -> None:
    character_ids = [str(item["id"]) for item in spec["characters"]]
    canon = {
        "world_id": spec["world_id"],
        "canon_policy": copy.deepcopy(world_assets["canon_policy"]),
        "evaluated_npcs": character_ids,
        "event_entities": copy.deepcopy(world_assets["event_entities"]),
        "sources": copy.deepcopy(world_assets["sources"]),
        "data_constraints": copy.deepcopy(world_assets["data_constraints"]),
    }
    environment = {
        "world_id": spec["world_id"],
        **copy.deepcopy(world_assets["environment"]),
        "language": spec["language"],
    }
    anchor_pool = {
        "world_id": spec["world_id"],
        "source": str(world_assets["canon_policy"]["primary_source"]),
        "events": copy.deepcopy(world_assets["anchor_events"]),
    }
    _write_yaml(world / "canon.yaml", canon)
    _write_yaml(world / "environment.yaml", environment)
    _write_yaml(world / "anchor_pool.yaml", anchor_pool)
    for item in profile_assets["characters"]:
        card = copy.deepcopy(item["card"])
        character_id = str(card["character_id"])
        _write_yaml(world / "characters" / f"{character_id}.yaml", card)
        _write_yaml(
            world / "frozen" / character_id / "initial_context.yaml",
            _initial_context(item["initial_state"], character_id),
        )


def _initial_context(
    state: Mapping[str, Any],
    character_id: str,
) -> dict[str, Any]:
    relationships = {
        target: {"level": level, "basis_event": "initial_state"}
        for target, level in state.get("relationships", {}).items()
    }
    relationships.setdefault(
        "player",
        {"level": "neutral", "basis_event": "initial_state"},
    )
    goals = [
        {**copy.deepcopy(goal), "basis_event": "initial_state"}
        for goal in state.get("goals", [])
    ]
    knowledge = {
        str(item["id"]): {
            "content": item["content"],
            "status": "active",
            "visibility": item.get("visibility", ["npc"]),
            "source_event": "initial_state",
        }
        for item in state.get("knowledge", [])
    }
    return {
        "runtime_state": {
            "claims": {},
            "commitments": [],
            "relationships": relationships,
            "goals": goals,
            "disclosures": [],
            "knowledge": knowledge,
        },
        "environment": {
            "health": {character_id: 100, "player": 100},
            "inventories": {character_id: [], "player": []},
            "locations": {
                character_id: state["location_id"],
                "player": state["location_id"],
            },
            "access": {},
            "offers": [],
            "task_status": "active",
            "dialogue_status": "active",
        },
    }


def _cached_or_generate(
    path: Path,
    *,
    cache_valid: bool,
    validator: Callable[[dict[str, Any]], None],
    generator: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if cache_valid and path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            validator(value)
            return value
        except (ValueError, TypeError, KeyError, AttributeError):
            path.unlink(missing_ok=True)
    value = generator()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return value


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _nonempty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )
