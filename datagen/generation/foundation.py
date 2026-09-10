"""Create a non-approved source-asset skeleton for a new world."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ..audit.source_assets import write_review_template


def scaffold_world(
    world_dir: str | Path,
    *,
    world_id: str,
    name: str,
    language: str,
    character_ids: list[str],
) -> dict[str, Any]:
    world = Path(world_dir)
    existing_files = (
        [
            path
            for path in world.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        ]
        if world.exists()
        else []
    )
    if existing_files:
        raise FileExistsError(f"{world}不是空目录")
    if world.exists():
        for marker in world.rglob(".gitkeep"):
            marker.unlink()
    if language not in {"zh", "en"}:
        raise ValueError("language必须是zh或en")
    if not world_id or not name or not character_ids:
        raise ValueError("world_id、name和character_ids不能为空")
    if len(character_ids) != len(set(character_ids)):
        raise ValueError("character_ids不能重复")

    (world / "characters").mkdir(parents=True, exist_ok=True)
    (world / "frozen").mkdir(parents=True, exist_ok=True)
    (world / "branches").mkdir(parents=True, exist_ok=True)
    (world / "scenarios").mkdir(parents=True, exist_ok=True)
    (world / "drafts").mkdir(parents=True, exist_ok=True)
    _write_yaml(
        world / "canon.yaml",
        {
            "world_id": world_id,
            "canon_policy": {
                "primary_source": "",
                "source_type": "",
                "excluded_versions": [],
                "shared_snapshot": "",
                "generated_timeline": "",
                "quotation_policy": "fact_level_paraphrase_only",
            },
            "evaluated_npcs": character_ids,
            "event_entities": [],
            "sources": [],
            "data_constraints": [],
        },
    )
    _write_yaml(
        world / "environment.yaml",
        {
            "world_id": world_id,
            "name": name,
            "language": language,
            "locations": [],
            "items": [],
        },
    )
    _write_yaml(
        world / "anchor_pool.yaml",
        {"world_id": world_id, "source": "", "events": []},
    )
    for character_id in character_ids:
        _write_yaml(
            world / "characters" / f"{character_id}.yaml",
            _empty_character(character_id),
        )
        frozen = world / "frozen" / character_id
        frozen.mkdir(parents=True, exist_ok=True)
        _write_yaml(
            frozen / "initial_context.yaml",
            {
                "runtime_state": {},
                "environment": {},
            },
        )
        write_review_template(world, character_id)
    return {
        "world": str(world),
        "world_id": world_id,
        "characters": character_ids,
        "approved": False,
        "next_step": "补全来源资产后运行python3 -m datagen.pipeline audit foundation",
    }


def _empty_character(character_id: str) -> dict[str, Any]:
    return {
        "character_id": character_id,
        "snapshot": "",
        "identity": {
            "name": "",
            "role": "",
            "affiliations": [],
            "public_background": [],
        },
        "worldview": {
            "principles": [],
            "stances": [],
            "core_goals": [],
            "prohibitions": [],
        },
        "testable_boundaries": [],
        "capability": {
            "skills": [],
            "authority": [],
            "limitations": [],
        },
        "private": {"secrets": [], "private_background": []},
        "social": {"relationship_rules": []},
        "style": {
            "register": "",
            "verbal_habits": [],
            "forbidden_styles": [],
        },
    }


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
