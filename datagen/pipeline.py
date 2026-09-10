"""Staged, review-gated data pipeline for any compatible world and NPC."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from llm import create_llm_client, load_llm_settings
from runners.pilot import DEFAULT_CONFIG, PROJECT_ROOT

from datagen.generation.anchor_draft import generate_anchor_draft
from datagen.generation.catalog_foundation import generate_catalog_foundations
from datagen.generation.foundation import scaffold_world
from datagen.generation.stage1_stage2 import generate_stage1_stage2
from datagen.audit.benchmark_assets import audit_executable_character
from datagen.audit.source_assets import (
    audit_anchor_plan,
    audit_foundation,
    write_review_template,
)
from datagen.audit.validation import validate_world_outputs


def inspect_sources(
    world_dir: str | Path,
    character_ids: list[str],
) -> dict[str, Any]:
    return {
        "foundation": audit_foundation(world_dir, character_ids),
        "anchors": {
            character_id: audit_anchor_plan(world_dir, character_id)
            for character_id in character_ids
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RPG-AgentBench数据主入口：生成与审核严格分组"
    )
    parser.add_argument("--world", type=Path)
    parser.add_argument("--character", action="append", dest="characters")
    actions = parser.add_subparsers(dest="action", required=True)

    generate = actions.add_parser(
        "generate",
        help="生成基础、共享前置或Stage 1/2数据",
    )
    generators = generate.add_subparsers(dest="target", required=True)
    catalog_parser = generators.add_parser(
        "catalog-foundation",
        help="按catalog批量生成未批准的World Foundation与NPC Profile",
    )
    catalog_parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "configs" / "datagen" / "world_catalog.yaml",
    )
    catalog_parser.add_argument(
        "--assets-root",
        type=Path,
        default=PROJECT_ROOT / "assets",
    )
    catalog_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    catalog_parser.add_argument("--workers", type=int, default=4)
    catalog_parser.add_argument(
        "--catalog-world",
        action="append",
        dest="catalog_worlds",
        help="只生成指定world_id；可重复传入",
    )
    catalog_parser.add_argument(
        "--replace-existing-drafts",
        action="store_true",
        help="仅覆盖尚未人工批准且带生成报告的Foundation草案",
    )

    foundation_parser = generators.add_parser(
        "foundation",
        help="生成基础资产空骨架：canon/environment/角色卡等",
    )
    foundation_parser.add_argument("--world-id", required=True)
    foundation_parser.add_argument("--name", required=True)
    foundation_parser.add_argument(
        "--language",
        choices=("zh", "en"),
        required=True,
    )

    anchor_parser = generators.add_parser(
        "anchor-draft",
        help="生成Stage 1/2共享前置：Anchor与Transition审核草案",
    )
    anchor_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    anchor_parser.add_argument("--total-anchors", type=int, default=30)
    anchor_parser.add_argument("--canon-anchors", type=int, default=10)
    anchor_parser.add_argument("--output-dir", type=Path)

    build_parser = generators.add_parser(
        "stage1-2",
        help=(
            "生成Stage 1历史/QA/开放任务与Stage 2最小分支Pair；"
            "不生成Stage 3"
        ),
    )
    build_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build_parser.add_argument("--rounds", type=int, default=600)
    build_parser.add_argument("--pairs", type=int, default=10)
    build_parser.add_argument("--workers", type=int, default=6)
    build_parser.add_argument("--replace", action="store_true")
    build_parser.add_argument("--no-progress", action="store_true")

    audit = actions.add_parser("audit", help="审核来源或已生成资产")
    auditors = audit.add_subparsers(dest="target", required=True)
    auditors.add_parser(
        "review-template",
        help="生成基础/Anchor人工审核清单",
    )
    auditors.add_parser(
        "foundation",
        help="审核基础资产：canon/environment/角色卡",
    )
    auditors.add_parser(
        "anchors",
        help="审核Stage 1/2共享的Anchor与Transition",
    )
    auditors.add_parser(
        "sources",
        help="同时审核基础资产与Anchor/Transition",
    )
    output_parser = auditors.add_parser(
        "outputs",
        help="审核已生成的Stage 1/2数据",
    )
    output_parser.add_argument("--rounds", type=int, default=600)
    output_parser.add_argument("--pairs", type=int, default=10)

    args = parser.parse_args()
    if args.action == "generate" and args.target == "catalog-foundation":
        settings = load_llm_settings(
            args.config,
            project_root=PROJECT_ROOT,
        )
        result = generate_catalog_foundations(
            args.catalog,
            assets_root=args.assets_root,
            client_factory=lambda: create_llm_client(
                settings,
                scene="foundation_draft",
            ),
            config=settings.generation,
            workers=args.workers,
            world_ids=args.catalog_worlds,
            replace_existing_drafts=args.replace_existing_drafts,
        )
        _print_result(result)
        return
    if args.world is None:
        raise ValueError("该命令必须提供--world")
    if args.action == "generate" and args.target == "foundation":
        if not args.characters:
            raise ValueError("生成基础骨架至少需要一个--character")
        result = scaffold_world(
            args.world,
            world_id=args.world_id,
            name=args.name,
            language=args.language,
            character_ids=args.characters,
        )
        _print_result(result)
        return

    characters = _selected_characters(args.world, args.characters)
    if args.action == "audit" and args.target == "review-template":
        result = {
            character_id: str(
                write_review_template(args.world, character_id)
            )
            for character_id in characters
        }
    elif args.action == "audit" and args.target == "foundation":
        result = audit_foundation(args.world, characters)
    elif args.action == "audit" and args.target == "anchors":
        result = {
            character_id: audit_anchor_plan(args.world, character_id)
            for character_id in characters
        }
    elif args.action == "audit" and args.target == "sources":
        result = inspect_sources(args.world, characters)
    elif args.action == "audit" and args.target == "outputs":
        result = {
            "characters": {
                character_id: audit_executable_character(
                    args.world,
                    character_id,
                    expected_rounds=args.rounds,
                )
                for character_id in characters
            },
            "validation": (
                validate_world_outputs(
                    args.world,
                    rounds=args.rounds,
                    pair_count=args.pairs,
                )
                if set(characters)
                == set(_selected_characters(args.world, None))
                else None
            ),
        }
    elif args.action == "generate" and args.target == "anchor-draft":
        if len(characters) != 1:
            raise ValueError("一次只能为一个NPC生成Anchor草案")
        settings = load_llm_settings(
            args.config,
            project_root=PROJECT_ROOT,
        )
        result = generate_anchor_draft(
            args.world,
            characters[0],
            client=create_llm_client(
                settings,
                scene="anchor_draft",
            ),
            config=settings.generation,
            total_anchors=args.total_anchors,
            canon_anchors=args.canon_anchors,
            output_dir=args.output_dir,
        )
    elif args.action == "generate" and args.target == "stage1-2":
        result = generate_stage1_stage2(
            args.world,
            config_path=args.config,
            rounds=args.rounds,
            pair_count=args.pairs,
            character_ids=characters,
            workers=args.workers,
            replace=args.replace,
            show_progress=not args.no_progress,
        )
    else:
        raise AssertionError(f"未处理的命令: {args.action} {args.target}")
    _print_result(result)


def _print_result(result: Any) -> None:
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))


def _selected_characters(
    world: Path,
    selected: list[str] | None,
) -> list[str]:
    canon_path = world / "canon.yaml"
    if not canon_path.is_file():
        raise ValueError(f"缺少{canon_path}")
    canon = yaml.safe_load(canon_path.read_text(encoding="utf-8")) or {}
    available = canon.get("evaluated_npcs")
    if not isinstance(available, list) or not available:
        raise ValueError("canon.yaml缺少evaluated_npcs")
    characters = selected or list(available)
    unknown = set(characters) - set(available)
    if unknown:
        raise ValueError(f"未知角色: {sorted(unknown)}")
    return characters


if __name__ == "__main__":
    main()
