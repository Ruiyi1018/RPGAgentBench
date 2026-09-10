"""Generate all Stage 1 assets and then Stage 2 branch-pair assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm import LLMClient, create_llm_client, load_llm_settings
from runners.pilot import DEFAULT_CONFIG, PROJECT_ROOT

from ..audit.benchmark_assets import (
    audit_executable_character,
    write_output_review_template,
)
from ..shared.io import load_yaml, read_jsonl, write_jsonl
from .stage2_tests import write_executable_pairs
from .stage1_tests import (
    build_executable_open_tasks,
    build_executable_qa,
)
from .stage1_history import generate_llm_history
from ..audit.source_assets import (
    audit_anchor_plan,
    audit_foundation,
    require_generation_ready,
)
from ..audit.validation import validate_world_outputs


def generate_stage1_stage2(
    world_dir: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    rounds: int = 600,
    pair_count: int = 10,
    character_ids: list[str] | None = None,
    workers: int = 6,
    replace: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    if pair_count != 10:
        raise ValueError("当前协议固定为7个敏感性+3个不变性Pair")
    world = Path(world_dir)
    canon = load_yaml(world / "canon.yaml")
    selected = character_ids or canon["evaluated_npcs"]
    unknown = set(selected) - set(canon["evaluated_npcs"])
    if unknown:
        raise ValueError(f"未知角色: {sorted(unknown)}")
    settings = load_llm_settings(config_path, project_root=PROJECT_ROOT)
    def client_factory() -> LLMClient:
        return create_llm_client(settings, scene="data_generation")

    generation_reports: dict[str, Any] = {}
    source_reviews: dict[str, Any] = {}
    for character_id in selected:
        require_generation_ready(world, character_id)
        source_reviews[character_id] = {
            "foundation": audit_foundation(world, [character_id]),
            "anchors": audit_anchor_plan(world, character_id),
        }
        frozen_dir = world / "frozen" / character_id
        history_path = frozen_dir / "history.jsonl"
        work_dir = (
            PROJECT_ROOT
            / "generation_work"
            / world.name
            / character_id
        )
        if history_path.exists() and not replace:
            raise FileExistsError(
                f"{history_path}已存在；确认删除旧数据后请传--replace"
            )
        seed_history = (
            read_jsonl(history_path)
            if replace and history_path.is_file() and not work_dir.exists()
            else None
        )
        history, generation_report = generate_llm_history(
            world,
            character_id,
            client_factory=client_factory,
            config=settings.generation,
            rounds=rounds,
            workers=workers,
            work_root=work_dir,
            seed_history=seed_history,
            show_progress=show_progress,
        )
        if not generation_report["quality"]["passed"]:
            (work_dir / "generation_report.yaml").write_text(
                yaml.safe_dump(
                    generation_report,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            raise ValueError(
                f"{character_id}历史质量门禁失败；候选数据未写入正式资产"
            )
        write_jsonl(history_path, history)
        write_jsonl(
            frozen_dir / "qa.jsonl",
            build_executable_qa(world, character_id, history),
        )
        write_jsonl(
            frozen_dir / "open_tasks.jsonl",
            build_executable_open_tasks(world, character_id, history),
        )
        write_executable_pairs(world, character_id, history)
        write_output_review_template(world, character_id)
        generation_reports[character_id] = generation_report

    validation = (
        validate_world_outputs(
            world,
            rounds=rounds,
            pair_count=pair_count,
        )
        if set(selected) == set(canon["evaluated_npcs"])
        else {
            "world_id": canon["world_id"],
            "validated_characters": selected,
        }
    )
    readiness = {
        character_id: audit_executable_character(
            world,
            character_id,
            expected_rounds=rounds,
        )
        for character_id in selected
    }
    manifest = {
        "schema_version": "0.4",
        "generation_mode": "llm_interactive_sessions",
        "formal_benchmark_ready": all(
            report["quality"]["passed"]
            and readiness[character_id]["formal_benchmark_ready"]
            for character_id, report in generation_reports.items()
        ),
        "world_id": canon["world_id"],
        "generated_characters": selected,
        "ready_for_api_smoke_test": all(
            item["ready_for_api_smoke_test"]
            for item in readiness.values()
        ),
        "rounds_per_character": rounds,
        "session_sizes": {
            character_id: dict(
                generation_reports[character_id]["quality"][
                    "session_size_distribution"
                ]
            )
            for character_id in selected
        },
        "structured_qa_per_character": 50,
        "open_tasks_per_character": 2,
        "pairs_per_character": pair_count,
        "source_reviews": source_reviews,
        "generation_reports": generation_reports,
        "validation": validation,
        "readiness_audit": readiness,
    }
    (world / "generated_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return manifest
