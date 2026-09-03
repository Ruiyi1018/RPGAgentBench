"""Generate LLM-realized histories, QA, and Stage2 probes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from llm import LLMClient, create_llm_client, load_llm_settings
from runners.pilot import DEFAULT_CONFIG, PROJECT_ROOT

from .audit import audit_executable_character
from .common import load_yaml, write_jsonl
from .executable_branches import write_executable_pairs
from .executable_qa import (
    build_executable_open_tasks,
    build_executable_qa,
)
from .llm_history import generate_llm_history
from .validate import validate_world_outputs


def generate_world(
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
    for character_id in selected:
        frozen_dir = world / "frozen" / character_id
        history_path = frozen_dir / "history.jsonl"
        if history_path.exists() and not replace:
            raise FileExistsError(
                f"{history_path}已存在；确认删除旧数据后请传--replace"
            )
        history, generation_report = generate_llm_history(
            world,
            character_id,
            client_factory=client_factory,
            config=settings.generation,
            rounds=rounds,
            workers=workers,
            show_progress=show_progress,
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
        "schema_version": "0.3",
        "generation_mode": "llm_interactive_sessions",
        "formal_benchmark_ready": all(
            report["quality"]["passed"]
            for report in generation_reports.values()
        ),
        "world_id": canon["world_id"],
        "generated_characters": selected,
        "ready_for_api_smoke_test": all(
            item["ready_for_api_smoke_test"]
            for item in readiness.values()
        ),
        "rounds_per_character": rounds,
        "session_size": rounds // 30,
        "structured_qa_per_character": 50,
        "open_tasks_per_character": 2,
        "pairs_per_character": pair_count,
        "generation_reports": generation_reports,
        "validation": validation,
        "readiness_audit": readiness,
    }
    (world / "generated_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--character", action="append", dest="characters")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    manifest = generate_world(
        args.world,
        config_path=args.config,
        rounds=args.rounds,
        pair_count=args.pairs,
        character_ids=args.characters,
        workers=args.workers,
        replace=args.replace,
        show_progress=not args.no_progress,
    )
    print(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
