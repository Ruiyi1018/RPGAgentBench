"""仅运行Stage 3的驱动入口，用于框架验证与失败分布采样。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents import PlayerMode
from datagen.shared.io import load_yaml
from gamecore import ActionRegistry
from llm import create_llm_client, load_llm_settings

from .full_experiment import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    STAGE3_PROTOCOL_VERSION,
    ProgressReporter,
    _run_stage3,
    _validate_scenario,
    _write_json,
)


def run_stage3_only(
    world_dir: Path,
    character_id: str,
    *,
    scenario_path: Path,
    config_path: Path,
    turns: int,
    seeds: tuple[int, ...],
    modes: tuple[PlayerMode, ...],
    checker_interval: int,
    run_dir: Path,
    workers: int,
    show_progress: bool,
) -> dict[str, object]:
    world = Path(world_dir)
    scenario = load_yaml(scenario_path)
    if scenario["character_id"] != character_id:
        raise ValueError("scenario.character_id与实验角色不一致")
    _validate_scenario(world, character_id, scenario)
    settings = load_llm_settings(config_path, project_root=PROJECT_ROOT)
    config = settings.generation
    registry = ActionRegistry.load_directory(
        PROJECT_ROOT / "gamecore" / "actions"
    )

    def client_factory():
        return create_llm_client(settings, scene="stage3_only")

    run_dir.mkdir(parents=True, exist_ok=True)
    checker_batches = (turns + checker_interval - 1) // checker_interval
    maximum_logical_calls = (
        len(seeds) * len(modes) * (turns * 2 + checker_batches * 2)
    )
    _write_json(
        run_dir / "run_config.json",
        {
            "world": str(world),
            "character_id": character_id,
            "scenario": str(scenario_path),
            "model": config.model,
            "base_url": settings.base_url,
            "turns": turns,
            "seeds": list(seeds),
            "modes": [mode.value for mode in modes],
            "checker_interval": checker_interval,
            "stage3_protocol": STAGE3_PROTOCOL_VERSION,
            "maximum_logical_calls": maximum_logical_calls,
        },
    )
    progress = ProgressReporter(
        total=maximum_logical_calls,
        completed=0,
        enabled=show_progress,
    )
    progress.render("start")
    stage3 = _run_stage3(
        world,
        character_id,
        scenario,
        registry,
        client_factory,
        config,
        turns,
        seeds,
        config.model,
        config.model,
        checker_interval,
        run_dir,
        workers,
        progress,
        modes=modes,
    )
    summary = {
        "character_id": character_id,
        "model": config.model,
        "stage3_protocol": STAGE3_PROTOCOL_VERSION,
        "turns": turns,
        "seeds": list(seeds),
        "modes": [mode.value for mode in modes],
        "stage3": stage3,
    }
    _write_json(run_dir / "stage3_only_summary.json", summary)
    progress.render("done")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--character", required=True)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--checker-interval", type=int, default=5)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--modes", default="normal,pressure")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    seeds = tuple(
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    )
    if not seeds:
        raise ValueError("--seeds至少包含一个整数")
    modes = tuple(
        PlayerMode(value.strip())
        for value in args.modes.split(",")
        if value.strip()
    )
    if not modes:
        raise ValueError("--modes至少包含一个模式")
    result = run_stage3_only(
        args.world,
        args.character,
        scenario_path=args.scenario,
        config_path=args.config,
        turns=args.turns,
        seeds=seeds,
        modes=modes,
        checker_interval=args.checker_interval,
        run_dir=args.run_dir,
        workers=args.workers,
        show_progress=not args.no_progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
