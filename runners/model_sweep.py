"""Expand candidate/reference model matrices into Stage 1-3 runs."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from agents import PlayerMode
from llm import load_model_registry

from .full_experiment import DEFAULT_STAGE3_SEEDS, run_full_experiment
from .pilot import PROJECT_ROOT


def run_model_sweep(
    experiment_path: str | Path,
    *,
    world_dir: str | Path,
    character_id: str,
    scenario_path: str | Path,
    dry_run: bool = False,
    output_root: str | Path = "runs",
    workers: int = 3,
    show_progress: bool = True,
) -> dict[str, Any]:
    source = Path(experiment_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("实验矩阵配置根节点必须是对象")
    registry_path = Path(_required_string(raw, "registry"))
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    registry = load_model_registry(
        registry_path,
        project_root=PROJECT_ROOT,
    )
    roles = _required_mapping(raw, "roles")
    matrix = raw.get("matrix", {})
    if not isinstance(matrix, Mapping):
        raise ValueError("matrix必须是对象")
    candidates = _matrix_values(matrix, roles, "candidate")
    players = _matrix_values(matrix, roles, "player")
    evaluators = _matrix_values(matrix, roles, "evaluator")
    combinations = list(itertools.product(candidates, players, evaluators))
    for names in combinations:
        for name in names:
            registry.model(name)
    stage3 = raw.get("stage3", {})
    if not isinstance(stage3, Mapping):
        raise ValueError("stage3必须是对象")
    seeds = tuple(
        int(value)
        for value in stage3.get("seeds", DEFAULT_STAGE3_SEEDS)
    )
    modes = tuple(
        PlayerMode(str(value))
        for value in stage3.get("modes", ["normal", "pressure"])
    )
    turns = int(stage3.get("turns", 40))
    experiment_id = (
        f"{source.stem}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    sweep_root = Path(output_root) / experiment_id
    results: list[dict[str, Any]] = []
    for candidate, player, evaluator in combinations:
        combination_id = "__".join((candidate, player, evaluator))
        try:
            result = run_full_experiment(
                world_dir,
                character_id,
                scenario_path=scenario_path,
                registry_path=registry_path,
                candidate_model=candidate,
                player_model=player,
                evaluator_model=evaluator,
                turns=turns,
                stage3_seeds=seeds,
                stage3_modes=modes,
                dry_run=dry_run,
                output_root=sweep_root / combination_id,
                workers=workers,
                show_progress=show_progress,
            )
        except Exception as error:
            results.append(
                {
                    "id": combination_id,
                    "candidate": candidate,
                    "player": player,
                    "evaluator": evaluator,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        results.append(
            {
                "id": combination_id,
                "candidate": candidate,
                "player": player,
                "evaluator": evaluator,
                "status": "preflight" if dry_run else "completed",
                "result": result,
            }
        )
    summary = {
        "experiment": str(source),
        "registry": str(registry_path),
        "combinations": len(combinations),
        "completed": sum(
            item["status"] in {"completed", "preflight"} for item in results
        ),
        "failed": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    if not dry_run:
        sweep_root.mkdir(parents=True, exist_ok=True)
        (sweep_root / "sweep_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def _matrix_values(
    matrix: Mapping[str, Any],
    roles: Mapping[str, Any],
    role: str,
) -> list[str]:
    value = matrix.get(role)
    if value is None:
        return [_required_string(roles, role)]
    if not isinstance(value, list) or not value:
        raise ValueError(f"matrix.{role}必须是非空数组")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"matrix.{role}只能包含非空模型名")
    return [str(item).strip() for item in value]


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key}必须是对象")
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key}必须是非空字符串")
    return value.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--character", required=True)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    result = run_model_sweep(
        args.experiment,
        world_dir=args.world,
        character_id=args.character,
        scenario_path=args.scenario,
        dry_run=args.dry_run,
        output_root=args.output_root,
        workers=args.workers,
        show_progress=not args.no_progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
