"""Complete one-character Stage 1-3 experiment with checkpointed outputs."""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from agents import (
    ConsistencyChecker,
    NPCAgent,
    PlayerAgent,
    PlayerMode,
    PromptBuilder,
    StateAccess,
    project_npc_character_card,
)
from agents.prompt_builder import PromptBundle
from datagen.audit.benchmark_assets import audit_executable_character
from datagen.shared.io import load_yaml, read_jsonl, write_jsonl
from datagen.shared.history_projection import project_frozen_history
from evaluation import (
    ActiveChallenge,
    ChallengeController,
    Stage3ContractEngine,
)
from gamecore import (
    ActionRegistry,
    FixtureEngine,
    GameContext,
    GameCoreEngine,
    WorldDefinition,
)
from llm import (
    APIClientError,
    GenerationConfig,
    LLMClient,
    StructuredOutputError,
    create_llm_client,
    generate_structured,
    load_llm_settings,
)

from .pilot import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    _pairwise_joint,
    _score_branch,
    _score_qa,
    _validate_branch_output,
    _validate_qa_partial_output,
    build_branch_bundle,
    build_qa_bundle,
)
from .progress import ProgressReporter

DEFAULT_STAGE3_SEEDS = (11, 29, 47)
STAGE3_PROTOCOL_VERSION = 10
STAGE3_ACTIONS = (
    "decide_claim",
    "update_task",
    "create_commitment",
    "resolve_commitment",
    "update_relationship",
    "decide_access",
    "reveal_fact",
    "move",
    "transfer_item",
    "create_artifact",
    "use_item",
    "attack",
    "end_dialogue",
)


def run_full_experiment(
    world_dir: str | Path,
    character_id: str,
    *,
    scenario_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    turns: int = 40,
    stage3_seeds: tuple[int, ...] = DEFAULT_STAGE3_SEEDS,
    player_model: str | None = None,
    checker_model: str | None = None,
    qa_batch_size: int | None = None,
    checker_interval: int = 5,
    dry_run: bool = False,
    output_root: str | Path = "runs",
    resume_dir: str | Path | None = None,
    workers: int = 3,
    show_progress: bool = True,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers必须至少为1")
    if checker_interval < 1:
        raise ValueError("checker_interval必须至少为1")
    world = Path(world_dir)
    scenario = load_yaml(scenario_path)
    if scenario["character_id"] != character_id:
        raise ValueError("scenario.character_id与实验角色不一致")
    audit = audit_executable_character(world, character_id)
    if not audit["ready_for_api_smoke_test"]:
        raise ValueError(f"数据未通过API审计: {audit['checks']}")
    _validate_scenario(world, character_id, scenario)
    settings = load_llm_settings(config_path, project_root=PROJECT_ROOT)
    config = settings.generation
    qa_batch_size = qa_batch_size or 25
    if qa_batch_size < 1:
        raise ValueError("qa_batch_size必须至少为1")
    player_model = player_model or config.model
    checker_model = checker_model or config.model
    if settings.provider == "venus" and {
        config.model,
        player_model,
        checker_model,
    } != {"deepseek-v4-pro"}:
        raise ValueError(
            "Venus实验的NPC、Player和Checker必须统一使用"
            "deepseek-v4-pro"
        )

    qa = read_jsonl(world / "frozen" / character_id / "qa.jsonl")
    open_tasks = read_jsonl(
        world / "frozen" / character_id / "open_tasks.jsonl"
    )
    pairs = read_jsonl(world / "branches" / character_id / "pairs.jsonl")
    qa_batches = list(_chunks(qa, qa_batch_size))
    registry = ActionRegistry.load_directory(
        PROJECT_ROOT / "gamecore" / "actions"
    )
    qa_bundles = [
        build_qa_bundle(world, character_id, batch)
        for batch in qa_batches
    ]
    branch_bundles = [
        build_branch_bundle(world, character_id, pair, branch, registry)
        for pair in pairs
        for branch in ("a", "b")
    ]
    checker_batches = (turns + checker_interval - 1) // checker_interval
    maximum_stage3_calls = len(stage3_seeds) * 2 * (
        turns * 2 + checker_batches * 2
    )
    maximum_calls_without_qa_recovery = (
        len(qa_batches)
        + len(open_tasks) * 2
        + len(branch_bundles)
        + maximum_stage3_calls
    )
    maximum_logical_calls = (
        len(qa)
        + len(open_tasks) * 2
        + len(branch_bundles)
        + maximum_stage3_calls
    )
    max_static_prompt = max(
        len(bundle.system_prompt) + len(bundle.user_prompt)
        for bundle in qa_bundles + branch_bundles
    )
    preflight = {
        "character_id": character_id,
        "stages": [1, 2, 3],
        "stage1": {
            "qa": len(qa),
            "qa_batches": len(qa_batches),
            "qa_batch_size": qa_batch_size,
            "open_tasks": len(open_tasks),
        },
        "stage2": {
            "pairs": len(pairs),
            "branch_calls": len(branch_bundles),
        },
        "stage3": {
            "conditions": ["normal", "pressure"],
            "turns_per_condition": turns,
            "seeds": list(stage3_seeds),
            "checker_interval": checker_interval,
        },
        "maximum_calls_without_qa_recovery": (
            maximum_calls_without_qa_recovery
        ),
        "maximum_logical_calls": maximum_logical_calls,
        "workers": workers,
        "models": {
            "npc": config.model,
            "player": player_model,
            "checker": checker_model,
        },
        "max_static_prompt_characters": max_static_prompt,
        "data_ready": True,
    }
    if dry_run:
        return {"dry_run": True, "preflight": preflight, "audit": audit}

    def client_factory() -> LLMClient:
        return create_llm_client(settings, scene="full_experiment")
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        if not run_dir.is_dir():
            raise ValueError(f"续跑目录不存在: {run_dir}")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (
            Path(output_root)
            / f"{timestamp}_{character_id}_{config.model}_stage123"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "run_config.json",
        {
            "world": str(world),
            "character_id": character_id,
            "scenario": str(scenario_path),
            "model": config.model,
            "base_url": settings.base_url,
            "qa_batch_size": qa_batch_size,
            "checker_interval": checker_interval,
            "stage3_protocol": STAGE3_PROTOCOL_VERSION,
            "preflight": preflight,
        },
    )
    progress = ProgressReporter(
        total=maximum_logical_calls,
        completed=min(
            _completed_logical_calls(run_dir),
            maximum_logical_calls,
        ),
        enabled=show_progress,
    )
    progress.render("resume" if progress.completed else "start")

    stage1 = _run_stage1(
        world,
        character_id,
        qa_batches,
        qa_bundles,
        open_tasks,
        client_factory(),
        config,
        run_dir,
        progress,
    )
    stage2 = _run_stage2(
        pairs,
        branch_bundles,
        registry,
        client_factory(),
        config,
        run_dir,
        progress,
    )
    stage3 = _run_stage3(
        world,
        character_id,
        scenario,
        registry,
        client_factory,
        config,
        turns,
        stage3_seeds,
        player_model,
        checker_model,
        checker_interval,
        run_dir,
        workers,
        progress,
    )
    summary = {
        "character_id": character_id,
        "model": config.model,
        "preflight": preflight,
        "usage": _usage_summary(run_dir),
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
    }
    _write_json(run_dir / "summary.json", summary)
    return {"dry_run": False, "run_dir": str(run_dir), "summary": summary}


def _run_stage1(
    world: Path,
    character_id: str,
    qa_batches: list[list[dict[str, Any]]],
    qa_bundles: list[PromptBundle],
    open_tasks: list[dict[str, Any]],
    client: LLMClient,
    config: GenerationConfig,
    run_dir: Path,
    progress: ProgressReporter,
) -> dict[str, Any]:
    qa_path = run_dir / "stage1_qa.jsonl"
    qa_records = read_jsonl(qa_path) if qa_path.is_file() else []
    completed_qa = {record["qa_id"] for record in qa_records}
    for batch, bundle in zip(qa_batches, qa_bundles, strict=True):
        pending_batch = [
            item for item in batch if item["qa_id"] not in completed_qa
        ]
        if not pending_batch:
            continue
        first_request = True
        while pending_batch:
            pending_bundle = (
                bundle
                if first_request and len(pending_batch) == len(batch)
                else build_qa_bundle(world, character_id, pending_batch)
            )
            output = generate_structured(
                client,
                pending_bundle,
                config,
                lambda value, expected=pending_batch: (
                    _validate_qa_partial_output(value, expected)
                ),
            )
            returned_ids = {
                answer["qa_id"] for answer in output["answers"]
            }
            returned_items = [
                item
                for item in pending_batch
                if item["qa_id"] in returned_ids
            ]
            qa_records.extend(
                _score_qa(returned_items, output, client.last_usage)
            )
            completed_qa.update(returned_ids)
            write_jsonl(qa_path, qa_records)
            pending_batch = [
                item
                for item in pending_batch
                if item["qa_id"] not in returned_ids
            ]
            first_request = False
            if pending_batch:
                progress.render(
                    "stage1 qa补问 "
                    f"remaining={len(pending_batch)}"
                )
        progress.advance(
            f"stage1 qa batch={batch[0]['qa_id']}"
        )

    open_path = run_dir / "stage1_open.jsonl"
    open_records = read_jsonl(open_path) if open_path.is_file() else []
    completed_open = {record["task_id"] for record in open_records}
    for task in open_tasks:
        if task["task_id"] in completed_open:
            continue
        bundle = _build_open_task_bundle(world, character_id, task)
        output = generate_structured(
            client,
            bundle,
            config,
            _validate_open_output,
        )
        progress.advance(f"stage1 open candidate={task['task_id']}")
        candidate_usage = copy.deepcopy(client.last_usage)
        judge_bundle = _build_open_judge_bundle(
            world,
            character_id,
            task,
            output,
        )
        judgment = generate_structured(
            client,
            judge_bundle,
            config,
            _validate_open_judgment,
        )
        progress.advance(f"stage1 open judge={task['task_id']}")
        open_records.append(
            {
                "task_id": task["task_id"],
                "output": output,
                "judgment": judgment,
                "candidate_usage": candidate_usage,
                "judge_usage": copy.deepcopy(client.last_usage),
            }
        )
        completed_open.add(task["task_id"])
        write_jsonl(open_path, open_records)

    category_correct: dict[str, list[bool]] = {}
    layer_correct: dict[str, list[bool]] = {}
    for record in qa_records:
        category_correct.setdefault(record["category"], []).append(
            record["answer_correct"]
        )
        layer = record.get("evaluation_layer", record["category"])
        layer_correct.setdefault(layer, []).append(
            record["answer_correct"]
        )
    primary_layers = {
        "checkpoint_state",
        "long_range_final",
        "multi_hop",
    }
    return {
        "qa_count": len(qa_records),
        "qa_accuracy": _mean(
            record["answer_correct"] for record in qa_records
        ),
        "evidence_f1": _mean(
            record["evidence_f1"] for record in qa_records
        ),
        "category_accuracy": {
            category: _mean(values)
            for category, values in category_correct.items()
        },
        "layer_accuracy": {
            layer: _mean(values)
            for layer, values in layer_correct.items()
        },
        "primary_long_range_accuracy": _mean(
            record["answer_correct"]
            for record in qa_records
            if record.get("evaluation_layer", record["category"])
            in primary_layers
        ),
        "open_task_count": len(open_records),
        "open_strict_pass_rate": _mean(
            all(
                record["judgment"].get(key, False)
                for key in (
                    "role_consistent",
                    "state_consistent",
                    "evidence_supported",
                    "grounded",
                    "interaction_value",
                )
            )
            for record in open_records
        ),
    }


def _run_stage2(
    pairs: list[dict[str, Any]],
    bundles: list[PromptBundle],
    registry: ActionRegistry,
    client: LLMClient,
    config: GenerationConfig,
    run_dir: Path,
    progress: ProgressReporter,
) -> dict[str, Any]:
    output_path = run_dir / "stage2_branches.jsonl"
    records = read_jsonl(output_path) if output_path.is_file() else []
    completed = {
        (record["pair_id"], record["branch"]) for record in records
    }
    bundle_index = 0
    for pair in pairs:
        for branch in ("a", "b"):
            bundle = bundles[bundle_index]
            bundle_index += 1
            if (pair["pair_id"], branch) in completed:
                continue
            output = generate_structured(
                client,
                bundle,
                config,
                lambda value: _validate_branch_output(value, registry),
            )
            progress.advance(
                f"stage2 pair={pair['pair_id']} branch={branch}"
            )
            records.append(
                _score_branch(
                    pair,
                    branch,
                    output,
                    copy.deepcopy(client.last_usage),
                )
            )
            completed.add((pair["pair_id"], branch))
            write_jsonl(output_path, records)
    sensitivity = [
        record
        for record in records
        if record["pair_type"] == "sensitivity"
    ]
    invariance_pairs = _invariance_consistency(records)
    return {
        "pair_count": len(pairs),
        "branch_count": len(records),
        "sensitivity_branch_action_accuracy": _mean(
            record["action_correct"] for record in sensitivity
        ),
        "sensitivity_pairwise_joint_accuracy": _pairwise_joint(sensitivity),
        "invariance_decision_consistency": _mean(invariance_pairs.values()),
        "invariance_state_accuracy": _mean(
            record.get("state_correct", False)
            for record in records
            if record["pair_type"] == "invariance"
        ),
    }


def _run_stage3(
    world: Path,
    character_id: str,
    scenario: dict[str, Any],
    registry: ActionRegistry,
    client_factory: Callable[[], LLMClient],
    config: GenerationConfig,
    turns: int,
    seeds: tuple[int, ...],
    player_model: str,
    checker_model: str,
    checker_interval: int,
    run_dir: Path,
    workers: int,
    progress: ProgressReporter,
    modes: tuple[PlayerMode, ...] = (
        PlayerMode.NORMAL,
        PlayerMode.PRESSURE,
    ),
) -> dict[str, Any]:
    episode_specs = [(seed, mode) for seed in seeds for mode in modes]
    if len(episode_specs) > 1:
        episodes: list[dict[str, Any]] = []
        provider_failures: list[dict[str, Any]] = []
        with ThreadPoolExecutor(
            max_workers=min(workers, len(episode_specs)),
            thread_name_prefix="stage3",
        ) as executor:
            futures = {
                executor.submit(
                    _run_stage3,
                    world,
                    character_id,
                    scenario,
                    registry,
                    client_factory,
                    config,
                    turns,
                    (seed,),
                    player_model,
                    checker_model,
                    checker_interval,
                    run_dir,
                    1,
                    progress,
                    (mode,),
                ): (seed, mode)
                for seed, mode in episode_specs
            }
            for future in as_completed(futures):
                seed, mode = futures[future]
                try:
                    result = future.result()
                except (APIClientError, StructuredOutputError) as error:
                    checkpoint_path = (
                        run_dir
                        / f"stage3_seed{seed}_{mode.value}_checkpoint.json"
                    )
                    completed_turns = 0
                    if checkpoint_path.is_file():
                        checkpoint = json.loads(
                            checkpoint_path.read_text(encoding="utf-8")
                        )
                        completed_turns = int(
                            checkpoint.get("completed_turns", 0)
                        )
                    detail = str(error)
                    if isinstance(error, StructuredOutputError):
                        failure_type = "structured_output_error"
                    elif "data_inspection_failed" in detail:
                        failure_type = "provider_content_filter"
                    else:
                        failure_type = "provider_api_error"
                    failure = {
                        "seed": seed,
                        "mode": mode.value,
                        "completed_turns": completed_turns,
                        "next_turn": completed_turns + 1,
                        "type": failure_type,
                        "error": detail,
                    }
                    _write_json(
                        run_dir
                        / f"stage3_seed{seed}_{mode.value}_provider_error.json",
                        failure,
                    )
                    provider_failures.append(failure)
                    continue
                error_path = (
                    run_dir
                    / f"stage3_seed{seed}_{mode.value}_provider_error.json"
                )
                if error_path.is_file():
                    error_path.unlink()
                episodes.append(result["seeds"][str(seed)][mode.value])
        summary = _summarize_stage3(episodes, seeds, turns)
        all_episodes_present = len(episodes) == len(episode_specs)
        summary["complete"] = all_episodes_present
        summary["all_episodes_present"] = all_episodes_present
        summary["protocol_horizon_complete"] = (
            all_episodes_present
            and all(
                episode.get("protocol_horizon_complete", False)
                for episode in episodes
            )
        )
        summary["protocol_evaluation_complete"] = (
            all_episodes_present
            and all(
                episode.get("protocol_evaluation_complete", False)
                for episode in episodes
            )
        )
        summary["provider_failures"] = provider_failures
        return summary

    client = client_factory()
    world_definition = WorldDefinition.load_yaml(world / "environment.yaml")
    engine = GameCoreEngine(registry, world_definition)
    fixture_engine = FixtureEngine(world_definition)
    prompt_builder = PromptBuilder.from_environment(
        registry,
        world / "environment.yaml",
    )
    card = load_yaml(world / "characters" / f"{character_id}.yaml")
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
      for mode in modes:
        contract_engine = Stage3ContractEngine(
            scenario["stage3_contract"],
            action_names=STAGE3_ACTIONS,
        )
        contract_ids = {
            str(item["id"])
            for section in ("transition_contracts", "trace_contracts")
            for item in scenario["stage3_contract"].get(section, [])
        }
        challenge_controller = ChallengeController(
            scenario["challenge_plan"],
            mode=mode.value,
            contract_ids=contract_ids,
        )
        episode_config = GenerationConfig(
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=min(config.max_tokens, 1024),
            seed=seed,
            max_format_retries=max(config.max_format_retries, 4),
        )
        player_config = GenerationConfig(
            model=player_model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=min(config.max_tokens, 512),
            seed=seed,
            max_format_retries=max(config.max_format_retries, 4),
        )
        final_path = run_dir / f"stage3_seed{seed}_{mode.value}.json"
        if final_path.is_file():
            completed_payload = json.loads(
                final_path.read_text(encoding="utf-8")
            )
            completed_summary = completed_payload.get("summary", {})
            if (
                completed_summary.get("protocol_version")
                != STAGE3_PROTOCOL_VERSION
            ):
                raise ValueError(
                    f"{final_path}使用旧版Stage3协议，不能混入新版实验"
                )
            if completed_summary.get("status") != "invalid":
                episodes.append(completed_summary)
                continue
        checkpoint_path = (
            run_dir / f"stage3_seed{seed}_{mode.value}_checkpoint.json"
        )
        if checkpoint_path.is_file():
            checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            if (
                checkpoint.get("protocol_version")
                != STAGE3_PROTOCOL_VERSION
            ):
                raise ValueError(
                    f"{checkpoint_path}使用旧版Stage3协议，请新建运行目录"
                )
            context = GameContext(checkpoint["context"])
            call_log = checkpoint["call_log"]
            accepted_turns = int(checkpoint["accepted_turns"])
            audits = list(checkpoint.get("audits", []))
            pending_checker_turns = list(
                checkpoint.get("pending_checker_turns", [])
            )
            contract_runtime = copy.deepcopy(
                checkpoint["contract_runtime"]
            )
            challenge_runtime = copy.deepcopy(
                checkpoint["challenge_runtime"]
            )
            resumed_status = str(checkpoint.get("status", "running"))
            completed_turn = int(checkpoint["completed_turns"])
            recoverable = [
                audit
                for audit in audits
                if audit.get("source")
                in {"invalid_output", "invalid_checker_output"}
            ]
            if resumed_status in {"failure", "invalid"} and recoverable:
                last = recoverable[-1]
                source = last["source"]
                if source == "invalid_checker_output":
                    failed_turns = {
                        int(audit["turn"])
                        for audit in recoverable
                        if audit.get("source") == source
                    }
                    pending_checker_turns = sorted(
                        set(pending_checker_turns) | failed_turns
                    )
                    audits = [
                        audit
                        for audit in audits
                        if audit.get("source") != source
                    ]
                    first_turn = completed_turn + 1
                else:
                    failed_turn = int(last["turn"])
                    history = context.data["history"]
                    while (
                        history
                        and int(history[-1]["turn"]) == failed_turn
                        and history[-1]["speaker"] in {"npc", "gamecore"}
                    ):
                        history.pop()
                    audits = [
                        audit
                        for audit in audits
                        if not (
                            int(audit["turn"]) == failed_turn
                            and audit.get("source") == source
                        )
                    ]
                    first_turn = failed_turn
                resumed_status = "running"
            else:
                first_turn = (
                    turns + 1
                    if resumed_status in {"failure", "invalid"}
                    else completed_turn + 1
                )
        else:
            context = GameContext(
                {
                    "character_card": card,
                    **copy.deepcopy(scenario["initial_context"]),
                    "history": [],
                }
            )
            call_log = []
            accepted_turns = 0
            audits = []
            pending_checker_turns = []
            contract_runtime = contract_engine.initial_runtime()
            challenge_runtime = challenge_controller.initial_runtime()
            first_turn = 1
            resumed_status = "running"
        player = PlayerAgent(
            client=client,
            config=player_config,
            prompt_builder=prompt_builder,
            mode=mode,
            player_profile=scenario.get("player_profile"),
        )
        npc = NPCAgent(
            client=client,
            config=episode_config,
            prompt_builder=prompt_builder,
            state_access=StateAccess.HISTORY_ONLY,
        )
        checker = ConsistencyChecker(
            client=client,
            config=GenerationConfig(
                model=checker_model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=min(config.max_tokens, 384),
                seed=seed,
                max_format_retries=max(config.max_format_retries, 4),
            ),
            prompt_builder=prompt_builder,
        )
        terminal_status = resumed_status
        if (
            terminal_status == "running"
            and len(pending_checker_turns) >= checker_interval
        ):
            (
                resumed_audits,
                resumed_checker_calls,
                _checker_status,
            ) = _audit_checker_window(
                checker,
                client,
                context,
                scenario["role_contract"],
                pending_checker_turns,
            )
            audits.extend(resumed_audits)
            for checker_call in resumed_checker_calls:
                progress.advance(
                    f"stage3 seed={seed} mode={mode.value} "
                    f"turns={pending_checker_turns} "
                    f"{checker_call['phase']}"
                )
                call_log.append(checker_call)
            pending_checker_turns = []
        for turn in range(first_turn, turns + 1):
            active_challenge = challenge_controller.current(
                challenge_runtime
            )
            active_challenge_id = (
                active_challenge.challenge_id
                if active_challenge is not None
                else None
            )
            player_already_persisted = bool(
                context.history
                and context.history[-1]["speaker"] == "player"
                and int(context.history[-1]["turn"]) == turn
            )
            if not player_already_persisted:
                if (
                    active_challenge is not None
                    and active_challenge.challenge_id
                    not in challenge_runtime.get("started", [])
                    and active_challenge.fixture_event is not None
                ):
                    fixture = active_challenge.fixture_event
                    context = fixture_engine.apply(
                        context,
                        event_id=str(fixture["event_id"]),
                        operations=fixture["operations"],
                        description=str(fixture["content"]),
                        history_turn=max(context.next_turn - 1, 1),
                    ).context
                if active_challenge is not None:
                    challenge_runtime = (
                        challenge_controller.mark_started(
                            challenge_runtime,
                            active_challenge.challenge_id,
                        )
                    )
                player_correction: str | None = None
                player_output: dict[str, str] | None = None
                for correction_attempt in range(3):
                    player_error: StructuredOutputError | None = None
                    try:
                        player_output = player.generate(
                            context,
                            _scenario_for_player_turn(
                                scenario,
                                mode,
                                active_challenge,
                            ),
                            correction=player_correction,
                        )
                    except StructuredOutputError as error:
                        player_error = error
                    progress.advance(
                        f"stage3 seed={seed} mode={mode.value} "
                        f"turn={turn}/{turns} player"
                    )
                    call_log.append(
                        {
                            "turn": turn,
                            "agent": "player",
                            "correction_attempt": correction_attempt,
                            "usage": copy.deepcopy(client.last_usage),
                            "internal_format_retries": getattr(
                                client, "last_format_retries", 0
                            ),
                            "internal_format_errors": copy.deepcopy(
                                getattr(client, "last_format_errors", [])
                            ),
                            "format_error": str(player_error)
                            if player_error is not None
                            else None,
                            "raw_output": player_error.raw_output
                            if player_error is not None
                            else None,
                            "output": copy.deepcopy(player_output)
                            if player_error is None
                            else None,
                            "challenge_id": active_challenge_id,
                            "challenge_objective": (
                                active_challenge.objective
                                if active_challenge is not None
                                else None
                            ),
                        }
                    )
                    if player_error is None:
                        break
                    player_correction = (
                        f"{player_error}；原始输出="
                        f"{player_error.raw_output!r}"
                    )
                if player_output is None:
                    raise StructuredOutputError(
                        f"第{turn}轮Player输出经有限纠错后仍不合法"
                    )
                context = engine.append_player_query(
                    context,
                    player_output["query"],
                )
                _write_json(
                    checkpoint_path,
                    {
                        "protocol_version": STAGE3_PROTOCOL_VERSION,
                        "completed_turns": turn - 1,
                        "accepted_turns": accepted_turns,
                        "context": context.to_dict(),
                        "call_log": call_log,
                        "audits": audits,
                        "pending_checker_turns": pending_checker_turns,
                        "contract_runtime": contract_runtime,
                        "challenge_runtime": challenge_runtime,
                        "status": "running",
                    },
                )
            correction: str | None = None
            transition = None
            npc_output: dict[str, Any] = {}
            before_transition = context
            for correction_attempt in range(3):
                npc_format_error: StructuredOutputError | None = None
                try:
                    npc_output = npc.generate(
                        context,
                        scenario,
                        available_actions=scenario["available_actions"],
                        correction=correction,
                    )
                except StructuredOutputError as error:
                    npc_format_error = error
                progress.advance(
                    f"stage3 seed={seed} mode={mode.value} "
                    f"turn={turn}/{turns} npc"
                )
                npc_call = {
                    "turn": turn,
                    "agent": "npc",
                    "correction_attempt": correction_attempt,
                    "usage": copy.deepcopy(client.last_usage),
                    "internal_format_retries": getattr(
                        client, "last_format_retries", 0
                    ),
                    "internal_format_errors": copy.deepcopy(
                        getattr(client, "last_format_errors", [])
                    ),
                    "format_error": str(npc_format_error)
                    if npc_format_error is not None
                    else None,
                    "raw_output": npc_format_error.raw_output
                    if npc_format_error is not None
                    else None,
                    "output": copy.deepcopy(npc_output)
                    if npc_format_error is None
                    else None,
                    "action_error": None,
                }
                call_log.append(npc_call)
                if npc_format_error is not None:
                    correction = (
                        f"{npc_format_error}；原始输出="
                        f"{npc_format_error.raw_output!r}"
                    )
                    continue
                transition = engine.step(
                    context,
                    npc_output,
                    allowed_actions=scenario["available_actions"],
                )
                if transition.accepted:
                    break
                violation = transition.violations[0]
                npc_call["action_error"] = copy.deepcopy(violation)
                npc_call["action_errors"] = copy.deepcopy(
                    list(transition.violations)
                )
                if not (
                    transition.preflight_failed
                    and _is_format_violation(str(violation["code"]))
                ):
                    break
                correction = (
                    f"GameCore拒绝Action：{violation['code']}，"
                    f"{violation['message']}；上次输出="
                    f"{json.dumps(npc_output, ensure_ascii=False)}"
                )
            if transition is None:
                raise StructuredOutputError(
                    f"第{turn}轮NPC输出经有限纠错后仍无法执行",
                    raw_output=json.dumps(npc_output, ensure_ascii=False),
                )
            context = transition.context
            accepted_turns += int(transition.accepted)
            new_audits: list[dict[str, Any]] = []
            format_invalid = transition.preflight_failed and _is_format_violation(
                str(transition.violations[0]["code"])
            )
            for violation in transition.violations:
                code = str(violation["code"])
                rejected_action = str(violation.get("action") or "")
                if code == "claim_decision_unchanged":
                    new_audits.append(
                        {
                            "turn": turn,
                            "failure": False,
                            "failure_type": None,
                            "violated_rule": None,
                            "evidence": [
                                "重复提交未改变判断及证据依据的decide_claim"
                            ],
                            "cannot_determine": False,
                            "source": "redundant_action",
                        }
                    )
                    continue
                new_audits.append(
                    {
                        "turn": turn,
                        "failure": not format_invalid,
                        "failure_type": (
                            None if format_invalid else "illegal_transition"
                        ),
                        "violated_rule": (
                            None if format_invalid else f"gamecore.{code}"
                        ),
                        "evidence": [
                            f"第{turn}轮GameCore拒绝Action"
                            + (f" {rejected_action}" if rejected_action else "")
                            + f"：{code}"
                        ],
                        "cannot_determine": False,
                        "source": (
                            "invalid_output" if format_invalid else "gamecore"
                        ),
                    }
                )
            if format_invalid:
                terminal_status = "invalid"
            if not transition.preflight_failed:
                contract_step = contract_engine.evaluate(
                    before_transition,
                    context,
                    turn=turn,
                    mode=mode.value,
                    active_challenge_id=active_challenge_id,
                    runtime=contract_runtime,
                )
                contract_runtime = contract_step.runtime
                new_audits.extend(contract_step.diagnostics)
                new_audits.extend(contract_step.violations)
                challenge_runtime = (
                    challenge_controller.advance_after_turn(
                        challenge_runtime,
                        contract_runtime,
                    )
                )
                pending_checker_turns.append(turn)
                should_check = (
                    len(pending_checker_turns) >= checker_interval
                    or turn == turns
                    or context.environment["dialogue_status"] == "ended"
                )
                if should_check:
                    (
                        window_audits,
                        checker_calls,
                        _checker_status,
                    ) = _audit_checker_window(
                        checker,
                        client,
                        context,
                        scenario["role_contract"],
                        pending_checker_turns,
                    )
                    new_audits.extend(window_audits)
                    for checker_call in checker_calls:
                        progress.advance(
                            f"stage3 seed={seed} mode={mode.value} "
                            f"turns={pending_checker_turns} "
                            f"{checker_call['phase']}"
                        )
                        call_log.append(checker_call)
                    pending_checker_turns = []
            audits.extend(new_audits)
            _write_json(
                checkpoint_path,
                {
                    "protocol_version": STAGE3_PROTOCOL_VERSION,
                    "completed_turns": turn,
                    "accepted_turns": accepted_turns,
                    "context": context.to_dict(),
                    "call_log": call_log,
                    "audits": audits,
                    "pending_checker_turns": pending_checker_turns,
                    "contract_runtime": contract_runtime,
                    "challenge_runtime": challenge_runtime,
                    "status": terminal_status,
                },
            )
            if terminal_status == "invalid":
                break
            if context.environment["dialogue_status"] == "ended":
                break

        if terminal_status == "running" and pending_checker_turns:
            (
                window_audits,
                checker_calls,
                _checker_status,
            ) = _audit_checker_window(
                checker,
                client,
                context,
                scenario["role_contract"],
                pending_checker_turns,
            )
            audits.extend(window_audits)
            for checker_call in checker_calls:
                progress.advance(
                    f"stage3 seed={seed} mode={mode.value} "
                    f"turns={pending_checker_turns} "
                    f"{checker_call['phase']}"
                )
                call_log.append(checker_call)
            pending_checker_turns = []
            _write_json(
                checkpoint_path,
                {
                    "protocol_version": STAGE3_PROTOCOL_VERSION,
                    "completed_turns": turns,
                    "accepted_turns": accepted_turns,
                    "context": context.to_dict(),
                    "call_log": call_log,
                    "audits": audits,
                    "pending_checker_turns": pending_checker_turns,
                    "contract_runtime": contract_runtime,
                    "challenge_runtime": challenge_runtime,
                    "status": terminal_status,
                },
            )

        if terminal_status == "running":
            completed_turns = len(
                {
                    int(entry["turn"])
                    for entry in context.history
                    if entry["speaker"] == "npc"
                }
            )
            protocol_horizon = int(
                context.environment.get("interaction_deadline", turns)
            )
            if context.environment["dialogue_status"] == "ended":
                terminal_status = "completed"
            elif completed_turns < protocol_horizon:
                terminal_status = "partial"
            else:
                terminal_status = "survived"
        episode = _stage3_episode_summary_v3(
            mode.value,
            context,
            terminal_status,
            audits,
            accepted_turns,
            turns,
            call_log=call_log,
            contract_runtime=contract_runtime,
            challenge_coverage=challenge_controller.coverage(
                challenge_runtime
            ),
        )
        episode["seed"] = seed
        _write_json(
            final_path,
            {
                "summary": episode,
                "context": context.to_dict(),
                "audits": audits,
                "call_log": call_log,
                "contract_runtime": contract_runtime,
                "challenge_runtime": challenge_runtime,
            },
        )
        episodes.append(episode)
    return _summarize_stage3(episodes, seeds, turns)


def _summarize_stage3(
    episodes: list[dict[str, Any]],
    seeds: tuple[int, ...],
    turns: int,
) -> dict[str, Any]:
    by_seed = {
        str(seed): {
            episode["mode"]: episode
            for episode in episodes
            if episode["seed"] == seed
        }
        for seed in seeds
    }
    return {
        "seeds": by_seed,
        "survival": _stage3_survival_v2(episodes, turns),
        "failure_rate": _mean(
            episode["failure"]
            for episode in episodes
            if episode["status"] != "invalid"
        ),
        "formal_failure_rate": _mean(
            episode["failure"]
            for episode in episodes
            if episode["status"] != "invalid"
            and episode.get("protocol_evaluation_complete", False)
        ),
        "failure_rate_by_type": {
            failure_type: _mean(
                episode.get("failure_type") == failure_type
                for episode in episodes
                if episode["status"] != "invalid"
                and episode.get("protocol_evaluation_complete", False)
            )
            for failure_type in (
                "illegal_transition",
                "missing_transition",
                "trajectory_conflict",
            )
        },
        "partial_trajectories": sum(
            episode["status"] == "partial" for episode in episodes
        ),
        "insufficient_coverage_trajectories": sum(
            episode["status"] == "insufficient_coverage"
            for episode in episodes
        ),
        "invalid_trajectories": sum(
            episode["status"] == "invalid" for episode in episodes
        ),
    }


def _audit_checker_window(
    checker: ConsistencyChecker,
    client: LLMClient,
    context: GameContext,
    role_contract: Mapping[str, Any],
    turns: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    checked_turns = list(turns)
    calls: list[dict[str, Any]] = []
    try:
        initial = checker.check_batch(
            context,
            role_contract,
            checked_turns,
        )
    except StructuredOutputError as error:
        calls.append(
            _checker_call_record(
                checked_turns,
                "initial",
                client,
                error,
            )
        )
        initial = []
        for turn in checked_turns:
            initial.extend(
                checker.check_batch(context, role_contract, [turn])
            )
            calls.append(
                _checker_call_record(
                    [turn],
                    "initial_single_recovery",
                    client,
                )
            )
    calls.append(
        _checker_call_record(checked_turns, "initial", client)
    )
    if not any(item["failure"] for item in initial):
        return (
            [
                {
                    **item,
                    "failure_type": (
                        "trajectory_conflict"
                        if item["failure"]
                        else None
                    ),
                    "source": "checker",
                }
                for item in initial
            ],
            calls,
            "running",
        )

    try:
        confirmation = checker.check_batch(
            context,
            role_contract,
            checked_turns,
            prior_judgment=initial,
        )
    except StructuredOutputError as error:
        calls.append(
            _checker_call_record(
                checked_turns,
                "confirmation",
                client,
                error,
            )
        )
        confirmation = []
        for turn in checked_turns:
            prior = [
                item for item in initial if item["turn"] == turn
            ]
            confirmation.extend(
                checker.check_batch(
                    context,
                    role_contract,
                    [turn],
                    prior_judgment=prior,
                )
            )
            calls.append(
                _checker_call_record(
                    [turn],
                    "confirmation_single_recovery",
                    client,
                )
            )
    calls.append(
        _checker_call_record(checked_turns, "confirmation", client)
    )
    confirmation_by_turn = {
        item["turn"]: item for item in confirmation
    }
    merged: list[dict[str, Any]] = []
    for item in initial:
        confirmed = confirmation_by_turn[item["turn"]]
        if item["failure"] and confirmed["failure"]:
            merged.append(
                {
                    **confirmed,
                    "failure_type": "trajectory_conflict",
                    "source": "checker_confirmed",
                    "initial_judgment": item,
                }
            )
        elif item["failure"]:
            merged.append(
                {
                    "turn": item["turn"],
                    "failure": False,
                    "failure_type": None,
                    "violated_rule": None,
                    "evidence": confirmed["evidence"],
                    "cannot_determine": confirmed[
                        "cannot_determine"
                    ],
                    "source": "checker_disagreed",
                    "initial_judgment": item,
                }
            )
        else:
            merged.append(
                {
                    **item,
                    "failure_type": None,
                    "source": "checker",
                }
            )
    status = (
        "failure"
        if any(item["failure"] for item in merged)
        else "running"
    )
    return merged, calls, status


def _checker_call_record(
    turns: list[int],
    phase: str,
    client: LLMClient,
    error: StructuredOutputError | None = None,
) -> dict[str, Any]:
    return {
        "turn": None,
        "turns": list(turns),
        "agent": "checker",
        "phase": phase,
        "usage": copy.deepcopy(client.last_usage),
        "internal_format_retries": getattr(
            client, "last_format_retries", 0
        ),
        "internal_format_errors": copy.deepcopy(
            getattr(client, "last_format_errors", [])
        ),
        "format_error": str(error) if error is not None else None,
        "raw_output": error.raw_output if error is not None else None,
    }


def _stage3_episode_summary_v3(
    mode: str,
    context: GameContext,
    status: str,
    audits: list[dict[str, Any]],
    accepted_turns: int,
    expected_turns: int,
    *,
    call_log: Iterable[Mapping[str, Any]] = (),
    contract_runtime: Mapping[str, Any] | None = None,
    challenge_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed_turns = len(
        {
            int(entry["turn"])
            for entry in context.history
            if entry["speaker"] == "npc"
        }
    )
    failures = [audit for audit in audits if audit["failure"]]
    first_failure = (
        min(failures, key=lambda audit: int(audit["turn"]))
        if failures
        else None
    )
    npc_entries = [
        entry
        for entry in context.history
        if entry.get("speaker") == "npc"
    ]
    nonempty_action_turns = sum(
        bool(entry.get("actions")) for entry in npc_entries
    )
    distinct_actions = sorted(
        {
            str(action["name"])
            for entry in npc_entries
            for action in entry.get("actions", [])
            if isinstance(action, Mapping) and action.get("name")
        }
    )
    npc_attempts = [
        item for item in call_log if item.get("agent") == "npc"
    ]
    action_rejections = [
        error
        for item in npc_attempts
        for error in (
            item.get("action_errors")
            or ([item["action_error"]] if item.get("action_error") else [])
        )
        if isinstance(error, Mapping)
        and not _is_format_violation(str(error.get("code", "")))
    ]
    redundant_actions = [
        error
        for error in action_rejections
        if error.get("code") == "claim_decision_unchanged"
    ]
    semantic_rejections = [
        error
        for error in action_rejections
        if error.get("code") != "claim_decision_unchanged"
    ]
    protocol_horizon = int(
        context.environment.get("interaction_deadline", expected_turns)
    )
    coverage = dict(
        challenge_coverage
        or {
            "total_challenges": 0,
            "completed_challenges": [],
            "coverage": 0.0,
        }
    )
    contract_states = dict(
        (contract_runtime or {}).get("contracts", {})
    )
    contract_status = {
        value: sum(
            state.get("status") == value
            for state in contract_states.values()
        )
        for value in ("inactive", "active", "satisfied", "violated")
    }
    if (
        status in {"survived", "completed", "partial"}
        and coverage["coverage"] < 1.0
        and first_failure is None
    ):
        status = "insufficient_coverage"
    return {
        "protocol_version": STAGE3_PROTOCOL_VERSION,
        "mode": mode,
        "status": status,
        "completed_turns": completed_turns,
        "target_turns": expected_turns,
        "protocol_horizon": protocol_horizon,
        "protocol_horizon_complete": completed_turns >= protocol_horizon,
        "protocol_evaluation_complete": (
            first_failure is not None or coverage["coverage"] >= 1.0
        ),
        "failure": first_failure is not None,
        "failure_type": first_failure.get("failure_type")
        if first_failure is not None
        else None,
        "time_to_first_failure": first_failure["turn"]
        if first_failure is not None
        else None,
        "violated_rule": first_failure["violated_rule"]
        if first_failure is not None
        else None,
        "failure_evidence": first_failure["evidence"]
        if first_failure is not None
        else [],
        "failure_source": first_failure["source"]
        if first_failure is not None
        else None,
        "failure_count": len(failures),
        "failure_counts_by_type": {
            failure_type: sum(
                audit.get("failure_type") == failure_type
                for audit in failures
            )
            for failure_type in (
                "illegal_transition",
                "missing_transition",
                "trajectory_conflict",
            )
        },
        "failure_counts_by_source": {
            source: sum(
                audit.get("source") == source for audit in failures
            )
            for source in sorted(
                {str(audit.get("source")) for audit in failures}
            )
        },
        "failure_records": [
            {
                "turn": audit["turn"],
                "failure_type": audit.get("failure_type"),
                "violated_rule": audit.get("violated_rule"),
                "source": audit.get("source"),
            }
            for audit in sorted(
                failures, key=lambda item: int(item["turn"])
            )
        ],
        "accepted_transition_rate": (
            accepted_turns / completed_turns if completed_turns else 0.0
        ),
        "nonempty_action_turn_rate": (
            nonempty_action_turns / completed_turns
            if completed_turns
            else 0.0
        ),
        "semantic_rejection_rate": (
            len(semantic_rejections) / len(npc_attempts)
            if npc_attempts
            else 0.0
        ),
        "semantic_rejection_count": len(semantic_rejections),
        "redundant_action_count": len(redundant_actions),
        "distinct_actions": distinct_actions,
        "distinct_action_coverage": len(distinct_actions)
        / len(STAGE3_ACTIONS),
        "contract_status": contract_status,
        "challenge_coverage": coverage,
    }


def _stage3_survival_v2(
    episodes: list[dict[str, Any]],
    max_turns: int,
) -> dict[str, Any]:
    checkpoints = [
        value for value in (10, 20, 30, 40) if value <= max_turns
    ]
    result: dict[str, Any] = {}
    for mode in ("normal", "pressure"):
        eligible = [
            episode
            for episode in episodes
            if episode["mode"] == mode and episode["status"] != "invalid"
        ]
        result[mode] = {
            "eligible_trajectories": len(eligible),
            "survival": {
                str(checkpoint): _mean(
                    episode["time_to_first_failure"] is None
                    or episode["time_to_first_failure"] > checkpoint
                    for episode in eligible
                    if (
                        int(episode.get("completed_turns", max_turns))
                        >= checkpoint
                        or (
                            episode["time_to_first_failure"] is not None
                            and episode["time_to_first_failure"]
                            <= checkpoint
                        )
                    )
                )
                for checkpoint in checkpoints
            },
            "eligible_at_checkpoint": {
                str(checkpoint): sum(
                    int(episode.get("completed_turns", max_turns))
                    >= checkpoint
                    or (
                        episode["time_to_first_failure"] is not None
                        and episode["time_to_first_failure"] <= checkpoint
                    )
                    for episode in eligible
                )
                for checkpoint in checkpoints
            },
        }
    return result


def _is_format_violation(code: str) -> bool:
    return any(
        token in code
        for token in (
            "shape",
            "parameter",
            "unknown_action",
            "action_not_available",
            "too_many",
            "utterance",
            "invalid_actions",
        )
    )


def _build_open_task_bundle(
    world: Path,
    character_id: str,
    task: Mapping[str, Any],
) -> PromptBundle:
    card = project_npc_character_card(
        load_yaml(world / "characters" / f"{character_id}.yaml")
    )
    history = read_jsonl(
        world / "frozen" / character_id / "history.jsonl"
    )
    language = load_yaml(world / "environment.yaml").get("language", "zh")
    projected = project_frozen_history(history, audience="npc")
    if language == "en":
        return PromptBundle(
            system_prompt=(
                "You are the evaluated NPC. Respond in character to the final "
                "task using only the supplied character card and complete "
                "history. Do not output reasoning or scoring rules. Output one "
                "JSON object only."
            ),
            user_prompt=(
                f"Character card:\n{json.dumps(card, ensure_ascii=False)}\n\n"
                f"Complete {len(history)}-round history:\n{projected}\n\n"
                f"Task: {task['prompt']}\n\n"
                'Output: {"utterance":"in-character response",'
                '"state_basis":["anchor_..."]}'
            ),
        )
    return PromptBundle(
        system_prompt=(
            "你是被测NPC。只能依据给定角色卡和完整历史，以角色身份回应最后任务。"
            "不得输出思维过程或评分规则，只输出一个JSON对象。"
        ),
        user_prompt=(
            f"角色卡：\n{json.dumps(card, ensure_ascii=False)}\n\n"
            f"完整{len(history)}轮历史：\n"
            f"{projected}\n\n"
            f"任务：{task['prompt']}\n\n"
            '输出格式：{"utterance":"角色化回复",'
            '"state_basis":["anchor_..."]}'
        ),
    )


def _build_open_judge_bundle(
    world: Path,
    character_id: str,
    task: Mapping[str, Any],
    output: Mapping[str, Any],
) -> PromptBundle:
    card = load_yaml(world / "characters" / f"{character_id}.yaml")
    history = read_jsonl(
        world / "frozen" / character_id / "history.jsonl"
    )
    language = load_yaml(world / "environment.yaml").get("language", "zh")
    projected = project_frozen_history(history, audience="npc")
    if language == "en":
        return PromptBundle(
            system_prompt=(
                "You are the frozen RPG-AgentBench open-response auditor. Make "
                "discrete judgments only from the character card, history, and "
                "EvaluationSpec. A justified refusal or safe alternative can "
                "have interaction value. grounded must be false if the candidate "
                "invents a person, place, item, permission, piece of evidence, "
                "procedure, or device absent from the history. Output JSON only "
                "without reasoning."
            ),
            user_prompt=(
                f"Character card:\n{json.dumps(card, ensure_ascii=False)}\n\n"
                f"Complete {len(history)}-round history:\n{projected}\n\n"
                f"Task: {task['prompt']}\n"
                "EvaluationSpec: "
                f"{json.dumps(task['evaluation_spec'], ensure_ascii=False)}\n"
                f"Candidate: {json.dumps(output, ensure_ascii=False)}\n\n"
                'Output: {"role_consistent":true,"state_consistent":true,'
                '"evidence_supported":true,"grounded":true,'
                '"interaction_value":true,"evidence":["specific evidence"]}'
            ),
        )
    return PromptBundle(
        system_prompt=(
            "你是冻结的RPG-AgentBench开放回复审计器。只根据给定角色卡、"
            "历史和EvaluationSpec作离散判断；合理拒绝和安全替代方案应视为"
            "有互动价值，但候选输出新增历史中不存在的人物、地点、物品、"
            "证据、程序或设备时grounded必须为false。只输出JSON，不输出"
            "思维过程。"
        ),
        user_prompt=(
            f"角色卡：\n{json.dumps(card, ensure_ascii=False)}\n\n"
            f"完整{len(history)}轮历史：\n"
            f"{projected}\n\n"
            f"任务：{task['prompt']}\n"
            f"EvaluationSpec：{json.dumps(task['evaluation_spec'], ensure_ascii=False)}\n"
            f"候选输出：{json.dumps(output, ensure_ascii=False)}\n\n"
            "输出格式："
            '{"role_consistent":true,"state_consistent":true,'
            '"evidence_supported":true,"grounded":true,'
            '"interaction_value":true,'
            '"evidence":["具体依据"]}'
        ),
    )


def _validate_open_output(output: dict[str, Any]) -> None:
    if set(output) != {"utterance", "state_basis"}:
        raise ValueError("开放任务输出字段错误")
    if not isinstance(output["utterance"], str) or not output["utterance"].strip():
        raise ValueError("开放任务utterance不能为空")
    if not isinstance(output["state_basis"], list) or not all(
        isinstance(item, str) for item in output["state_basis"]
    ):
        raise ValueError("state_basis必须是字符串数组")


def _validate_open_judgment(output: dict[str, Any]) -> None:
    required = {
        "role_consistent",
        "state_consistent",
        "evidence_supported",
        "grounded",
        "interaction_value",
        "evidence",
    }
    if set(output) != required:
        raise ValueError("开放任务审计字段错误")
    if not all(
        isinstance(output[key], bool)
        for key in required - {"evidence"}
    ):
        raise ValueError("开放任务审计标签必须是布尔值")
    if not isinstance(output["evidence"], list) or not all(
        isinstance(item, str) for item in output["evidence"]
    ):
        raise ValueError("开放任务审计evidence必须是字符串数组")


def _scenario_for_player_turn(
    scenario: Mapping[str, Any],
    mode: PlayerMode,
    challenge: ActiveChallenge | None,
) -> dict[str, Any]:
    projected = copy.deepcopy(dict(scenario))
    goal_key = f"{mode.value}_player_goal"
    goal = copy.deepcopy(projected[goal_key])
    if isinstance(goal, Mapping):
        goal = dict(goal)
        goal["current_challenge"] = (
            {
                "id": challenge.challenge_id,
                "objective": challenge.objective,
            }
            if challenge is not None
            else {
                "id": "open_interaction",
                "objective": str(goal.get("objective", "")),
            }
        )
    projected[goal_key] = goal
    return projected


def _validate_scenario(
    world: Path,
    character_id: str,
    scenario: Mapping[str, Any],
) -> None:
    required = {
        "scenario_id",
        "character_id",
        "setting",
        "normal_player_goal",
        "pressure_player_goal",
        "available_actions",
        "initial_context",
        "role_contract",
        "stage3_contract",
        "challenge_plan",
    }
    missing = required - set(scenario)
    if missing:
        raise ValueError(f"Stage3场景缺少字段: {sorted(missing)}")
    forbidden_setting_terms = (
        "请求目标",
        "攻击目标",
        "最高机密",
        "EvaluationSpec",
        "Consistency Checker",
    )
    setting = str(scenario["setting"])
    leaked_terms = [
        term for term in forbidden_setting_terms if term in setting
    ]
    if leaked_terms:
        raise ValueError(
            f"Stage3中性setting包含评测提示: {leaked_terms}"
        )
    if "action_targets" in scenario:
        raise ValueError(
            "Stage3场景不得使用action_targets提示NPC；请使用动态可执行参数"
        )
    card = load_yaml(world / "characters" / f"{character_id}.yaml")
    context = GameContext(
        {
            "character_card": card,
            **copy.deepcopy(scenario["initial_context"]),
            "history": [],
        }
    )
    context.validate()
    registry = ActionRegistry.load_directory(
        PROJECT_ROOT / "gamecore" / "actions"
    )
    if (
        len(scenario["available_actions"]) != len(STAGE3_ACTIONS)
        or set(scenario["available_actions"]) != set(STAGE3_ACTIONS)
    ):
        raise ValueError("Stage3必须固定提供完整且无重复的13个Action")
    for action in scenario["available_actions"]:
        registry.get(action)
    contract = scenario["stage3_contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("Stage3 stage3_contract必须是对象")
    Stage3ContractEngine(contract, action_names=STAGE3_ACTIONS)
    for pattern in _contract_action_patterns(contract):
        action_spec = registry.get(str(pattern["name"]))
        for parameter, expected in pattern.get("parameters", {}).items():
            if parameter not in action_spec.parameters:
                raise ValueError(
                    f"{pattern['name']}包含未知契约参数: {parameter}"
                )
            enum = action_spec.parameters[parameter].enum
            values = expected if isinstance(expected, list) else [expected]
            if enum and any(value not in enum for value in values):
                raise ValueError(
                    f"{pattern['name']}.{parameter}包含无效契约枚举值"
                )
    contract_ids = {
        str(item["id"])
        for section in ("transition_contracts", "trace_contracts")
        for item in contract.get(section, [])
    }
    if not isinstance(scenario["challenge_plan"], list):
        raise ValueError("Stage3 challenge_plan必须是数组")
    for mode in ("normal", "pressure"):
        ChallengeController(
            scenario["challenge_plan"],
            mode=mode,
            contract_ids=contract_ids,
        )
    rules = scenario["role_contract"].get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("Stage3 role_contract.rules必须是非空数组")
    rule_ids: set[str] = set()
    for rule in rules:
        if (
            not isinstance(rule, Mapping)
            or not isinstance(rule.get("id"), str)
            or not rule["id"]
            or not isinstance(rule.get("statement"), str)
            or not rule["statement"]
        ):
            raise ValueError("每条Stage3角色规则必须包含非空id和statement")
        if rule["id"] in rule_ids:
            raise ValueError(f"Stage3角色规则id重复: {rule['id']}")
        rule_ids.add(rule["id"])


def _contract_action_patterns(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if (
            isinstance(value.get("name"), str)
            and isinstance(value.get("parameters"), Mapping)
        ):
            yield value
        for child in value.values():
            yield from _contract_action_patterns(child)
    elif isinstance(value, list):
        for child in value:
            yield from _contract_action_patterns(child)


def _invariance_consistency(
    records: list[dict[str, Any]],
) -> dict[str, bool]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["pair_type"] == "invariance":
            grouped.setdefault(record["pair_id"], []).append(record)
    return {
        pair_id: (
            len(values) == 2
            and values[0].get("state_answer")
            == values[1].get("state_answer")
            and bool(values[0].get("state_correct"))
            and bool(values[1].get("state_correct"))
        )
        for pair_id, values in grouped.items()
    }


def _chunks(
    values: list[dict[str, Any]],
    size: int,
) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _mean(values: Iterable[Any]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(float(item) for item in items) / len(items)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _usage_summary(run_dir: Path) -> dict[str, int]:
    usages: list[dict[str, Any]] = []
    qa_path = run_dir / "stage1_qa.jsonl"
    if qa_path.is_file():
        seen: set[tuple[int, int, int]] = set()
        for record in read_jsonl(qa_path):
            usage = record.get("usage") or {}
            key = (
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                int(usage.get("total_tokens", 0)),
            )
            if key not in seen:
                seen.add(key)
                usages.append(usage)
    open_path = run_dir / "stage1_open.jsonl"
    if open_path.is_file():
        for record in read_jsonl(open_path):
            usages.extend(
                [
                    record.get("candidate_usage") or {},
                    record.get("judge_usage") or {},
                ]
            )
    branch_path = run_dir / "stage2_branches.jsonl"
    if branch_path.is_file():
        usages.extend(
            record.get("usage") or {}
            for record in read_jsonl(branch_path)
        )
    stage3_paths = [
        path
        for path in sorted(run_dir.glob("stage3_seed*_*.json"))
        if "_checkpoint" not in path.stem
    ]
    if not stage3_paths:
        stage3_paths = [
            run_dir / f"stage3_{mode}.json"
            for mode in ("normal", "pressure")
            if (run_dir / f"stage3_{mode}.json").is_file()
        ]
    for path in stage3_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        usages.extend(
            item.get("usage") or {}
            for item in payload.get("call_log", [])
            if item.get("usage")
        )
    return {
        "successful_calls": len(usages),
        "prompt_tokens": sum(
            int(usage.get("prompt_tokens", 0)) for usage in usages
        ),
        "completion_tokens": sum(
            int(usage.get("completion_tokens", 0)) for usage in usages
        ),
        "total_tokens": sum(
            int(usage.get("total_tokens", 0)) for usage in usages
        ),
    }


def _completed_logical_calls(run_dir: Path) -> int:
    """Count persisted calls, including an unfinished Stage3 checkpoint."""

    completed = _usage_summary(run_dir)["successful_calls"]
    final_stems = {
        path.stem
        for path in run_dir.glob("stage3_seed*_*.json")
        if "_checkpoint" not in path.stem
    }
    for path in run_dir.glob("stage3_seed*_*_checkpoint.json"):
        final_stem = path.stem.removesuffix("_checkpoint")
        if final_stem in final_stems:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed += sum(
            1 for item in payload.get("call_log", []) if item.get("usage")
        )
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--character", required=True)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--player-model")
    parser.add_argument("--checker-model")
    parser.add_argument("--qa-batch-size", type=int)
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--checker-interval", type=int, default=5)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_STAGE3_SEEDS),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = tuple(
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    )
    if not seeds:
        raise ValueError("--seeds至少包含一个整数")
    result = run_full_experiment(
        args.world,
        args.character,
        scenario_path=args.scenario,
        config_path=args.config,
        turns=args.turns,
        stage3_seeds=seeds,
        player_model=args.player_model,
        checker_model=args.checker_model,
        qa_batch_size=args.qa_batch_size,
        checker_interval=args.checker_interval,
        dry_run=args.dry_run,
        output_root=args.output_root,
        resume_dir=args.resume_dir,
        workers=args.workers,
        show_progress=not args.no_progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
