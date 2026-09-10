"""Generate Stage 1 continuous-session history.jsonl records with an LLM."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping

from agents import project_npc_character_card
from agents.prompt_builder import PromptBundle
from gamecore import FixtureEngine, GameContext, WorldDefinition
from llm import (
    GenerationConfig,
    LLMClient,
    StructuredOutputError,
    generate_structured,
)
from runners.progress import ProgressReporter

from ..shared.io import load_yaml, read_jsonl, write_jsonl
from ..shared.reviewed_sources import load_reviewed_anchors, load_transition_plan
from ..audit.source_assets import require_generation_ready


_SCHEMA_TERMS = (
    "state_transition",
    "runtime_state",
    "context_delta",
    "GameCore",
    "EvaluationSpec",
    "权限或义务",
    "关系保持不变",
    "目标状态",
)
_BAD_PHRASES = (
    "把能确认的和不能确认的分开",
    "这件事没有形成新的权限或义务",
    "原来的任务和关系都不变",
    "先回答事实问题",
    "separate what can and cannot be confirmed",
    "does not create any new permission or obligation",
    "the original task and relationship remain unchanged",
)
_TARGET_CHUNK_SIZE = 5
_MIN_CHUNK_SIZE = 3
_MAX_CHUNK_SIZE = 7
_SESSION_OFFSET_PATTERN = (-4, 2, 5, -1, 3, -5, 1, 4, -2, -3)
_HISTORY_LAYOUT_VERSION = 2


def generate_llm_history(
    world_dir: str | Path,
    character_id: str,
    *,
    client_factory: Callable[[], LLMClient],
    config: GenerationConfig,
    rounds: int = 600,
    workers: int = 6,
    work_root: str | Path | None = None,
    seed_history: list[dict[str, Any]] | None = None,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one frozen history through planned Player/NPC interaction."""

    if workers < 1:
        raise ValueError("workers必须至少为1")
    world = Path(world_dir)
    require_generation_ready(world, character_id)
    blueprints = build_session_blueprints(world, character_id, rounds=rounds)
    temporary_work: tempfile.TemporaryDirectory[str] | None = None
    if work_root is None:
        temporary_work = tempfile.TemporaryDirectory(
            prefix=f"rpg_agent_bench_{character_id}_",
        )
        work_dir = Path(temporary_work.name)
    else:
        work_dir = Path(work_root)
    plans_dir = work_dir / "plans"
    sessions_dir = work_dir / "sessions"
    audits_dir = work_dir / "audits"
    plans_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)
    _prepare_work_cache(
        work_dir,
        plans_dir,
        sessions_dir,
        audits_dir,
        blueprints,
    )
    if seed_history and not any(sessions_dir.glob("*.jsonl")):
        _seed_session_cache(seed_history, blueprints, sessions_dir)
    completed_calls = _completed_generation_calls(
        plans_dir,
        sessions_dir,
        audits_dir,
        blueprints,
    )
    dialogue_calls = sum(len(item["act_ranges"]) for item in blueprints)
    progress = ProgressReporter(
        total=len(blueprints) + dialogue_calls + 1,
        completed=completed_calls,
        enabled=show_progress,
    )
    progress.render("history resume" if completed_calls else "history start")

    plans = _generate_plans(
        world,
        character_id,
        blueprints,
        plans_dir,
        client_factory,
        config,
        workers,
        progress,
    )
    _attach_cross_session_memories(plans)
    _write_plans(plans_dir, plans)

    histories = _generate_sessions(
        world,
        character_id,
        blueprints,
        plans,
        sessions_dir,
        client_factory,
        config,
        workers,
        progress,
    )
    history = sorted(
        [record for session in histories for record in session],
        key=lambda record: int(record["round"]),
    )
    if len(history) != rounds:
        raise ValueError(f"历史轮数应为{rounds}，实际为{len(history)}")
    surface_quality = audit_surface_quality(history)
    history_value = _audit_history_value(history)
    sample_audit = _audit_history_sample(
        world,
        character_id,
        blueprints,
        histories,
        audits_dir,
        client_factory,
        config,
        progress,
    )
    quality = {
        **surface_quality,
        "sample_audit": sample_audit,
        "sample_audit_passed": _sample_audit_passed(sample_audit),
        "history_value": history_value,
    }
    quality["passed"] = bool(
        surface_quality["passed"]
        and quality["sample_audit_passed"]
        and history_value["passed"]
    )
    report = {
        "generation_mode": "llm_interactive_sessions",
        "sessions": len(blueprints),
        "rounds": len(history),
        "logical_api_calls": len(blueprints) + dialogue_calls + 1,
        "semantic_audit_calls": 1,
        "workers": workers,
        "quality": quality,
    }
    if temporary_work is not None:
        temporary_work.cleanup()
    return history, report


def _prepare_work_cache(
    work_dir: Path,
    plans_dir: Path,
    sessions_dir: Path,
    audits_dir: Path,
    blueprints: list[dict[str, Any]],
) -> None:
    signature = hashlib.sha256(
        json.dumps(
            {
                "layout_version": _HISTORY_LAYOUT_VERSION,
                "blueprints": blueprints,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    marker = work_dir / "layout.sha256"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == signature:
        return
    for directory, pattern in (
        (plans_dir, "*.json"),
        (sessions_dir, "*.jsonl"),
        (audits_dir, "*.json"),
    ):
        for path in directory.glob(pattern):
            path.unlink()
    marker.write_text(signature + "\n", encoding="utf-8")


def build_session_blueprints(
    world_dir: str | Path,
    character_id: str,
    *,
    rounds: int = 600,
) -> list[dict[str, Any]]:
    world_path = Path(world_dir)
    plan = load_transition_plan(world_path, character_id)
    anchors = {
        anchor["id"]: anchor
        for anchor in load_reviewed_anchors(world_path, character_id)
    }
    transitions = plan["transitions"]
    player_profile = _load_history_player_profile(
        world_path,
        character_id,
    )
    session_count = len(transitions)
    session_sizes = _allocate_session_sizes(rounds, session_count)
    anchor_loads = _allocate_anchor_loads(
        session_count,
        len(transitions),
        f"{world_path.name}:{character_id}",
    )

    world = WorldDefinition.load_yaml(world_path / "environment.yaml")
    context = GameContext(
        {
            "character_card": load_yaml(
                world_path / "characters" / f"{character_id}.yaml"
            ),
            **copy.deepcopy(plan["initial_context"]),
            "history": [],
        }
    )
    fixture = FixtureEngine(world)
    blueprints: list[dict[str, Any]] = []
    prior_anchors: list[dict[str, Any]] = []
    transition_index = 0
    for index in range(1, session_count + 1):
        session_size = session_sizes[index - 1]
        start_round = sum(session_sizes[: index - 1]) + 1
        act_ranges = _build_act_ranges(session_size, index)
        pre_state = _state_projection(context)
        session_anchors: list[dict[str, Any]] = []
        session_transitions = transitions[
            transition_index : transition_index + anchor_loads[index - 1]
        ]
        transition_index += len(session_transitions)
        anchor_turns = _allocate_anchor_turns(
            session_size,
            session_transitions,
            f"{world_path.name}:{character_id}:{index}",
        )
        for transition, anchor_turn in zip(
            session_transitions,
            anchor_turns,
            strict=True,
        ):
            anchor = anchors[transition["anchor_id"]]
            global_anchor_round = start_round + anchor_turn - 1
            event_id = f"anchor_{anchor['id']}"
            anchor_pre_state = _state_projection(context)
            applied = fixture.apply(
                context,
                event_id=event_id,
                operations=transition["operations"],
                description=anchor["event"],
                history_turn=global_anchor_round,
            )
            context = applied.context
            delay = _anchor_effect_delay(
                str(anchor["id"]),
                index,
                session_count,
            )
            session_anchors.append(
                {
                    "turn": anchor_turn,
                    "anchor": anchor,
                    "transition": transition,
                    "event_id": event_id,
                    "pre_state": anchor_pre_state,
                    "post_state": _state_projection(context),
                    "context_delta": applied.context_delta,
                    "effect_start_session": index + delay,
                }
            )
        post_state = _state_projection(context)
        active_prior_anchors = [
            item
            for item in prior_anchors
            if int(item["effect_start_session"]) <= index
        ]
        blueprints.append(
            {
                "session_index": index,
                "session_id": f"{character_id}_session_{index:02d}",
                "start_round": start_round,
                "session_size": session_size,
                "act_ranges": act_ranges,
                "anchors": session_anchors,
                "pre_state": pre_state,
                "post_state": post_state,
                "prior_anchors": copy.deepcopy(active_prior_anchors[-5:]),
                "cross_event_probe": (
                    copy.deepcopy(
                        active_prior_anchors[
                            -min(
                                2 + index % 4,
                                len(active_prior_anchors),
                            )
                        ]
                    )
                    if len(active_prior_anchors) >= 2 and index % 2 == 0
                    else None
                ),
                "player_profile": copy.deepcopy(player_profile),
                "player_forbidden_knowledge": [
                    {
                        "anchor_id": item["anchor_id"],
                        "event": item["event"],
                    }
                    for item in prior_anchors
                    if not item["player_visible"]
                ]
                + [
                    {
                        "anchor_id": item["anchor"]["id"],
                        "event": item["anchor"]["event"],
                    }
                    for item in session_anchors
                    if not item["transition"]["player_visible"]
                ],
            }
        )
        prior_anchors.extend(
            {
                "event_id": item["event_id"],
                "anchor_id": item["anchor"]["id"],
                "event": item["anchor"]["event"],
                "behavioral_consequence": str(
                    item["transition"]["npc_surface"]
                ),
                "player_visible": bool(
                    item["transition"]["player_visible"]
                ),
                "effect_start_session": item["effect_start_session"],
            }
            for item in session_anchors
        )
    if transition_index != len(transitions):
        raise ValueError("Anchor分配未覆盖全部transition")
    _attach_anchor_effect_probes(blueprints)
    return blueprints


def _stable_order(seed: str, count: int) -> list[int]:
    return sorted(
        range(count),
        key=lambda index: hashlib.sha256(
            f"{seed}:{index}".encode()
        ).digest(),
    )


def _allocate_anchor_loads(
    session_count: int,
    anchor_count: int,
    seed: str,
) -> list[int]:
    if session_count < 4 or anchor_count != session_count:
        raise ValueError("当前长历史要求Anchor数与Session数相同且至少为4")
    empty_count = max(2, round(session_count * 0.27))
    triple_count = 1 if session_count >= 12 else 0
    double_count = empty_count - 2 * triple_count
    single_count = session_count - empty_count - double_count - triple_count
    loads = (
        [0] * empty_count
        + [1] * single_count
        + [2] * double_count
        + [3] * triple_count
    )
    order = _stable_order(seed, session_count)
    shuffled = [0] * session_count
    for target, load in zip(order, loads, strict=True):
        shuffled[target] = load
    if shuffled[-1] == 0:
        donor = max(
            (index for index, load in enumerate(shuffled[:-1]) if load > 0),
            key=lambda index: (shuffled[index], index),
        )
        shuffled[-1], shuffled[donor] = shuffled[donor], shuffled[-1]
    if sum(shuffled) != anchor_count:
        raise ValueError("Anchor分配数量错误")
    return shuffled


def _allocate_anchor_turns(
    session_size: int,
    transitions: list[dict[str, Any]],
    seed: str,
) -> list[int]:
    if not transitions:
        return []
    available = list(range(2, session_size))
    ordered = _stable_order(seed, len(available))
    turns = sorted(available[index] for index in ordered[: len(transitions)])
    if any(
        operation.get("op") == "set_dialogue_status"
        and operation.get("status") == "ended"
        for operation in transitions[-1]["operations"]
    ):
        turns[-1] = session_size
    if len(set(turns)) != len(turns):
        raise ValueError("同一Session中的Anchor位置必须唯一")
    return turns


def _anchor_effect_delay(
    anchor_id: str,
    session_index: int,
    session_count: int,
) -> int:
    digest = hashlib.sha256(anchor_id.encode()).digest()[0]
    delay = (0, 0, 0, 1, 1, 2)[digest % 6]
    return min(delay, session_count - session_index)


def _attach_anchor_effect_probes(
    blueprints: list[dict[str, Any]],
) -> None:
    for blueprint in blueprints:
        blueprint["effect_probes"] = []
    for source in blueprints:
        for item in source["anchors"]:
            target_index = int(item["effect_start_session"]) - 1
            target = blueprints[target_index]
            occupied = {
                int(probe["turn"]) for probe in target["effect_probes"]
            }
            minimum_turn = (
                int(item["turn"]) + 1
                if target_index + 1 == source["session_index"]
                else 3
            )
            candidates = list(
                range(minimum_turn, target["session_size"])
            )
            turn = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate not in occupied
                    and all(
                        abs(candidate - int(anchor["turn"])) > 1
                        for anchor in target["anchors"]
                    )
                ),
                int(item["turn"]) if not candidates else candidates[0],
            )
            target["effect_probes"].append(
                {
                    "anchor_id": item["anchor"]["id"],
                    "source_session": source["session_id"],
                    "turn": turn,
                    "behavioral_consequence": item["transition"]["npc_surface"],
                    "delayed": target_index + 1 > source["session_index"],
                }
            )


def _state_before_turn(
    blueprint: dict[str, Any],
    turn: int,
) -> dict[str, Any]:
    state = blueprint["pre_state"]
    for item in blueprint["anchors"]:
        if int(item["turn"]) >= turn:
            break
        state = item["post_state"]
    return state


def _state_after_turn(
    blueprint: dict[str, Any],
    turn: int,
) -> dict[str, Any]:
    state = blueprint["pre_state"]
    for item in blueprint["anchors"]:
        if int(item["turn"]) > turn:
            break
        state = item["post_state"]
    return state


def _allocate_session_sizes(rounds: int, count: int) -> list[int]:
    if count < 1 or rounds < count * 12:
        raise ValueError("每个session至少需要12轮以形成连续事件")
    base, extra = divmod(rounds, count)
    amplitude = min(5, base - 12)
    offsets = [
        round(
            _SESSION_OFFSET_PATTERN[index % len(_SESSION_OFFSET_PATTERN)]
            * amplitude
            / 5
        )
        for index in range(count)
    ]
    sizes = [
        base + offsets[index] + (1 if index < extra else 0)
        for index in range(count)
    ]
    delta = rounds - sum(sizes)
    direction = 1 if delta > 0 else -1
    for index in range(abs(delta)):
        target = (index * 7 + 3) % count
        if direction < 0 and sizes[target] <= 12:
            target = max(range(count), key=sizes.__getitem__)
        sizes[target] += direction
    if sum(sizes) != rounds or min(sizes) < 12:
        raise ValueError("无法分配合法的session长度")
    return sizes


def _build_act_ranges(session_size: int, session_index: int) -> list[list[int]]:
    count = max(2, round(session_size / _TARGET_CHUNK_SIZE))
    base, extra = divmod(session_size, count)
    lengths = [base + (1 if index < extra else 0) for index in range(count)]
    if len(set(lengths)) == 1 and count >= 2:
        lengths[0] -= 1
        lengths[-1] += 1
    shift = session_index % count
    lengths = lengths[shift:] + lengths[:shift]
    if not all(_MIN_CHUNK_SIZE <= length <= _MAX_CHUNK_SIZE for length in lengths):
        raise ValueError("session无法划分为3到7轮的自然段")
    ranges: list[list[int]] = []
    start = 1
    for length in lengths:
        ranges.append([start, start + length - 1])
        start += length
    return ranges


def _load_history_player_profile(
    world_path: Path,
    character_id: str,
) -> dict[str, str]:
    for scenario_path in sorted((world_path / "scenarios").glob("*.yaml")):
        scenario = load_yaml(scenario_path)
        if (
            scenario.get("character_id", scenario.get("npc"))
            != character_id
        ):
            continue
        profile = scenario.get("player_profile")
        if isinstance(profile, Mapping) and profile.get("role"):
            return {
                "role": str(profile["role"]),
                "public_identity": str(
                    profile.get("public_identity", "")
                ),
            }
    language = load_yaml(world_path / "environment.yaml").get(
        "language",
        "zh",
    )
    if language == "en":
        return {
            "role": "ordinary acquaintance or coworker of the NPC",
            "public_identity": (
                "Handles routine communication and verification without "
                "extra rank, powers, or privileged access."
            ),
        }
    return {
        "role": "与NPC共事的普通办事员",
        "public_identity": "负责传递和核对日常事务，没有额外职务或特殊权限。",
    }


def audit_surface_quality(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    utterances = [
        message["utterance"]
        for record in history
        for message in record["messages"]
    ]
    schema_leaks = [
        {"round": record["round"], "term": term}
        for record in history
        for message in record["messages"]
        for term in _SCHEMA_TERMS
        if term in message["utterance"]
    ]
    bad_phrases = [
        {"round": record["round"], "phrase": phrase}
        for record in history
        for message in record["messages"]
        for phrase in _BAD_PHRASES
        if phrase in message["utterance"]
    ]
    repeated = [
        {"round": record["round"], "utterance": message["utterance"]}
        for record in history
        for message in record["messages"]
        if _has_bad_repetition(message["utterance"])
    ]
    exact_duplicates: list[dict[str, Any]] = []
    seen_utterances: dict[tuple[str, str], int] = {}
    for record in history:
        for message in record["messages"]:
            normalized = _normalized_utterance(message["utterance"])
            key = (str(message["speaker"]), normalized)
            if len(normalized) >= 12 and key in seen_utterances:
                exact_duplicates.append(
                    {
                        "round": int(record["round"]),
                        "first_round": seen_utterances[key],
                        "speaker": str(message["speaker"]),
                    }
                )
            else:
                seen_utterances[key] = int(record["round"])
    invalid_dialogue = []
    for record in history:
        for message in record["messages"]:
            speaker = str(message["speaker"])
            npc_name = str(message.get("name", ""))
            try:
                _validate_utterance(
                    message["utterance"],
                    speaker=speaker,
                    npc_name=npc_name,
                )
            except ValueError as exc:
                invalid_dialogue.append(
                    {
                        "round": record["round"],
                        "speaker": speaker,
                        "reason": str(exc),
                    }
                )
    session_sizes = Counter(
        record["session_id"] for record in history
    )
    act_lengths = {
        (
            int(record["generation_meta"]["beat"]["turn_range"][1])
            - int(record["generation_meta"]["beat"]["turn_range"][0])
            + 1
        )
        for record in history
        if record.get("generation_meta", {}).get("beat", {}).get("turn_range")
    }
    anchor_counts = Counter(
        record["session_id"]
        for record in history
        if record.get("is_anchor", False)
    )
    all_sessions = list(dict.fromkeys(record["session_id"] for record in history))
    anchor_turn_ratios = [
        float(record["turn_in_session"])
        / float(session_sizes[record["session_id"]])
        for record in history
        if record.get("is_anchor", False)
    ]
    effect_refs = {
        str(anchor_id)
        for record in history
        for anchor_id in record.get("generation_meta", {}).get(
            "anchor_effect_refs",
            [],
        )
    }
    anchor_ids = {
        str(record["anchor_id"])
        for record in history
        if record.get("is_anchor", False)
    }
    delayed_anchors = [
        record
        for record in history
        if record.get("is_anchor", False)
        and int(
            record.get("generation_meta", {}).get(
                "anchor_effect_start_session",
                0,
            )
            or 0
        )
        > all_sessions.index(record["session_id"]) + 1
    ]
    anchor_layout_safe = bool(
        any(anchor_counts[session_id] == 0 for session_id in all_sessions)
        and any(anchor_counts[session_id] > 1 for session_id in all_sessions)
        and len({round(ratio, 1) for ratio in anchor_turn_ratios}) >= 5
        and anchor_ids.issubset(effect_refs)
        and delayed_anchors
    )
    pattern_safe = (
        len(set(session_sizes.values())) >= min(4, len(session_sizes))
        and len(act_lengths) >= 2
        and anchor_layout_safe
    )
    return {
        "unique_utterance_ratio": (
            len(set(utterances)) / len(utterances) if utterances else 0.0
        ),
        "schema_leaks": schema_leaks,
        "bad_phrases": bad_phrases,
        "repeated_word_lines": repeated,
        "exact_duplicate_utterances": exact_duplicates,
        "invalid_dialogue": invalid_dialogue,
        "session_size_distribution": dict(
            Counter(session_sizes.values())
        ),
        "act_lengths": sorted(act_lengths),
        "anchor_count_distribution": dict(
            Counter(anchor_counts[session_id] for session_id in all_sessions)
        ),
        "anchor_position_buckets": len(
            {round(ratio, 1) for ratio in anchor_turn_ratios}
        ),
        "delayed_anchor_effects": len(delayed_anchors),
        "anchors_with_effect_evidence": len(anchor_ids & effect_refs),
        "anchor_layout_safe": anchor_layout_safe,
        "generation_pattern_safe": pattern_safe,
        "sessions_with_memory_recall": len(
            {
                record["session_id"]
                for record in history
                if record.get("generation_meta", {}).get(
                    "recalled_memories", []
                )
            }
        ),
        "passed": (
            not schema_leaks
            and not bad_phrases
            and not repeated
            and not exact_duplicates
            and not invalid_dialogue
            and pattern_safe
        ),
    }


def _audit_history_value(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    introduced: dict[str, str] = {}
    recalled: set[str] = set()
    cross_session_records: set[str] = set()
    recall_turns: Counter[int] = Counter()
    meaningful_rounds: set[int] = set()
    for record in history:
        meta = record.get("generation_meta", {})
        session_id = str(record["session_id"])
        introduced_ids = meta.get("introduced_memories", [])
        recalled_ids = meta.get("recalled_memories", [])
        for memory_id in introduced_ids:
            introduced[str(memory_id)] = session_id
        for memory_id in recalled_ids:
            memory_key = str(memory_id)
            recalled.add(memory_key)
            if introduced.get(memory_key) not in {None, session_id}:
                cross_session_records.add(session_id)
        if introduced_ids or recalled_ids:
            meaningful_rounds.add(int(record["round"]))
        if recalled_ids:
            recall_turns[int(record["turn_in_session"])] += len(recalled_ids)
    introduced_count = len(introduced)
    unresolved_count = len(set(introduced) - recalled)
    unresolved_ratio = (
        unresolved_count / introduced_count if introduced_count else 0.0
    )
    recall_total = sum(recall_turns.values())
    top_three_share = (
        sum(count for _, count in recall_turns.most_common(3)) / recall_total
        if recall_total
        else 1.0
    )
    session_count = len({record["session_id"] for record in history})
    meaningful_ratio = len(meaningful_rounds) / len(history) if history else 0.0
    passed = bool(
        meaningful_ratio >= 0.1
        and len(cross_session_records) >= max(2, round(session_count * 0.3))
        and 0.1 <= unresolved_ratio <= 0.7
        and top_three_share <= 0.45
    )
    return {
        "meaningful_round_ratio": round(meaningful_ratio, 4),
        "cross_session_recall_sessions": len(cross_session_records),
        "introduced_memories": introduced_count,
        "unresolved_memory_ratio": round(unresolved_ratio, 4),
        "top_three_recall_turn_share": round(top_three_share, 4),
        "passed": passed,
    }


def _generate_plans(
    world: Path,
    character_id: str,
    blueprints: list[dict[str, Any]],
    plans_dir: Path,
    client_factory: Callable[[], LLMClient],
    config: GenerationConfig,
    workers: int,
    progress: ProgressReporter,
) -> list[dict[str, Any]]:
    card = _history_character_card(
        load_yaml(world / "characters" / f"{character_id}.yaml")
    )
    environment = load_yaml(world / "environment.yaml")

    def generate(blueprint: dict[str, Any]) -> dict[str, Any]:
        path = plans_dir / f"{blueprint['session_id']}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        bundle = _plan_bundle(card, environment, blueprint)
        plan_config = _config_for_session(
            config,
            blueprint["session_index"],
            max_tokens=2048,
        )
        client = client_factory()
        output = _generate_history_structured(
            client,
            bundle,
            plan_config,
            lambda value: _validate_plan(
                value,
                blueprint["session_id"],
                blueprint["act_ranges"],
                [
                    int(item["turn"])
                    for item in blueprint["anchors"]
                ],
                blueprint["player_forbidden_knowledge"],
            ),
        )
        for index, memory in enumerate(output["memory_seeds"], start=1):
            memory["memory_id"] = (
                f"{blueprint['session_id']}_memory_{index:02d}"
            )
        output["usage"] = copy.deepcopy(client.last_usage)
        path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        progress.advance(f"history plan={blueprint['session_id']}")
        return output

    plans: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(blueprints))) as pool:
        futures = {
            pool.submit(generate, blueprint): blueprint["session_index"]
            for blueprint in blueprints
        }
        for future in as_completed(futures):
            plans[futures[future]] = future.result()
    return [plans[index] for index in sorted(plans)]


def _generate_sessions(
    world: Path,
    character_id: str,
    blueprints: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    sessions_dir: Path,
    client_factory: Callable[[], LLMClient],
    config: GenerationConfig,
    workers: int,
    progress: ProgressReporter,
) -> list[list[dict[str, Any]]]:
    card = load_yaml(world / "characters" / f"{character_id}.yaml")
    npc_card = _history_character_card(card)
    public_card = {
        "name": card["identity"]["name"],
        "role": card["identity"]["role"],
    }
    environment = load_yaml(world / "environment.yaml")

    def generate(
        blueprint: dict[str, Any],
        plan: dict[str, Any],
    ) -> list[dict[str, Any]]:
        path = sessions_dir / f"{blueprint['session_id']}.jsonl"
        records = read_jsonl(path) if path.is_file() else []
        if len(records) == blueprint["session_size"]:
            return records
        if len(records) > blueprint["session_size"]:
            raise ValueError(f"{path}包含过多轮次")
        completed_ranges = [
            turn_range
            for turn_range in blueprint["act_ranges"]
            if turn_range[1] <= len(records)
        ]
        if len(records) and (
            not completed_ranges or completed_ranges[-1][1] != len(records)
        ):
            raise ValueError(f"{path}未停在完整自然段边界")
        client = client_factory()
        transcript = [
            message
            for record in records
            for message in record["messages"]
        ]
        for act in plan["acts"][len(completed_ranges) :]:
            chunk_start, chunk_end = [
                int(value) for value in act["turn_range"]
            ]
            due_memories = [
                memory
                for memory in plan["memory_seeds"]
                if chunk_start
                <= int(memory["introduced_turn"])
                <= chunk_end
            ]
            recalled = [
                memory
                for memory in (
                    plan["memory_seeds"]
                    + plan.get("carry_in_memories", [])
                )
                if memory.get("recall_turn") is not None
                and chunk_start <= int(memory["recall_turn"]) <= chunk_end
            ]
            due_effects = [
                probe
                for probe in blueprint["effect_probes"]
                if chunk_start <= int(probe["turn"]) <= chunk_end
            ]
            bundle = _dialogue_chunk_bundle(
                public_card,
                npc_card,
                environment,
                blueprint,
                plan,
                act,
                transcript,
                chunk_start,
                chunk_end,
                due_memories,
                recalled,
                due_effects,
            )
            required_memories: dict[int, list[str]] = {}
            for memory in due_memories:
                required_memories.setdefault(
                    int(memory["introduced_turn"]),
                    [],
                ).append(str(memory["memory_id"]))
            for memory in recalled:
                required_memories.setdefault(
                    int(memory["recall_turn"]),
                    [],
                ).append(str(memory["memory_id"]))
            required_effects = {
                int(probe["turn"]): str(probe["anchor_id"])
                for probe in due_effects
            }
            output = _generate_history_structured(
                client,
                bundle,
                _config_for_session(config, blueprint["session_index"]),
                lambda value, start=chunk_start, end=chunk_end: (
                    _validate_dialogue_chunk(
                        value,
                        start,
                        end,
                        str(card["identity"]["name"]),
                        required_memories,
                        required_effects,
                        transcript,
                        blueprint["player_forbidden_knowledge"],
                    )
                ),
            )
            usage = copy.deepcopy(client.last_usage)
            progress.advance(
                f"history {blueprint['session_id']} "
                f"turns={chunk_start}-{chunk_end}/"
                f"{blueprint['session_size']}"
            )
            for turn_output in output["turns"]:
                turn = int(turn_output["turn_in_session"])
                turn_due = [
                    memory
                    for memory in due_memories
                    if int(memory["introduced_turn"]) == turn
                ]
                turn_recalled = [
                    memory
                    for memory in recalled
                    if int(memory["recall_turn"]) == turn
                ]
                record = _history_record(
                    character_id,
                    card["identity"]["name"],
                    environment["world_id"],
                    blueprint,
                    plan,
                    act,
                    turn,
                    turn_output["player"],
                    turn_output["npc"],
                    turn_output["anchor_effect_refs"],
                    turn_due,
                    turn_recalled,
                    usage if turn == chunk_start else None,
                )
                records.append(record)
                transcript.extend(record["messages"])
            write_jsonl(path, records)
        return records

    sessions: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(plans))) as pool:
        futures = {
            pool.submit(generate, blueprint, plan): blueprint["session_index"]
            for blueprint, plan in zip(blueprints, plans, strict=True)
        }
        for future in as_completed(futures):
            sessions[futures[future]] = future.result()
    return [sessions[index] for index in sorted(sessions)]


def _audit_history_sample(
    world: Path,
    character_id: str,
    blueprints: list[dict[str, Any]],
    histories: list[list[dict[str, Any]]],
    audits_dir: Path,
    client_factory: Callable[[], LLMClient],
    config: GenerationConfig,
    progress: ProgressReporter,
) -> dict[str, Any]:
    card = project_npc_character_card(
        load_yaml(world / "characters" / f"{character_id}.yaml")
    )
    world_language = load_yaml(world / "environment.yaml").get(
        "language",
        "zh",
    )

    flat_history = [record for session in histories for record in session]
    digest = hashlib.sha256(
        json.dumps(flat_history, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    path = audits_dir / "summary.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        candidate = {
            key: value
            for key, value in cached.items()
            if key not in {"usage", "history_digest", "sampled_sessions"}
        }
        try:
            _validate_sample_audit(candidate)
            if cached.get("history_digest") == digest:
                return cached
        except ValueError:
            pass
    sample_indices = _audit_sample_indices(blueprints)
    samples = [
        {
            "session_id": blueprints[index]["session_id"],
            "anchors": [
                {
                    "anchor_id": item["anchor"]["id"],
                    "turn": item["turn"],
                    "event": item["anchor"]["event"],
                    "visible_to_player": item["transition"]["player_visible"],
                    "effect_start_session": item["effect_start_session"],
                    "behavioral_consequence": item["transition"]["npc_surface"],
                }
                for item in blueprints[index]["anchors"]
            ],
            "effect_probes": blueprints[index]["effect_probes"],
            "cross_event_probe": blueprints[index]["cross_event_probe"],
            "dialogue": _sample_history_projection(histories[index]),
        }
        for index in sample_indices
    ]
    bundle = PromptBundle(
        system_prompt=(
            "你是长历史角色数据抽样质检员。只做角色级总体判断，不逐轮打分。"
            "依据四个分层抽取的Session样本（覆盖无Anchor、多Anchor和延迟"
            "显现），检查：Session连续推进；Anchor在规定时间后的行为中生效；"
            "旧Anchor在不同事件中以行为而非复述显现；普通"
            "细节有后续价值；Player能主动推动局势；无状态机语言；角色风险"
            "偏好、信任、目标和承诺一致；无明显机械剧情模式。若隐藏Anchor"
            "向Player泄露代号、身份、组织关系或来源，必须判角色一致性失败。"
            "这是粗粒度抽样审计，只输出JSON。"
        ),
        user_prompt=(
            f"world_language：{world_language}\n"
            f"角色指导：{json.dumps(card, ensure_ascii=False)}\n"
            f"抽样Session：{json.dumps(samples, ensure_ascii=False)}\n\n"
            "输出boolean字段：continuity_pass、anchor_effect_pass、"
            "cross_event_pass、history_value_pass、player_agency_pass、"
            "schema_safe_pass、role_consistency_pass、pattern_diversity_pass；"
            "另输出issues字符串数组。"
        ),
    )
    client = client_factory()
    result = generate_structured(
        client,
        bundle,
        _config_for_session(config, 0),
        _validate_sample_audit,
    )
    result["usage"] = copy.deepcopy(client.last_usage)
    result["history_digest"] = digest
    result["sampled_sessions"] = [
        blueprints[index]["session_id"] for index in sample_indices
    ]
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress.advance("history sampled audit")
    return result


def _audit_history_projection(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "turn": record["turn_in_session"],
            "player": record["messages"][0]["utterance"],
            "npc": record["messages"][1]["utterance"],
            "is_anchor": record["is_anchor"],
            "introduced_memories": record["generation_meta"][
                "introduced_memories"
            ],
            "recalled_memories": record["generation_meta"][
                "recalled_memories"
            ],
            "anchor_effect_refs": record["generation_meta"].get(
                "anchor_effect_refs",
                [],
            ),
        }
        for record in history
    ]


def _sample_history_projection(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indices = {
        0,
        1,
        len(history) - 2,
        len(history) - 1,
    }
    focal = [
        index
        for index, record in enumerate(history)
        if record["is_anchor"]
        or record.get("generation_meta", {}).get("anchor_effect_refs")
    ]
    for index in focal:
        indices.update(
            range(max(0, index - 1), min(len(history), index + 2))
        )
    selected = [history[index] for index in sorted(indices)]
    return _audit_history_projection(selected)


def _audit_sample_indices(
    blueprints: list[dict[str, Any]],
) -> list[int]:
    candidates = [
        next(
            index
            for index, item in enumerate(blueprints)
            if not item["anchors"]
        ),
        next(
            index
            for index, item in enumerate(blueprints)
            if len(item["anchors"]) > 1
        ),
        next(
            index
            for index, item in enumerate(blueprints)
            if any(probe["delayed"] for probe in item["effect_probes"])
        ),
        len(blueprints) - 1,
    ]
    selected = set(candidates)
    for index in (
        0,
        len(blueprints) // 3,
        len(blueprints) * 2 // 3,
        len(blueprints) - 1,
    ):
        if len(selected) >= 4:
            break
        selected.add(index)
    return sorted(selected)


def _history_character_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Keep stable characterization while removing final-snapshot chronology."""

    projected = project_npc_character_card(card)
    projected.pop("snapshot", None)
    identity = projected.get("identity")
    if isinstance(identity, dict):
        identity.pop("public_background", None)
    background = projected.get("background")
    if isinstance(background, dict):
        background.pop("personal_history", None)
        background.pop("self_knowledge", None)
        if not background:
            projected.pop("background", None)
    return projected


def _plan_bundle(
    card: dict[str, Any],
    environment: dict[str, Any],
    blueprint: dict[str, Any],
) -> PromptBundle:
    npc_name = str(
        card.get("identity", {}).get("name")
        or card.get("character_id")
        or "NPC"
    )
    payload = {
        "session_id": blueprint["session_id"],
        "turns": blueprint["session_size"],
        "required_act_ranges": blueprint["act_ranges"],
        "fixed_player_profile": blueprint["player_profile"],
        "anchors": [
            {
                "turn": item["turn"],
                "event": item["anchor"]["event"],
                "visible_to_player": bool(
                    item["transition"]["player_visible"]
                ),
                "behavioral_consequence": item["transition"]["npc_surface"],
                "state_change": item["context_delta"],
                "effect_start_session": item["effect_start_session"],
            }
            for item in blueprint["anchors"]
        ],
        "effect_probes": blueprint["effect_probes"],
        "prior_events": blueprint["prior_anchors"],
        "cross_event_behavior_probe": blueprint["cross_event_probe"],
        "player_forbidden_knowledge": blueprint[
            "player_forbidden_knowledge"
        ],
        "character": card,
        "npc_name": npc_name,
        "world_name": environment.get("name", environment.get("world_id")),
        "world_language": environment.get("language", "zh"),
        "locations": environment["locations"],
    }
    return PromptBundle(
        system_prompt=(
            "你是长篇角色扮演数据的session规划器。根据给定世界、角色和"
            "语言规划一段真正连续的"
            "多轮对话。一个session只能围绕一项具体局部事务，不能并列家庭、"
            "站内派系和多条案件线。不得把每轮写成独立问答。局部事件应逐步"
            "出现新证据、回应和后果；Player下一轮必须能承接NPC上一轮。"
            "本Session可能没有Anchor，也可能有多个。Anchor发生前任何人都不得"
            "预知其结果；只有达到effect_start_session后，NPC处理方式才应体现"
            "变化。effect_probes指定的轮次必须通过具体决策体现对应后果，例如"
            "改变核查渠道、保护对象、信息流向或承担风险的方式；不能只写成"
            "更谨慎、更警觉。若本Session有Anchor，memory_seeds中至少一项必须"
            "在某个Anchor发生时或之后引入；无Anchor时则围绕局部事件留下记忆。"
            "若cross_event_behavior_probe非空，local_event不得重复该旧事件，"
            "但NPC在新事务中的一项具体处理必须体现其behavioral_consequence。"
            "character只提供稳定人格和能力，不代表其中任何经历已在本Session"
            "发生；Anchor时间线拥有最高优先级。setting、local_event、stakes和"
            "Anchor发生前的act不得把尚未发生的Anchor写成既成事实。"
            "player_forbidden_knowledge中的事实无论发生前后都不得出现在"
            "Player可知的背景、player_intent、公开线索或Player台词中；隐藏"
            "Anchor发生后也只能让NPC通过不泄密的处理变化体现。"
            "所有自然语言字段必须使用world_language指定的语言。"
            "不要使用状态机、标注或评测术语。"
            "只输出JSON对象。"
        ),
        user_prompt=(
            f"规划约束：{json.dumps(payload, ensure_ascii=False)}\n\n"
            "输出字段必须为：session_id；local_event；setting；stakes；"
            "acts（按required_act_ranges划分自然阶段，含act、turn_range、development、"
            "player_intent、npc_response_strategy）；memory_seeds（2到3项，含memory_id、"
            "content、introduced_turn、recall_turn、introduced_by，其中introduced_by"
            "只能是player或npc）。memory内容应是后文可重新引用的"
            "人、文件、时间、异常或临时决定；recall_turn可以是null，表示该线索"
            "暂未回收；非null时必须晚于introduced_turn，并让人物用该细节作出"
            "判断或推进事件。每个session至少有一项被回收，但不能让所有线索都"
            "整齐回收。"
            "任何Anchor若对Player不可见，"
            "不得让Player主动说出该事件。player_intent专指与NPC对话的"
            "Player本阶段会提出什么请求、问题或压力；npc_response_strategy"
            f"专指{npc_name}本人如何回应和处理，不能写成其他配角的目标。"
            "所有session中的Player必须始终采用fixed_player_profile，"
            "不能自行获得新的职业、职位、能力或权限。"
            "local_event、setting、stakes"
            "必须各自是一个字符串，不能输出数组或对象。acts必须依次使用"
            "required_act_ranges中的范围，每个范围对应一个act。"
        ),
    )


def _dialogue_chunk_bundle(
    public_card: dict[str, Any],
    npc_card: dict[str, Any],
    environment: dict[str, Any],
    blueprint: dict[str, Any],
    plan: dict[str, Any],
    act: dict[str, Any],
    transcript: list[dict[str, str]],
    chunk_start: int,
    chunk_end: int,
    due_memories: list[dict[str, Any]],
    recalled: list[dict[str, Any]],
    due_effects: list[dict[str, Any]],
) -> PromptBundle:
    anchors_in_chunk = [
        item
        for item in blueprint["anchors"]
        if chunk_start <= int(item["turn"]) <= chunk_end
    ]
    anchors_already_occurred = [
        item
        for item in blueprint["anchors"]
        if int(item["turn"]) < chunk_start
    ]
    payload = {
        "npc_public_identity": public_card,
        "npc_private_character_guidance": npc_card,
        "fixed_player_profile": blueprint["player_profile"],
        "world": environment["name"],
        "world_language": environment.get("language", "zh"),
        "session": {
            "local_event": plan["local_event"],
            "setting": plan["setting"],
            "stakes": plan["stakes"],
        },
        "current_act": act,
        "turn_range": [chunk_start, chunk_end],
        "previous_dialogue": transcript,
        "details_to_introduce_naturally": due_memories,
        "past_details_to_recall_naturally": recalled,
        "anchors": [
            {
                "turn": item["turn"],
                "event": item["anchor"]["event"],
                "visible_to_player": bool(
                    item["transition"]["player_visible"]
                ),
                "behavior_after_event": item["transition"]["npc_surface"],
                "effect_start_session": item["effect_start_session"],
            }
            for item in anchors_already_occurred + anchors_in_chunk
        ],
        "anchor_effects_to_show": due_effects,
        "current_npc_state": _state_before_turn(blueprint, chunk_start),
        "state_after_chunk": _state_after_turn(blueprint, chunk_end),
        "prior_long_term_events": blueprint["prior_anchors"],
        "cross_event_behavior_probe": blueprint["cross_event_probe"],
        "player_forbidden_knowledge": blueprint[
            "player_forbidden_knowledge"
        ],
    }
    return PromptBundle(
        system_prompt=(
            "你是长对话编剧，根据给定世界、角色和语言同时生成"
            "Player与NPC一段长度可变的连续真实对话。"
            "每个Player发言必须承接上一句NPC，每个NPC回答既回应当前问题又"
            "推进同一局部事件。不得写成独立问答集合。NPC内部状态只能通过"
            "具体选择、渠道、人物和语气体现，禁止解释状态字段或规则。若Anchor"
            "不对Player可见，Player绝不能说出或暗示自己知道该事件；NPC也不得"
            "向Player说出事件、代号、秘密身份、组织关系或信息来源，只能以"
            "不泄密的具体选择体现其后果。不得使用"
            "括号舞台动作，不得复用谨慎套话。Player必须始终符合"
            "fixed_player_profile，不能临时获得新的职务或权限。"
            "anchor_effects_to_show中的每项必须在指定轮通过NPC的具体决定体现，"
            "并在该轮anchor_effect_refs列出对应anchor_id；不得只填ID或解释"
            "内心状态、政治标签。Anchor刚发生不代表其行为影响必须立即显现。"
            "若cross_event_behavior_probe非空，必须让其中的既有行为后果在"
            "当前不同局部事务的一项具体决定中自然显现，不得复述旧事件。"
            "player_forbidden_knowledge中的事实不得由Player说出、确认、猜中或"
            "作为消息转述；即使session设定或NPC人物经历中存在相似表述，也以"
            "该禁止列表和Anchor时间线为准。"
            "details_to_introduce_naturally必须在指定轮首次出现，"
            "past_details_to_recall_naturally必须在指定轮被再次引用并影响判断。"
            "引用记忆时必须在台词中实际体现其事实或行为后果，并在该轮"
            "memory_refs准确列出对应memory_id；不能只填ID而把台词写成泛指。"
            "Player不得原样重复已被拒绝的请求，不得在公开环境直接说出"
            "角色设定中需要保密的身份、凭据、位置或其他敏感内容；不得擅自"
            "伪造、涂改或销毁世界中的记录，"
            "也不得做超出fixed_player_profile权限的决定。新消息必须交代可信"
            "来源或承认无法核验，不能只为推动剧情突然抛出。"
            "Player和NPC台词必须使用world_language指定的语言。"
            "只输出JSON。"
        ),
        user_prompt=(
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            '输出：{"turns":[{"turn_in_session":整数,'
            '"player":"Player自然台词","npc":"NPC角色化台词",'
            '"memory_refs":["本轮实际引入或调用的memory_id"],'
            '"anchor_effect_refs":["本轮实际体现的anchor_id"]}]}。'
            "必须完整覆盖turn_range，顺序连续。"
        ),
    )


def _history_record(
    character_id: str,
    npc_name: str,
    world_id: str,
    blueprint: dict[str, Any],
    plan: dict[str, Any],
    beat: dict[str, Any],
    turn: int,
    player_utterance: str,
    npc_utterance: str,
    anchor_effect_refs: list[str],
    due_memories: list[dict[str, Any]],
    recalled: list[dict[str, Any]],
    generation_usage: dict[str, Any] | None,
) -> dict[str, Any]:
    round_number = blueprint["start_round"] + turn - 1
    anchor_item = next(
        (
            item
            for item in blueprint["anchors"]
            if int(item["turn"]) == turn
        ),
        None,
    )
    is_anchor = anchor_item is not None
    transition = anchor_item["transition"] if anchor_item else None
    anchor = anchor_item["anchor"] if anchor_item else None
    return {
        "world_id": world_id,
        "character_id": character_id,
        "round": round_number,
        "session_id": blueprint["session_id"],
        "turn_in_session": turn,
        "messages": [
            {"speaker": "player", "utterance": player_utterance},
            {"speaker": "npc", "name": npc_name, "utterance": npc_utterance},
        ],
        "observation": (
            {
                "event_id": anchor_item["event_id"],
                "content": anchor["event"],
                "visible_to": (
                    ["npc", "player"]
                    if transition["player_visible"]
                    else ["npc"]
                ),
            }
            if is_anchor
            else None
        ),
        "is_anchor": is_anchor,
        "anchor_id": anchor["id"] if is_anchor else None,
        "source_anchor_id": anchor["id"] if is_anchor else None,
        "state_transition": (
            {
                "pre_state": anchor_item["pre_state"],
                "operations": transition["operations"],
                "context_delta": anchor_item["context_delta"],
                "post_state": anchor_item["post_state"],
            }
            if is_anchor
            else None
        ),
        "generation_meta": {
            "history_layout_version": _HISTORY_LAYOUT_VERSION,
            "local_event": plan["local_event"],
            "beat": beat,
            "introduced_memories": [
                memory["memory_id"] for memory in due_memories
            ],
            "recalled_memories": [
                memory["memory_id"] for memory in recalled
            ],
            "anchor_effect_refs": anchor_effect_refs,
            "anchor_effect_start_session": (
                anchor_item["effect_start_session"]
                if anchor_item is not None
                else None
            ),
            "introduced_memory_facts": [
                {
                    "memory_id": memory["memory_id"],
                    "content": memory["content"],
                }
                for memory in due_memories
            ],
            "recalled_memory_facts": [
                {
                    "memory_id": memory["memory_id"],
                    "content": memory["content"],
                }
                for memory in recalled
            ],
            "generation_usage": generation_usage,
        },
    }


def _validate_plan(
    value: dict[str, Any],
    session_id: str,
    act_ranges: list[list[int]],
    anchor_turns: list[int],
    player_forbidden_knowledge: list[dict[str, Any]],
) -> None:
    session_size = act_ranges[-1][-1]
    required = {
        "session_id",
        "local_event",
        "setting",
        "stakes",
        "acts",
        "memory_seeds",
    }
    if not required.issubset(value) or value["session_id"] != session_id:
        raise ValueError("session计划字段或session_id错误")
    normalized_plan = {key: value[key] for key in required}
    value.clear()
    value.update(normalized_plan)
    for key in ("local_event", "setting", "stakes"):
        value[key] = _coerce_plan_text(value[key])
    acts = value["acts"]
    expected_count = len(act_ranges)
    if not isinstance(acts, list) or len(acts) != expected_count:
        raise ValueError(f"acts必须恰好包含{expected_count}项")
    act_fields = {
        "act",
        "turn_range",
        "development",
        "player_intent",
        "npc_response_strategy",
    }
    for index, act in enumerate(acts, start=1):
        expected_range = act_ranges[index - 1]
        if not isinstance(act, Mapping) or not act_fields.issubset(act):
            raise ValueError(f"第{index}个act字段不符合契约")
        normalized_act = {key: act[key] for key in act_fields}
        acts[index - 1] = normalized_act
        act = normalized_act
        try:
            act_number = int(act["act"])
            turn_range = [int(value) for value in act["turn_range"]]
        except (TypeError, ValueError):
            raise ValueError(f"第{index}个act编号必须是整数") from None
        if act_number != index or turn_range != expected_range:
            raise ValueError(
                f"第{index}个act应为act={index}, "
                f"turn_range={expected_range}"
            )
        act["act"] = act_number
        act["turn_range"] = turn_range
    memories = value["memory_seeds"]
    if not isinstance(memories, list) or not 2 <= len(memories) <= 3:
        raise ValueError("memory_seeds必须包含2到3项")
    memory_ids: set[str] = set()
    memory_fields = {
        "content",
        "introduced_turn",
        "recall_turn",
        "introduced_by",
    }
    for index, memory in enumerate(memories, start=1):
        if not isinstance(memory, Mapping) or not memory_fields.issubset(
            memory
        ):
            raise ValueError("memory_seed字段不符合契约")
        normalized_memory = {
            "memory_id": str(
                memory.get("memory_id")
                or f"{session_id}_draft_memory_{index:02d}"
            ),
            **{key: memory[key] for key in memory_fields},
        }
        memories[index - 1] = normalized_memory
        memory = normalized_memory
        if not isinstance(memory["memory_id"], str) or not isinstance(
            memory["content"], str
        ):
            raise ValueError("memory_seed文本字段错误")
        introduced = min(
            max(int(memory["introduced_turn"]), 2),
            session_size - 2,
        )
        raw_recall = memory["recall_turn"]
        recalled = (
            None
            if raw_recall is None
            else min(max(int(raw_recall), introduced + 1), session_size)
        )
        memory["introduced_turn"] = introduced
        memory["recall_turn"] = recalled
        if memory["introduced_by"] not in {"player", "npc"}:
            raise ValueError("memory_seed.introduced_by必须为player或npc")
        memory_ids.add(memory["memory_id"])
    if len(memory_ids) != len(memories):
        raise ValueError("memory_id必须唯一")
    recalled_count = sum(
        memory["recall_turn"] is not None for memory in memories
    )
    if recalled_count < 1 or recalled_count == len(memories):
        raise ValueError("每个session必须自然保留至少一项未回收细节")
    earliest_anchor = min(anchor_turns) if anchor_turns else None
    if (
        earliest_anchor is not None
        and earliest_anchor <= session_size - 2
        and not any(
            int(memory["introduced_turn"]) >= earliest_anchor
            for memory in memories
        )
    ):
        candidate = next(
            (
                memory
                for memory in memories
                if memory["recall_turn"] is None
            ),
            memories[-1],
        )
        candidate["introduced_turn"] = earliest_anchor
        if (
            candidate["recall_turn"] is not None
            and int(candidate["recall_turn"]) <= earliest_anchor
        ):
            candidate["recall_turn"] = min(
                earliest_anchor + 1,
                session_size,
            )
    player_facing_text = [
        str(value[key]) for key in ("local_event", "setting", "stakes")
    ]
    player_facing_text.extend(
        str(act["player_intent"]) for act in acts
    )
    player_facing_text.extend(
        str(memory["content"])
        for memory in memories
        if memory["introduced_by"] == "player"
    )
    leaked_event = _hidden_event_overlap(
        "\n".join(player_facing_text),
        player_forbidden_knowledge,
    )
    if leaked_event is not None:
        raise ValueError(
            "session计划向Player侧泄漏隐藏Anchor："
            f"{leaked_event}"
        )


def _validate_dialogue_chunk(
    value: dict[str, Any],
    chunk_start: int,
    chunk_end: int,
    npc_name: str,
    required_memories: Mapping[int, list[str]],
    required_effects: Mapping[int, str],
    previous_dialogue: list[dict[str, str]],
    player_forbidden_knowledge: list[dict[str, Any]],
) -> None:
    if set(value) != {"turns"} or not isinstance(value["turns"], list):
        raise ValueError("对话分块只能包含turns数组")
    expected = list(range(chunk_start, chunk_end + 1))
    if [
        turn.get("turn_in_session")
        for turn in value["turns"]
        if isinstance(turn, Mapping)
    ] != expected:
        raise ValueError("对话分块未完整覆盖连续turn_range")
    seen_by_speaker: dict[str, set[str]] = {"player": set(), "npc": set()}
    for message in previous_dialogue:
        speaker = str(message.get("speaker", ""))
        normalized = _normalized_utterance(str(message.get("utterance", "")))
        if speaker in seen_by_speaker and len(normalized) >= 12:
            seen_by_speaker[speaker].add(normalized)
    for turn in value["turns"]:
        if set(turn) != {
            "turn_in_session",
            "player",
            "npc",
            "memory_refs",
            "anchor_effect_refs",
        }:
            raise ValueError("分块中的单轮字段不符合契约")
        expected_refs = set(
            required_memories.get(int(turn["turn_in_session"]), [])
        )
        if turn["memory_refs"] is None and not expected_refs:
            turn["memory_refs"] = []
        elif (
            isinstance(turn["memory_refs"], str)
            and turn["memory_refs"] in expected_refs
        ):
            turn["memory_refs"] = [turn["memory_refs"]]
        if not isinstance(turn["memory_refs"], list) or not all(
            isinstance(memory_id, str)
            for memory_id in turn["memory_refs"]
        ):
            raise ValueError(
                "memory_refs必须是memory_id字符串数组；"
                f"turn={turn['turn_in_session']}，"
                f"expected={sorted(expected_refs)}，"
                f"actual={turn['memory_refs']!r}"
            )
        _validate_utterance(turn["player"], speaker="player")
        leaked_event = _hidden_event_overlap(
            turn["player"],
            player_forbidden_knowledge,
        )
        if leaked_event is not None:
            raise ValueError(
                "Player台词泄漏隐藏Anchor："
                f"{leaked_event}"
            )
        _validate_utterance(
            turn["npc"],
            speaker="npc",
            npc_name=npc_name,
        )
        for speaker in ("player", "npc"):
            normalized = _normalized_utterance(turn[speaker])
            if (
                len(normalized) >= 12
                and normalized in seen_by_speaker[speaker]
            ):
                raise ValueError(
                    f"{speaker}台词与本Session已有长台词精确重复"
                )
            seen_by_speaker[speaker].add(normalized)
        actual_refs = set(turn["memory_refs"])
        if not expected_refs.issubset(actual_refs):
            raise ValueError(
                "memory_refs缺少本轮计划引入或调用的memory_id；"
                f"turn={turn['turn_in_session']}，"
                f"expected={sorted(expected_refs)}，"
                f"actual={sorted(actual_refs)}"
            )
        turn["memory_refs"] = sorted(expected_refs)
        expected_effect = required_effects.get(
            int(turn["turn_in_session"])
        )
        effect_refs = turn["anchor_effect_refs"]
        if effect_refs is None and expected_effect is None:
            effect_refs = []
        elif isinstance(effect_refs, str):
            effect_refs = [effect_refs]
        if not isinstance(effect_refs, list) or not all(
            isinstance(anchor_id, str) for anchor_id in effect_refs
        ):
            raise ValueError("anchor_effect_refs必须是anchor_id字符串数组")
        if expected_effect is not None and expected_effect not in effect_refs:
            raise ValueError("anchor_effect_refs缺少本轮要求体现的anchor_id")
        turn["anchor_effect_refs"] = (
            [expected_effect] if expected_effect is not None else []
        )


def _coerce_plan_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value:
        text = "；".join(str(item).strip() for item in value if str(item).strip())
        if text:
            return text
    if isinstance(value, Mapping) and value:
        text = "；".join(
            f"{key}：{item}" for key, item in value.items()
        )
        if text:
            return text
    raise ValueError("session主题字段必须是非空文本")


def _hidden_event_overlap(
    text: str,
    forbidden_knowledge: list[dict[str, Any]],
) -> str | None:
    normalized_text = _normalized_utterance(text)
    if not normalized_text:
        return None
    text_grams = {
        normalized_text[index : index + 4]
        for index in range(max(0, len(normalized_text) - 3))
    }
    for item in forbidden_knowledge:
        event = str(item.get("event", ""))
        normalized_event = _normalized_utterance(event)
        event_grams = {
            normalized_event[index : index + 4]
            for index in range(max(0, len(normalized_event) - 3))
        }
        if text_grams & event_grams:
            return str(item.get("anchor_id") or event)
    return None


def _validate_utterance(
    utterance: Any,
    *,
    speaker: str,
    npc_name: str = "",
) -> None:
    if not isinstance(utterance, str) or not 4 <= len(utterance.strip()) <= 300:
        raise ValueError(
            f"{speaker} utterance必须是4到300字的有效台词；"
            f"actual={utterance!r}"
        )
    text = utterance.strip()
    if (
        speaker == "npc"
        and npc_name
        and any(text.startswith(name) for name in _npc_name_variants(npc_name))
    ):
        raise ValueError("NPC台词不得以角色名或第三人称角色叙述开头")
    if re.match(r"^(player|npc|玩家|角色)\s*[:：]", text, re.IGNORECASE):
        raise ValueError("utterance不得携带说话人标签")
    for left, right in (("“", "”"), ("‘", "’")):
        if text.count(left) != text.count(right):
            raise ValueError("utterance包含不配对引号")
    if text.count('"') % 2:
        raise ValueError("utterance包含不配对引号")
    if any(term in utterance for term in _SCHEMA_TERMS):
        raise ValueError("utterance泄漏内部schema术语")
    if any(phrase in utterance for phrase in _BAD_PHRASES):
        raise ValueError("utterance使用了禁止的模板化套话")
    if _has_bad_repetition(utterance):
        raise ValueError("utterance包含明显重复词")
    if utterance.lstrip().startswith(("（", "(", "[")):
        raise ValueError("utterance不得包含舞台动作或括号旁白")


def _npc_name_variants(npc_name: str) -> set[str]:
    variants = {npc_name.strip()}
    if (
        len(npc_name.strip()) in {3, 4}
        and all("\u4e00" <= char <= "\u9fff" for char in npc_name.strip())
    ):
        variants.add(npc_name.strip()[1:])
    return {value for value in variants if value}


def _normalized_utterance(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def _validate_sample_audit(value: dict[str, Any]) -> None:
    boolean_fields = {
        "continuity_pass",
        "anchor_effect_pass",
        "cross_event_pass",
        "history_value_pass",
        "player_agency_pass",
        "schema_safe_pass",
        "role_consistency_pass",
        "pattern_diversity_pass",
    }
    if set(value) != boolean_fields | {"issues"}:
        raise ValueError("抽样审计字段不符合契约")
    if not all(isinstance(value[key], bool) for key in boolean_fields):
        raise ValueError("抽样审计标签必须为boolean")
    if not isinstance(value["issues"], list) or not all(
        isinstance(issue, str) for issue in value["issues"]
    ):
        raise ValueError("抽样审计issues必须是字符串数组")


def _sample_audit_passed(audit: Mapping[str, Any]) -> bool:
    return all(
        bool(audit[key])
        for key in (
            "continuity_pass",
            "anchor_effect_pass",
            "cross_event_pass",
            "history_value_pass",
            "player_agency_pass",
            "schema_safe_pass",
            "role_consistency_pass",
            "pattern_diversity_pass",
        )
    )


def _attach_cross_session_memories(
    plans: list[dict[str, Any]],
) -> None:
    for index, plan in enumerate(plans):
        carry: list[dict[str, Any]] = []
        offsets = (1, 2 + index % 3, 5 + index % 4)
        carry_count = index % 3
        for position, offset in enumerate(offsets[:carry_count]):
            source_index = index - offset
            if source_index < 0:
                continue
            source = plans[source_index]["memory_seeds"][0]
            act_ranges = plan["acts"]
            target_act = act_ranges[
                (index + position * 2) % len(act_ranges)
            ]["turn_range"]
            recall_turn = target_act[
                (index + position) % len(target_act)
            ]
            carry.append(
                {
                    "memory_id": source["memory_id"],
                    "content": source["content"],
                    "source_session_id": plans[source_index]["session_id"],
                    "recall_turn": int(recall_turn),
                }
            )
        plan["carry_in_memories"] = carry


def _write_plans(plans_dir: Path, plans: list[dict[str, Any]]) -> None:
    for plan in plans:
        path = plans_dir / f"{plan['session_id']}.json"
        path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _seed_session_cache(
    history: list[dict[str, Any]],
    blueprints: list[dict[str, Any]],
    sessions_dir: Path,
) -> None:
    if not history or any(
        record.get("generation_meta", {}).get("history_layout_version")
        != _HISTORY_LAYOUT_VERSION
        for record in history
    ):
        return
    by_session: dict[str, list[dict[str, Any]]] = {}
    for record in history:
        by_session.setdefault(str(record["session_id"]), []).append(record)
    expected_ids = {str(item["session_id"]) for item in blueprints}
    if set(by_session) != expected_ids:
        return
    for blueprint in blueprints:
        session_id = str(blueprint["session_id"])
        records = sorted(
            by_session[session_id],
            key=lambda record: int(record["turn_in_session"]),
        )
        if len(records) != int(blueprint["session_size"]):
            return
        if [int(record["round"]) for record in records] != list(
            range(
                int(blueprint["start_round"]),
                int(blueprint["start_round"]) + int(blueprint["session_size"]),
            )
        ):
            return
    for session_id, records in by_session.items():
        write_jsonl(
            sessions_dir / f"{session_id}.jsonl",
            sorted(records, key=lambda record: int(record["turn_in_session"])),
        )


def _completed_generation_calls(
    plans_dir: Path,
    sessions_dir: Path,
    audits_dir: Path,
    blueprints: list[dict[str, Any]],
) -> int:
    completed = len(list(plans_dir.glob("*.json")))
    completed += int((audits_dir / "summary.json").is_file())
    ranges_by_session = {
        blueprint["session_id"]: blueprint["act_ranges"]
        for blueprint in blueprints
    }
    for path in sessions_dir.glob("*.jsonl"):
        record_count = len(read_jsonl(path))
        completed += sum(
            turn_range[1] <= record_count
            for turn_range in ranges_by_session.get(path.stem, [])
        )
    return completed


def _config_for_session(
    config: GenerationConfig,
    session_index: int,
    *,
    max_tokens: int | None = None,
) -> GenerationConfig:
    return GenerationConfig(
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=max_tokens or min(config.max_tokens, 2048),
        seed=(config.seed or 0) + session_index,
        max_format_retries=config.max_format_retries,
    )


def _generate_history_structured(
    client: LLMClient,
    bundle: PromptBundle,
    config: GenerationConfig,
    validator: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Regenerate once when format repair cannot fix generated content."""

    last_error: StructuredOutputError | None = None
    for fresh_attempt in range(2):
        try:
            result = generate_structured(
                client,
                bundle,
                config,
                validator,
            )
            setattr(client, "last_fresh_retries", fresh_attempt)
            return result
        except StructuredOutputError as error:
            last_error = error
    setattr(client, "last_fresh_retries", 1)
    if last_error is not None:
        raise last_error
    raise StructuredOutputError("历史生成失败")


def _state_projection(context: GameContext) -> dict[str, Any]:
    return {
        "runtime_state": copy.deepcopy(context.runtime_state),
        "environment": copy.deepcopy(context.environment),
    }


def _has_bad_repetition(text: str) -> bool:
    if re.search(r"(目前目前|先先|继续继续|已经已经|现在现在)", text):
        return True
    compact = re.sub(r"\s+", "", text)
    return any(
        compact[index : index + size]
        == compact[index + size : index + size * 2]
        == compact[index + size * 2 : index + size * 3]
        for size in range(2, 9)
        for index in range(0, len(compact) - size * 3 + 1)
    )
