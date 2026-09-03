"""LLM-driven, session-coherent long-history generation."""

from __future__ import annotations

import copy
import json
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping

from agents import project_npc_character_card
from agents.prompt_builder import PromptBundle
from gamecore import FixtureEngine, GameContext, WorldDefinition
from llm import (
    GenerationConfig,
    LLMClient,
    generate_structured,
)
from runners.progress import ProgressReporter

from .common import load_yaml, read_jsonl, write_jsonl
from .history_sources import load_reviewed_anchors, load_transition_plan


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
)
_CHUNK_SIZE = 5


def generate_llm_history(
    world_dir: str | Path,
    character_id: str,
    *,
    client_factory: Callable[[], LLMClient],
    config: GenerationConfig,
    rounds: int = 600,
    workers: int = 6,
    work_root: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one frozen history through planned Player/NPC interaction."""

    if workers < 1:
        raise ValueError("workers必须至少为1")
    world = Path(world_dir)
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

    completed_calls = _completed_generation_calls(
        plans_dir,
        sessions_dir,
        audits_dir,
    )
    progress = ProgressReporter(
        total=len(blueprints) * 2 + rounds // _CHUNK_SIZE,
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
    llm_audits = _audit_sessions(
        world,
        character_id,
        blueprints,
        histories,
        audits_dir,
        client_factory,
        config,
        workers,
        progress,
    )
    surface_quality = audit_surface_quality(history)
    quality = {
        **surface_quality,
        "session_audits": llm_audits,
        "llm_session_pass_rate": sum(
            all(
                audit[key]
                for key in (
                    "continuity_pass",
                    "player_naturalness_pass",
                    "npc_style_pass",
                    "state_effect_visible",
                    "memory_value_pass",
                )
            )
            for audit in llm_audits
        )
        / len(llm_audits),
    }
    quality["passed"] = bool(
        surface_quality["passed"]
        and quality["llm_session_pass_rate"] >= 0.9
    )
    report = {
        "generation_mode": "llm_interactive_sessions",
        "sessions": len(blueprints),
        "rounds": len(history),
        "logical_api_calls": (
            len(blueprints) * 2 + rounds // _CHUNK_SIZE
        ),
        "workers": workers,
        "quality": quality,
    }
    if temporary_work is not None:
        temporary_work.cleanup()
    return history, report


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
    if rounds % len(transitions) != 0:
        raise ValueError("rounds必须能被transition数量整除")
    session_size = rounds // len(transitions)
    if session_size < 12:
        raise ValueError("每个session至少需要12轮以形成连续事件")
    if session_size % _CHUNK_SIZE != 0:
        raise ValueError(f"每个session轮数必须能被{_CHUNK_SIZE}整除")

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
    prior_anchors: list[dict[str, str]] = []
    for index, transition in enumerate(transitions, start=1):
        anchor = anchors[transition["anchor_id"]]
        anchor_low = max(3, round(session_size * 0.4))
        anchor_high = min(session_size - 3, round(session_size * 0.7))
        anchor_turn = anchor_low + (
            (index * 3) % (anchor_high - anchor_low + 1)
        )
        global_anchor_round = (index - 1) * session_size + anchor_turn
        pre_state = _state_projection(context)
        event_id = f"anchor_{anchor['id']}"
        applied = fixture.apply(
            context,
            event_id=event_id,
            operations=transition["operations"],
            description=anchor["event"],
            history_turn=global_anchor_round,
        )
        context = applied.context
        post_state = _state_projection(context)
        blueprints.append(
            {
                "session_index": index,
                "session_id": f"{character_id}_session_{index:02d}",
                "start_round": (index - 1) * session_size + 1,
                "session_size": session_size,
                "anchor_turn": anchor_turn,
                "anchor": anchor,
                "transition": transition,
                "event_id": event_id,
                "pre_state": pre_state,
                "post_state": post_state,
                "context_delta": applied.context_delta,
                "prior_anchors": copy.deepcopy(prior_anchors[-5:]),
                "player_profile": copy.deepcopy(player_profile),
            }
        )
        prior_anchors.append(
            {
                "event_id": event_id,
                "event": anchor["event"],
                "behavioral_consequence": str(transition["npc_surface"]),
            }
        )
    return blueprints


def _load_history_player_profile(
    world_path: Path,
    character_id: str,
) -> dict[str, str]:
    for scenario_path in sorted((world_path / "scenarios").glob("*.yaml")):
        scenario = load_yaml(scenario_path)
        if scenario.get("character_id") != character_id:
            continue
        profile = scenario.get("player_profile")
        if isinstance(profile, Mapping) and profile.get("role"):
            return {
                "role": str(profile["role"]),
                "public_identity": str(
                    profile.get("public_identity", "")
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
    return {
        "unique_utterance_ratio": (
            len(set(utterances)) / len(utterances) if utterances else 0.0
        ),
        "schema_leaks": schema_leaks,
        "bad_phrases": bad_phrases,
        "repeated_word_lines": repeated,
        "sessions_with_memory_recall": len(
            {
                record["session_id"]
                for record in history
                if record.get("generation_meta", {}).get(
                    "recalled_memories", []
                )
            }
        ),
        "passed": not schema_leaks and not bad_phrases and not repeated,
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
    card = project_npc_character_card(
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
        output = generate_structured(
            client,
            bundle,
            plan_config,
            lambda value: _validate_plan(
                value,
                blueprint["session_id"],
                blueprint["session_size"],
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
    npc_card = project_npc_character_card(card)
    public_card = {
        "name": card["identity"]["name"],
        "role": card["identity"]["role"],
        "public_background": card["identity"].get("public_background", []),
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
        if len(records) % _CHUNK_SIZE != 0:
            raise ValueError(f"{path}不是完整的{_CHUNK_SIZE}轮分块")
        client = client_factory()
        transcript = [
            message
            for record in records
            for message in record["messages"]
        ]
        for chunk_start in range(
            len(records) + 1,
            blueprint["session_size"] + 1,
            _CHUNK_SIZE,
        ):
            chunk_end = min(
                chunk_start + _CHUNK_SIZE - 1,
                blueprint["session_size"],
            )
            act = plan["acts"][(chunk_start - 1) // _CHUNK_SIZE]
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
                if chunk_start <= int(memory["recall_turn"]) <= chunk_end
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
            )
            output = generate_structured(
                client,
                bundle,
                _config_for_session(config, blueprint["session_index"]),
                lambda value, start=chunk_start, end=chunk_end: (
                    _validate_dialogue_chunk(value, start, end)
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


def _audit_sessions(
    world: Path,
    character_id: str,
    blueprints: list[dict[str, Any]],
    histories: list[list[dict[str, Any]]],
    audits_dir: Path,
    client_factory: Callable[[], LLMClient],
    config: GenerationConfig,
    workers: int,
    progress: ProgressReporter,
) -> list[dict[str, Any]]:
    card = project_npc_character_card(
        load_yaml(world / "characters" / f"{character_id}.yaml")
    )

    def audit(
        blueprint: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        path = audits_dir / f"{blueprint['session_id']}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        bundle = PromptBundle(
            system_prompt=(
                "你是中文长篇角色对话数据质检员。判断session是否为连续发生的"
                "一个局部事件，Player是否承接NPC，NPC是否像具体角色而非"
                "评测器，Anchor状态是否在后续语言行为中可观察，普通细节是否"
                "具有后续记忆价值。严格审核，只输出JSON。"
            ),
            user_prompt=(
                f"角色指导：{json.dumps(card, ensure_ascii=False)}\n"
                f"Anchor：{json.dumps(blueprint['anchor'], ensure_ascii=False)}\n"
                f"状态变化：{json.dumps(blueprint['context_delta'], ensure_ascii=False)}\n"
                "Anchor后的行为要求："
                f"{blueprint['transition']['npc_surface']}\n"
                "Session："
                f"{json.dumps(_audit_history_projection(history), ensure_ascii=False)}\n\n"
                "输出字段：continuity_pass、player_naturalness_pass、"
                "npc_style_pass、state_effect_visible、memory_value_pass"
                "（均为boolean），issues（字符串数组）。"
            ),
        )
        client = client_factory()
        result = generate_structured(
            client,
            bundle,
            _config_for_session(config, blueprint["session_index"]),
            _validate_session_audit,
        )
        result["usage"] = copy.deepcopy(client.last_usage)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        progress.advance(f"history audit={blueprint['session_id']}")
        return result

    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(histories))) as pool:
        futures = {
            pool.submit(audit, blueprint, history): blueprint[
                "session_index"
            ]
            for blueprint, history in zip(
                blueprints,
                histories,
                strict=True,
            )
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[index] for index in sorted(results)]


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
        }
        for record in history
    ]


def _plan_bundle(
    card: dict[str, Any],
    environment: dict[str, Any],
    blueprint: dict[str, Any],
) -> PromptBundle:
    anchor = blueprint["anchor"]
    transition = blueprint["transition"]
    payload = {
        "session_id": blueprint["session_id"],
        "turns": blueprint["session_size"],
        "fixed_player_profile": blueprint["player_profile"],
        "anchor_turn": blueprint["anchor_turn"],
        "anchor_event": anchor["event"],
        "anchor_visible_to_player": bool(transition["player_visible"]),
        "anchor_behavioral_consequence": transition["npc_surface"],
        "prior_events": blueprint["prior_anchors"],
        "character": card,
        "locations": environment["locations"],
    }
    return PromptBundle(
        system_prompt=(
            "你是长篇谍战角色扮演数据的session规划器。规划一段真正连续的"
            "多轮对话。一个session只能围绕一项具体局部事务，不能并列家庭、"
            "站内派系和多条案件线。不得把每轮写成独立问答。局部事件应逐步"
            "出现新证据、回应和后果；Player下一轮必须能承接NPC上一轮。"
            "Anchor发生前，任何人都不得预知其结果；Anchor发生后，NPC后续"
            "处理方式应持续体现变化。包含Anchor及之后的每个act都必须写出"
            "至少一个由Anchor导致的具体决策差异，例如改变核查渠道、保护对象、"
            "信息流向或承担风险的方式；不能只写成更谨慎、更警觉。"
            "不要使用状态机、标注或评测术语。"
            "只输出JSON对象。"
        ),
        user_prompt=(
            f"规划约束：{json.dumps(payload, ensure_ascii=False)}\n\n"
            "输出字段必须为：session_id；local_event；setting；stakes；"
            "acts（每5轮一个阶段，含act、turn_range、development、"
            "player_intent、npc_response_strategy）；memory_seeds（2到3项，含memory_id、"
            "content、introduced_turn、recall_turn、introduced_by，其中introduced_by"
            "只能是player或npc）。memory内容应是后文可重新引用的"
            "人、文件、时间、异常或临时决定；recall_turn必须晚于"
            "introduced_turn，后文必须让人物用该细节作出判断或推进事件。"
            "Anchor若对Player不可见，"
            "不得让Player主动说出该事件。player_intent专指与NPC对话的"
            "Player本阶段会提出什么请求、问题或压力；npc_response_strategy"
            "专指余则成本人如何回应和处理，不能写成吴敬中、李涯等配角的"
            "目标。所有session中的Player必须始终采用fixed_player_profile，"
            "不能自行改成主任、站长、行动人员或其他身份。"
            "local_event、setting、stakes"
            "必须各自是一个字符串，不能输出数组或对象。acts必须依次使用"
            "act=1..4和turn_range=[1,5],[6,10],[11,15],[16,20]。"
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
) -> PromptBundle:
    anchor_in_chunk = (
        chunk_start <= blueprint["anchor_turn"] <= chunk_end
    )
    anchor_already_occurred = chunk_start > blueprint["anchor_turn"]
    payload = {
        "npc_public_identity": public_card,
        "npc_private_character_guidance": npc_card,
        "fixed_player_profile": blueprint["player_profile"],
        "world": environment["name"],
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
        "anchor": (
            {
                "turn": blueprint["anchor_turn"],
                "event": blueprint["anchor"]["event"],
                "visible_to_player": bool(
                    blueprint["transition"]["player_visible"]
                ),
                "behavior_after_event": blueprint["transition"][
                    "npc_surface"
                ],
            }
            if anchor_in_chunk or anchor_already_occurred
            else None
        ),
        "anchor_phase": (
            "occurs_in_this_chunk"
            if anchor_in_chunk
            else "already_occurred"
            if anchor_already_occurred
            else "not_yet_occurred"
        ),
        "current_npc_state": (
            blueprint["post_state"]
            if anchor_already_occurred
            else blueprint["pre_state"]
        ),
        "state_after_anchor": (
            blueprint["post_state"] if anchor_in_chunk else None
        ),
        "prior_long_term_events": blueprint["prior_anchors"],
    }
    return PromptBundle(
        system_prompt=(
            "你是谍战长对话编剧，同时生成Player与NPC连续5轮左右的真实对话。"
            "每个Player发言必须承接上一句NPC，每个NPC回答既回应当前问题又"
            "推进同一局部事件。不得写成独立问答集合。NPC内部状态只能通过"
            "具体选择、渠道、人物和语气体现，禁止解释状态字段或规则。若Anchor"
            "不对Player可见，Player绝不能说出或暗示自己知道该事件。不得使用"
            "括号舞台动作，不得复用谨慎套话。Player必须始终符合"
            "fixed_player_profile，不能临时获得新的职务或权限。"
            "若anchor_phase为already_occurred，本段至少一个NPC决定必须具体"
            "体现behavior_after_event，但不得直接解释内心状态或政治标签。"
            "details_to_introduce_naturally必须在指定轮首次出现，"
            "past_details_to_recall_naturally必须在指定轮被再次引用并影响判断。"
            "Player不得原样重复已被拒绝的请求，不得在走廊等公开环境直说"
            "接头暗号、地下身份或其他敏感内容；不得擅自伪造、涂改或销毁记录，"
            "也不得做超出fixed_player_profile权限的决定。新消息必须交代可信"
            "来源或承认无法核验，不能只为推动剧情突然抛出。"
            "只输出JSON。"
        ),
        user_prompt=(
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            '输出：{"turns":[{"turn_in_session":整数,'
            '"player":"Player自然台词","npc":"NPC角色化台词"}]}。'
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
    due_memories: list[dict[str, Any]],
    recalled: list[dict[str, Any]],
    generation_usage: dict[str, Any] | None,
) -> dict[str, Any]:
    round_number = blueprint["start_round"] + turn - 1
    is_anchor = turn == blueprint["anchor_turn"]
    transition = blueprint["transition"]
    anchor = blueprint["anchor"]
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
                "event_id": blueprint["event_id"],
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
                "pre_state": blueprint["pre_state"],
                "operations": transition["operations"],
                "context_delta": blueprint["context_delta"],
                "post_state": blueprint["post_state"],
            }
            if is_anchor
            else None
        ),
        "generation_meta": {
            "local_event": plan["local_event"],
            "beat": beat,
            "introduced_memories": [
                memory["memory_id"] for memory in due_memories
            ],
            "recalled_memories": [
                memory["memory_id"] for memory in recalled
            ],
            "generation_usage": generation_usage,
        },
    }


def _validate_plan(
    value: dict[str, Any],
    session_id: str,
    session_size: int,
) -> None:
    required = {
        "session_id",
        "local_event",
        "setting",
        "stakes",
        "acts",
        "memory_seeds",
    }
    if set(value) != required or value["session_id"] != session_id:
        raise ValueError("session计划字段或session_id错误")
    for key in ("local_event", "setting", "stakes"):
        value[key] = _coerce_plan_text(value[key])
    acts = value["acts"]
    expected_count = session_size // _CHUNK_SIZE
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
        expected_range = [
            (index - 1) * _CHUNK_SIZE + 1,
            index * _CHUNK_SIZE,
        ]
        if not isinstance(act, Mapping) or set(act) != act_fields:
            raise ValueError(f"第{index}个act字段不符合契约")
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
    for memory in memories:
        if set(memory) != {
            "memory_id",
            "content",
            "introduced_turn",
            "recall_turn",
            "introduced_by",
        }:
            raise ValueError("memory_seed字段不符合契约")
        if not isinstance(memory["memory_id"], str) or not isinstance(
            memory["content"], str
        ):
            raise ValueError("memory_seed文本字段错误")
        introduced = min(
            max(int(memory["introduced_turn"]), 2),
            session_size - 2,
        )
        recalled = min(
            max(int(memory["recall_turn"]), introduced + 1),
            session_size,
        )
        memory["introduced_turn"] = introduced
        memory["recall_turn"] = recalled
        if memory["introduced_by"] not in {"player", "npc"}:
            raise ValueError("memory_seed.introduced_by必须为player或npc")
        memory_ids.add(memory["memory_id"])
    if len(memory_ids) != len(memories):
        raise ValueError("memory_id必须唯一")


def _validate_dialogue_chunk(
    value: dict[str, Any],
    chunk_start: int,
    chunk_end: int,
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
    for turn in value["turns"]:
        if set(turn) != {"turn_in_session", "player", "npc"}:
            raise ValueError("分块中的单轮字段不符合契约")
        _validate_utterance(turn["player"])
        _validate_utterance(turn["npc"])


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


def _validate_utterance(utterance: Any) -> None:
    if not isinstance(utterance, str) or not 2 <= len(utterance.strip()) <= 300:
        raise ValueError("utterance必须是2到300字的非空字符串")
    if any(term in utterance for term in _SCHEMA_TERMS):
        raise ValueError("utterance泄漏内部schema术语")
    if any(phrase in utterance for phrase in _BAD_PHRASES):
        raise ValueError("utterance使用了禁止的模板化套话")
    if _has_bad_repetition(utterance):
        raise ValueError("utterance包含明显重复词")
    if utterance.lstrip().startswith(("（", "(", "[")):
        raise ValueError("utterance不得包含舞台动作或括号旁白")


def _validate_session_audit(value: dict[str, Any]) -> None:
    boolean_fields = {
        "continuity_pass",
        "player_naturalness_pass",
        "npc_style_pass",
        "state_effect_visible",
        "memory_value_pass",
    }
    if set(value) != boolean_fields | {"issues"}:
        raise ValueError("session审计字段不符合契约")
    if not all(isinstance(value[key], bool) for key in boolean_fields):
        raise ValueError("session审计标签必须为boolean")
    if not isinstance(value["issues"], list) or not all(
        isinstance(issue, str) for issue in value["issues"]
    ):
        raise ValueError("session审计issues必须是字符串数组")


def _attach_cross_session_memories(
    plans: list[dict[str, Any]],
) -> None:
    offsets = (1, 3, 7)
    recall_turns = (4, 10, 16)
    for index, plan in enumerate(plans):
        carry: list[dict[str, Any]] = []
        for offset, recall_turn in zip(offsets, recall_turns, strict=True):
            source_index = index - offset
            if source_index < 0:
                continue
            source = plans[source_index]["memory_seeds"][0]
            carry.append(
                {
                    "memory_id": source["memory_id"],
                    "content": source["content"],
                    "source_session_id": plans[source_index]["session_id"],
                    "recall_turn": min(
                        recall_turn,
                        int(plan["acts"][-1]["turn_range"][-1]),
                    ),
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


def _completed_generation_calls(
    plans_dir: Path,
    sessions_dir: Path,
    audits_dir: Path,
) -> int:
    completed = len(list(plans_dir.glob("*.json")))
    completed += len(list(audits_dir.glob("*.json")))
    for path in sessions_dir.glob("*.jsonl"):
        completed += len(read_jsonl(path)) // _CHUNK_SIZE
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
        for size in range(2, 7)
        for index in range(0, len(compact) - size * 2 + 1)
    )
