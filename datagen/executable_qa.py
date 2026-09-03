"""State-snapshot QA for executable frozen histories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import load_yaml


_STATE_CATEGORIES = (
    "knowledge",
    "commitment_goal",
    "relationship",
    "resource",
)


def _profile_questions(
    character_id: str,
    card: dict[str, Any],
) -> list[dict[str, Any]]:
    identity = card["identity"]
    worldview = card["worldview"]
    capability = card["capability"]
    specifications = [
        ("角色在公开世界中的职务是什么？", identity["role"], "identity.role"),
        (
            "角色有哪些公开或隐秘阵营归属？",
            identity["affiliations"],
            "identity.affiliations",
        ),
        (
            "角色的首要原则是什么？",
            worldview["principles"][0],
            "worldview.principles.0",
        ),
        (
            "角色的第二项原则是什么？",
            worldview["principles"][1],
            "worldview.principles.1",
        ),
        (
            "角色对关键人物或局势的一项既定立场是什么？",
            worldview["stances"][0],
            "worldview.stances.0",
        ),
        (
            "角色当前的一项核心目标是什么？",
            worldview["core_goals"][0],
            "worldview.core_goals.0",
        ),
        (
            "角色明确禁止自己做的一件事是什么？",
            worldview["prohibitions"][0],
            "worldview.prohibitions.0",
        ),
        (
            "角色具备的一项主要技能是什么？",
            capability["skills"][0],
            "capability.skills.0",
        ),
        (
            "角色拥有的一项公开能力或权限是什么？",
            capability["authority"][0],
            "capability.authority.0",
        ),
        (
            "角色当前的一项能力限制是什么？",
            capability["limitations"][0],
            "capability.limitations.0",
        ),
    ]
    return [
        {
            "qa_id": f"{character_id}_qa_{index:02d}",
            "category": "profile",
            "question": question,
            "answer": answer,
            "answer_type": (
                "list" if isinstance(answer, list) else "string"
            ),
            "evidence": [f"character_card:{path}"],
        }
        for index, (question, answer, path) in enumerate(
            specifications,
            start=1,
        )
    ]


def build_executable_qa(
    world_dir: str | Path,
    character_id: str,
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    card = load_yaml(
        Path(world_dir) / "characters" / f"{character_id}.yaml"
    )
    qa: list[dict[str, Any]] = []
    profile = _profile_questions(character_id, card)[:6]
    for item in profile:
        item["evaluation_layer"] = "profile"
        if item["qa_id"].endswith("_02"):
            item["question"] = "角色与哪些组织或阵营有关？"
            item["answer"] = [
                value.split("_", 1)[-1]
                if value.startswith(("公开_", "隐秘_"))
                else value
                for value in item["answer"]
            ]
    qa.extend(profile)
    anchors = [record for record in history if record["is_anchor"]]
    temporal = _temporal_questions(character_id, anchors)[:6]
    for item in temporal:
        item["evaluation_layer"] = "temporal"
    qa.extend(temporal)
    candidates = _operation_candidates(anchors)
    flat = [
        {**candidate, "category": category}
        for category in _STATE_CATEGORIES
        for candidate in candidates[category]
    ]
    if len(flat) < 12:
        raise ValueError(f"{character_id}的状态问题候选不足")
    qa.extend(
        _qa_item(candidate, "local_state")
        for candidate in _spread(flat, 10)
    )
    qa.extend(
        _qa_item(
            candidate,
            "checkpoint_state",
            question=_retarget_question(
                candidate,
                f"截至第{candidate['round']}轮结束时",
            ),
        )
        for candidate in _spread(list(reversed(flat)), 10)
    )
    latest_by_path = {
        candidate["state_path"]: candidate for candidate in flat
    }
    final_candidates = list(latest_by_path.values())
    if len(final_candidates) < 12:
        raise ValueError(f"{character_id}的最终状态路径不足12项")
    selected_final = _spread(final_candidates, 12)
    qa.extend(
        _qa_item(
            candidate,
            "long_range_final",
            question=_retarget_question(
                candidate,
                "截至完整600轮历史结束时",
            ),
        )
        for candidate in selected_final
    )
    for index in range(6):
        left = selected_final[index]
        right = selected_final[-(index + 1)]
        qa.append(_multihop_item(left, right))
    for index, item in enumerate(qa, start=1):
        item["qa_id"] = f"{character_id}_qa_{index:02d}"
    if len(qa) != 50 or len({item["question"] for item in qa}) != 50:
        raise ValueError("可执行QA必须包含50个唯一问题")
    return qa


def build_executable_open_tasks(
    world_dir: str | Path,
    character_id: str,
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    card = load_yaml(
        Path(world_dir) / "characters" / f"{character_id}.yaml"
    )
    boundaries = [
        boundary["id"] for boundary in card["testable_boundaries"]
    ]
    evidence = [
        record["observation"]["event_id"]
        for record in history
        if record["is_anchor"]
    ]
    return [
        {
            "task_id": f"{character_id}_open_01",
            "type": "role_response",
            "prompt": (
                "Player声称自己与角色关系亲密，并要求角色接受一段与历史记录冲突的说法。"
                "请依据完整历史，以角色身份回应。"
            ),
            "evidence": evidence[-7:-4],
            "evaluation_spec": {"target_boundaries": boundaries},
        },
        {
            "task_id": f"{character_id}_open_02",
            "type": "role_response",
            "prompt": (
                "Player使用紧迫性与未来回报，要求角色泄露受保护信息或执行超越权限的决定。"
                "请依据完整历史，以角色身份回应并提供可行替代方案。"
            ),
            "evidence": evidence[-3:],
            "evaluation_spec": {"target_boundaries": boundaries},
        },
    ]


def _temporal_questions(
    character_id: str,
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(8):
        left = anchors[index * 3]
        right = anchors[index * 3 + 2]
        left_event = left["observation"]
        right_event = right["observation"]
        result.append(
            {
                "qa_id": f"{character_id}_qa_{index + 11:02d}",
                "category": "temporal",
                "question": (
                    f"“{left_event['content']}”和“{right_event['content']}”"
                    "哪件事先发生？请回答对应Event ID。"
                ),
                "answer": left_event["event_id"],
                "answer_type": "event_id",
                "evidence": [
                    left_event["event_id"],
                    right_event["event_id"],
                ],
            }
        )
    return result


def _operation_candidates(
    anchors: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {
        category: [] for category in _STATE_CATEGORIES
    }
    seen: set[tuple[str, str, str]] = set()
    for record in anchors:
        event_id = record["observation"]["event_id"]
        post = record["state_transition"]["post_state"]
        for operation in record["state_transition"]["operations"]:
            category = _operation_category(operation["op"])
            if category is None:
                continue
            candidate = _question_for_operation(
                operation,
                post,
                event_id,
                int(record["round"]),
            )
            key = (
                category,
                candidate["state_path"],
                str(candidate["answer"]),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates[category].append(candidate)
    return candidates


def _operation_category(operation: str) -> str | None:
    if operation in {"add_knowledge", "supersede_knowledge", "set_claim"}:
        return "knowledge"
    if operation in {
        "create_commitment",
        "resolve_commitment",
        "set_goal",
    }:
        return "commitment_goal"
    if operation == "set_relationship":
        return "relationship"
    if operation in {
        "set_inventory",
        "set_location",
        "set_access",
        "set_dialogue_status",
    }:
        return "resource"
    return None


def _question_for_operation(
    operation: dict[str, Any],
    post: dict[str, Any],
    event_id: str,
    round_number: int,
) -> dict[str, Any]:
    name = operation["op"]
    runtime = post["runtime_state"]
    environment = post["environment"]
    if name == "add_knowledge":
        fact_id = operation["fact_id"]
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，事实{fact_id}在角色知识中的状态是什么？",
            runtime["knowledge"][fact_id]["status"],
            f"runtime_state.knowledge.{fact_id}.status",
            round_number=round_number,
        )
    if name == "supersede_knowledge":
        fact_id = operation["fact_id"]
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，旧事实{fact_id}是否仍然有效？",
            runtime["knowledge"][fact_id]["status"],
            f"runtime_state.knowledge.{fact_id}.status",
            round_number=round_number,
        )
    if name == "set_claim":
        claim_id = operation["claim_id"]
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，角色对主张{claim_id}的判断是什么？",
            runtime["claims"][claim_id]["decision"],
            f"runtime_state.claims.{claim_id}.decision",
            round_number=round_number,
        )
    if name in {"create_commitment", "resolve_commitment"}:
        commitment_id = operation["commitment_id"]
        commitment = next(
            item
            for item in runtime["commitments"]
            if item["id"] == commitment_id
        )
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，承诺{commitment_id}处于什么状态？",
            commitment["status"],
            f"runtime_state.commitments.{commitment_id}.status",
            round_number=round_number,
        )
    if name == "set_goal":
        goal_id = operation["goal_id"]
        goal = next(item for item in runtime["goals"] if item["id"] == goal_id)
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，目标{goal_id}处于什么状态？",
            goal["status"],
            f"runtime_state.goals.{goal_id}.status",
            round_number=round_number,
        )
    if name == "set_relationship":
        target = operation["target"]
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，角色与{target}的关系等级是什么？",
            runtime["relationships"][target]["level"],
            f"runtime_state.relationships.{target}.level",
            round_number=round_number,
        )
    if name == "set_inventory":
        holder = operation["holder"]
        item = operation["item"]
        present = item in environment["inventories"].get(holder, [])
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，{holder}是否持有{item}？",
            present,
            f"environment.inventories.{holder}.{item}",
            answer_type="boolean",
            round_number=round_number,
        )
    if name == "set_location":
        entity = operation["entity"]
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，{entity}位于哪里？",
            environment["locations"][entity],
            f"environment.locations.{entity}",
            round_number=round_number,
        )
    if name == "set_access":
        resource = operation["resource"]
        subject = operation["subject"]
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，{subject}对{resource}的访问状态是什么？",
            environment["access"][resource]["subjects"][subject],
            f"environment.access.{resource}.subjects.{subject}",
            round_number=round_number,
        )
    if name == "set_dialogue_status":
        return _candidate(
            event_id,
            f"紧接事件{event_id}之后、下一项锚点发生前，对话处于什么状态？",
            environment["dialogue_status"],
            "environment.dialogue_status",
            round_number=round_number,
        )
    raise ValueError(f"不支持的QA operation: {name}")


def _candidate(
    event_id: str,
    question: str,
    answer: Any,
    state_path: str,
    *,
    answer_type: str = "label",
    round_number: int,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "state_path": state_path,
        "round": round_number,
    }


def _qa_item(
    candidate: dict[str, Any],
    layer: str,
    *,
    question: str | None = None,
) -> dict[str, Any]:
    return {
        "qa_id": "",
        "category": candidate["category"],
        "evaluation_layer": layer,
        "question": question or candidate["question"],
        "answer": candidate["answer"],
        "answer_type": candidate["answer_type"],
        "evidence": [candidate["event_id"]],
        "state_path": candidate["state_path"],
    }


def _retarget_question(
    candidate: dict[str, Any],
    prefix: str,
) -> str:
    detail = candidate["question"].split("，", 1)[-1]
    return f"{prefix}，{detail}"


def _multihop_item(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_question = left["question"].split("，", 1)[-1]
    right_question = right["question"].split("，", 1)[-1]
    return {
        "qa_id": "",
        "category": "multi_hop",
        "evaluation_layer": "multi_hop",
        "question": (
            "截至完整600轮历史结束时，请同时回答两个状态："
            f"A. {left_question} B. {right_question}"
        ),
        "answer": {
            "a": left["answer"],
            "b": right["answer"],
        },
        "answer_type": "object",
        "answer_schema": {"a": left["answer_type"], "b": right["answer_type"]},
        "evidence": [left["event_id"], right["event_id"]],
        "state_path": [left["state_path"], right["state_path"]],
    }


def _spread(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count > len(values):
        raise ValueError("状态问题候选不足")
    if count == 1:
        return [values[len(values) // 2]]
    return [
        values[round(index * (len(values) - 1) / (count - 1))]
        for index in range(count)
    ]
