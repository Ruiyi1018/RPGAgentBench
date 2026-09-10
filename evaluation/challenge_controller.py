"""State-driven challenge sequencing for Stage 3 player agents."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ActiveChallenge:
    challenge_id: str
    objective: str
    contract_ids: tuple[str, ...]
    fixture_event: dict[str, Any] | None
    opportunity_budget: int


class ChallengePlanError(ValueError):
    """Raised when a challenge plan cannot be executed."""


class ChallengeController:
    """Advance a reviewed challenge plan from runtime outcomes."""

    def __init__(
        self,
        plan: list[Mapping[str, Any]],
        *,
        mode: str,
        contract_ids: set[str],
    ) -> None:
        if mode not in {"normal", "pressure"}:
            raise ChallengePlanError(f"未知Stage3模式: {mode}")
        self.mode = mode
        self.plan = [
            copy.deepcopy(dict(item))
            for item in plan
            if mode in item.get("modes", ["normal", "pressure"])
        ]
        validate_challenge_plan(
            self.plan,
            mode=mode,
            contract_ids=contract_ids,
        )

    def initial_runtime(self) -> dict[str, Any]:
        return {
            "index": 0,
            "started": [],
            "completed": [],
            "opportunities": {},
        }

    def current(
        self,
        runtime: Mapping[str, Any],
    ) -> ActiveChallenge | None:
        index = int(runtime.get("index", 0))
        if index >= len(self.plan):
            return None
        item = self.plan[index]
        fixture = item.get("fixture_event")
        return ActiveChallenge(
            challenge_id=str(item["id"]),
            objective=str(item["objectives"][self.mode]),
            contract_ids=tuple(
                str(value) for value in item.get("contract_ids", [])
            ),
            fixture_event=(
                copy.deepcopy(dict(fixture))
                if isinstance(fixture, Mapping)
                else None
            ),
            opportunity_budget=int(item["opportunity_budget"]),
        )

    def mark_started(
        self,
        runtime: Mapping[str, Any],
        challenge_id: str,
    ) -> dict[str, Any]:
        updated = copy.deepcopy(dict(runtime))
        started = updated.setdefault("started", [])
        if challenge_id not in started:
            started.append(challenge_id)
        return updated

    def advance_after_turn(
        self,
        runtime: Mapping[str, Any],
        contract_runtime: Mapping[str, Any],
    ) -> dict[str, Any]:
        updated = copy.deepcopy(dict(runtime))
        active = self.current(updated)
        if active is None:
            return updated
        opportunities = updated.setdefault("opportunities", {})
        opportunities[active.challenge_id] = (
            int(opportunities.get(active.challenge_id, 0)) + 1
        )
        linked_states = [
            contract_runtime.get("contracts", {}).get(contract_id, {})
            for contract_id in active.contract_ids
        ]
        resolved = bool(linked_states) and all(
            state.get("status") in {"satisfied", "violated"}
            for state in linked_states
        )
        budget_used = (
            int(opportunities[active.challenge_id])
            >= active.opportunity_budget
        )
        if resolved or budget_used:
            completed = updated.setdefault("completed", [])
            if active.challenge_id not in completed:
                completed.append(active.challenge_id)
            updated["index"] = int(updated.get("index", 0)) + 1
        return updated

    def coverage(
        self,
        runtime: Mapping[str, Any],
    ) -> dict[str, Any]:
        completed = {
            str(value) for value in runtime.get("completed", [])
        }
        ids = [str(item["id"]) for item in self.plan]
        return {
            "total_challenges": len(ids),
            "completed_challenges": [
                challenge_id
                for challenge_id in ids
                if challenge_id in completed
            ],
            "coverage": (
                len(completed & set(ids)) / len(ids) if ids else 0.0
            ),
        }


def validate_challenge_plan(
    plan: list[Mapping[str, Any]],
    *,
    mode: str,
    contract_ids: set[str],
) -> None:
    seen: set[str] = set()
    for item in plan:
        challenge_id = item.get("id")
        if not isinstance(challenge_id, str) or not challenge_id:
            raise ChallengePlanError("challenge_plan条目缺少有效id")
        if challenge_id in seen:
            raise ChallengePlanError(f"challenge id重复: {challenge_id}")
        seen.add(challenge_id)
        objectives = item.get("objectives")
        if (
            not isinstance(objectives, Mapping)
            or not isinstance(objectives.get(mode), str)
            or not objectives[mode]
        ):
            raise ChallengePlanError(
                f"{challenge_id}.objectives缺少{mode}目标"
            )
        budget = item.get("opportunity_budget")
        if not isinstance(budget, int) or budget < 1:
            raise ChallengePlanError(
                f"{challenge_id}.opportunity_budget必须为正整数"
            )
        linked = item.get("contract_ids", [])
        if not isinstance(linked, list) or any(
            value not in contract_ids for value in linked
        ):
            raise ChallengePlanError(
                f"{challenge_id}.contract_ids包含未知契约"
            )
        fixture = item.get("fixture_event")
        if fixture is not None and (
            not isinstance(fixture, Mapping)
            or not isinstance(fixture.get("event_id"), str)
            or not fixture["event_id"]
            or not isinstance(fixture.get("content"), str)
            or not isinstance(fixture.get("operations"), list)
        ):
            raise ChallengePlanError(
                f"{challenge_id}.fixture_event格式错误"
            )
