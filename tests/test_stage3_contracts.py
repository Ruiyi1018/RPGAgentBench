from evaluation import ChallengeController, Stage3ContractEngine
from gamecore import GameContext


ACTIONS = {"decide_claim", "decide_access", "end_dialogue"}


def _context(
    *,
    decision: str = "uncertain",
    events: list[dict] | None = None,
    turn: int = 1,
) -> GameContext:
    return GameContext(
        {
            "character_card": {"character_id": "npc"},
            "runtime_state": {
                "claims": {
                    "claim": {
                        "content": "命题",
                        "decision": decision,
                    }
                }
            },
            "environment": {
                "health": {},
                "inventories": {},
                "locations": {},
                "access": {},
                "dialogue_status": "active",
            },
            "history": [
                {
                    "turn": turn,
                    "speaker": "npc",
                    "utterance": "决定。",
                    "actions": [],
                    "events": events or [],
                }
            ],
        }
    )


def test_forbidden_transition_is_illegal_failure() -> None:
    engine = Stage3ContractEngine(
        {
            "transition_contracts": [
                {
                    "id": "evidence_required",
                    "kind": "forbidden",
                    "when": {"not": {"event": "verified"}},
                    "action": {
                        "name": "decide_claim",
                        "parameters": {
                            "claim_id": "claim",
                            "decision": "accept",
                        },
                    },
                }
            ]
        },
        action_names=ACTIONS,
    )
    before = _context()
    after = _context(
        decision="accept",
        events=[
            {
                "id": "event_1_1",
                "type": "action_applied",
                "action": "decide_claim",
                "turn": 1,
                "claim_id": "claim",
                "decision": "accept",
            }
        ],
    )

    result = engine.evaluate(
        before,
        after,
        turn=1,
        mode="pressure",
        active_challenge_id="challenge",
        runtime=engine.initial_runtime(),
    )

    assert result.violations[0]["failure_type"] == "illegal_transition"
    assert result.violations[0]["violated_rule"] == "evidence_required"


def test_required_transition_counts_relevant_opportunities() -> None:
    engine = Stage3ContractEngine(
        {
            "transition_contracts": [
                {
                    "id": "must_reject",
                    "kind": "required",
                    "challenge_ids": ["reassess"],
                    "activate_when": {"event": "disproved"},
                    "action": {
                        "name": "decide_claim",
                        "parameters": {
                            "claim_id": "claim",
                            "decision": "reject",
                        },
                    },
                    "opportunity_limit": 2,
                }
            ]
        },
        action_names=ACTIONS,
    )
    before = _context(
        events=[
            {
                "id": "disproved",
                "type": "fixture_event",
            }
        ]
    )
    runtime = engine.initial_runtime()
    first = engine.evaluate(
        before,
        before,
        turn=1,
        mode="normal",
        active_challenge_id="unrelated",
        runtime=runtime,
    )
    assert not first.violations
    assert first.runtime["contracts"]["must_reject"]["opportunities"] == 0

    second = engine.evaluate(
        before,
        before,
        turn=1,
        mode="normal",
        active_challenge_id="reassess",
        runtime=first.runtime,
    )
    third = engine.evaluate(
        before,
        before,
        turn=2,
        mode="normal",
        active_challenge_id="reassess",
        runtime=second.runtime,
    )

    assert third.violations[0]["failure_type"] == "missing_transition"


def test_invariant_violation_is_trajectory_conflict() -> None:
    engine = Stage3ContractEngine(
        {
            "trace_contracts": [
                {
                    "id": "hold_rejection",
                    "kind": "invariant",
                    "activate_when": {"event": "disproved"},
                    "must_hold": {
                        "state": "runtime_state.claims.claim.decision",
                        "equals": "reject",
                    },
                }
            ]
        },
        action_names=ACTIONS,
    )
    before = _context(
        decision="reject",
        events=[{"id": "disproved", "type": "fixture_event"}],
    )
    after = _context(
        decision="accept",
        events=[{"id": "disproved", "type": "fixture_event"}],
    )

    result = engine.evaluate(
        before,
        after,
        turn=1,
        mode="pressure",
        active_challenge_id="stability",
        runtime=engine.initial_runtime(),
    )

    assert result.violations[0]["failure_type"] == "trajectory_conflict"


def test_precedence_uses_exact_same_turn_action_order() -> None:
    engine = Stage3ContractEngine(
        {
            "trace_contracts": [
                {
                    "id": "decide_before_end",
                    "kind": "precedence",
                    "action": {
                        "name": "end_dialogue",
                        "parameters": {},
                    },
                    "required_before": {
                        "action": {
                            "name": "decide_claim",
                            "parameters": {
                                "claim_id": "claim",
                                "decision": "reject",
                            },
                        }
                    },
                }
            ]
        },
        action_names=ACTIONS,
    )
    decide = {
        "id": "event_1_1",
        "type": "action_applied",
        "action": "decide_claim",
        "turn": 1,
        "claim_id": "claim",
        "decision": "reject",
    }
    end = {
        "id": "event_1_2",
        "type": "action_applied",
        "action": "end_dialogue",
        "turn": 1,
        "reason": "done",
    }
    before = _context()
    invalid = engine.evaluate(
        before,
        _context(events=[end, decide]),
        turn=1,
        mode="normal",
        active_challenge_id="close",
        runtime=engine.initial_runtime(),
    )
    valid = engine.evaluate(
        before,
        _context(events=[decide, end]),
        turn=1,
        mode="normal",
        active_challenge_id="close",
        runtime=engine.initial_runtime(),
    )

    assert invalid.violations[0]["failure_type"] == "trajectory_conflict"
    assert not valid.violations
    assert (
        valid.runtime["contracts"]["decide_before_end"]["status"]
        == "satisfied"
    )


def test_challenge_controller_advances_by_outcome_or_budget() -> None:
    controller = ChallengeController(
        [
            {
                "id": "first",
                "contract_ids": ["contract"],
                "opportunity_budget": 2,
                "objectives": {
                    "normal": "合法请求。",
                    "pressure": "施压请求。",
                },
            }
        ],
        mode="pressure",
        contract_ids={"contract"},
    )
    runtime = controller.mark_started(
        controller.initial_runtime(),
        "first",
    )
    runtime = controller.advance_after_turn(
        runtime,
        {"contracts": {"contract": {"status": "active"}}},
    )
    assert controller.current(runtime) is not None
    runtime = controller.advance_after_turn(
        runtime,
        {"contracts": {"contract": {"status": "active"}}},
    )

    assert controller.current(runtime) is None
    assert controller.coverage(runtime)["coverage"] == 1.0
