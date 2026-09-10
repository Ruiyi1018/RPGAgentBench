from gamecore import GameContext
from runners.full_experiment import (
    _invariance_consistency,
    _stage3_episode_summary_v3,
    _stage3_survival_v2,
)


def _context() -> GameContext:
    return GameContext(
        {
            "character_card": {"character_id": "npc"},
            "runtime_state": {},
            "environment": {
                "health": {},
                "inventories": {},
                "locations": {},
                "access": {},
                "dialogue_status": "active",
            },
            "history": [
                {
                    "turn": 1,
                    "speaker": "npc",
                    "utterance": "拒绝。",
                    "actions": [],
                    "events": [],
                }
            ],
        }
    )


def test_invariance_requires_correct_equivalent_decision_and_state() -> None:
    records = [
        {
            "pair_id": "pair_08",
            "pair_type": "invariance",
            "decision": {"type": "respond_only", "parameters": {}},
            "action_correct": True,
            "state_answer": "active",
            "state_correct": True,
        },
        {
            "pair_id": "pair_08",
            "pair_type": "invariance",
            "decision": {"type": "respond_only", "parameters": {}},
            "action_correct": False,
            "state_answer": "active",
            "state_correct": True,
        },
    ]

    assert _invariance_consistency(records) == {"pair_08": False}


def test_stage3_summary_reports_failure_type_and_coverage() -> None:
    summary = _stage3_episode_summary_v3(
        "pressure",
        _context(),
        "failure",
        [
            {
                "turn": 1,
                "failure": True,
                "failure_type": "illegal_transition",
                "violated_rule": "protect_document",
                "evidence": ["非法转移"],
                "source": "contract_engine",
            }
        ],
        accepted_turns=1,
        expected_turns=40,
        contract_runtime={
            "contracts": {
                "protect_document": {"status": "violated"}
            }
        },
        challenge_coverage={
            "total_challenges": 2,
            "completed_challenges": [],
            "coverage": 0.0,
        },
    )

    assert summary["protocol_version"] == 10
    assert summary["failure"] is True
    assert summary["failure_type"] == "illegal_transition"
    assert summary["time_to_first_failure"] == 1
    assert summary["contract_status"]["violated"] == 1
    assert summary["protocol_evaluation_complete"] is True


def test_stage3_summary_reports_all_failures_not_only_first() -> None:
    summary = _stage3_episode_summary_v3(
        "pressure",
        _context(),
        "survived",
        [
            {
                "turn": 12,
                "failure": True,
                "failure_type": "illegal_transition",
                "violated_rule": "gamecore.access_not_authorized",
                "evidence": ["GameCore拒绝"],
                "source": "gamecore",
            },
            {
                "turn": 26,
                "failure": True,
                "failure_type": "missing_transition",
                "violated_rule": "revocation_requires_access_removal",
                "evidence": ["缺失转移"],
                "source": "contract_engine",
            },
            {
                "turn": 31,
                "failure": True,
                "failure_type": "trajectory_conflict",
                "violated_rule": "words_actions_state_alignment",
                "evidence": ["言行不一致"],
                "source": "checker_confirmed",
            },
        ],
        accepted_turns=40,
        expected_turns=40,
        challenge_coverage={
            "total_challenges": 2,
            "completed_challenges": ["first", "second"],
            "coverage": 1.0,
        },
    )

    assert summary["failure_count"] == 3
    assert summary["failure_type"] == "illegal_transition"
    assert summary["time_to_first_failure"] == 12
    assert summary["failure_counts_by_type"] == {
        "illegal_transition": 1,
        "missing_transition": 1,
        "trajectory_conflict": 1,
    }
    assert summary["failure_counts_by_source"]["checker_confirmed"] == 1
    assert [
        record["turn"] for record in summary["failure_records"]
    ] == [12, 26, 31]


def test_incomplete_nonfailure_is_insufficient_coverage() -> None:
    summary = _stage3_episode_summary_v3(
        "normal",
        _context(),
        "partial",
        [],
        accepted_turns=1,
        expected_turns=16,
        challenge_coverage={
            "total_challenges": 2,
            "completed_challenges": ["first"],
            "coverage": 0.5,
        },
    )

    assert summary["failure"] is False
    assert summary["status"] == "insufficient_coverage"
    assert summary["protocol_evaluation_complete"] is False


def test_stage3_survival_excludes_invalid_trajectories() -> None:
    episodes = [
        {
            "mode": "pressure",
            "status": "failure",
            "completed_turns": 12,
            "time_to_first_failure": 12,
        },
        {
            "mode": "pressure",
            "status": "survived",
            "completed_turns": 40,
            "time_to_first_failure": None,
        },
        {
            "mode": "pressure",
            "status": "invalid",
            "completed_turns": 1,
            "time_to_first_failure": None,
        },
    ]

    survival = _stage3_survival_v2(episodes, 40)["pressure"]

    assert survival["eligible_trajectories"] == 2
    assert survival["survival"]["10"] == 1.0
    assert survival["survival"]["20"] == 0.5
