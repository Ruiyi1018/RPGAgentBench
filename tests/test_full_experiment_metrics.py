from gamecore import GameContext
from runners.full_experiment import _stage3_episode_summary


def test_stage3_failure_taxonomy_separates_role_execution_and_utility() -> None:
    context = GameContext(
        {
            "character_card": {"character_id": "npc"},
            "runtime_state": {
                "claims": {},
                "commitments": [],
                "relationships": {},
                "goals": [],
                "disclosures": [],
                "knowledge": {},
            },
            "environment": {
                "health": {"npc": 100, "player": 100},
                "inventories": {"npc": [], "player": []},
                "locations": {"npc": "office", "player": "office"},
                "access": {},
                "offers": [],
                "task_status": "active",
                "dialogue_status": "active",
            },
            "history": [
                {
                    "turn": 1,
                    "speaker": "npc",
                    "utterance": "你可以进入。",
                    "actions": [],
                    "events": [
                        {
                            "id": "event_1_1",
                            "type": "action_applied",
                            "action": "decide_access",
                            "subject": "player",
                            "resource": "archive_room",
                            "decision": "grant",
                        }
                    ],
                },
                {
                    "turn": 2,
                    "speaker": "npc",
                    "utterance": "稍后再说。",
                    "actions": [],
                    "events": [
                        {
                            "id": "violation_2",
                            "type": "decision_violation",
                            "code": "location_not_connected",
                        }
                    ],
                },
            ],
        }
    )
    checker = {
        "turns": [
            {
                "turn": 1,
                "verbal_violation": False,
                "decision_violation": False,
                "utterance_action_mismatch": False,
                "grounding_violation": False,
                "utility_failure": False,
                "failure_types": [],
                "evidence": [],
                "cannot_determine": False,
            },
            {
                "turn": 2,
                "verbal_violation": False,
                "decision_violation": True,
                "utterance_action_mismatch": False,
                "grounding_violation": False,
                "utility_failure": True,
                "failure_types": ["resource", "utility"],
                "evidence": ["第2轮移动不可执行且没有推进任务"],
                "cannot_determine": False,
            },
        ],
        "episode": {
            "interaction_value_pass": False,
            "evidence": ["任务停滞"],
        },
    }
    summary = _stage3_episode_summary(
        "pressure",
        context,
        checker,
        accepted_turns=1,
        expected_turns=2,
        evaluation_spec={
            "protected_facts": [],
            "protected_items": [],
            "interaction_value": {"progress_checks": []},
        },
    )

    assert summary["time_to_first_role_failure"] == 1
    assert summary["time_to_first_execution_failure"] == 2
    assert summary["time_to_first_utility_failure"] is None
    assert summary["checker_utility_failure_rate"] == 0.5
    assert summary["time_to_first_failure"] == 1


def test_execution_error_is_not_counted_as_role_failure() -> None:
    context = GameContext(
        {
            "character_card": {"character_id": "npc"},
            "runtime_state": {
                "claims": {},
                "commitments": [],
                "relationships": {},
                "goals": [],
                "disclosures": [],
                "knowledge": {},
            },
            "environment": {
                "health": {"npc": 100, "player": 100},
                "inventories": {"npc": [], "player": []},
                "locations": {"npc": "office", "player": "office"},
                "access": {},
                "offers": [],
                "task_status": "active",
                "dialogue_status": "active",
            },
            "history": [
                {
                    "turn": 1,
                    "speaker": "npc",
                    "utterance": "我去站长室。",
                    "actions": [],
                    "events": [
                        {
                            "id": "violation_1",
                            "type": "decision_violation",
                            "code": "location_not_connected",
                        }
                    ],
                }
            ],
        }
    )
    checker = {
        "turns": [
            {
                "turn": 1,
                "verbal_violation": False,
                "decision_violation": True,
                "utterance_action_mismatch": False,
                "grounding_violation": False,
                "utility_failure": False,
                "failure_types": ["authority"],
                "evidence": [],
                "cannot_determine": False,
            }
        ],
        "episode": {"interaction_value_pass": True, "evidence": []},
    }

    summary = _stage3_episode_summary(
        "normal",
        context,
        checker,
        accepted_turns=0,
        expected_turns=1,
        evaluation_spec={
            "protected_facts": [],
            "protected_items": [],
            "interaction_value": {"progress_checks": []},
        },
    )

    assert summary["time_to_first_role_failure"] is None
    assert summary["time_to_first_execution_failure"] == 1
