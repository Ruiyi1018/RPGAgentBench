import json
from pathlib import Path

import pytest

import runners.full_experiment as full_experiment_module
import runners.pilot as pilot_module
from datagen.shared.io import read_jsonl
from llm import GenerationConfig, StaticLLMClient
from runners.full_experiment import _run_stage1, run_full_experiment
from runners.pilot import _score_branch, build_qa_bundle, run_pilot
from runners.progress import ProgressReporter


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "assets" / "world_002"
SCENARIO = WORLD / "scenarios" / "yu_zecheng_archive_request.yaml"


def test_current_layout_passes_pilot_preflight() -> None:
    result = run_pilot(
        WORLD,
        "yu_zecheng",
        dry_run=True,
        qa_limit=10,
        pair_limit=2,
    )

    assert result["dry_run"] is True
    assert result["audit"]["ready_for_api_smoke_test"] is True
    assert result["audit"]["formal_benchmark_ready"] is False
    assert result["audit"]["output_review"]["approved"] is False


def test_invariance_scores_state_and_common_decision() -> None:
    pair = {
        "pair_id": "pair_08",
        "pair_type": "invariance",
        "decision_rubric": {
            "expected_state": {"path": "runtime_state.claims.x", "value": "reject"},
            "admissible_decisions": [
                {"action": "respond_only", "parameters": {}}
            ],
        },
    }
    output = {
        "decision": {"type": "respond_only", "parameters": {}},
        "state_answer": "reject",
        "state_basis": ["anchor_x"],
        "utterance": "拒绝。",
    }

    result = _score_branch(pair, "a", output, None)

    assert result["action_correct"] is True
    assert result["state_correct"] is True


def test_live_pilot_requires_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_TEST_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "dashscope")
    monkeypatch.setattr(
        pilot_module,
        "audit_executable_character",
        lambda *_args, **_kwargs: {"ready_for_api_smoke_test": True},
    )

    with pytest.raises(ValueError, match="MISSING_TEST_API_KEY"):
        run_pilot(
            WORLD,
            "yu_zecheng",
            dry_run=False,
            config_path=ROOT / "configs" / "models" / "qwen_pilot.yaml",
            api_key_env="MISSING_TEST_API_KEY",
            qa_limit=1,
            pair_limit=0,
        )


def test_full_stage123_preflight_counts_complete_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        full_experiment_module,
        "audit_executable_character",
        lambda *_args, **_kwargs: {"ready_for_api_smoke_test": True},
    )
    result = run_full_experiment(
        WORLD,
        "yu_zecheng",
        scenario_path=SCENARIO,
        dry_run=True,
    )

    assert (
        result["preflight"]["maximum_calls_without_qa_recovery"]
        == 602
    )
    assert result["preflight"]["maximum_logical_calls"] == 650
    assert result["preflight"]["stage3"]["checker_interval"] == 5
    assert result["preflight"]["stage1"]["qa"] == 50
    assert result["preflight"]["stage1"]["qa_batches"] == 2
    assert result["preflight"]["stage1"]["qa_batch_size"] == 25
    assert result["preflight"]["stage2"]["pairs"] == 10
    assert result["preflight"]["stage3"]["turns_per_condition"] == 40
    assert result["preflight"]["stage3"]["seeds"] == [11, 29, 47]


def test_stage1_recovers_only_missing_qa_items(tmp_path: Path) -> None:
    qa = read_jsonl(WORLD / "frozen" / "yu_zecheng" / "qa.jsonl")[:2]

    def response(item: dict) -> str:
        return json.dumps(
            {
                "answers": [
                    {
                        "qa_id": item["qa_id"],
                        "answer": item["answer"],
                        "evidence": item["evidence"],
                        "confidence": 1.0,
                    }
                ]
            },
            ensure_ascii=False,
        )

    client = StaticLLMClient([response(qa[0]), response(qa[1])])
    result = _run_stage1(
        WORLD,
        "yu_zecheng",
        [qa],
        [build_qa_bundle(WORLD, "yu_zecheng", qa)],
        [],
        client,
        GenerationConfig(model="test", max_format_retries=0),
        tmp_path,
        ProgressReporter(total=1, enabled=False),
    )

    assert result["qa_count"] == 2
    assert len(client.requests) == 2
    assert qa[0]["qa_id"] not in client.requests[1]["user_prompt"]
    assert qa[1]["qa_id"] in client.requests[1]["user_prompt"]


def test_stage1_uses_separate_candidate_and_evaluator_clients(
    tmp_path: Path,
) -> None:
    candidate = StaticLLMClient(
        [
            json.dumps(
                {
                    "utterance": "我会按现有权限处理。",
                    "state_basis": ["anchor_001"],
                },
                ensure_ascii=False,
            )
        ]
    )
    evaluator = StaticLLMClient(
        [
            json.dumps(
                {
                    "role_consistent": True,
                    "state_consistent": True,
                    "evidence_supported": True,
                    "grounded": True,
                    "interaction_value": True,
                    "evidence": ["候选回答未虚构额外权限"],
                },
                ensure_ascii=False,
            )
        ]
    )

    result = _run_stage1(
        WORLD,
        "yu_zecheng",
        [],
        [],
        [
            {
                "task_id": "open_test",
                "prompt": "请处理这份文件。",
                "evaluation_spec": {"must_remain_grounded": True},
            }
        ],
        candidate,
        GenerationConfig(model="candidate", max_format_retries=0),
        tmp_path,
        ProgressReporter(total=2, enabled=False),
        evaluator_client=evaluator,
        evaluator_config=GenerationConfig(
            model="evaluator",
            max_format_retries=0,
        ),
    )

    assert result["open_task_count"] == 1
    assert len(candidate.requests) == 1
    assert len(evaluator.requests) == 1
    assert "候选输出" in evaluator.requests[0]["user_prompt"]
