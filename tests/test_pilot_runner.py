from pathlib import Path

import pytest

from runners.pilot import run_pilot
from runners.full_experiment import run_full_experiment


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "assets" / "world_002"
SCENARIO = WORLD / "scenarios" / "yu_zecheng_archive_request.yaml"


def test_yu_zecheng_pilot_preflight_never_requires_key() -> None:
    result = run_pilot(
        WORLD,
        "yu_zecheng",
        dry_run=True,
        qa_limit=10,
        pair_limit=2,
    )

    assert result["preflight"]["data_ready"] is True
    assert result["preflight"]["api_calls"] == 5
    assert result["preflight"]["within_documented_context"] is True


def test_live_pilot_requires_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_TEST_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_TEST_API_KEY"):
        run_pilot(
            WORLD,
            "yu_zecheng",
            dry_run=False,
            api_key_env="MISSING_TEST_API_KEY",
            qa_limit=1,
            pair_limit=0,
        )


def test_full_stage123_preflight_counts_complete_protocol() -> None:
    result = run_full_experiment(
        WORLD,
        "yu_zecheng",
        scenario_path=SCENARIO,
        dry_run=True,
    )

    assert result["preflight"]["expected_api_calls"] == 512
    assert result["preflight"]["stage1"]["qa"] == 50
    assert result["preflight"]["stage2"]["pairs"] == 10
    assert result["preflight"]["stage3"]["turns_per_condition"] == 40
    assert result["preflight"]["stage3"]["seeds"] == [11, 29, 47]
