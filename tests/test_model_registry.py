from pathlib import Path
import threading
import time

from llm import VenusClient, load_model_registry
from llm.rate_limit import model_concurrency_gate
from datagen.pipeline import _generation_runtime
from llm.venus import build_venus_request
from runners import model_sweep


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "llm" / "registry.yaml"


def test_registry_resolves_backend_model_and_capabilities() -> None:
    registry = load_model_registry(REGISTRY, project_root=ROOT)

    profile = registry.model("venus-gpt6-astra")
    settings = registry.settings("venus-gpt6-astra")

    assert profile.backend == "venus"
    assert settings.provider == "venus"
    assert settings.api_key_env == "ENV_VENUS_OPENAPI_SECRET_ID"
    assert settings.api_key_suffix == "@5784"
    assert settings.generation.model == "gpt-6-astra"
    assert (
        settings.generation.capabilities.token_parameter
        == "max_completion_tokens"
    )


def test_registered_capabilities_control_venus_request() -> None:
    config = load_model_registry(
        REGISTRY,
        project_root=ROOT,
    ).generation_config("venus-gpt6-astra")

    request = build_venus_request(
        [{"role": "user", "content": "JSON"}],
        config.model,
        {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.seed,
        },
        config.capabilities,
    )

    assert request["max_completion_tokens"] == 8192
    assert "max_tokens" not in request
    assert "temperature" not in request
    assert "top_p" not in request
    assert "seed" not in request


def test_sweep_changes_candidate_and_keeps_references(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment = tmp_path / "sweep.yaml"
    experiment.write_text(
        "\n".join(
            [
                f"registry: {REGISTRY}",
                "roles:",
                "  candidate: venus-deepseek-v4-pro",
                "  player: venus-deepseek-v4-pro",
                "  evaluator: venus-deepseek-v4-pro",
                "matrix:",
                "  candidate:",
                "    - venus-deepseek-v4-flash",
                "    - venus-gpt6-astra",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[dict] = []

    def fake_run(*_args, **kwargs):
        calls.append(kwargs)
        return {"dry_run": True}

    monkeypatch.setattr(model_sweep, "run_full_experiment", fake_run)

    result = model_sweep.run_model_sweep(
        experiment,
        world_dir="world",
        character_id="npc",
        scenario_path="scenario.yaml",
        dry_run=True,
    )

    assert result["combinations"] == 2
    assert [call["candidate_model"] for call in calls] == [
        "venus-deepseek-v4-flash",
        "venus-gpt6-astra",
    ]
    assert {
        call["player_model"] for call in calls
    } == {"venus-deepseek-v4-pro"}
    assert {
        call["evaluator_model"] for call in calls
    } == {"venus-deepseek-v4-pro"}


def test_process_wide_model_concurrency_gate_caps_workers() -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def work() -> None:
        nonlocal active, maximum_active
        with model_concurrency_gate("test-model-gate", 2):
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum_active == 2


def test_registry_builds_venus_token_from_configured_suffix(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VENUS_API_KEY", raising=False)
    monkeypatch.delenv("VENUS_TOKEN_SUFFIX", raising=False)
    monkeypatch.setenv("ENV_VENUS_OPENAPI_SECRET_ID", "secret-id")
    monkeypatch.setattr("llm.registry.load_env_file", lambda *_args: None)
    registry = load_model_registry(REGISTRY, project_root=ROOT)

    client = registry.create_client(
        "venus-gpt-5.5",
        scene="token-test",
    )

    assert isinstance(client, VenusClient)
    assert client.api_key == "secret-id@5784"


def test_datagen_runtime_resolves_registered_model() -> None:
    _factory, config = _generation_runtime(
        config_path=ROOT / "configs" / "models" / "venus_deepseek.yaml",
        registry_path=REGISTRY,
        model_name="venus-gpt-5.5",
        scene="data_generation_test",
    )

    assert config.model == "gpt-5.5"
    assert config.capabilities.token_parameter == "max_completion_tokens"
