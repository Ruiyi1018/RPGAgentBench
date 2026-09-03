import os
from pathlib import Path

from llm import load_llm_settings, parse_json_object


def test_parse_json_object_ignores_surrounding_model_commentary() -> None:
    assert parse_json_object(
        '结果如下：\\n{"continuity_pass": true}\\n以上为审核结果。'
    ) == {"continuity_pass": True}


def test_llm_config_loads_local_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("TEST_DASHSCOPE_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "TEST_DASHSCOPE_KEY='secret-for-test'\n",
        encoding="utf-8",
    )
    config = tmp_path / "qwen.yaml"
    config.write_text(
        "\n".join(
            [
                "provider: dashscope",
                "base_url: https://example.test/v1",
                "api_key_env: TEST_DASHSCOPE_KEY",
                "env_file: .env",
                "enable_thinking: false",
                "generation:",
                "  model: qwen-test",
                "  temperature: 0",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_llm_settings(config, project_root=tmp_path)

    assert settings.generation.model == "qwen-test"
    assert settings.api_key_env == "TEST_DASHSCOPE_KEY"
    assert os.environ["TEST_DASHSCOPE_KEY"] == "secret-for-test"


def test_venus_config_forces_internal_deepseek(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "venus")
    monkeypatch.setenv("VENUS_MODEL", "deepseek-v4-pro")
    config = tmp_path / "venus.yaml"
    config.write_text(
        "\n".join(
            [
                "provider: venus",
                "base_url: https://v2.open.venus.woa.com/llmproxy",
                "api_key_env: VENUS_API_KEY",
                "timeout_seconds: 300",
                "max_attempts: 3",
                "generation:",
                "  model: deepseek-v4-pro",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_llm_settings(config, project_root=tmp_path)

    assert settings.provider == "venus"
    assert settings.generation.model == "deepseek-v4-pro"
    assert settings.timeout_seconds == 300
    assert settings.max_attempts == 3
