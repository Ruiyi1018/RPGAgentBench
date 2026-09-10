import http.client
import io
import json
import os
import urllib.error
from pathlib import Path

from llm import (
    GenerationConfig,
    OpenAICompatibleClient,
    load_llm_settings,
    parse_json_object,
)
from llm import openai_compatible


def test_parse_json_object_ignores_surrounding_model_commentary() -> None:
    assert parse_json_object(
        '结果如下：\\n{"continuity_pass": true}\\n以上为审核结果。'
    ) == {"continuity_pass": True}


def test_parse_json_object_accepts_literal_control_characters() -> None:
    assert parse_json_object('{"explanation":"第一行\n第二行\t缩进"}') == {
        "explanation": "第一行\n第二行\t缩进"
    }


def test_parse_json_object_skips_invalid_earlier_brace() -> None:
    assert parse_json_object(
        '先说明一个集合{这不是JSON}，正式结果是{"ok":true}'
    ) == {"ok": True}


def test_parse_json_object_recovers_missing_outer_left_brace() -> None:
    assert parse_json_object(
        '"turns":[{"turn":1,"failure":false}]}'
    ) == {"turns": [{"turn": 1, "failure": False}]}


def test_parse_json_object_prefers_content_after_thinking_block() -> None:
    assert parse_json_object(
        '<think>{"draft":false}</think>{"failures":[]}'
    ) == {"failures": []}


def test_openai_client_retries_remote_disconnect(monkeypatch) -> None:
    calls = 0

    class Response:
        headers = {"x-request-id": "request-2"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": '{"ok":true}'}}],
                    "usage": {"total_tokens": 3},
                }
            ).encode()

    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise http.client.RemoteDisconnected(
                "Remote end closed connection without response"
            )
        return Response()

    monkeypatch.setattr(
        openai_compatible.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(openai_compatible.time, "sleep", lambda _: None)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://example.test/v1",
        max_retries=1,
    )

    output = client.generate(
        system_prompt="system",
        user_prompt="user",
        config=GenerationConfig(model="test"),
    )

    assert output == '{"ok":true}'
    assert calls == 2
    assert client.last_request_id == "request-2"


def test_openai_client_waits_for_allocation_quota_window(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                "https://example.test",
                429,
                "Too Many Requests",
                {},
                io.BytesIO(
                    b'{"error":{"code":"insufficient_quota"}}'
                ),
            )
        return Response()

    monkeypatch.setattr(
        openai_compatible.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        openai_compatible.time,
        "sleep",
        sleeps.append,
    )
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://example.test/v1",
        max_retries=1,
    )

    assert client.generate(
        system_prompt="system",
        user_prompt="user",
        config=GenerationConfig(model="test"),
    ) == "ok"
    assert calls == 2
    assert sleeps == [60.0]


def test_llm_config_loads_local_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("TEST_DASHSCOPE_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("VENUS_MODEL", raising=False)
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
