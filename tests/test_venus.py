from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from llm import GenerationConfig, LLMSettings, create_llm_client
from llm.venus import (
    DEFAULT_MODEL,
    VenusClient,
    VenusResult,
    build_venus_request,
    build_venus_token,
    call_venus_api_result_async,
)


def _completion(content: str = "OK") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=1,
            total_tokens=4,
        ),
        _request_id="request-test",
    )


class _FakeCompletions:
    def __init__(self, outcomes: list[Any], requests: list[dict[str, Any]]):
        self.outcomes = outcomes
        self.requests = requests

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(
        self,
        outcomes: list[Any],
        requests: list[dict[str, Any]],
    ) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(outcomes, requests)
        )
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_complete_venus_token_has_priority() -> None:
    assert (
        build_venus_token(
            {
                "VENUS_API_KEY": "complete-test-token",
                "ENV_VENUS_OPENAPI_SECRET_ID": "secret-id",
                "VENUS_TOKEN_SUFFIX": "@suffix",
            }
        )
        == "complete-test-token"
    )


def test_secret_id_and_suffix_build_token() -> None:
    assert (
        build_venus_token(
            {
                "VENUS_API_KEY": "",
                "ENV_VENUS_OPENAPI_SECRET_ID": "secret-id",
                "VENUS_TOKEN_SUFFIX": "@application-group",
            }
        )
        == "secret-id@application-group"
    )


def test_request_preserves_deepseek_parameters() -> None:
    messages = [{"role": "user", "content": "只回复OK"}]
    request = build_venus_request(
        messages,
        DEFAULT_MODEL,
        {
            "max_tokens": 10,
            "temperature": 0,
            "top_p": 0.9,
            "seed": 7,
            "response_format": {"type": "json_object"},
        },
    )

    assert request["model"] == "deepseek-v4-pro"
    assert request["messages"] == messages
    assert request["max_tokens"] == 10
    assert request["temperature"] == 0
    assert request["top_p"] == 0.9
    assert request["seed"] == 7


def test_missing_token_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VENUS_API_KEY",
        "ENV_VENUS_OPENAPI_SECRET_ID",
        "VENUS_TOKEN_SUFFIX",
    ):
        monkeypatch.delenv(name, raising=False)

    result = asyncio.run(
        call_venus_api_result_async(
            [{"role": "user", "content": "OK"}],
        )
    )

    assert result is None


def test_venus_retries_and_closes_every_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    clients: list[_FakeClient] = []
    outcomes: list[Any] = [RuntimeError("temporary"), _completion()]

    def factory(**_: Any) -> _FakeClient:
        client = _FakeClient([outcomes.pop(0)], requests)
        clients.append(client)
        return client

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("llm.venus.asyncio.sleep", no_sleep)
    result = asyncio.run(
        call_venus_api_result_async(
            [{"role": "user", "content": "只回复OK"}],
            token="test-token",
            model_name=DEFAULT_MODEL,
            model_args={"temperature": 0, "max_tokens": 10},
            scene="smoke",
            max_attempts=2,
            client_factory=factory,
        )
    )

    assert result is not None and result.text == "OK"
    assert result.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }
    assert len(requests) == 2
    assert requests[-1]["model"] == "deepseek-v4-pro"
    assert all(client.closed for client in clients)


def test_invalid_provider_never_falls_back() -> None:
    settings = LLMSettings(
        provider="unknown",
        base_url="https://example.invalid",
        api_key_env="UNKNOWN_KEY",
        generation=GenerationConfig(model=DEFAULT_MODEL),
    )

    with pytest.raises(ValueError, match="不会回退"):
        create_llm_client(settings, scene="test")


def test_factory_routes_venus_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VENUS_API_KEY", "factory-test-token")
    settings = LLMSettings(
        provider="venus",
        base_url="https://v2.open.venus.woa.com/llmproxy",
        api_key_env="VENUS_API_KEY",
        generation=GenerationConfig(model=DEFAULT_MODEL),
    )

    client = create_llm_client(settings, scene="data_generation")

    assert isinstance(client, VenusClient)
    assert client.scene == "data_generation"
    assert client.timeout_seconds == 300.0
    assert client.max_attempts == 3


def test_sync_client_routes_deepseek_and_preserves_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_call(
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> VenusResult:
        captured["messages"] = messages
        captured.update(kwargs)
        return VenusResult(
            text='{"ok":true}',
            usage={"total_tokens": 4},
            request_id="request-test",
        )

    monkeypatch.setattr(
        "llm.venus.call_venus_api_result_async",
        fake_call,
    )
    client = VenusClient(
        api_key="sync-test-token",
        scene="data_generation",
    )

    result = client.generate(
        system_prompt="system",
        user_prompt="user",
        config=GenerationConfig(
            model=DEFAULT_MODEL,
            temperature=0,
            max_tokens=10,
        ),
    )

    assert result == '{"ok":true}'
    assert captured["model_name"] == "deepseek-v4-pro"
    assert captured["scene"] == "data_generation"
    assert captured["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert client.last_usage == {"total_tokens": 4}
