"""Live, low-volume contract checks for registered model deployments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm import load_model_registry, parse_json_object

from .pilot import PROJECT_ROOT


def smoke_models(
    registry_path: str | Path,
    model_names: list[str],
    *,
    long_context_path: str | Path | None = None,
    long_context_characters: int = 120_000,
    output_characters: int = 0,
) -> dict[str, Any]:
    registry = load_model_registry(
        registry_path,
        project_root=PROJECT_ROOT,
    )
    context = ""
    expected_markers: list[str] = []
    if long_context_path is not None:
        context = Path(long_context_path).read_text(encoding="utf-8")
        if len(context) < long_context_characters:
            repeats = (long_context_characters // max(len(context), 1)) + 1
            context = (context + "\n") * repeats
        context = context[:long_context_characters]
        expected_markers = [
            "RPG_SMOKE_START_7F3A",
            "RPG_SMOKE_MIDDLE_91C2",
            "RPG_SMOKE_END_4D8E",
        ]
        midpoint = len(context) // 2
        context = (
            expected_markers[0]
            + "\n"
            + context[:midpoint]
            + "\n"
            + expected_markers[1]
            + "\n"
            + context[midpoint:]
            + "\n"
            + expected_markers[2]
        )
    results: dict[str, Any] = {}
    for name in model_names:
        profile = registry.model(name)
        client = registry.create_client(name, scene="model_smoke")
        config = replace(
            profile.generation,
            max_tokens=(
                profile.generation.max_tokens
                if output_characters > 0
                else min(profile.generation.max_tokens, 2048)
            ),
            max_format_retries=0,
            response_schema=(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                }
                if output_characters > 0
                else {
                    "type": "object",
                    "additionalProperties": False,
                    "required": (
                        ["ok", "model", "markers"]
                        if expected_markers
                        else ["ok", "model"]
                    ),
                    "properties": {
                        "ok": {"type": "boolean"},
                        "model": {"type": "string"},
                        **(
                            {
                                "markers": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            }
                            if expected_markers
                            else {}
                        ),
                    },
                }
            ),
        )
        started = time.monotonic()
        try:
            if output_characters > 0:
                instruction = (
                    "Return exactly one JSON object with key text. The text "
                    f"value must contain exactly {output_characters} lowercase "
                    "x characters and nothing else."
                )
            else:
                instruction = (
                    'Return exactly {"ok":true,"model":"'
                    + profile.remote_model
                    + '"'
                    + (
                        ',"markers":'
                        + json.dumps(expected_markers)
                        if expected_markers
                        else ""
                    )
                    + "}. Output JSON only."
                )
            user_prompt = instruction
            if context:
                user_prompt = (
                    "Reference context follows. Do not summarize it.\n"
                    f"<context>\n{context}\n</context>\n\n{instruction}"
                )
            raw = client.generate(
                system_prompt="You are an API contract test assistant.",
                user_prompt=user_prompt,
                config=config,
            )
            parsed = parse_json_object(raw)
            if output_characters > 0:
                text = parsed.get("text")
                if not isinstance(text, str) or len(text) != output_characters:
                    raise ValueError(
                        "输出长度不符合请求: "
                        f"expected={output_characters}, "
                        f"actual={len(text) if isinstance(text, str) else None}"
                    )
            elif parsed.get("ok") is not True:
                raise ValueError("模型未返回ok=true")
            elif expected_markers and parsed.get("markers") != expected_markers:
                raise ValueError("长上下文首中尾marker未完整保留")
        except Exception as error:
            results[name] = {
                "status": "failed",
                "backend": profile.backend,
                "remote_model": profile.remote_model,
                "latency_seconds": round(time.monotonic() - started, 3),
                "error": f"{type(error).__name__}: {error}",
                "usage": getattr(client, "last_usage", None),
                "request_id": getattr(client, "last_request_id", None),
                "finish_reason": getattr(client, "last_finish_reason", None),
            }
            continue
        results[name] = {
            "status": "passed",
            "backend": profile.backend,
            "remote_model": profile.remote_model,
            "latency_seconds": round(time.monotonic() - started, 3),
            "input_characters": len(user_prompt),
            "output_characters": len(raw),
            "usage": getattr(client, "last_usage", None),
            "request_id": getattr(client, "last_request_id", None),
            "finish_reason": getattr(client, "last_finish_reason", None),
        }
    return {
        "registry": str(registry_path),
        "long_context_characters": len(context),
        "requested_output_characters": output_characters,
        "passed": sum(
            result["status"] == "passed" for result in results.values()
        ),
        "failed": sum(
            result["status"] == "failed" for result in results.values()
        ),
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "configs" / "llm" / "registry.yaml",
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--long-context", type=Path)
    parser.add_argument("--long-context-characters", type=int, default=120_000)
    parser.add_argument("--output-characters", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry = load_model_registry(
        args.registry,
        project_root=PROJECT_ROOT,
    )
    models = args.models or [
        name
        for name, profile in registry.models.items()
        if profile.backend == "venus"
    ]
    result = smoke_models(
        args.registry,
        models,
        long_context_path=args.long_context,
        long_context_characters=args.long_context_characters,
        output_characters=args.output_characters,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
