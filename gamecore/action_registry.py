"""Load and validate contracts from gamecore/actions/*.md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ActionValidationError


_PYTHON_TYPES: dict[str, type[Any]] = {
    "string": str,
    "array": list,
    "integer": int,
    "number": (int, float),  # type: ignore[dict-item]
    "boolean": bool,
    "object": dict,
}


@dataclass(frozen=True)
class ParameterSpec:
    type: str
    required: bool
    enum: tuple[Any, ...] = ()
    meaning: str = ""


@dataclass(frozen=True)
class ActionSpec:
    name: str
    category: str
    description: str
    parameters: dict[str, ParameterSpec]
    updates: tuple[str, ...]
    handler: str
    body: str
    source: Path


class ActionRegistry:
    """Immutable registry assembled from Action Markdown files."""

    def __init__(self, specs: Mapping[str, ActionSpec]) -> None:
        self._specs = dict(specs)

    @classmethod
    def load_directory(cls, path: str | Path) -> "ActionRegistry":
        directory = Path(path)
        specs: dict[str, ActionSpec] = {}
        for source in sorted(directory.glob("*.md")):
            if source.name.lower() == "readme.md":
                continue
            spec = cls._parse_file(source)
            if spec.name in specs:
                raise ActionValidationError(
                    "duplicate_action", f"动作名称重复: {spec.name}"
                )
            specs[spec.name] = spec
        if not specs:
            raise ActionValidationError("empty_registry", "未找到Action Markdown")
        return cls(specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def get(self, name: str) -> ActionSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ActionValidationError("unknown_action", f"未知动作: {name}") from exc

    def validate_call(self, action: Mapping[str, Any]) -> ActionSpec:
        if set(action) != {"name", "parameters"}:
            raise ActionValidationError(
                "invalid_action_shape", "Action只能包含name和parameters"
            )
        name = action.get("name")
        parameters = action.get("parameters")
        if not isinstance(name, str) or not name:
            raise ActionValidationError("invalid_action_name", "name必须是非空字符串")
        if not isinstance(parameters, Mapping):
            raise ActionValidationError("invalid_parameters", "parameters必须是对象")

        spec = self.get(name)
        unknown = set(parameters) - set(spec.parameters)
        if unknown:
            raise ActionValidationError(
                "unknown_parameter",
                f"{name}包含未知参数: {', '.join(sorted(unknown))}",
            )

        for parameter_name, parameter_spec in spec.parameters.items():
            if parameter_spec.required and parameter_name not in parameters:
                raise ActionValidationError(
                    "missing_parameter",
                    f"{name}缺少参数: {parameter_name}",
                )
            if parameter_name not in parameters:
                continue
            value = parameters[parameter_name]
            expected_type = _PYTHON_TYPES[parameter_spec.type]
            if not isinstance(value, expected_type) or (
                parameter_spec.type in {"integer", "number"}
                and isinstance(value, bool)
            ):
                raise ActionValidationError(
                    "invalid_parameter_type",
                    f"{name}.{parameter_name}类型应为{parameter_spec.type}",
                )
            if parameter_spec.enum and value not in parameter_spec.enum:
                raise ActionValidationError(
                    "invalid_parameter_value",
                    f"{name}.{parameter_name}不在允许值中",
                )
        if name == "create_commitment":
            expected_action = parameters.get("expected_action")
            expected_parameters = parameters.get("expected_parameters")
            if expected_action in {"create_commitment", "resolve_commitment"}:
                raise ActionValidationError(
                    "invalid_expected_action",
                    "承诺不能以创建或收束另一承诺作为履行效果",
                )
            if not isinstance(expected_action, str):
                raise ActionValidationError(
                    "invalid_expected_action",
                    "create_commitment.expected_action必须是Action名称",
                )
            expected_spec = self.get(expected_action)
            if not isinstance(expected_parameters, Mapping):
                raise ActionValidationError(
                    "invalid_expected_parameters",
                    "expected_parameters必须是对象",
                )
            unknown_expected = set(expected_parameters) - set(
                expected_spec.parameters
            )
            if unknown_expected:
                raise ActionValidationError(
                    "invalid_expected_parameters",
                    "expected_parameters包含目标Action未知参数: "
                    f"{', '.join(sorted(unknown_expected))}",
                )
        return spec

    @staticmethod
    def _parse_file(source: Path) -> ActionSpec:
        text = source.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ActionValidationError(
                "missing_front_matter", f"{source.name}缺少YAML front matter"
            )
        try:
            raw_header, body = text[4:].split("\n---", 1)
        except ValueError as exc:
            raise ActionValidationError(
                "invalid_front_matter", f"{source.name}的front matter未闭合"
            ) from exc
        header = yaml.safe_load(raw_header)
        if not isinstance(header, Mapping):
            raise ActionValidationError(
                "invalid_front_matter", f"{source.name}的front matter必须是对象"
            )

        required_fields = {
            "name",
            "category",
            "description",
            "parameters",
            "updates",
            "handler",
        }
        missing = required_fields - set(header)
        if missing:
            raise ActionValidationError(
                "invalid_action_spec",
                f"{source.name}缺少字段: {', '.join(sorted(missing))}",
            )

        raw_parameters = header["parameters"]
        if not isinstance(raw_parameters, Mapping):
            raise ActionValidationError(
                "invalid_action_spec", f"{source.name}.parameters必须是对象"
            )
        parameters: dict[str, ParameterSpec] = {}
        for name, raw_spec in raw_parameters.items():
            if not isinstance(raw_spec, Mapping):
                raise ActionValidationError(
                    "invalid_action_spec", f"{source.name}.{name}参数定义无效"
                )
            parameter_type = raw_spec.get("type")
            if parameter_type not in _PYTHON_TYPES:
                raise ActionValidationError(
                    "invalid_action_spec",
                    f"{source.name}.{name}使用不支持的类型: {parameter_type}",
                )
            meaning = str(raw_spec.get("meaning", "")).strip()
            if not meaning:
                raise ActionValidationError(
                    "invalid_action_spec",
                    f"{source.name}.{name}缺少参数含义meaning",
                )
            parameters[str(name)] = ParameterSpec(
                type=str(parameter_type),
                required=bool(raw_spec.get("required", False)),
                enum=tuple(raw_spec.get("enum", [])),
                meaning=meaning,
            )

        updates = header["updates"]
        if not isinstance(updates, list) or not all(
            isinstance(item, str) for item in updates
        ):
            raise ActionValidationError(
                "invalid_action_spec", f"{source.name}.updates必须是字符串数组"
            )
        return ActionSpec(
            name=str(header["name"]),
            category=str(header["category"]),
            description=str(header["description"]),
            parameters=parameters,
            updates=tuple(updates),
            handler=str(header["handler"]),
            body=body.lstrip("\n"),
            source=source,
        )
