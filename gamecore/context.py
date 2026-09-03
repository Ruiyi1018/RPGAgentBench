"""GameContext and static world-definition loading."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from jsonschema import Draft202012Validator
except ModuleNotFoundError:  # pragma: no cover - depends on installation mode
    Draft202012Validator = None  # type: ignore[assignment,misc]

from .errors import ContextValidationError


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class WorldDefinition:
    """Static locations, items, and deterministic environment rules."""

    world_id: str
    language: str
    locations: dict[str, dict[str, Any]]
    items: dict[str, dict[str, Any]]
    combat: dict[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorldDefinition":
        locations = {
            entry["id"]: copy.deepcopy(dict(entry))
            for entry in data.get("locations", [])
        }
        items = {
            entry["id"]: copy.deepcopy(dict(entry))
            for entry in data.get("items", [])
        }
        return cls(
            world_id=str(data.get("world_id", "")),
            language=str(data.get("language", "zh")),
            locations=locations,
            items=items,
            combat=copy.deepcopy(dict(data.get("combat", {}))),
        )

    @classmethod
    def load_yaml(cls, path: str | Path) -> "WorldDefinition":
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise ContextValidationError("environment.yaml根节点必须是对象")
        return cls.from_mapping(data)

    def connected(self, source: str, destination: str) -> bool:
        location = self.locations.get(source)
        return bool(location and destination in location.get("connected_to", []))

    def attack_damage(self, method: str) -> int:
        by_method = self.combat.get("damage_by_method", {})
        damage = by_method.get(method, self.combat.get("default_damage", 10))
        if not isinstance(damage, int) or damage <= 0:
            raise ContextValidationError("攻击伤害规则必须是正整数")
        return damage


class GameContext:
    """Mutable implementation wrapper; only GameCore may mutate its data."""

    def __init__(self, data: Mapping[str, Any], *, validate: bool = True) -> None:
        self.data: dict[str, Any] = copy.deepcopy(dict(data))
        self._apply_defaults()
        if validate:
            self.validate()

    @classmethod
    def load_json(cls, path: str | Path) -> "GameContext":
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def clone(self) -> "GameContext":
        return GameContext(self.data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    @property
    def character_id(self) -> str:
        card = self.data["character_card"]
        character_id = card.get("character_id") or card.get("id")
        if not character_id:
            raise ContextValidationError("character_card缺少character_id")
        return str(character_id)

    @property
    def runtime_state(self) -> dict[str, Any]:
        return self.data["runtime_state"]

    @property
    def environment(self) -> dict[str, Any]:
        return self.data["environment"]

    @property
    def history(self) -> list[dict[str, Any]]:
        return self.data["history"]

    @property
    def next_turn(self) -> int:
        return max((int(entry["turn"]) for entry in self.history), default=0) + 1

    def validate(self) -> None:
        self._validate_core_contract()
        if Draft202012Validator is None:
            return
        schema_path = _project_root() / "schemas" / "game_context.json"
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(self.data),
            key=lambda error: list(error.path),
        )
        if errors:
            details = "; ".join(
                f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
                for error in errors
            )
            raise ContextValidationError(details)

    def _validate_core_contract(self) -> None:
        allowed_root = {
            "character_card",
            "runtime_state",
            "environment",
            "history",
        }
        if set(self.data) - allowed_root:
            raise ContextValidationError("GameContext包含未知顶层字段")
        if not isinstance(self.data["character_card"], dict):
            raise ContextValidationError("character_card必须是对象")
        if not isinstance(self.data["runtime_state"], dict):
            raise ContextValidationError("runtime_state必须是对象")
        if not isinstance(self.data["environment"], dict):
            raise ContextValidationError("environment必须是对象")
        if not isinstance(self.data["history"], list):
            raise ContextValidationError("history必须是数组")
        for entry in self.data["history"]:
            if not isinstance(entry, dict):
                raise ContextValidationError("history条目必须是对象")
            if not {"turn", "speaker", "utterance"} <= set(entry):
                raise ContextValidationError("history条目缺少必需字段")
            if (
                not isinstance(entry["turn"], int)
                or entry["turn"] < 1
                or entry["speaker"] not in {"player", "npc", "gamecore"}
                or not isinstance(entry["utterance"], str)
            ):
                raise ContextValidationError("history条目字段无效")

    def event_exists(self, event_id: str) -> bool:
        for entry in self.history:
            for event in entry.get("events", []):
                if event == event_id:
                    return True
                if isinstance(event, Mapping) and event.get("id") == event_id:
                    return True
        return False

    def append_history(self, entry: Mapping[str, Any]) -> None:
        self.history.append(copy.deepcopy(dict(entry)))

    def _apply_defaults(self) -> None:
        self.data.setdefault("character_card", {})
        runtime = self.data.setdefault("runtime_state", {})
        runtime.setdefault("claims", {})
        runtime.setdefault("commitments", [])
        runtime.setdefault("relationships", {})
        runtime.setdefault("goals", [])
        runtime.setdefault("disclosures", [])
        runtime.setdefault("knowledge", {})

        environment = self.data.setdefault("environment", {})
        environment.setdefault("health", {})
        environment.setdefault("inventories", {})
        environment.setdefault("locations", {})
        environment.setdefault("access", {})
        environment.setdefault("offers", [])
        environment.setdefault("task_status", "active")
        environment.setdefault("dialogue_status", "active")
        self.data.setdefault("history", [])
