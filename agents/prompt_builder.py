"""Strict prompt construction and information isolation."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from gamecore import ActionRegistry, GameContext, WorldDefinition

_PROMPT_HIDDEN_KEYS = {
    "evaluation_spec",
    "testable_boundaries",
    "target_boundaries",
    "challenge_space",
    "forbidden_outcomes",
    "allowed_change",
    "protected_facts",
    "protected_items",
    "authority_rules",
    "role_boundaries",
    "interaction_value",
    "productive_behaviors",
    "required_progress",
    "progress_checks",
    "eval_setting",
}


def project_npc_character_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Natural NPC self-context without evaluation labels or hard prohibitions."""

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: strip(item)
                for key, item in value.items()
                if key not in _PROMPT_HIDDEN_KEYS
            }
        if isinstance(value, list):
            return [strip(item) for item in value]
        return copy.deepcopy(value)

    visible = strip(card)
    identity = visible.get("identity", {})
    private = visible.pop("private", {})
    worldview = visible.get("worldview", {})
    if isinstance(worldview, dict):
        worldview.pop("prohibitions", None)
    affiliations = (
        identity.pop("affiliations", [])
        if isinstance(identity, dict)
        else []
    )
    background: dict[str, Any] = {}
    if affiliations:
        background["affiliations"] = [
            re.sub(r"^(公开|隐秘)_", "", value)
            if isinstance(value, str)
            else copy.deepcopy(value)
            for value in affiliations
        ]
    if isinstance(private, Mapping):
        if private.get("private_background"):
            background["personal_history"] = copy.deepcopy(
                private["private_background"]
            )
        if private.get("secrets"):
            background["self_knowledge"] = copy.deepcopy(private["secrets"])
    if background:
        visible["background"] = background
    return visible


class StateAccess(str, Enum):
    HISTORY_ONLY = "history_only"
    STATE_ONLY = "state_only"
    HISTORY_AND_STATE = "history_and_state"


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str


class PromptIsolationError(ValueError):
    """Raised when a prompt cannot be built without violating its contract."""


class PromptBuilder:
    """Build role-specific prompts from whitelisted GameContext projections."""

    _PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
    _HIDDEN_KEYS = _PROMPT_HIDDEN_KEYS
    _PUBLIC_ACTIONS = {
        "create_commitment",
        "reveal_fact",
        "move",
        "transfer_item",
        "create_offer",
        "respond_offer",
        "use_item",
        "decide_access",
        "attack",
        "end_dialogue",
    }

    def __init__(
        self,
        registry: ActionRegistry,
        *,
        template_root: str | Path | None = None,
        language: str = "zh",
        world: WorldDefinition | None = None,
    ) -> None:
        self.registry = registry
        self.language = self._normalize_language(language)
        self.world = world
        self.template_root = (
            Path(template_root)
            if template_root is not None
            else Path(__file__).resolve().parent / "prompts"
        )

    @classmethod
    def from_environment(
        cls,
        registry: ActionRegistry,
        environment_path: str | Path,
        *,
        template_root: str | Path | None = None,
    ) -> "PromptBuilder":
        world = WorldDefinition.load_yaml(environment_path)
        return cls(
            registry,
            template_root=template_root,
            language=world.language,
            world=world,
        )

    def build_npc(
        self,
        context: GameContext,
        scenario: Mapping[str, Any],
        *,
        available_actions: Sequence[str],
        state_access: StateAccess = StateAccess.HISTORY_ONLY,
    ) -> PromptBundle:
        current_query, prior_history = self._split_current_player_query(context)
        actions = [self._action_for_prompt(name) for name in available_actions]
        unavailable = "未提供" if self.language == "zh" else "Not provided"
        visible_history: Any = unavailable
        runtime_state: Any = unavailable
        if state_access in {
            StateAccess.HISTORY_ONLY,
            StateAccess.HISTORY_AND_STATE,
        }:
            visible_history = self._history_projection(
                prior_history, audience="npc"
            )
        if state_access in {
            StateAccess.STATE_ONLY,
            StateAccess.HISTORY_AND_STATE,
        }:
            runtime_state = self._strip_hidden(context.runtime_state)

        values = {
            "character_card": self._json(
                self._npc_character_card(context.data["character_card"])
            ),
            "scenario": self._json(self._npc_scenario(scenario)),
            "observation": self._json(
                self._npc_observation(context, available_actions)
            ),
            "runtime_state": self._json(runtime_state),
            "history": self._json(visible_history),
            "current_query": self._json(current_query),
            "actions": self._json(actions),
        }
        return PromptBundle(
            system_prompt=self._load(
                f"npc/online_system.{self.language}.j2"
            ),
            user_prompt=self._render(
                f"npc/online_user.{self.language}.j2", values
            ),
        )

    def build_player(
        self,
        context: GameContext,
        scenario: Mapping[str, Any],
        *,
        mode: str,
        player_profile: Mapping[str, Any] | None = None,
    ) -> PromptBundle:
        if mode not in {"normal", "pressure"}:
            raise PromptIsolationError(f"未知Player模式: {mode}")
        goal_key = f"{mode}_player_goal"
        if goal_key not in scenario:
            raise PromptIsolationError(f"场景缺少{goal_key}")
        values = {
            "player_profile": self._json(
                self._strip_hidden(player_profile or {"role": "player"})
            ),
            "npc_public_profile": self._json(
                self._public_character_profile(context.data["character_card"])
            ),
            "scenario": self._json(self._player_scenario(scenario)),
            "player_goal": self._json(scenario[goal_key]),
            "observation": self._json(self._player_observation(context)),
            "history": self._json(
                self._history_projection(context.history, audience="player")
            ),
        }
        return PromptBundle(
            system_prompt=self._load(
                f"player/{mode}_system.{self.language}.j2"
            ),
            user_prompt=self._render(
                f"player/online_user.{self.language}.j2", values
            ),
        )

    def build_checker(
        self,
        context: GameContext,
        evaluation_spec: Mapping[str, Any],
    ) -> PromptBundle:
        if not evaluation_spec:
            raise PromptIsolationError("Consistency Checker需要EvaluationSpec")
        values = {
            "character_card": self._json(context.data["character_card"]),
            "evaluation_spec": self._json(evaluation_spec),
            "trajectory": self._json(context.to_dict()),
        }
        return PromptBundle(
            system_prompt=self._load(
                f"checker/consistency_system.{self.language}.j2"
            ),
            user_prompt=self._render(
                f"checker/consistency_user.{self.language}.j2", values
            ),
        )

    def _load(self, relative_path: str) -> str:
        path = self.template_root / relative_path
        return path.read_text(encoding="utf-8").strip()

    def _render(
        self,
        relative_path: str,
        values: Mapping[str, str],
    ) -> str:
        template = self._load(relative_path)
        required = set(self._PLACEHOLDER.findall(template))
        missing = required - set(values)
        extra = set(values) - required
        if missing or extra:
            raise PromptIsolationError(
                f"模板变量不匹配 missing={sorted(missing)} extra={sorted(extra)}"
            )
        return self._PLACEHOLDER.sub(
            lambda match: values[match.group(1)],
            template,
        )

    def _action_for_prompt(self, name: str) -> dict[str, Any]:
        spec = self.registry.get(name)
        parameters: dict[str, Any] = {}
        for parameter_name, parameter in spec.parameters.items():
            item: dict[str, Any] = {
                "type": parameter.type,
                "required": parameter.required,
            }
            if parameter.enum:
                item["enum"] = list(parameter.enum)
            parameters[parameter_name] = item
        return {
            "name": spec.name,
            "description": spec.description,
            "parameters": parameters,
            "usage": spec.body.strip(),
        }

    def _npc_character_card(
        self,
        card: Mapping[str, Any],
    ) -> dict[str, Any]:
        visible = project_npc_character_card(card)
        if "character_id" not in visible:
            raise PromptIsolationError("角色卡缺少character_id")
        return visible

    @staticmethod
    def _public_character_profile(
        card: Mapping[str, Any],
    ) -> dict[str, Any]:
        identity = card.get("identity", {})
        if not isinstance(identity, Mapping):
            return {"character_id": card.get("character_id")}
        return {
            "character_id": card.get("character_id"),
            "name": identity.get("name"),
            "role": identity.get("role"),
            "public_background": copy.deepcopy(
                identity.get("public_background", [])
            ),
        }

    @staticmethod
    def _npc_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: copy.deepcopy(scenario[key])
            for key in ("scenario_id", "setting")
            if key in scenario
        }

    @staticmethod
    def _player_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: copy.deepcopy(scenario[key])
            for key in ("scenario_id", "setting")
            if key in scenario
        }

    def _npc_observation(
        self,
        context: GameContext,
        available_actions: Sequence[str],
    ) -> dict[str, Any]:
        environment = context.environment
        actor = context.character_id
        player_claims = {
            claim_id: {
                "content": copy.deepcopy(claim.get("content")),
                "decision": claim.get("decision"),
            }
            for claim_id, claim in context.runtime_state["claims"].items()
            if isinstance(claim, Mapping) and claim.get("source") == "player"
        }
        offers = [
            copy.deepcopy(offer)
            for offer in environment["offers"]
            if offer.get("proposer") == actor or offer.get("recipient") == actor
        ]
        access = {
            resource: copy.deepcopy(record)
            for resource, record in environment["access"].items()
            if actor in record.get("controllers", [])
            or actor in record.get("subjects", {})
        }
        valid_arguments = self.npc_valid_action_arguments(
            context,
            available_actions,
        )
        return {
            "location": environment["locations"].get(actor),
            "inventory": copy.deepcopy(
                environment["inventories"].get(actor, [])
            ),
            "health": environment["health"].get(actor),
            "participant_locations": {
                key: value
                for key, value in environment["locations"].items()
                if key in {actor, "player"}
            },
            "pending_offers": offers,
            "access": access,
            "task_status": environment["task_status"],
            "dialogue_status": environment["dialogue_status"],
            "player_claims": player_claims,
            "valid_action_arguments": valid_arguments,
        }

    def npc_valid_action_arguments(
        self,
        context: GameContext,
        available_actions: Sequence[str],
    ) -> dict[str, list[str]]:
        """Return runtime-constrained IDs that NPC actions may reference."""

        environment = context.environment
        actor = context.character_id
        player_claims = {
            claim_id
            for claim_id, claim in context.runtime_state["claims"].items()
            if isinstance(claim, Mapping) and claim.get("source") == "player"
        }
        valid_event_ids = {
            str(event["id"])
            for entry in context.history
            for event in entry.get("events", [])
            if isinstance(event, Mapping) and event.get("id")
        }
        access = {
            resource
            for resource, record in environment["access"].items()
            if actor in record.get("controllers", [])
            or actor in record.get("subjects", {})
        }
        valid_arguments: dict[str, list[str]] = {}
        if "accept_claim" in available_actions:
            valid_arguments["accept_claim.claim_id"] = sorted(player_claims)
            valid_arguments["accept_claim.evidence_event"] = sorted(
                valid_event_ids
            )
        if "decide_access" in available_actions:
            valid_arguments["decide_access.resource"] = sorted(access)
            valid_arguments["decide_access.subject"] = ["player"]
        if "reveal_fact" in available_actions:
            valid_arguments["reveal_fact.fact_id"] = sorted(
                context.runtime_state["knowledge"]
            )
            valid_arguments["reveal_fact.recipient"] = ["player"]
        if "update_relationship" in available_actions:
            valid_arguments["update_relationship.target"] = sorted(
                context.runtime_state["relationships"]
            )
            valid_arguments["update_relationship.reason_event"] = sorted(
                valid_event_ids
            )
        if "resolve_commitment" in available_actions:
            commitments = context.runtime_state["commitments"]
            valid_arguments["resolve_commitment.commitment_id"] = sorted(
                str(item["id"])
                for item in commitments
                if isinstance(item, Mapping)
                and item.get("id")
                and item.get("status") == "active"
            )
            valid_arguments["resolve_commitment.reason_event"] = sorted(
                valid_event_ids
            )
        if "respond_offer" in available_actions:
            valid_arguments["respond_offer.offer_id"] = sorted(
                str(offer["id"])
                for offer in environment["offers"]
                if offer.get("id")
                and offer.get("status", "pending") == "pending"
            )
        inventory = sorted(environment["inventories"].get(actor, []))
        if "transfer_item" in available_actions:
            valid_arguments["transfer_item.item"] = inventory
        if "use_item" in available_actions:
            valid_arguments["use_item.item"] = inventory
        if "move" in available_actions:
            current_location = environment["locations"].get(actor)
            valid_arguments["move.destination"] = (
                sorted(
                    self.world.locations.get(current_location, {}).get(
                        "connected_to", []
                    )
                )
                if self.world is not None
                else []
            )
        return valid_arguments

    def _player_observation(self, context: GameContext) -> dict[str, Any]:
        environment = context.environment
        actor = context.character_id
        offers = [
            copy.deepcopy(offer)
            for offer in environment["offers"]
            if offer.get("proposer") == "player"
            or offer.get("recipient") == "player"
        ]
        access: dict[str, Any] = {}
        for resource, record in environment["access"].items():
            decision = record.get("subjects", {}).get("player")
            if decision is not None:
                access[resource] = decision
        return {
            "location": environment["locations"].get("player"),
            "inventory": copy.deepcopy(
                environment["inventories"].get("player", [])
            ),
            "health": environment["health"].get("player"),
            "npc_location": environment["locations"].get(actor),
            "npc_health": environment["health"].get(actor),
            "pending_offers": offers,
            "access": access,
            "dialogue_status": environment["dialogue_status"],
        }

    def _history_projection(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        audience: str,
    ) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        for entry in history:
            item = {
                "turn": entry["turn"],
                "speaker": entry["speaker"],
                "utterance": entry["utterance"],
            }
            if entry["speaker"] == "npc" and entry.get("actions"):
                actions = copy.deepcopy(entry["actions"])
                if audience == "player":
                    actions = [
                        action
                        for action in actions
                        if action.get("name") in self._PUBLIC_ACTIONS
                    ]
                if actions:
                    item["actions"] = actions
            projected.append(item)
        return projected

    @staticmethod
    def _split_current_player_query(
        context: GameContext,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not context.history or context.history[-1]["speaker"] != "player":
            raise PromptIsolationError(
                "构建NPC prompt前，history末条必须是Player query"
            )
        current = str(context.history[-1]["utterance"])
        return current, copy.deepcopy(context.history[:-1])

    def _strip_hidden(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: self._strip_hidden(item)
                for key, item in value.items()
                if key not in self._HIDDEN_KEYS
            }
        if isinstance(value, list):
            return [self._strip_hidden(item) for item in value]
        return copy.deepcopy(value)

    @staticmethod
    def _json(value: Any) -> str:
        if isinstance(value, str) and value in {"未提供", "Not provided"}:
            return value
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)

    @staticmethod
    def _normalize_language(language: str) -> str:
        normalized = language.strip().lower().replace("_", "-")
        aliases = {
            "zh": "zh",
            "zh-cn": "zh",
            "chinese": "zh",
            "en": "en",
            "en-us": "en",
            "en-gb": "en",
            "english": "en",
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise PromptIsolationError(
                f"当前只支持zh或en Prompt，收到: {language}"
            ) from exc
