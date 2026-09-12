"""Human-reviewed source loading for LLM history generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_yaml


def load_reviewed_anchors(
    world_dir: str | Path,
    character_id: str,
) -> list[dict[str, Any]]:
    world = Path(world_dir)
    selected = load_yaml(world / "frozen" / character_id / "anchors.yaml")
    pool = load_yaml(world / "anchor_pool.yaml")
    canon_by_id = {event["id"]: event for event in pool["events"]}

    anchors: list[dict[str, Any]] = []
    for anchor_id in selected["canon_anchors"]:
        source = canon_by_id[anchor_id]
        anchors.append(
            {
                "id": anchor_id,
                "source": "canon",
                "event": source["fact"],
                "state_change": source["state_change"],
                "cf_edit": source["cf_edit"],
                "visibility": {
                    key: source.get(key, [])
                    for key in (
                        "known_by",
                        "believed_by",
                        "suspected_by",
                        "outcome_known_by",
                        "later_known_by",
                    )
                    if key in source
                },
            }
        )
    for source in selected["generated_anchors"]:
        anchor = {
            "id": source["id"],
            "source": "controlled",
            "event": source["event"],
            "state_change": source["state_change"],
            "cf_edit": source["cf_edit"],
            "visibility": {"known_by": [character_id, "player"]},
        }
        if "foundation_refs" in source:
            anchor["foundation_refs"] = source["foundation_refs"]
        anchors.append(anchor)
    if len(anchors) != selected["anchor_count"]:
        raise ValueError(f"{character_id}锚点数量与anchor_count不一致")
    return anchors


def load_transition_plan(
    world_dir: str | Path,
    character_id: str,
) -> dict[str, Any]:
    return load_yaml(
        Path(world_dir) / "frozen" / character_id / "transitions.yaml"
    )
