"""Model-visible projections for frozen histories and branches."""

from __future__ import annotations

from typing import Any, Sequence


def project_frozen_history(
    history: Sequence[dict[str, Any]],
    *,
    audience: str = "npc",
) -> str:
    """Serialize only messages, visible observations, and citable event IDs."""

    if audience not in {"npc", "player"}:
        raise ValueError("audience必须是npc或player")
    lines: list[str] = []
    current_session: str | None = None
    for record in history:
        if record["session_id"] != current_session:
            current_session = record["session_id"]
            lines.append(f"[Session {current_session}]")
        observation = record.get("observation")
        if observation and audience in observation.get("visible_to", []):
            lines.append(
                f"[Event {observation['event_id']}] {observation['content']}"
            )
        for message in record["messages"]:
            speaker = "Player" if message["speaker"] == "player" else "NPC"
            lines.append(
                f"[Round {record['round']}][{speaker}] {message['utterance']}"
            )
    return "\n".join(lines)
