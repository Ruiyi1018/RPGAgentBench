"""Process-local concurrency gates shared by clients for one deployment."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class _Gate:
    limit: int
    active: int = 0
    condition: threading.Condition = field(
        default_factory=threading.Condition
    )

    def tighten(self, limit: int) -> None:
        with self.condition:
            self.limit = min(self.limit, limit)
            self.condition.notify_all()

    def acquire(self) -> None:
        with self.condition:
            while self.active >= self.limit:
                self.condition.wait()
            self.active += 1

    def release(self) -> None:
        with self.condition:
            self.active -= 1
            self.condition.notify()


_LOCK = threading.Lock()
_GATES: dict[str, _Gate] = {}


@contextmanager
def model_concurrency_gate(key: str, limit: int) -> Iterator[None]:
    """Bound in-flight requests even when runners create many clients."""

    resolved_limit = max(1, int(limit))
    with _LOCK:
        gate = _GATES.get(key)
        if gate is None:
            gate = _Gate(limit=resolved_limit)
            _GATES[key] = gate
        else:
            gate.tighten(resolved_limit)
    gate.acquire()
    try:
        yield
    finally:
        gate.release()
