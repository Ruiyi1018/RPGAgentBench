"""Thread-safe terminal progress reporting for long experiment runs."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field


@dataclass
class ProgressReporter:
    total: int
    completed: int = 0
    enabled: bool = True
    width: int = 24
    _started_at: float = field(default_factory=time.monotonic, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def advance(self, label: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.completed += 1
            self._render(label)

    def render(self, label: str = "ready") -> None:
        if not self.enabled:
            return
        with self._lock:
            self._render(label)

    def _render(self, label: str) -> None:
        elapsed = max(time.monotonic() - self._started_at, 0.001)
        fraction = min(self.completed / max(self.total, 1), 1.0)
        filled = round(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.completed / elapsed
        remaining = max(self.total - self.completed, 0)
        eta = remaining / rate if rate > 0 else 0.0
        message = (
            f"\r[{bar}] {self.completed}/{self.total} "
            f"{fraction:6.1%} elapsed={_duration(elapsed)} "
            f"eta={_duration(eta)} {label}"
        )
        print(
            message,
            end="\n" if self.completed >= self.total else "",
            file=sys.stderr,
            flush=True,
        )


def _duration(seconds: float) -> str:
    value = max(int(seconds), 0)
    minutes, seconds = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
