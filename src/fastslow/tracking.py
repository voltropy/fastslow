from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    mode: str

    def phase(self, name: str) -> None: ...

    def log(self, step: int, metrics: dict[str, float]) -> None: ...

    def complete(self, summary: str | None = None) -> None: ...

    def fail(self, summary: str | None = None) -> None: ...


class JsonlTracker:
    def __init__(self, metrics_path: Path) -> None:
        self.mode = "jsonl"
        self.metrics_path = metrics_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: dict[str, object]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def phase(self, name: str) -> None:
        self._write({"event": "phase", "phase": name, "time": time.time()})

    def log(self, step: int, metrics: dict[str, float]) -> None:
        self._write({"event": "metrics", "step": step, "metrics": metrics, "time": time.time()})

    def complete(self, summary: str | None = None) -> None:
        self._write({"event": "complete", "summary": summary, "time": time.time()})

    def fail(self, summary: str | None = None) -> None:
        self._write({"event": "fail", "summary": summary, "time": time.time()})


class VoltaTracker:
    def __init__(self) -> None:
        import volta

        self.mode = "volta"
        self._run = volta.init()

    def phase(self, name: str) -> None:
        self._run.phase(name)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        self._run.log(step=step, **metrics)

    def complete(self, summary: str | None = None) -> None:
        self._run.complete(summary=summary)

    def fail(self, summary: str | None = None) -> None:
        self._run.fail(summary=summary)


def build_tracker(artifacts_dir: Path) -> Tracker:
    want_volta = os.getenv("VOLTA_PROJECT") and os.getenv("VOLTA_EXPERIMENT")
    if want_volta and os.getenv("VOLTA_SPANNER_DATABASE"):
        try:
            return VoltaTracker()
        except Exception:
            pass
    return JsonlTracker(artifacts_dir / "metrics.jsonl")
