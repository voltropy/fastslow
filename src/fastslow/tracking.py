from __future__ import annotations

import importlib.util
import json
import math
import os
from statistics import median
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


class NullTracker:
    def __init__(self) -> None:
        self.mode = "disabled"

    def phase(self, name: str) -> None:
        return None

    def log(self, step: int, metrics: dict[str, float]) -> None:
        return None

    def complete(self, summary: str | None = None) -> None:
        return None

    def fail(self, summary: str | None = None) -> None:
        return None


def _history_values(history: object, key: str) -> list[float]:
    if not isinstance(history, list):
        return []

    values: list[float] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def default_anomaly_check(entry: object, history: object) -> str | None:
    if not isinstance(entry, dict):
        return None

    numeric_metrics = {
        key: float(value)
        for key, value in entry.items()
        if isinstance(value, (int, float)) and key != "step"
    }

    for key, value in numeric_metrics.items():
        if not math.isfinite(value):
            return f"{key} became non-finite"

    train_loss = numeric_metrics.get("train_loss")
    loss_history = _history_values(history, "train_loss")
    if train_loss is not None and len(loss_history) >= 5:
        baseline = median(loss_history[:10])
        if baseline > 0.0 and train_loss > baseline * 3.0 and train_loss - baseline > 5.0:
            return f"train_loss spiked from baseline {baseline:.3f} to {train_loss:.3f}"

    grad_norm = numeric_metrics.get("grad_norm")
    grad_history = _history_values(history, "grad_norm")
    if grad_norm is not None and len(grad_history) >= 5:
        baseline = median(grad_history[:10])
        if baseline > 0.0 and grad_norm > baseline * 5.0 and grad_norm > 1000.0:
            return f"grad_norm spiked from baseline {baseline:.3f} to {grad_norm:.3f}"

    return None


class VoltaTracker:
    def __init__(self) -> None:
        import volta

        self.mode = "volta"
        self._run = volta.init(anomaly_check=default_anomaly_check)

    def phase(self, name: str) -> None:
        self._run.phase(name)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        self._run.log(step=step, **metrics)

    def complete(self, summary: str | None = None) -> None:
        self._run.complete(summary=summary)

    def fail(self, summary: str | None = None) -> None:
        self._run.fail(summary=summary)


STRICT_VOLTA_ENV_VARS = ("VOLTA_PROJECT", "VOLTA_EXPERIMENT", "VOLTA_SPANNER_DATABASE")
STRICT_VOLTA_AUTH_VARS = ("GOOGLE_APPLICATION_CREDENTIALS",)


def _should_use_volta(tracking_backend: str) -> bool:
    if tracking_backend == "volta":
        return True
    if tracking_backend == "jsonl":
        return False
    if tracking_backend != "auto":
        raise ValueError(f"unknown tracking backend: {tracking_backend}")
    return any(os.getenv(key) for key in STRICT_VOLTA_ENV_VARS)


def validate_tracking_backend(tracking_backend: str) -> None:
    if not _should_use_volta(tracking_backend):
        return

    missing = [key for key in STRICT_VOLTA_ENV_VARS if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Volta tracking requested but required environment variables are missing: "
            + ", ".join(missing)
        )
    missing_auth = [key for key in STRICT_VOLTA_AUTH_VARS if not os.getenv(key)]
    if missing_auth:
        raise RuntimeError(
            "Volta tracking requested but authentication is not configured: "
            + ", ".join(missing_auth)
            + ". Mount a service account key and set GOOGLE_APPLICATION_CREDENTIALS."
        )
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_path and not Path(creds_path).exists():
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to {creds_path!r} which does not exist. "
            "Check that the service account key is mounted correctly."
        )
    if importlib.util.find_spec("volta") is None:
        raise RuntimeError(
            "Volta tracking requested but the `volta` Python package is not installed. "
            "Use the `volta-runtime:latest` base image or install `fastslow[volta]`."
        )


def build_tracker(artifacts_dir: Path, tracking_backend: str) -> Tracker:
    validate_tracking_backend(tracking_backend)
    if _should_use_volta(tracking_backend):
        return VoltaTracker()
    return JsonlTracker(artifacts_dir / "metrics.jsonl")
