from __future__ import annotations

from pathlib import Path

import pytest

from fastslow.tracking import JsonlTracker, build_tracker, default_anomaly_check, validate_tracking_backend


def test_build_tracker_defaults_to_jsonl_without_volta_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VOLTA_PROJECT", raising=False)
    monkeypatch.delenv("VOLTA_EXPERIMENT", raising=False)
    monkeypatch.delenv("VOLTA_SPANNER_DATABASE", raising=False)

    tracker = build_tracker(tmp_path, "auto")
    assert isinstance(tracker, JsonlTracker)


def test_validate_tracking_backend_requires_strict_volta_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLTA_PROJECT", "fastslow")
    monkeypatch.setenv("VOLTA_EXPERIMENT", "run-1")
    monkeypatch.delenv("VOLTA_SPANNER_DATABASE", raising=False)

    with pytest.raises(RuntimeError, match="VOLTA_SPANNER_DATABASE"):
        validate_tracking_backend("auto")


def test_default_anomaly_check_flags_non_finite() -> None:
    message = default_anomaly_check(
        {"step": 12, "train_loss": float("inf")},
        [{"step": 11, "train_loss": 5.0}],
    )
    assert message is not None
