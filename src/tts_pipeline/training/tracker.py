"""Experiment-tracking abstraction; local JSONL is always available."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    def log_params(self, values: dict[str, Any]) -> None: ...
    def log_metrics(self, values: dict[str, float], step: int) -> None: ...
    def log_artifact(self, path: Path) -> None: ...


class LocalTracker:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = self.run_dir / "events.jsonl"

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"time": time.time(), "type": kind, **payload}, default=str) + "\n"
            )

    def log_params(self, values: dict[str, Any]) -> None:
        self._write("parameters", {"values": values})

    def log_metrics(self, values: dict[str, float], step: int) -> None:
        self._write("metrics", {"step": step, "values": values})

    def log_artifact(self, path: Path) -> None:
        self._write("artifact", {"path": str(path), "size": path.stat().st_size})


class MLflowTracker:
    def __init__(self, experiment: str, run_name: str | None = None) -> None:
        import mlflow  # type: ignore[import-not-found]

        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        self.client = mlflow

    def log_params(self, values: dict[str, Any]) -> None:
        self.client.log_params(values)

    def log_metrics(self, values: dict[str, float], step: int) -> None:
        self.client.log_metrics(values, step=step)

    def log_artifact(self, path: Path) -> None:
        self.client.log_artifact(str(path))
