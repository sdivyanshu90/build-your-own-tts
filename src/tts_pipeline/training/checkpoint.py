"""Atomic checkpoint save/restore with explicit integrity manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from tts_pipeline.errors import CompatibilityError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any = None,
    scaler: Any = None,
    **state: Any,
) -> Path:
    """Write same-filesystem temporary data then atomically replace the checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "state": state,
    }
    torch.save(payload, temporary, _use_new_zipfile_serialization=True)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(target)
    manifest = {"sha256": sha256_file(target), "size": target.stat().st_size, "format_version": 1}
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
) -> dict[str, Any]:
    source = Path(path)
    manifest_path = source.with_suffix(source.suffix + ".json")
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())["sha256"]
        if sha256_file(source) != expected:
            raise CompatibilityError(f"checkpoint integrity failure: {source}")
    # weights_only constrains global construction; optimizer tensors and primitives remain valid.
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise CompatibilityError("unsupported checkpoint format")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    return dict(payload.get("state", {}))
