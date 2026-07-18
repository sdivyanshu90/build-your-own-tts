"""Versioned model bundle export/load with compatibility and integrity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from tts_pipeline.config import Settings
from tts_pipeline.errors import CompatibilityError
from tts_pipeline.models.acoustic import FastSpeech2
from tts_pipeline.models.vocoder import HiFiGANGenerator
from tts_pipeline.text import Vocabulary


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@dataclass(frozen=True)
class ModelBundle:
    acoustic: FastSpeech2
    vocoder: HiFiGANGenerator
    vocabulary: Vocabulary
    version: str
    speakers: tuple[str, ...]
    languages: tuple[str, ...]


def export_bundle(
    directory: str | Path,
    acoustic: FastSpeech2,
    vocoder: HiFiGANGenerator,
    vocabulary: Vocabulary,
    settings: Settings,
    version: str = "development",
    speakers: tuple[str, ...] = ("default",),
    languages: tuple[str, ...] = ("en-US",),
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    vocabulary.save(target / "vocabulary.json")
    torch.save(acoustic.state_dict(), target / "acoustic.pt")
    torch.save(vocoder.state_dict(), target / "vocoder.pt")
    manifest = {
        "format_version": 1,
        "model_version": version,
        "config_fingerprint": settings.artifact_fingerprint(),
        "vocabulary_checksum": vocabulary.checksum,
        "speakers": speakers,
        "languages": languages,
        "files": {
            name: _digest(target / name)
            for name in ("vocabulary.json", "acoustic.pt", "vocoder.pt")
        },
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def load_bundle(directory: str | Path, settings: Settings, device: torch.device) -> ModelBundle:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("format_version") != 1:
        raise CompatibilityError("unsupported model bundle format")
    if manifest.get("config_fingerprint") != settings.artifact_fingerprint():
        raise CompatibilityError("model bundle configuration fingerprint mismatch")
    for name, digest in manifest["files"].items():
        if _digest(source / name) != digest:
            raise CompatibilityError(f"model bundle integrity failure: {name}")
    vocabulary = Vocabulary.load(source / "vocabulary.json", manifest["vocabulary_checksum"])
    acoustic = FastSpeech2(len(vocabulary.symbols), settings.model, settings.audio)
    vocoder = HiFiGANGenerator(settings.audio.n_mels, settings.vocoder)
    acoustic.load_state_dict(
        torch.load(source / "acoustic.pt", map_location="cpu", weights_only=True)
    )
    vocoder.load_state_dict(
        torch.load(source / "vocoder.pt", map_location="cpu", weights_only=True)
    )
    acoustic.to(device).eval()
    vocoder.to(device).eval()
    return ModelBundle(
        acoustic,
        vocoder,
        vocabulary,
        manifest["model_version"],
        tuple(manifest["speakers"]),
        tuple(manifest["languages"]),
    )
