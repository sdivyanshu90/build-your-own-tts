"""Dataset-agnostic JSONL manifest schema and validation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tts_pipeline.audio.io import audio_info
from tts_pipeline.errors import ValidationError


@dataclass(frozen=True)
class ManifestRecord:
    audio_path: Path
    transcript: str
    speaker_id: str = "default"
    language: str = "en-US"
    normalized_transcript: str | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        payload = f"{self.audio_path.resolve()}\0{self.transcript}\0{self.speaker_id}"
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ValidationReport:
    total: int
    valid: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    speakers: dict[str, int]
    duration_seconds: float
    fingerprint: str


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValidationError(f"manifest not found: {source}")
    records: list[ManifestRecord] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            audio_path = Path(raw["audio_path"])
            if not audio_path.is_absolute():
                audio_path = (source.parent / audio_path).resolve()
            records.append(
                ManifestRecord(
                    audio_path=audio_path,
                    transcript=str(raw["transcript"]),
                    speaker_id=str(raw.get("speaker_id", "default")),
                    language=str(raw.get("language", "en-US")),
                    normalized_transcript=raw.get("normalized_transcript"),
                    duration=float(raw["duration"]) if raw.get("duration") is not None else None,
                    metadata=dict(raw.get("metadata", {})),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid manifest line {line_number}: {exc}") from exc
    if not records:
        raise ValidationError("manifest contains no records")
    return records


def validate_manifest(
    records: Iterable[ManifestRecord],
    expected_sample_rate: int | None = None,
    min_duration: float = 0.1,
    max_duration: float = 30.0,
) -> ValidationReport:
    rows = list(records)
    errors: list[str] = []
    warnings: list[str] = []
    seen_audio: set[Path] = set()
    seen_identity: set[str] = set()
    durations = 0.0
    valid = 0
    identities: list[str] = []
    for index, record in enumerate(rows):
        prefix = f"record {index} ({record.audio_path})"
        row_errors = 0
        if not record.transcript.strip():
            errors.append(f"{prefix}: empty transcript")
            row_errors += 1
        if record.audio_path in seen_audio or record.identity in seen_identity:
            errors.append(f"{prefix}: duplicate sample")
            row_errors += 1
        seen_audio.add(record.audio_path)
        seen_identity.add(record.identity)
        identities.append(record.identity)
        if not record.audio_path.is_file():
            errors.append(f"{prefix}: missing audio")
            continue
        try:
            info = audio_info(record.audio_path)
        except (RuntimeError, OSError, ValueError) as exc:
            errors.append(f"{prefix}: corrupt audio: {exc}")
            continue
        duration = info.duration
        durations += duration
        if not min_duration <= duration <= max_duration:
            errors.append(
                f"{prefix}: duration {duration:.3f}s outside [{min_duration}, {max_duration}]"
            )
            row_errors += 1
        if expected_sample_rate and info.samplerate != expected_sample_rate:
            warnings.append(f"{prefix}: sample rate {info.samplerate}, will resample")
        if info.channels > 2:
            warnings.append(f"{prefix}: {info.channels} channels will be mixed to mono")
        if row_errors == 0:
            valid += 1
    digest = hashlib.sha256("\n".join(sorted(identities)).encode()).hexdigest()
    return ValidationReport(
        len(rows),
        valid,
        tuple(errors),
        tuple(warnings),
        dict(Counter(r.speaker_id for r in rows)),
        durations,
        digest,
    )


def split_manifest(
    records: Iterable[ManifestRecord],
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 1337,
) -> dict[str, list[ManifestRecord]]:
    """Speaker-stratified, deterministic split with audio-identity leakage prevention."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("split fractions must be non-negative and sum to less than one")
    by_speaker: dict[str, list[ManifestRecord]] = {}
    for row in records:
        by_speaker.setdefault(row.speaker_id, []).append(row)
    result: dict[str, list[ManifestRecord]] = {"train": [], "validation": [], "test": []}
    # Dataset splitting needs reproducibility, not cryptographic randomness.
    rng = random.Random(seed)  # noqa: S311
    for speaker in sorted(by_speaker):
        group = sorted(by_speaker[speaker], key=lambda r: r.identity)
        rng.shuffle(group)
        test_count = round(len(group) * test_fraction)
        validation_count = round(len(group) * validation_fraction)
        result["test"].extend(group[:test_count])
        result["validation"].extend(group[test_count : test_count + validation_count])
        result["train"].extend(group[test_count + validation_count :])
    return result


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in records:
        lines.append(
            json.dumps(
                {
                    "audio_path": str(row.audio_path),
                    "transcript": row.transcript,
                    "normalized_transcript": row.normalized_transcript,
                    "speaker_id": row.speaker_id,
                    "language": row.language,
                    "duration": row.duration,
                    "metadata": row.metadata,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
