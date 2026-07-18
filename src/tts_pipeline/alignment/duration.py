"""Isolated alignment adapter with a deterministic fixture backend.

Production datasets should use MFA or another forced aligner to write this same artifact schema.
The uniform backend is intentionally restricted to local fixtures and smoke tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tts_pipeline.errors import ValidationError


@dataclass(frozen=True)
class AlignmentArtifact:
    durations: tuple[int, ...]
    token_count: int
    frame_count: int
    backend: str
    backend_version: str
    source_fingerprint: str

    def validate(self) -> None:
        if len(self.durations) != self.token_count:
            raise ValidationError("duration count does not equal token count")
        if any(value < 0 for value in self.durations):
            raise ValidationError("durations cannot be negative")
        if sum(self.durations) != self.frame_count:
            raise ValidationError("durations do not sum to mel frame count")

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({**self.__dict__, "durations": self.durations}, indent=2) + "\n"
        )

    @classmethod
    def load(cls, path: str | Path) -> AlignmentArtifact:
        raw = json.loads(Path(path).read_text())
        artifact = cls(
            tuple(raw["durations"]),
            raw["token_count"],
            raw["frame_count"],
            raw["backend"],
            raw["backend_version"],
            raw["source_fingerprint"],
        )
        artifact.validate()
        return artifact


class AlignmentBackend(Protocol):
    def align(self, token_ids: list[int], frame_count: int, source: str) -> AlignmentArtifact: ...


class UniformAlignmentBackend:
    """Working smoke-test backend that assigns frames uniformly across non-padding tokens."""

    def align(self, token_ids: list[int], frame_count: int, source: str) -> AlignmentArtifact:
        if not token_ids or frame_count < len(token_ids):
            raise ValidationError("uniform alignment needs at least one frame per token")
        base, remainder = divmod(frame_count, len(token_ids))
        durations = tuple(base + (1 if i < remainder else 0) for i in range(len(token_ids)))
        artifact = AlignmentArtifact(
            durations,
            len(token_ids),
            frame_count,
            "uniform-fixture",
            "1",
            hashlib.sha256(source.encode()).hexdigest(),
        )
        artifact.validate()
        return artifact
