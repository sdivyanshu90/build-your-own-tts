from __future__ import annotations

import pytest

from tts_pipeline.alignment import AlignmentArtifact, UniformAlignmentBackend
from tts_pipeline.errors import ValidationError


def test_uniform_alignment_and_roundtrip(tmp_path) -> None:
    artifact = UniformAlignmentBackend().align([2, 4, 3], 10, "fixture")
    path = tmp_path / "alignment.json"
    artifact.save(path)
    assert AlignmentArtifact.load(path) == artifact
    assert sum(artifact.durations) == 10


def test_bad_alignment_rejected() -> None:
    with pytest.raises(ValidationError):
        AlignmentArtifact((1, 1), 2, 3, "x", "1", "hash").validate()
