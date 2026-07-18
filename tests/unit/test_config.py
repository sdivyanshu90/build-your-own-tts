from __future__ import annotations

import pytest
from pydantic import ValidationError

from tts_pipeline.config import Settings


def test_configuration_resolves_and_vocoder_matches_hop(settings) -> None:
    assert settings.audio.hop_length == 256
    assert settings.fingerprint() == settings.fingerprint()
    changed = settings.model_copy(deep=True)
    changed.serving.model_dir = settings.serving.model_dir / "relocated"
    assert changed.fingerprint() != settings.fingerprint()
    assert changed.artifact_fingerprint() == settings.artifact_fingerprint()


def test_configuration_rejects_mismatched_upsampling(settings) -> None:
    raw = settings.model_dump()
    raw["vocoder"]["upsample_rates"] = [2, 2]
    raw["vocoder"]["upsample_kernel_sizes"] = [4, 4]
    with pytest.raises(ValidationError, match="hop_length"):
        Settings.model_validate(raw)
