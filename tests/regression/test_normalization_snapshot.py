from __future__ import annotations

import pytest

from tts_pipeline.text.normalizer import TextNormalizer


@pytest.mark.regression
def test_normalization_snapshot() -> None:
    source = "Dr. Li paid €1.05 at 09:00 (25%)."
    assert (
        TextNormalizer().normalize(source).normalized
        == "doctor Li paid one euro and five cents at nine o'clock a m (twenty five percent)."
    )
