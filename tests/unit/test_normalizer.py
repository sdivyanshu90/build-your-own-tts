from __future__ import annotations

import pytest

from tts_pipeline.errors import ValidationError
from tts_pipeline.text.normalizer import TextNormalizer


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Dr. Rao paid $12.50.", "doctor Rao paid twelve dollars and fifty cents."),
        ("It is 25% at 14:30.", "It is twenty five percent at two thirty p m."),
        ("Date: 2026-07-18.", "Date: July eighteenth, two thousand twenty six."),
        ("Version 3.14", "Version three point one four"),
        ("NASA", "N A S A"),
    ],
)
def test_normalization_table(source: str, expected: str) -> None:
    assert TextNormalizer().normalize(source).normalized == expected


def test_trace_and_control_rejection() -> None:
    result = TextNormalizer().normalize("１  2", trace=True)
    assert result.normalized == "one two"
    assert result.stages[0] == ("unicode", "1  2")
    with pytest.raises(ValidationError):
        TextNormalizer().normalize("bad\x00text")
