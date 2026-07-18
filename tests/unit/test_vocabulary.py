from __future__ import annotations

import json

import pytest

from tts_pipeline.errors import CompatibilityError
from tts_pipeline.text import Vocabulary


def test_roundtrip_and_unknown(tmp_path) -> None:
    vocab = Vocabulary.build([["h", "i"], ["|"]])
    path = tmp_path / "vocabulary.json"
    vocab.save(path)
    loaded = Vocabulary.load(path)
    ids = loaded.encode(["h", "missing"])
    assert loaded.decode(ids) == ["<bos>", "h", "<unk>", "<eos>"]


def test_checksum_detects_tampering(tmp_path) -> None:
    path = tmp_path / "vocabulary.json"
    Vocabulary.default_graphemes().save(path)
    raw = json.loads(path.read_text())
    raw["symbols"].append("tampered")
    path.write_text(json.dumps(raw))
    with pytest.raises(CompatibilityError):
        Vocabulary.load(path)
