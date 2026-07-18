from __future__ import annotations

import json

import numpy as np
from scipy.io import wavfile

from tts_pipeline.data import load_manifest, split_manifest, validate_manifest


def test_manifest_validation_and_reproducible_split(tmp_path) -> None:
    rows = []
    for index in range(20):
        path = tmp_path / f"{index}.wav"
        wavfile.write(path, 22050, np.zeros(11025, dtype=np.int16))
        rows.append(
            {"audio_path": path.name, "transcript": f"sample {index}", "speaker_id": str(index % 2)}
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    records = load_manifest(manifest)
    report = validate_manifest(records, 22050)
    assert report.valid == 20 and not report.errors
    first = split_manifest(records, 0.1, 0.1, seed=7)
    second = split_manifest(records, 0.1, 0.1, seed=7)
    assert [[x.identity for x in first[key]] for key in first] == [
        [x.identity for x in second[key]] for key in second
    ]
