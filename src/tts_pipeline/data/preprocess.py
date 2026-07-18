"""Manifest-to-training-feature preprocessing with versioned alignments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tts_pipeline.alignment import UniformAlignmentBackend
from tts_pipeline.audio import AudioProcessor
from tts_pipeline.config import Settings
from tts_pipeline.data import ManifestRecord
from tts_pipeline.text import Phonemizer, TextNormalizer, Vocabulary


def preprocess_records(
    records: list[ManifestRecord],
    output_dir: str | Path,
    settings: Settings,
    vocabulary: Vocabulary,
    fixture_alignment: bool = False,
) -> Path:
    """Create compressed tensors; fixture alignment must be explicitly opted into."""
    if not fixture_alignment:
        raise ValueError(
            "no production aligner configured; supply precomputed alignments or explicitly "
            "use fixture_alignment"
        )
    output = Path(output_dir)
    feature_dir = output / "features"
    alignment_dir = output / "alignments"
    feature_dir.mkdir(parents=True, exist_ok=True)
    processor = AudioProcessor(settings.audio)
    normalizer = TextNormalizer(settings.text.unicode_form, settings.text.max_characters)
    phonemizer = Phonemizer()
    aligner = UniformAlignmentBackend()
    index_rows = []
    for row in records:
        normalized = row.normalized_transcript or normalizer.normalize(row.transcript).normalized
        tokens = vocabulary.encode(phonemizer.phonemize(normalized, row.language))
        waveform = processor.load(row.audio_path)
        mel = processor.mel(waveform)
        if mel.shape[1] < len(tokens):
            raise ValueError(f"audio has fewer mel frames than tokens: {row.audio_path}")
        source = f"{row.identity}:{processor.fingerprint}:{vocabulary.checksum}"
        alignment = aligner.align(tokens, mel.shape[1], source)
        key = hashlib.sha256(source.encode()).hexdigest()
        alignment.save(alignment_dir / f"{key}.json")
        frame_energy = torch.linalg.vector_norm(mel, dim=0)
        # The fixture backend uses a stable zero pitch target; production preprocessing should
        # replace this with voiced F0 from a configured estimator before duration averaging.
        frame_pitch = torch.zeros_like(frame_energy)
        pitch = _average_by_duration(frame_pitch, alignment.durations)
        energy = _average_by_duration(frame_energy, alignment.durations)
        relative = Path("features") / f"{key}.npz"
        np.savez_compressed(
            output / relative,
            tokens=np.asarray(tokens, dtype=np.int64),
            durations=np.asarray(alignment.durations, dtype=np.int64),
            pitch=pitch.numpy(),
            energy=energy.numpy(),
            mel=mel.numpy(),
            feature_fingerprint=processor.fingerprint,
            vocabulary_checksum=vocabulary.checksum,
        )
        index_rows.append(
            {
                "features": str(relative),
                "speaker_id": row.speaker_id,
                "language": row.language,
                "identity": row.identity,
            }
        )
    index = output / "index.jsonl"
    index.write_text("\n".join(json.dumps(item, sort_keys=True) for item in index_rows) + "\n")
    return index


def _average_by_duration(frames: torch.Tensor, durations: tuple[int, ...]) -> torch.Tensor:
    values = []
    start = 0
    for duration in durations:
        section = frames[start : start + duration]
        values.append(section.mean() if section.numel() else frames.new_tensor(0))
        start += duration
    return torch.stack(values)
