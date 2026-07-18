from __future__ import annotations

import torch

from tts_pipeline.audio import AudioProcessor
from tts_pipeline.audio.features import crossfade


def test_mel_shape_determinism_and_finite(settings) -> None:
    processor = AudioProcessor(settings.audio)
    waveform = torch.sin(
        torch.arange(settings.audio.sample_rate) * (2 * torch.pi * 220 / settings.audio.sample_rate)
    )
    first = processor.mel(waveform)
    second = processor.mel(waveform)
    assert first.shape[0] == settings.audio.n_mels
    assert first.shape[1] > 1
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_wav_and_crossfade(settings) -> None:
    processor = AudioProcessor(settings.audio)
    joined = crossfade([torch.ones(10), torch.zeros(10)], 4)
    assert joined.numel() == 16
    wav = processor.encode_wav(joined)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
