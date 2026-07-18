from __future__ import annotations

import torch

from tts_pipeline.losses.acoustic import acoustic_loss
from tts_pipeline.models.acoustic import FastSpeech2
from tts_pipeline.models.vocoder import HiFiGANGenerator


def test_acoustic_training_and_inference_shapes(settings) -> None:
    model = FastSpeech2(30, settings.model, settings.audio).eval()
    tokens = torch.tensor([[2, 5, 6, 3], [2, 7, 3, 0]])
    lengths = torch.tensor([4, 3])
    durations = torch.tensor([[1, 2, 1, 1], [1, 2, 2, 0]])
    pitch = torch.zeros(2, 4)
    energy = torch.ones(2, 4)
    output = model(
        tokens, lengths, duration_targets=durations, pitch_targets=pitch, energy_targets=energy
    )
    assert output.mel.shape == (2, 5, settings.audio.n_mels)
    losses = acoustic_loss(output, torch.zeros_like(output.mel), durations, pitch, energy)
    assert torch.isfinite(losses["total"])
    predicted = model(tokens[:1], lengths[:1])
    assert predicted.mel.shape[1] >= 1


def test_vocoder_upsample_shape(settings) -> None:
    model = HiFiGANGenerator(settings.audio.n_mels, settings.vocoder).eval()
    mel = torch.randn(1, settings.audio.n_mels, 3)
    with torch.inference_mode():
        waveform = model(mel)
    assert waveform.shape == (1, 1, 3 * settings.audio.hop_length)
    assert waveform.abs().max() <= 1
