from __future__ import annotations

import torch

from tts_pipeline.inference.artifacts import ModelBundle
from tts_pipeline.inference.synthesizer import Synthesizer
from tts_pipeline.models.acoustic import FastSpeech2
from tts_pipeline.models.vocoder import HiFiGANGenerator
from tts_pipeline.text import Phonemizer, Vocabulary


def test_end_to_end_random_smoke(settings) -> None:
    vocabulary = Vocabulary.default_graphemes()
    acoustic = FastSpeech2(len(vocabulary.symbols), settings.model, settings.audio).eval()
    vocoder = HiFiGANGenerator(settings.audio.n_mels, settings.vocoder).eval()
    bundle = ModelBundle(acoustic, vocoder, vocabulary, "smoke", ("default",), ("en-US",))
    result = Synthesizer(settings, bundle, Phonemizer("grapheme")).synthesize("Hello.", seed=3)
    assert result.wav.startswith(b"RIFF")
    assert result.metadata.token_count > 0
    assert torch.isfinite(result.waveform).all()
