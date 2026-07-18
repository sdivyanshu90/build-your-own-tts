from __future__ import annotations

import torch

from tts_pipeline.inference.artifacts import export_bundle, load_bundle
from tts_pipeline.models.acoustic import FastSpeech2
from tts_pipeline.models.vocoder import HiFiGANGenerator
from tts_pipeline.text import Vocabulary


def test_bundle_export_load(settings, tmp_path) -> None:
    vocabulary = Vocabulary.default_graphemes()
    acoustic = FastSpeech2(len(vocabulary.symbols), settings.model, settings.audio)
    vocoder = HiFiGANGenerator(settings.audio.n_mels, settings.vocoder)
    export_bundle(tmp_path, acoustic, vocoder, vocabulary, settings, "test")
    loaded = load_bundle(tmp_path, settings, torch.device("cpu"))
    assert loaded.version == "test"
    assert loaded.vocabulary.checksum == vocabulary.checksum
