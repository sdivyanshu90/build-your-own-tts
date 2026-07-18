"""End-to-end, long-text-aware synthesis orchestration."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from tts_pipeline.audio.features import AudioProcessor, crossfade
from tts_pipeline.config import Settings
from tts_pipeline.errors import ValidationError
from tts_pipeline.inference.artifacts import ModelBundle, load_bundle
from tts_pipeline.inference.watermark import NoopWatermarker, Watermarker
from tts_pipeline.observability.tracing import NoopTracer, Tracer
from tts_pipeline.text import Phonemizer, TextNormalizer
from tts_pipeline.text.normalizer import segment_sentences
from tts_pipeline.utils.runtime import seed_everything


@dataclass(frozen=True)
class SynthesisMetadata:
    request_id: str
    audio_duration: float
    input_characters: int
    token_count: int
    processing_latency: float
    real_time_factor: float
    model_version: str
    sample_rate: int
    speaker_id: str
    language: str
    normalized_text: str | None


@dataclass(frozen=True)
class SynthesisResult:
    wav: bytes
    waveform: torch.Tensor
    metadata: SynthesisMetadata


class Synthesizer:
    def __init__(
        self,
        settings: Settings,
        bundle: ModelBundle,
        phonemizer: Phonemizer | None = None,
        tracer: Tracer | None = None,
        watermarker: Watermarker | None = None,
    ) -> None:
        self.settings = settings
        self.bundle = bundle
        self.device = next(bundle.acoustic.parameters()).device
        self.normalizer = TextNormalizer(settings.text.unicode_form, settings.text.max_characters)
        self.phonemizer = phonemizer or Phonemizer()
        self.audio = AudioProcessor(settings.audio)
        self.tracer = tracer or NoopTracer()
        self.watermarker = watermarker or NoopWatermarker()

    @classmethod
    def from_directory(cls, settings: Settings, directory: str | Path) -> Synthesizer:
        device = torch.device(settings.runtime.device)
        return cls(settings, load_bundle(directory, settings, device))

    def synthesize(
        self,
        text: str,
        language: str = "en-US",
        speaker_id: str = "default",
        rate: float = 1.0,
        pitch: float = 1.0,
        energy: float = 1.0,
        output_sample_rate: int | None = None,
        seed: int | None = None,
        request_id: str | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> SynthesisResult:
        self._validate_controls(language, speaker_id, rate, pitch, energy, output_sample_rate)
        started = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        seed_everything(
            seed if seed is not None else self.settings.runtime.seed,
            self.settings.runtime.deterministic,
        )
        with self.tracer.span("tts.text", {"character_count": len(text), "language": language}):
            normalized = self.normalizer.normalize(text).normalized
            chunks = self._chunks(normalized, language)
        waveforms: list[torch.Tensor] = []
        token_count = 0
        with torch.inference_mode():
            for token_ids in chunks:
                if cancelled and cancelled():
                    raise TimeoutError("synthesis request was cancelled")
                tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                lengths = torch.tensor([len(token_ids)], device=self.device)
                speaker = torch.tensor([self.bundle.speakers.index(speaker_id)], device=self.device)
                language_id = torch.tensor(
                    [self.bundle.languages.index(language)], device=self.device
                )
                with self.tracer.span("tts.acoustic", {"token_count": len(token_ids)}):
                    output = self.bundle.acoustic(
                        tokens,
                        lengths,
                        speaker,
                        language_id,
                        rate=rate,
                        pitch_scale=pitch,
                        energy_scale=energy,
                    )
                with self.tracer.span("tts.vocoder", {"frame_count": output.mel.shape[1]}):
                    waveform = self.bundle.vocoder(output.mel_postnet.transpose(1, 2))[0, 0].cpu()
                waveforms.append(waveform)
                token_count += len(token_ids)
        waveform = crossfade(waveforms, int(self.settings.audio.sample_rate * 0.015))
        peak = waveform.abs().max()
        if peak > 1e-6:
            waveform = waveform * (0.95 / peak)
        waveform = self.watermarker.apply(waveform, self.settings.audio.sample_rate, request_id)
        target_rate = output_sample_rate or self.settings.audio.sample_rate
        wav = self.audio.encode_wav(waveform, target_rate)
        latency = time.perf_counter() - started
        duration = waveform.numel() / self.settings.audio.sample_rate
        metadata = SynthesisMetadata(
            request_id,
            duration,
            len(text),
            token_count,
            latency,
            latency / duration if duration else math.inf,
            self.bundle.version,
            target_rate,
            speaker_id,
            language,
            normalized if self.settings.serving.expose_normalized_text else None,
        )
        return SynthesisResult(wav, waveform, metadata)

    def _chunks(self, text: str, language: str) -> list[list[int]]:
        maximum = self.settings.text.max_tokens_per_chunk
        result: list[list[int]] = []
        for sentence in segment_sentences(text):
            symbols = self.phonemizer.phonemize(sentence, language)
            ids = self.bundle.vocabulary.encode(symbols)
            if len(ids) <= maximum:
                result.append(ids)
                continue
            # Fixed token windows are the final fallback when one sentence is unusually long.
            for start in range(0, len(ids), maximum - 2):
                window = ids[start : start + maximum - 2]
                result.append([2, *[value for value in window if value not in (2, 3)], 3])
        if not result:
            raise ValidationError("normalization produced no speakable tokens")
        return result

    def _validate_controls(
        self,
        language: str,
        speaker: str,
        rate: float,
        pitch: float,
        energy: float,
        sample_rate: int | None,
    ) -> None:
        if language not in self.bundle.languages:
            raise ValidationError(f"unsupported language: {language}")
        if speaker not in self.bundle.speakers:
            raise ValidationError(f"unknown or restricted speaker: {speaker}")
        for name, value in (("rate", rate), ("pitch", pitch), ("energy", energy)):
            if not math.isfinite(value) or not 0.5 <= value <= 2.0:
                raise ValidationError(f"{name} must be finite and between 0.5 and 2.0")
        if sample_rate is not None and not 8000 <= sample_rate <= 48000:
            raise ValidationError("output sample rate must be between 8000 and 48000")
