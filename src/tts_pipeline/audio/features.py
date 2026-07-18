"""Torch-based, configuration-addressed audio feature processing."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from tts_pipeline.audio.io import read_audio, wav_bytes
from tts_pipeline.config import AudioConfig
from tts_pipeline.errors import ValidationError


class AudioProcessor:
    """Load audio and extract log magnitude mel features shaped ``[n_mels, frames]``."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.window = torch.hann_window(config.win_length)
        self.mel_filters = _mel_filterbank(config)

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.config.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def load(self, path: str | Path) -> torch.Tensor:
        try:
            waveform, sample_rate = read_audio(path)
        except (RuntimeError, OSError, ValueError) as exc:
            raise ValidationError(f"cannot load audio {path}: {exc}") from exc
        if not np.isfinite(waveform).all():
            raise ValidationError(f"audio contains NaN or infinity: {path}")
        mono = waveform.mean(axis=1)
        if sample_rate != self.config.sample_rate:
            gcd = math.gcd(sample_rate, self.config.sample_rate)
            mono = resample_poly(mono, self.config.sample_rate // gcd, sample_rate // gcd).astype(
                np.float32
            )
        return self.process_waveform(torch.from_numpy(np.asarray(mono, dtype=np.float32)))

    def process_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValidationError("waveform must be a non-empty mono tensor")
        waveform = torch.nan_to_num(waveform.float()).clamp(-1, 1)
        if self.config.trim_silence:
            waveform = self._trim(waveform)
        if self.config.peak_normalize:
            peak = waveform.abs().amax()
            if peak > 1e-6:
                waveform = waveform * (0.95 / peak)
        return waveform

    def _trim(self, waveform: torch.Tensor) -> torch.Tensor:
        amplitude = waveform.abs()
        if amplitude.amax() <= 1e-8:
            return waveform
        threshold = amplitude.amax() * (10 ** (-self.config.trim_db / 20))
        active = torch.where(amplitude >= threshold)[0]
        if active.numel() == 0:
            return waveform
        padding = self.config.hop_length
        start = max(0, int(active[0]) - padding)
        end = min(waveform.numel(), int(active[-1]) + padding + 1)
        return waveform[start:end]

    def mel(self, waveform: torch.Tensor) -> torch.Tensor:
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        filters = self.mel_filters.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = (
            torch.stft(
                waveform,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=window,
                center=self.config.center,
                pad_mode="reflect",
                return_complex=True,
            )
            .abs()
            .pow(self.config.power)
        )
        mel = torch.matmul(filters, spectrum)
        return torch.log(torch.clamp(mel, min=self.config.log_floor))

    def cache_features(self, audio_path: str | Path, cache_dir: str | Path) -> Path:
        source = Path(audio_path).resolve()
        key = hashlib.sha256(
            f"{source}:{source.stat().st_mtime_ns}:{self.fingerprint}".encode()
        ).hexdigest()
        target = Path(cache_dir) / f"{key}.npz"
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        mel = self.mel(self.load(source)).cpu().numpy()
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, mel=mel, feature_fingerprint=self.fingerprint)
        temporary.replace(target)
        return target

    def encode_wav(self, waveform: torch.Tensor, sample_rate: int | None = None) -> bytes:
        rate = sample_rate or self.config.sample_rate
        audio = waveform.detach().cpu().float().clamp(-1, 1).numpy()
        if rate != self.config.sample_rate:
            gcd = math.gcd(rate, self.config.sample_rate)
            audio = resample_poly(audio, rate // gcd, self.config.sample_rate // gcd)
        return wav_bytes(audio, rate)


def _mel_filterbank(config: AudioConfig) -> torch.Tensor:
    """Create a Slaney-style triangular mel bank shaped ``[M, N/2+1]``."""

    def hz_to_mel(value: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + value / 700.0)

    def mel_to_hz(value: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)

    low = hz_to_mel(torch.tensor(config.f_min))
    high = hz_to_mel(torch.tensor(config.f_max))
    points = mel_to_hz(torch.linspace(low, high, config.n_mels + 2))
    frequencies = torch.linspace(0, config.sample_rate / 2, config.n_fft // 2 + 1)
    lower, center, upper = points[:-2, None], points[1:-1, None], points[2:, None]
    bank = torch.minimum(
        (frequencies - lower) / torch.clamp(center - lower, min=1e-8),
        (upper - frequencies) / torch.clamp(upper - center, min=1e-8),
    ).clamp(min=0)
    # Area normalization prevents energy scale from increasing with wider high-frequency bands.
    result: torch.Tensor = bank * (
        2.0 / torch.clamp(upper[:, 0] - lower[:, 0], min=1e-8)
    ).unsqueeze(1)
    return result


def crossfade(chunks: list[torch.Tensor], overlap: int) -> torch.Tensor:
    if not chunks:
        return torch.zeros(0)
    result = chunks[0]
    for chunk in chunks[1:]:
        width = min(overlap, result.numel(), chunk.numel())
        if width:
            fade_out = torch.linspace(1, 0, width, device=result.device)
            fade_in = 1 - fade_out
            joined = result[-width:] * fade_out + chunk[:width] * fade_in
            result = torch.cat((result[:-width], joined, chunk[width:]))
        else:
            result = torch.cat((result, chunk))
    return result
