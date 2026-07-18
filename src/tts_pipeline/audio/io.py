"""SoundFile-backed audio I/O with a WAV-only SciPy fallback for minimal environments."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile  # type: ignore[import-untyped]

try:
    import soundfile as sf  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only by minimal installations
    sf = None


@dataclass(frozen=True)
class AudioInfo:
    samplerate: int
    channels: int
    duration: float


def read_audio(path: str | Path) -> tuple[NDArray[np.float32], int]:
    if sf is not None:
        waveform, rate = sf.read(path, dtype="float32", always_2d=True)
        return waveform, int(rate)
    rate, raw = wavfile.read(path)
    if np.issubdtype(raw.dtype, np.integer):
        scale = max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
        raw = raw.astype(np.float32) / scale
    else:
        raw = raw.astype(np.float32)
    if raw.ndim == 1:
        raw = raw[:, None]
    return raw, int(rate)


def audio_info(path: str | Path) -> AudioInfo:
    if sf is not None:
        value = sf.info(path)
        return AudioInfo(value.samplerate, value.channels, value.duration)
    waveform, rate = read_audio(path)
    return AudioInfo(rate, waveform.shape[1], waveform.shape[0] / rate)


def wav_bytes(waveform: NDArray[np.float32], sample_rate: int) -> bytes:
    audio = np.clip(waveform, -1, 1)
    pcm = np.round(audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, pcm)
    return buffer.getvalue()
