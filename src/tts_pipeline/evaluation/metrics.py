"""Objective, offline-friendly audio and latency metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AudioMetrics:
    duration_seconds: float
    peak: float
    rms: float
    clipping_fraction: float
    silence_fraction: float
    real_time_factor: float | None


def evaluate_audio(
    waveform: torch.Tensor, sample_rate: int, latency_seconds: float | None = None
) -> AudioMetrics:
    audio = waveform.detach().float().flatten()
    if audio.numel() == 0 or not torch.isfinite(audio).all():
        raise ValueError("audio must be finite and non-empty")
    duration = audio.numel() / sample_rate
    rms = math.sqrt(float(torch.mean(audio.square())))
    return AudioMetrics(
        duration_seconds=duration,
        peak=float(audio.abs().max()),
        rms=rms,
        clipping_fraction=float((audio.abs() >= 0.999).float().mean()),
        silence_fraction=float((audio.abs() < 1e-4).float().mean()),
        real_time_factor=latency_seconds / duration
        if latency_seconds is not None and duration
        else None,
    )
