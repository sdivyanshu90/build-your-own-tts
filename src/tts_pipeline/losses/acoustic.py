"""Mask-aware FastSpeech2 objectives."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tts_pipeline.models.acoustic import FastSpeech2Output


def _masked(loss: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(loss)
    if not expanded.any():
        raise ValueError(f"{name} mask is empty")
    selected = loss.masked_select(expanded)
    if not torch.isfinite(selected).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return selected.mean()


def acoustic_loss(
    output: FastSpeech2Output,
    mel_target: torch.Tensor,
    duration_target: torch.Tensor,
    pitch_target: torch.Tensor,
    energy_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if (duration_target < 0).any():
        raise ValueError("duration target cannot be negative")
    losses = {
        "mel_l1": _masked(
            F.l1_loss(output.mel, mel_target, reduction="none"), output.mel_mask, "mel_l1"
        ),
        "mel_mse": _masked(
            F.mse_loss(output.mel_postnet, mel_target, reduction="none"), output.mel_mask, "mel_mse"
        ),
        "duration": _masked(
            F.mse_loss(
                output.log_durations, torch.log1p(duration_target.float()), reduction="none"
            ),
            output.token_mask,
            "duration",
        ),
        "pitch": _masked(
            F.mse_loss(output.pitch, pitch_target, reduction="none"), output.token_mask, "pitch"
        ),
        "energy": _masked(
            F.mse_loss(output.energy, energy_target, reduction="none"), output.token_mask, "energy"
        ),
    }
    losses["total"] = torch.stack(list(losses.values())).sum()
    return losses
