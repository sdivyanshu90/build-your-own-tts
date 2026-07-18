"""Disclosure-watermark integration contract.

The default intentionally leaves samples unchanged. Deployments can inject an independently
reviewed, audible or robust implementation appropriate to their policy and signal-quality targets.
"""

from __future__ import annotations

from typing import Protocol

import torch


class Watermarker(Protocol):
    def apply(self, waveform: torch.Tensor, sample_rate: int, request_id: str) -> torch.Tensor: ...


class NoopWatermarker:
    def apply(self, waveform: torch.Tensor, sample_rate: int, request_id: str) -> torch.Tensor:
        del sample_rate, request_id
        return waveform
