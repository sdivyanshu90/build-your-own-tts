"""HiFi-GAN-style generator and adversarial discriminators."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from tts_pipeline.config import VocoderConfig


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            weight_norm(
                nn.Conv1d(channels, channels, kernel, 1, dilation=d, padding=(kernel * d - d) // 2)
            )
            for d in dilations
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            value = value + conv(F.leaky_relu(value, 0.1))
        return value


class HiFiGANGenerator(nn.Module):
    """Map mel ``[B,M,F]`` to waveform ``[B,1,F*hop_length]``."""

    def __init__(self, n_mels: int, config: VocoderConfig) -> None:
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(n_mels, config.channels, 7, padding=3))
        channels = config.channels
        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for rate, kernel in zip(config.upsample_rates, config.upsample_kernel_sizes, strict=True):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        channels, channels // 2, kernel, rate, padding=(kernel - rate) // 2
                    )
                )
            )
            channels //= 2
            self.resblocks.append(
                nn.ModuleList(ResBlock(channels, k) for k in config.resblock_kernel_sizes)
            )
        self.post = weight_norm(nn.Conv1d(channels, 1, 7, padding=3))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        value = self.pre(mel)
        for upsample, blocks in zip(self.upsamples, self.resblocks, strict=True):
            value = F.leaky_relu(value, 0.1)
            value = upsample(value)
            block_list = cast(nn.ModuleList, blocks)
            value = sum(cast(ResBlock, block)(value) for block in block_list) / len(block_list)
        return torch.tanh(self.post(F.leaky_relu(value, 0.1)))


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        channels = [1, 32, 128, 512, 1024, 1024]
        self.layers = nn.ModuleList(
            weight_norm(
                nn.Conv2d(
                    channels[i], channels[i + 1], (5, 1), (3 if i < 4 else 1, 1), padding=(2, 0)
                )
            )
            for i in range(len(channels) - 1)
        )
        self.final = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        length = waveform.shape[-1]
        if length % self.period:
            waveform = F.pad(waveform, (0, self.period - length % self.period), mode="reflect")
        value = waveform.view(waveform.shape[0], 1, -1, self.period)
        features = []
        for layer in self.layers:
            value = F.leaky_relu(layer(value), 0.1)
            features.append(value)
        value = self.final(value)
        features.append(value)
        return value.flatten(1), features


class ScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = [1, 128, 128, 256, 512, 1024, 1024, 1024]
        groups = [1, 4, 16, 16, 16, 16, 1]
        kernels = [15, 41, 41, 41, 41, 41, 5]
        strides = [1, 2, 2, 4, 4, 1, 1]
        self.layers = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernels[i],
                    strides[i],
                    padding=kernels[i] // 2,
                    groups=groups[i],
                )
            )
            for i in range(len(groups))
        )
        self.final = weight_norm(nn.Conv1d(1024, 1, 3, padding=1))

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        value = waveform
        features = []
        for layer in self.layers:
            value = F.leaky_relu(layer(value), 0.1)
            features.append(value)
        value = self.final(value)
        features.append(value)
        return value.flatten(1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            PeriodDiscriminator(period) for period in (2, 3, 5, 7, 11)
        )

    def forward(self, waveform: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        return [model(waveform) for model in self.discriminators]


class MultiScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(ScaleDiscriminator() for _ in range(3))
        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, waveform: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        outputs = []
        value = waveform
        for index, model in enumerate(self.discriminators):
            if index:
                value = self.pool(value)
            outputs.append(model(value))
        return outputs
