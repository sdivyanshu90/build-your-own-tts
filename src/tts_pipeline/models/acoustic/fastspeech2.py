"""Compact, complete FastSpeech2-style acoustic model.

Shapes use batch-first notation: tokens ``[B, T]``, encoder states ``[B, T, H]``,
expanded states ``[B, F, H]``, and mel output ``[B, F, M]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from tts_pipeline.config import AudioConfig, ModelConfig


def sequence_mask(lengths: torch.Tensor, maximum: int | None = None) -> torch.Tensor:
    maximum = int(lengths.max()) if maximum is None and lengths.numel() else (maximum or 0)
    return torch.arange(maximum, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class PositionalEncoding(nn.Module):
    encoding: torch.Tensor

    def __init__(self, hidden: int, maximum: int) -> None:
        super().__init__()
        position = torch.arange(maximum).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
        encoding = torch.zeros(maximum, hidden)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] > self.encoding.shape[0]:
            raise ValueError("sequence exceeds configured max_positions")
        return value + self.encoding[: value.shape[1]]


class FFTBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_dim, config.attention_heads, config.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.conv_filter_size, 9, padding=4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.conv_filter_size, config.hidden_dim, 1),
            nn.Dropout(config.dropout),
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, value: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            value, value, value, key_padding_mask=padding_mask, need_weights=False
        )
        value = self.norm1(value + attended)
        convolved = self.conv(value.transpose(1, 2)).transpose(1, 2)
        return cast(
            torch.Tensor,
            self.norm2(value + convolved).masked_fill(padding_mask.unsqueeze(-1), 0),
        )


class VariancePredictor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        kernel = config.variance_kernel_size
        if kernel % 2 == 0:
            raise ValueError("variance_kernel_size must be odd")
        self.conv1 = nn.Conv1d(
            config.hidden_dim, config.variance_filter_size, kernel, padding=kernel // 2
        )
        self.conv2 = nn.Conv1d(
            config.variance_filter_size, config.variance_filter_size, kernel, padding=kernel // 2
        )
        self.norm1 = nn.LayerNorm(config.variance_filter_size)
        self.norm2 = nn.LayerNorm(config.variance_filter_size)
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.variance_filter_size, 1)

    def forward(self, value: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value.transpose(1, 2)).transpose(1, 2)
        value = self.dropout(self.norm1(torch.relu(value)))
        value = self.conv2(value.transpose(1, 2)).transpose(1, 2)
        value = self.dropout(self.norm2(torch.relu(value)))
        return cast(torch.Tensor, self.projection(value).squeeze(-1).masked_fill(padding_mask, 0))


class LengthRegulator(nn.Module):
    def forward(
        self, value: torch.Tensor, durations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        for states, row_durations in zip(value, durations, strict=True):
            expanded = torch.repeat_interleave(states, row_durations.clamp(min=0).long(), dim=0)
            if expanded.shape[0] == 0:
                expanded = states[:1]
            rows.append(expanded)
            lengths.append(expanded.shape[0])
        maximum = max(lengths)
        output = value.new_zeros((len(rows), maximum, value.shape[-1]))
        for index, row in enumerate(rows):
            output[index, : row.shape[0]] = row
        return output, torch.tensor(lengths, device=value.device, dtype=torch.long)


class PostNet(nn.Module):
    def __init__(self, n_mels: int, layers: int, dropout: float) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for index in range(layers):
            incoming = n_mels if index == 0 else 512
            outgoing = n_mels if index == layers - 1 else 512
            modules.extend([nn.Conv1d(incoming, outgoing, 5, padding=2), nn.BatchNorm1d(outgoing)])
            if index != layers - 1:
                modules.append(nn.Tanh())
            modules.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*modules)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(mel.transpose(1, 2)).transpose(1, 2))


@dataclass
class FastSpeech2Output:
    mel: torch.Tensor
    mel_postnet: torch.Tensor
    log_durations: torch.Tensor
    pitch: torch.Tensor
    energy: torch.Tensor
    durations: torch.Tensor
    token_mask: torch.Tensor
    mel_mask: torch.Tensor


class FastSpeech2(nn.Module):
    def __init__(self, vocabulary_size: int, model: ModelConfig, audio: AudioConfig) -> None:
        super().__init__()
        self.config = model
        self.embedding = nn.Embedding(vocabulary_size, model.hidden_dim, padding_idx=0)
        self.position = PositionalEncoding(model.hidden_dim, model.max_positions)
        self.encoder = nn.ModuleList(FFTBlock(model) for _ in range(model.encoder_layers))
        self.decoder = nn.ModuleList(FFTBlock(model) for _ in range(model.decoder_layers))
        self.duration_predictor = VariancePredictor(model)
        self.pitch_predictor = VariancePredictor(model)
        self.energy_predictor = VariancePredictor(model)
        self.pitch_embedding = nn.Conv1d(1, model.hidden_dim, 3, padding=1)
        self.energy_embedding = nn.Conv1d(1, model.hidden_dim, 3, padding=1)
        self.speaker_embedding = nn.Embedding(model.speaker_count, model.hidden_dim)
        self.language_embedding = nn.Embedding(model.language_count, model.hidden_dim)
        self.length_regulator = LengthRegulator()
        self.mel_projection = nn.Linear(model.hidden_dim, audio.n_mels)
        self.postnet = PostNet(audio.n_mels, model.postnet_layers, model.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        duration_targets: torch.Tensor | None = None,
        pitch_targets: torch.Tensor | None = None,
        energy_targets: torch.Tensor | None = None,
        rate: float = 1.0,
        pitch_scale: float = 1.0,
        energy_scale: float = 1.0,
    ) -> FastSpeech2Output:
        if tokens.ndim != 2 or token_lengths.ndim != 1:
            raise ValueError("tokens must be [B,T] and lengths [B]")
        batch, token_count = tokens.shape
        token_valid = sequence_mask(token_lengths, token_count)
        token_padding = ~token_valid
        value = self.position(self.embedding(tokens))
        if speaker_ids is None:
            speaker_ids = torch.zeros(batch, dtype=torch.long, device=tokens.device)
        if language_ids is None:
            language_ids = torch.zeros(batch, dtype=torch.long, device=tokens.device)
        value = value + self.speaker_embedding(speaker_ids).unsqueeze(1)
        value = value + self.language_embedding(language_ids).unsqueeze(1)
        for block in self.encoder:
            value = block(value, token_padding)
        log_durations = self.duration_predictor(value, token_padding)
        pitch_prediction = self.pitch_predictor(value, token_padding)
        energy_prediction = self.energy_predictor(value, token_padding)
        pitch = pitch_targets if pitch_targets is not None else pitch_prediction * pitch_scale
        energy = energy_targets if energy_targets is not None else energy_prediction * energy_scale
        value = value + self.pitch_embedding(pitch.unsqueeze(1)).transpose(1, 2)
        value = value + self.energy_embedding(energy.unsqueeze(1)).transpose(1, 2)
        if duration_targets is None:
            # Bound exponentiation and duration expansion so corrupted/untrained weights cannot
            # allocate an unbounded frame tensor during a public request.
            durations = torch.clamp(
                torch.round(torch.expm1(log_durations.clamp(max=math.log1p(50))) / rate),
                min=0,
                max=50,
            ).long()
            durations = durations.masked_fill(token_padding, 0)
            # Untrained models otherwise emit empty audio; each valid token gets at least one frame.
            durations = torch.where(
                token_valid & (durations == 0), torch.ones_like(durations), durations
            )
            for row_index in range(batch):
                total = int(durations[row_index].sum())
                if total > self.config.max_positions:
                    scale = self.config.max_positions / total
                    scaled = torch.floor(durations[row_index].float() * scale).long()
                    durations[row_index] = torch.where(
                        token_valid[row_index], scaled.clamp(min=1), scaled
                    )
        else:
            durations = duration_targets.long().masked_fill(token_padding, 0)
        value, mel_lengths = self.length_regulator(value, durations)
        mel_padding = ~sequence_mask(mel_lengths, value.shape[1])
        value = self.position(value)
        for block in self.decoder:
            value = block(value, mel_padding)
        mel = self.mel_projection(value).masked_fill(mel_padding.unsqueeze(-1), 0)
        mel_postnet = (mel + self.postnet(mel)).masked_fill(mel_padding.unsqueeze(-1), 0)
        return FastSpeech2Output(
            mel,
            mel_postnet,
            log_durations,
            pitch_prediction,
            energy_prediction,
            durations,
            token_valid,
            ~mel_padding,
        )
