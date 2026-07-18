"""Fixed-length waveform/mel segment dataset for adversarial vocoder training."""

from __future__ import annotations

import hashlib

import torch
from torch.utils.data import Dataset

from tts_pipeline.audio import AudioProcessor
from tts_pipeline.data.manifest import ManifestRecord


class VocoderDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, records: list[ManifestRecord], processor: AudioProcessor, segment_frames: int = 32
    ) -> None:
        self.records = records
        self.processor = processor
        self.segment_frames = segment_frames
        self.segment_samples = segment_frames * processor.config.hop_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        waveform = self.processor.load(self.records[index].audio_path)
        if waveform.numel() < self.segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.numel())
            )
        else:
            maximum = waveform.numel() - self.segment_samples
            digest = hashlib.sha256(self.records[index].identity.encode()).digest()
            start = int.from_bytes(digest[:8], "little") % (maximum + 1)
            waveform = waveform[start : start + self.segment_samples]
        mel = self.processor.mel(waveform)
        return {"waveform": waveform.unsqueeze(0), "mel": mel}
