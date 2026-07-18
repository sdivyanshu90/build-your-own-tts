"""Preprocessed feature dataset and variable-length collation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, index_path: str | Path) -> None:
        source = Path(index_path)
        self.rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        self.root = source.parent
        if not self.rows:
            raise ValueError("preprocessed index is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.root / self.rows[index]["features"], allow_pickle=False)
        return {
            "tokens": torch.from_numpy(data["tokens"]).long(),
            "durations": torch.from_numpy(data["durations"]).long(),
            "pitch": torch.from_numpy(data["pitch"]).float(),
            "energy": torch.from_numpy(data["energy"]).float(),
            "mel": torch.from_numpy(data["mel"]).float().transpose(0, 1),
        }


def collate_acoustic(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    token_max = max(row["tokens"].numel() for row in rows)
    frame_max = max(row["mel"].shape[0] for row in rows)
    mel_bins = rows[0]["mel"].shape[1]
    batch = {
        "tokens": torch.zeros(len(rows), token_max, dtype=torch.long),
        "token_lengths": torch.tensor([row["tokens"].numel() for row in rows]),
        "durations": torch.zeros(len(rows), token_max, dtype=torch.long),
        "pitch": torch.zeros(len(rows), token_max),
        "energy": torch.zeros(len(rows), token_max),
        "mel": torch.zeros(len(rows), frame_max, mel_bins),
    }
    for index, row in enumerate(rows):
        tokens, frames = row["tokens"].numel(), row["mel"].shape[0]
        for key in ("tokens", "durations", "pitch", "energy"):
            batch[key][index, :tokens] = row[key]
        batch["mel"][index, :frames] = row["mel"]
    return batch
