#!/usr/bin/env python3
"""Create deterministic, synthetic, consent-free audio and a JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/tiny"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    transcripts = [
        "Hello, this is synthetic fixture one.",
        "The price is $12.50.",
        "Meet me at 14:30 on 2026-07-18.",
        "CPU usage is 25 percent.",
    ]
    for index in range(args.count):
        duration = 1.0 + index * 0.1
        time = np.arange(round(args.sample_rate * duration), dtype=np.float32) / args.sample_rate
        envelope = np.minimum(1.0, time * 20) * np.minimum(1.0, (duration - time) * 20)
        waveform = (0.15 * np.sin(2 * np.pi * (180 + 30 * index) * time) * envelope).astype(
            np.float32
        )
        name = f"synthetic-{index:03d}.wav"
        wavfile.write(
            args.output / name, args.sample_rate, np.round(waveform * 32767).astype(np.int16)
        )
        rows.append(
            {
                "audio_path": name,
                "transcript": transcripts[index % len(transcripts)],
                "speaker_id": "default",
                "language": "en-US",
                "duration": duration,
                "metadata": {
                    "source": "generated-sine-fixture",
                    "consent": "not-applicable-synthetic",
                },
            }
        )
    (args.output / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(args.output / "manifest.jsonl")


if __name__ == "__main__":
    main()
