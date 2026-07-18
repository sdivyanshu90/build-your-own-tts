"""Operational command-line interface for the complete TTS lifecycle."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import torch
import typer
import uvicorn
from torch.utils.data import DataLoader, random_split

from tts_pipeline.audio import AudioProcessor
from tts_pipeline.audio.io import read_audio
from tts_pipeline.config import Settings, load_settings
from tts_pipeline.data import load_manifest, validate_manifest
from tts_pipeline.data.preprocess import preprocess_records
from tts_pipeline.data.preprocessed import PreprocessedDataset, collate_acoustic
from tts_pipeline.data.vocoder import VocoderDataset
from tts_pipeline.evaluation import evaluate_audio
from tts_pipeline.inference import Synthesizer
from tts_pipeline.inference.artifacts import export_bundle
from tts_pipeline.models.acoustic import FastSpeech2
from tts_pipeline.models.vocoder import HiFiGANGenerator
from tts_pipeline.observability import configure_logging
from tts_pipeline.serving import create_app
from tts_pipeline.text import Phonemizer, TextNormalizer, Vocabulary
from tts_pipeline.training.checkpoint import load_checkpoint
from tts_pipeline.training.engine import AcousticTrainer
from tts_pipeline.training.tracker import LocalTracker
from tts_pipeline.training.vocoder import VocoderTrainer

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Train, evaluate, and serve the modular TTS platform.",
)
Config = Annotated[Path, typer.Option("--config", "-c", help="YAML configuration path")]


def _settings(path: Path) -> Settings:
    return load_settings(path)


@app.command("validate-config")
def validate_config(config: Config = Path("configs/development.yaml")) -> None:
    """Resolve and validate every configuration invariant."""
    settings = _settings(config)
    typer.echo(
        json.dumps(
            {
                "valid": True,
                "fingerprint": settings.fingerprint(),
                "resolved": settings.model_dump(mode="json"),
            },
            indent=2,
        )
    )


@app.command("validate-dataset")
def validate_dataset(manifest: Path, config: Config = Path("configs/development.yaml")) -> None:
    """Check JSONL schema, duplicates, missing/corrupt audio, duration, and sample rate."""
    settings = _settings(config)
    report = validate_manifest(load_manifest(manifest), settings.audio.sample_rate)
    typer.echo(json.dumps(asdict(report), indent=2))
    if report.errors:
        raise typer.Exit(2)


@app.command("build-vocabulary")
def build_vocabulary(manifest: Path, output: Path = Path("artifacts/vocabulary.json")) -> None:
    """Build and checksum a vocabulary from normalized manifest transcripts."""
    normalizer, phonemizer = TextNormalizer(), Phonemizer()
    sequences = [
        phonemizer.phonemize(normalizer.normalize(row.transcript).normalized, row.language)
        for row in load_manifest(manifest)
    ]
    vocabulary = Vocabulary.build(sequences)
    vocabulary.save(output)
    typer.echo(f"wrote {len(vocabulary.symbols)} symbols to {output} ({vocabulary.checksum})")


@app.command()
def preprocess(
    manifest: Path,
    output: Path,
    vocabulary: Path,
    config: Config = Path("configs/development.yaml"),
    fixture_alignment: bool = typer.Option(False, help="Use uniform development-only alignments"),
) -> None:
    """Extract log-mels and versioned duration/variance targets."""
    index = preprocess_records(
        load_manifest(manifest),
        output,
        _settings(config),
        Vocabulary.load(vocabulary),
        fixture_alignment,
    )
    typer.echo(str(index))


@app.command()
def align(
    manifest: Path,
    output: Path,
    vocabulary: Path,
    config: Config = Path("configs/development.yaml"),
    fixture: bool = typer.Option(False, help="Explicitly permit fixture alignments"),
) -> None:
    """Generate duration artifacts through the configured alignment adapter."""
    preprocess_records(
        load_manifest(manifest), output, _settings(config), Vocabulary.load(vocabulary), fixture
    )


@app.command("train-acoustic")
def train_acoustic(
    index: Path,
    vocabulary: Path,
    run_dir: Path = Path("runs/acoustic"),
    config: Config = Path("configs/development.yaml"),
    resume: Path | None = None,
) -> None:
    """Train FastSpeech2 from a preprocessed index and optionally resume all state."""
    settings, vocab = _settings(config), Vocabulary.load(vocabulary)
    dataset = PreprocessedDataset(index)
    validation_size = max(1, round(len(dataset) * 0.1)) if len(dataset) > 1 else 0
    train_size = len(dataset) - validation_size
    train_data, validation_data = (
        random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(settings.runtime.seed),
        )
        if validation_size
        else (dataset, dataset)
    )
    train_loader = DataLoader(
        train_data,
        batch_size=settings.training.batch_size,
        shuffle=True,
        num_workers=settings.training.num_workers,
        collate_fn=collate_acoustic,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=settings.training.batch_size,
        num_workers=settings.training.num_workers,
        collate_fn=collate_acoustic,
    )
    model = FastSpeech2(len(vocab.symbols), settings.model, settings.audio)
    tracker = LocalTracker(run_dir)
    trainer = AcousticTrainer(model, settings, tracker, run_dir)
    report = trainer.startup_report(
        train_size, validation_size, settings.model.speaker_count, len(vocab.symbols)
    )
    typer.echo(json.dumps(report, indent=2, default=str))
    tracker.log_params(report)
    if resume:
        trainer.resume(resume)
    state = trainer.fit(train_loader, validation_loader)
    typer.echo(json.dumps(state.__dict__, indent=2))


@app.command("train-vocoder")
def train_vocoder(
    manifest: Path,
    run_dir: Path = Path("runs/vocoder"),
    resume: Path | None = None,
    config: Config = Path("configs/development.yaml"),
) -> None:
    """Train HiFi-GAN with separate generator/discriminator optimizers and full resume."""
    settings = _settings(config)
    dataset = VocoderDataset(load_manifest(manifest), AudioProcessor(settings.audio))
    loader = DataLoader(
        dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,
        num_workers=settings.training.num_workers,
    )
    trainer = VocoderTrainer(
        HiFiGANGenerator(settings.audio.n_mels, settings.vocoder), settings, run_dir
    )
    if resume:
        trainer.resume(resume)
    trainer.fit(loader)


@app.command("export-model")
def export_model(
    output: Path,
    vocabulary: Path | None = None,
    acoustic_checkpoint: Path | None = None,
    version: str = "development",
    config: Config = Path("configs/development.yaml"),
) -> None:
    """Export a bundle; absent checkpoints create a smoke-test-only random model."""
    settings = _settings(config)
    vocab = Vocabulary.load(vocabulary) if vocabulary else Vocabulary.default_graphemes()
    acoustic = FastSpeech2(len(vocab.symbols), settings.model, settings.audio)
    vocoder = HiFiGANGenerator(settings.audio.n_mels, settings.vocoder)
    if acoustic_checkpoint:
        load_checkpoint(acoustic_checkpoint, acoustic)
    export_bundle(output, acoustic, vocoder, vocab, settings, version)
    typer.echo(f"exported model bundle to {output}")


@app.command()
def synthesize(
    text: str,
    output: Path = Path("output.wav"),
    speaker: str = "default",
    language: str = "en-US",
    rate: float = 1.0,
    pitch: float = 1.0,
    energy: float = 1.0,
    seed: int | None = None,
    config: Config = Path("configs/development.yaml"),
) -> None:
    """Synthesize text to PCM-16 WAV from the configured model bundle."""
    settings = _settings(config)
    result = Synthesizer.from_directory(settings, settings.serving.model_dir).synthesize(
        text, language, speaker, rate, pitch, energy, seed=seed
    )
    output.write_bytes(result.wav)
    typer.echo(json.dumps(asdict(result.metadata), indent=2))


@app.command()
def evaluate(audio: Path, latency: float | None = None) -> None:
    """Report duration, peak, RMS, clipping, silence, and optional real-time factor."""
    waveform, sample_rate = read_audio(audio)
    typer.echo(
        json.dumps(
            asdict(evaluate_audio(torch.as_tensor(waveform), sample_rate, latency)), indent=2
        )
    )


@app.command("inspect-checkpoint")
def inspect_checkpoint(path: Path) -> None:
    """Safely inspect checkpoint metadata without constructing arbitrary Python globals."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state", {})
    typer.echo(
        json.dumps(
            {
                "format_version": payload.get("format_version"),
                "state": state,
                "model_tensors": len(payload.get("model", {})),
            },
            indent=2,
            default=str,
        )
    )


@app.command()
def benchmark(
    text: str = "The quick brown fox jumps over the lazy dog.",
    iterations: int = 3,
    config: Config = Path("configs/development.yaml"),
) -> None:
    """Measure cold/warm latency, real-time factor, throughput, and device memory."""
    settings = _settings(config)
    model = Synthesizer.from_directory(settings, settings.serving.model_dir)
    records = []
    for index in range(iterations):
        started = time.perf_counter()
        result = model.synthesize(text, seed=0)
        records.append(
            {
                "iteration": index,
                "latency": time.perf_counter() - started,
                "duration": result.metadata.audio_duration,
                "rtf": result.metadata.real_time_factor,
            }
        )
    typer.echo(
        json.dumps(
            {
                "device": settings.runtime.device,
                "cold": records[0],
                "warm": records[1:],
                "python": platform.python_version(),
                "cuda_memory": torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0,
            },
            indent=2,
        )
    )


@app.command()
def serve(config: Config = Path("configs/development.yaml")) -> None:
    """Run the authenticated, rate-limited FastAPI service."""
    configure_logging()
    settings = _settings(config)
    uvicorn.run(create_app(settings), host=settings.serving.host, port=settings.serving.port)


if __name__ == "__main__":
    app()
