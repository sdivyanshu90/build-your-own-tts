"""Typed YAML configuration with recursive merging and environment overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tts_pipeline.errors import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(StrictModel):
    seed: int = 1337
    device: str = "cpu"
    deterministic: bool = True


class AudioConfig(StrictModel):
    sample_rate: int = Field(22050, ge=8000, le=192000)
    n_fft: int = Field(1024, ge=64)
    win_length: int = Field(1024, ge=64)
    hop_length: int = Field(256, ge=1)
    n_mels: int = Field(80, ge=16, le=512)
    f_min: float = Field(0.0, ge=0)
    f_max: float = Field(8000.0, gt=0)
    center: bool = True
    power: float = Field(1.0, gt=0)
    log_floor: float = Field(1e-5, gt=0)
    peak_normalize: bool = True
    trim_silence: bool = True
    trim_db: float = Field(45.0, gt=0)

    @model_validator(mode="after")
    def consistent(self) -> AudioConfig:
        if self.win_length > self.n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if self.f_max > self.sample_rate / 2:
            raise ValueError("f_max must not exceed Nyquist frequency")
        return self


class TextConfig(StrictModel):
    language: str = "en-US"
    unicode_form: Literal["NFC", "NFKC"] = "NFKC"
    max_characters: int = Field(5000, ge=1, le=100000)
    max_tokens_per_chunk: int = Field(256, ge=8, le=4096)
    unknown_policy: Literal["replace", "reject", "drop"] = "replace"


class ModelConfig(StrictModel):
    vocabulary_path: Path = Path("artifacts/vocabulary.json")
    hidden_dim: int = Field(192, ge=16)
    encoder_layers: int = Field(4, ge=1)
    decoder_layers: int = Field(4, ge=1)
    attention_heads: int = Field(2, ge=1)
    conv_filter_size: int = Field(768, ge=32)
    dropout: float = Field(0.1, ge=0, lt=1)
    max_positions: int = Field(2048, ge=32)
    speaker_count: int = Field(1, ge=1)
    language_count: int = Field(1, ge=1)
    variance_filter_size: int = Field(256, ge=16)
    variance_kernel_size: int = Field(3, ge=1)
    postnet_layers: int = Field(5, ge=1)

    @model_validator(mode="after")
    def attention_divides_hidden(self) -> ModelConfig:
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        return self


class VocoderConfig(StrictModel):
    channels: int = Field(256, ge=16)
    upsample_rates: list[int]
    upsample_kernel_sizes: list[int]
    resblock_kernel_sizes: list[int]

    @model_validator(mode="after")
    def compatible(self) -> VocoderConfig:
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("upsample rates and kernel sizes must have equal length")
        return self


class TrainingConfig(StrictModel):
    batch_size: int = Field(16, ge=1)
    epochs: int = Field(100, ge=1)
    learning_rate: float = Field(2e-4, gt=0)
    weight_decay: float = Field(1e-6, ge=0)
    gradient_accumulation: int = Field(1, ge=1)
    gradient_clip: float = Field(1.0, gt=0)
    mixed_precision: bool = False
    num_workers: int = Field(2, ge=0)
    patience: int = Field(10, ge=1)
    checkpoint_every: int = Field(1000, ge=1)


class ServingConfig(StrictModel):
    host: str = "0.0.0.0"  # noqa: S104
    port: int = Field(8000, ge=1, le=65535)
    max_concurrency: int = Field(2, ge=1)
    queue_timeout_seconds: float = Field(10.0, gt=0)
    request_timeout_seconds: float = Field(60.0, gt=0)
    max_request_bytes: int = Field(32768, ge=1024)
    expose_normalized_text: bool = False
    allow_base64: bool = True
    api_key_env: str = "TTS_API_KEY"
    model_dir: Path = Path("artifacts/demo")


class Settings(StrictModel):
    runtime: RuntimeConfig
    audio: AudioConfig
    text: TextConfig
    model: ModelConfig
    vocoder: VocoderConfig
    training: TrainingConfig
    serving: ServingConfig

    @model_validator(mode="after")
    def hop_matches_vocoder(self) -> Settings:
        factor = 1
        for rate in self.vocoder.upsample_rates:
            factor *= rate
        if factor != self.audio.hop_length:
            raise ValueError(
                f"vocoder upsample product {factor} must equal hop_length {self.audio.hop_length}"
            )
        return self

    def fingerprint(self) -> str:
        import hashlib

        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def artifact_fingerprint(self) -> str:
        """Hash only fields that determine inference tensor compatibility.

        Training schedules and serving paths may change without invalidating model weights.
        """
        import hashlib

        payload = json.dumps(
            {
                "audio": self.audio.model_dump(mode="json"),
                "model": self.model.model_dump(mode="json"),
                "vocoder": self.vocoder.model_dump(mode="json"),
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"configuration does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        parent_path = (config_path.parent / str(parent)).resolve()
        parent_raw = yaml.safe_load(parent_path.read_text(encoding="utf-8")) or {}
        raw = _merge(parent_raw, raw)
    overrides = {
        "runtime": {k: v for k, v in {"device": os.getenv("TTS_DEVICE")}.items() if v},
        "serving": {
            k: v
            for k, v in {
                "max_concurrency": os.getenv("TTS_MAX_CONCURRENCY"),
                "model_dir": os.getenv("TTS_MODEL_DIR"),
            }.items()
            if v
        },
    }
    try:
        return Settings.model_validate(_merge(raw, overrides))
    except Exception as exc:
        raise ConfigurationError(f"invalid configuration {config_path}: {exc}") from exc
