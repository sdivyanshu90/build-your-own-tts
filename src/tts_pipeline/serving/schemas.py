"""Strict public HTTP schemas and validation bounds."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=5000, examples=["Hello from the TTS pipeline."])
    language: str = Field("en-US", pattern=r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
    speaker_id: str = Field("default", min_length=1, max_length=128)
    rate: float = Field(1.0, ge=0.5, le=2.0)
    pitch: float = Field(1.0, ge=0.5, le=2.0)
    energy: float = Field(1.0, ge=0.5, le=2.0)
    output_sample_rate: int | None = Field(None, ge=8000, le=48000)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    response_format: Literal["wav", "json", "storage"] = "wav"

    @field_validator("rate", "pitch", "energy")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value


class NormalizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=5000)
    trace: bool = False


class ErrorResponse(BaseModel):
    request_id: str
    code: str
    message: str
