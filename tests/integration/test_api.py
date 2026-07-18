from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import pytest
import torch

from tts_pipeline.inference import SynthesisMetadata, SynthesisResult
from tts_pipeline.serving import create_app


@dataclass
class FakeSynthesizer:
    bundle: object = field(
        default_factory=lambda: SimpleNamespace(version="test", speakers=("default",))
    )

    def synthesize(
        self, text, language, speaker, rate, pitch, energy, output_rate, seed, request_id
    ):  # noqa: ANN001, ANN201
        wav = b"RIFF\x00\x00\x00\x00WAVE"
        metadata = SynthesisMetadata(
            request_id,
            0.1,
            len(text),
            3,
            0.01,
            0.1,
            "test",
            output_rate or 22050,
            speaker,
            language,
            None,
        )
        return SynthesisResult(wav, torch.zeros(2205), metadata)


@pytest.mark.asyncio
async def test_health_normalize_and_synthesize(settings) -> None:
    api = create_app(settings, FakeSynthesizer())
    async with (
        api.router.lifespan_context(api),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as client,
    ):
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/ready")).status_code == 200
        normalized = await client.post("/v1/normalize", json={"text": "I paid $2."})
        assert normalized.json()["normalized"] == "I paid two dollars."
        response = await client.post("/v1/synthesize", json={"text": "hello"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")


@pytest.mark.asyncio
async def test_api_rejects_invalid_controls(settings) -> None:
    api = create_app(settings, FakeSynthesizer())
    async with (
        api.router.lifespan_context(api),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as client,
    ):
        response = await client.post("/v1/synthesize", json={"text": "hello", "rate": "NaN"})
        assert response.status_code == 422
