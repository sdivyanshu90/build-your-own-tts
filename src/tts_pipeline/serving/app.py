"""FastAPI application with bounded concurrency, auth, rate limits, metrics, and errors."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from tts_pipeline.config import Settings
from tts_pipeline.errors import CompatibilityError, TTSError
from tts_pipeline.inference import SynthesisResult, Synthesizer
from tts_pipeline.observability import configure_logging
from tts_pipeline.security import APIKeyAuthenticator, TokenBucketRateLimiter
from tts_pipeline.serving.schemas import ErrorResponse, NormalizeRequest, SynthesisRequest
from tts_pipeline.storage import LocalStorage
from tts_pipeline.text import TextNormalizer

LOGGER = logging.getLogger(__name__)
REQUESTS = Counter("tts_requests_total", "TTS requests", ("endpoint", "status"))
LATENCY = Histogram("tts_synthesis_seconds", "Synthesis processing latency")
ACTIVE = Gauge("tts_active_syntheses", "Active synthesis operations")
CHARACTERS = Counter("tts_characters_total", "Accepted synthesis characters")


class RequestContextMiddleware:
    """Pure ASGI middleware; it never buffers request bodies or disrupts streaming."""

    def __init__(self, app, max_request_bytes: int) -> None:  # type: ignore[no-untyped-def]
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        request_id = headers.get(b"x-request-id", b"").decode() or str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                too_large = int(raw_length) > self.max_request_bytes
            except ValueError:
                too_large = True
            if too_large:
                response = _error(
                    request_id, 413, "payload_too_large", "request body exceeds configured limit"
                )
                await response(scope, receive, send)
                return
        started = time.perf_counter()
        status = 500

        async def send_with_headers(message):  # type: ignore[no-untyped-def]
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        await self.app(scope, receive, send_with_headers)
        LOGGER.info(
            "request complete",
            extra={
                "request_id": request_id,
                "status_code": status,
                "latency_seconds": time.perf_counter() - started,
            },
        )


def create_app(settings: Settings, synthesizer: Synthesizer | None = None) -> FastAPI:
    configure_logging()
    auth = APIKeyAuthenticator(settings.serving.api_key_env)
    limiter = TokenBucketRateLimiter()
    semaphore = asyncio.Semaphore(settings.serving.max_concurrency)
    storage = LocalStorage(settings.serving.model_dir / "outputs")
    idempotency: dict[str, tuple[str, bytes, dict[str, object]]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if synthesizer is not None:
            app.state.synthesizer = synthesizer
        else:
            try:
                app.state.synthesizer = Synthesizer.from_directory(
                    settings, settings.serving.model_dir
                )
            except (OSError, ValueError, CompatibilityError) as exc:
                LOGGER.error("model load failed", exc_info=exc)
                app.state.synthesizer = None
                app.state.model_error = str(exc)
        yield
        app.state.synthesizer = None

    app = FastAPI(
        title="TTS Pipeline API",
        version="1.0.0",
        lifespan=lifespan,
        description="Consent-gated modular speech synthesis. Raw WAV is the production default.",
        responses={
            400: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    app.add_middleware(
        RequestContextMiddleware, max_request_bytes=settings.serving.max_request_bytes
    )

    async def authorize(request: Request, x_api_key: str | None = Header(None)) -> None:
        identity = request.client.host if request.client else "unknown"
        if not auth.authorize(x_api_key):
            raise HTTPException(401, "invalid API key")
        if not limiter.allow(identity):
            raise HTTPException(429, "rate limit exceeded")

    @app.exception_handler(TTSError)
    async def expected_error(request: Request, exc: TTSError) -> JSONResponse:
        return _error(request.state.request_id, 400, "invalid_request", str(exc))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return _error(request.state.request_id, exc.status_code, "http_error", str(exc.detail))

    def ready_synthesizer(request: Request) -> Synthesizer:
        instance = getattr(request.app.state, "synthesizer", None)
        if instance is None:
            raise HTTPException(503, getattr(request.app.state, "model_error", "model unavailable"))
        return cast(Synthesizer, instance)

    async def run_synthesis(request: Request, body: SynthesisRequest) -> SynthesisResult:
        instance = ready_synthesizer(request)
        try:
            await asyncio.wait_for(semaphore.acquire(), settings.serving.queue_timeout_seconds)
        except TimeoutError as exc:
            raise HTTPException(503, "synthesis queue is full") from exc
        ACTIVE.inc()
        try:
            work = asyncio.to_thread(
                instance.synthesize,
                body.text,
                body.language,
                body.speaker_id,
                body.rate,
                body.pitch,
                body.energy,
                body.output_sample_rate,
                body.seed,
                request.state.request_id,
            )
            return await asyncio.wait_for(work, settings.serving.request_timeout_seconds)
        except TimeoutError as exc:
            raise HTTPException(504, "synthesis timed out") from exc
        finally:
            ACTIVE.dec()
            semaphore.release()

    @app.post("/v1/synthesize", dependencies=[Depends(authorize)])
    async def synthesize(
        request: Request, body: SynthesisRequest, idempotency_key: str | None = Header(None)
    ) -> Response:
        request_hash = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        if idempotency_key and idempotency_key in idempotency:
            prior_hash, prior_wav, prior_meta = idempotency[idempotency_key]
            if prior_hash != request_hash:
                raise HTTPException(409, "idempotency key reused with different request")
            return _response(body, prior_wav, prior_meta, storage, settings)
        result = await run_synthesis(request, body)
        metadata = asdict(result.metadata)
        if idempotency_key:
            if len(idempotency) >= 128:
                idempotency.pop(next(iter(idempotency)))
            idempotency[idempotency_key] = (request_hash, result.wav, metadata)
        REQUESTS.labels("synthesize", "success").inc()
        LATENCY.observe(result.metadata.processing_latency)
        CHARACTERS.inc(len(body.text))
        return _response(body, result.wav, metadata, storage, settings)

    @app.post("/v1/synthesize/stream", dependencies=[Depends(authorize)])
    async def synthesize_stream(request: Request, body: SynthesisRequest) -> StreamingResponse:
        body.response_format = "wav"
        result = await run_synthesis(request, body)

        async def chunks() -> AsyncIterator[bytes]:
            for start in range(0, len(result.wav), 64 * 1024):
                if await request.is_disconnected():
                    break
                yield result.wav[start : start + 64 * 1024]

        return StreamingResponse(
            chunks(), media_type="audio/wav", headers={"X-Request-ID": result.metadata.request_id}
        )

    @app.post("/v1/normalize", dependencies=[Depends(authorize)])
    async def normalize(body: NormalizeRequest) -> dict[str, object]:
        result = TextNormalizer(settings.text.unicode_form, settings.text.max_characters).normalize(
            body.text, body.trace
        )
        return {
            "normalized": result.normalized,
            "stages": [{"name": k, "text": v} for k, v in result.stages],
        }

    @app.get("/v1/models")
    async def models(request: Request) -> list[dict[str, str]]:
        model = ready_synthesizer(request)
        return [
            {
                "id": "default",
                "version": model.bundle.version,
                "architecture": "FastSpeech2+HiFiGAN",
            }
        ]

    @app.get("/v1/speakers")
    async def speakers(request: Request) -> list[dict[str, str]]:
        model = ready_synthesizer(request)
        return [{"id": value, "status": "authorized"} for value in model.bundle.speakers]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/ready")
    async def ready(request: Request) -> Response:
        available = getattr(request.app.state, "synthesizer", None) is not None
        return JSONResponse(
            {"status": "ready" if available else "not_ready"}, status_code=200 if available else 503
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _response(
    body: SynthesisRequest,
    wav: bytes,
    metadata: dict[str, object],
    storage: LocalStorage,
    settings: Settings,
) -> Response:
    if body.response_format == "wav":
        safe = base64.urlsafe_b64encode(json.dumps(metadata).encode()).decode()
        return Response(wav, media_type="audio/wav", headers={"X-TTS-Metadata": safe})
    if body.response_format == "json":
        if not settings.serving.allow_base64:
            raise HTTPException(406, "base64 responses are disabled")
        return JSONResponse({"audio_base64": base64.b64encode(wav).decode(), "metadata": metadata})
    return JSONResponse({"object_key": storage.put(wav), "metadata": metadata})


def _error(request_id: str, status: int, code: str, message: str) -> JSONResponse:
    REQUESTS.labels("request", str(status)).inc()
    return JSONResponse(
        {"request_id": request_id, "code": code, "message": message}, status_code=status
    )


def app_from_environment() -> FastAPI:
    import os

    from tts_pipeline.config import load_settings

    return create_app(load_settings(os.getenv("TTS_CONFIG", "configs/development.yaml")))
