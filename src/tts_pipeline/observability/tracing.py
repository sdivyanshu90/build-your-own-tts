"""OpenTelemetry-compatible tracing boundary with an offline no-op default."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol, cast


class Tracer(Protocol):
    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> AbstractContextManager[Any]: ...


class NoopTracer:
    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> AbstractContextManager[None]:
        del name, attributes
        return nullcontext()


class OpenTelemetryTracer:
    """Adapter loaded only when the optional OpenTelemetry SDK is installed."""

    def __init__(self, instrumentation_name: str = "tts_pipeline") -> None:
        from opentelemetry import trace

        self._tracer = trace.get_tracer(instrumentation_name)

    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> AbstractContextManager[Any]:
        return cast(
            AbstractContextManager[Any],
            self._tracer.start_as_current_span(name, attributes=attributes or {}),
        )
