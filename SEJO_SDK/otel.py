"""OpenTelemetry integration for SEJO SDK.

Exports agent turns as OTEL spans.  Compatible with any OTEL-compliant
backend (Jaeger, Zipkin, Grafana Tempo, Langfuse OTEL endpoint, etc.).

Usage::

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from SEJO_SDK.otel import OTelHooks

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    hooks = OTelHooks(provider=provider, service_name="ba-crew-assistant")
    agent = Agent(model=..., hooks=hooks)
    agent.run("What is LHR?")

No OTEL backend? Use the built-in console exporter for development::

    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    hooks = OTelHooks.with_console_exporter()
    agent = Agent(model=..., hooks=hooks)

Requires ``pip install sejo-sdk[otel]``
(``opentelemetry-sdk opentelemetry-api``).
"""

from __future__ import annotations

from typing import Any

from SEJO_SDK.hooks import HookRegistry


class OTelHooks(HookRegistry):
    """A HookRegistry that emits OpenTelemetry spans for every agent turn.

    Each turn produces one span named ``sejo.agent.turn`` with attributes:

    * ``sejo.turn``          — turn number
    * ``sejo.input``         — first 512 chars of user input
    * ``sejo.output``        — first 512 chars of model output
    * ``sejo.duration_ms``   — wall-clock time in milliseconds
    * ``sejo.error``         — error message if the turn raised
    * ``sejo.model``         — model class name (if detectable)
    """

    def __init__(
        self,
        provider: Any | None = None,
        service_name: str = "sejo-sdk",
        tracer_name: str = "sejo.agent",
    ) -> None:
        super().__init__()
        self._tracer = self._build_tracer(provider, service_name, tracer_name)
        self._spans: dict[int, Any] = {}

        self.on_turn_start(self._start_span)
        self.on_turn_end(self._end_span)

    @staticmethod
    def _build_tracer(provider: Any | None, service_name: str, tracer_name: str) -> Any:
        try:
            from opentelemetry import trace
        except ImportError as exc:
            from SEJO_SDK.errors import ProviderDependencyError
            raise ProviderDependencyError(
                "Install OpenTelemetry with `pip install sejo-sdk[otel]`."
            ) from exc

        if provider is None:
            provider = trace.get_tracer_provider()
        return provider.get_tracer(tracer_name)

    @classmethod
    def with_console_exporter(
        cls,
        service_name: str = "sejo-sdk",
    ) -> OTelHooks:
        """Return an OTelHooks that prints spans to stdout (dev/debug use)."""
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )
        except ImportError as exc:
            from SEJO_SDK.errors import ProviderDependencyError
            raise ProviderDependencyError(
                "Install OpenTelemetry with `pip install sejo-sdk[otel]`."
            ) from exc

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        return cls(provider=provider, service_name=service_name)

    def _start_span(self, ctx: dict[str, Any]) -> None:
        span = self._tracer.start_span("sejo.agent.turn")
        span.set_attribute("sejo.turn", ctx["turn"])
        span.set_attribute("sejo.input", str(ctx["input"])[:512])
        model = ctx.get("agent") and type(ctx["agent"].model).__name__
        if model:
            span.set_attribute("sejo.model", model)
        self._spans[ctx["turn"]] = span

    def _end_span(self, ctx: dict[str, Any]) -> None:
        span = self._spans.pop(ctx["turn"], None)
        if span is None:
            return
        try:
            from opentelemetry.trace import StatusCode
            span.set_attribute("sejo.output", str(ctx.get("output", ""))[:512])
            dur = round(ctx.get("duration_ms", 0.0), 1)
            span.set_attribute("sejo.duration_ms", dur)
            if ctx.get("error"):
                span.set_attribute("sejo.error", str(ctx["error"]))
                span.set_status(StatusCode.ERROR, str(ctx["error"]))
            else:
                span.set_status(StatusCode.OK)
        finally:
            span.end()
