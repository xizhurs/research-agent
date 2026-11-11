"""OpenTelemetry tracing, exported to Langfuse (or any OTLP backend). Off unless configured.

Code everywhere else uses ``observe`` and ``set_content``; without a configured provider the
OpenTelemetry API makes them no-ops, so nothing is recorded or sent (tests, CI, default runs).

Switching it on (``configure_tracing``, called by ``ask``, ``eval`` and ``ui``):
- Langfuse: ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_HOST``, all three or
  none, from the environment or the .env file. Spans go to ``{host}/api/public/otel/v1/traces``
  with Basic auth.
- Any other OTLP/HTTP backend: ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or ``..._TRACES_ENDPOINT``) and
  ``..._HEADERS`` in the process environment; the OpenTelemetry SDK reads them itself.

Content (question, answer, model messages, tool arguments and results) is attached by default
when tracing is on; ``RESEARCH_AGENT_TRACE_CONTENT=false`` keeps only metadata. Exception text is
redacted and never recorded with a traceback.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from research_agent.credentials import validate_secret
from research_agent.logging_setup import REDACTOR

LANGFUSE_PUBLIC_VAR = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_VAR = "LANGFUSE_SECRET_KEY"
LANGFUSE_HOST_VAR = "LANGFUSE_HOST"
CONTENT_VAR = "RESEARCH_AGENT_TRACE_CONTENT"
OTEL_ENDPOINT_VARS = ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
OTEL_HEADER_VARS = ("OTEL_EXPORTER_OTLP_HEADERS", "OTEL_EXPORTER_OTLP_TRACES_HEADERS")
LANGFUSE_TRACES_PATH = "/api/public/otel/v1/traces"
SERVICE_NAME = "research-agent"

_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}

_tracer = trace.get_tracer("research_agent")
# Set once by configure_tracing (tests may patch it). Only read while a span is recording.
_capture_content = True


class TracingError(ValueError):
    """Tracing is misconfigured. The message never contains a key."""


@dataclass(frozen=True)
class ExportTarget:
    """Where spans go. ``endpoint`` None means the SDK reads the ``OTEL_*`` variables itself."""

    endpoint: str | None
    headers: dict[str, str] | None
    secrets: tuple[str, ...]  # values to redact from logs and span text


def _set(env: Mapping[str, str | None], name: str) -> str | None:
    value = env.get(name)
    return value.strip() or None if value else None


def capture_content_setting(env: Mapping[str, str | None]) -> bool:
    raw = _set(env, CONTENT_VAR)
    if raw is None or raw.lower() in _TRUE:
        return True
    if raw.lower() in _FALSE:
        return False
    raise TracingError(f"{CONTENT_VAR} must be true or false")


def export_target(
    env: Mapping[str, str | None], process_env: Mapping[str, str]
) -> ExportTarget | None:
    """Pure: the export target from the settings, or None when tracing is off.

    ``env`` is the .env file merged with the environment (for the Langfuse variables);
    ``process_env`` is the process environment only, because the SDK reads ``OTEL_*`` from there.
    """
    names = (LANGFUSE_PUBLIC_VAR, LANGFUSE_SECRET_VAR, LANGFUSE_HOST_VAR)
    present = [n for n in names if _set(env, n)]
    otel = any(_set(process_env, n) for n in OTEL_ENDPOINT_VARS)
    if present and otel:
        raise TracingError(
            "both LANGFUSE_* and OTEL_EXPORTER_OTLP_*ENDPOINT are set; choose one backend"
        )
    if otel:
        secrets = []
        for name in OTEL_HEADER_VARS:
            for pair in (_set(process_env, name) or "").split(","):
                secrets.append(pair.partition("=")[2].strip())
        return ExportTarget(None, None, tuple(s for s in secrets if s))
    if not present:
        return None
    missing = [n for n in names if n not in present]
    if missing:
        raise TracingError(f"Langfuse tracing needs all three of {', '.join(names)}; "
                           f"missing {', '.join(missing)}")
    public = validate_secret(LANGFUSE_PUBLIC_VAR, env[LANGFUSE_PUBLIC_VAR])
    secret = validate_secret(LANGFUSE_SECRET_VAR, env[LANGFUSE_SECRET_VAR])
    assert public is not None and secret is not None  # present, so not blank
    host = _set(env, LANGFUSE_HOST_VAR) or ""
    if not host.startswith(("https://", "http://")):
        raise TracingError(f"{LANGFUSE_HOST_VAR} must start with https:// (or http://)")
    auth = base64.b64encode(f"{public.reveal()}:{secret.reveal()}".encode()).decode()
    return ExportTarget(
        endpoint=host.rstrip("/") + LANGFUSE_TRACES_PATH,
        headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
        secrets=(secret.reveal(), auth),
    )


def configure_tracing(env: Mapping[str, str | None], process_env: Mapping[str, str]) -> bool:
    """Install the global tracer provider if tracing is configured. Returns whether it is on.

    Raises TracingError when misconfigured or when the ``tracing`` extra is not installed.
    """
    global _capture_content
    target = export_target(env, process_env)
    if target is None:
        return False
    capture = capture_content_setting(env)
    for value in target.secrets:
        REDACTOR.add(value)
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ModuleNotFoundError as exc:
        raise TracingError("tracing needs the extra: uv sync --extra tracing") from exc
    # The SDK flushes and shuts the provider down at interpreter exit.
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=target.endpoint, headers=target.headers))
    )
    trace.set_tracer_provider(provider)
    _capture_content = capture
    return True


# --- recording ----------------------------------------------------------------------------------


@contextmanager
def observe(
    name: str,
    kind: str,
    attributes: Mapping[str, Any] | None = None,
    tags: Sequence[str] = (),
) -> Iterator[Span]:
    """A span shown in Langfuse as ``kind`` (span, agent, generation, tool).

    An exception leaving the block marks the span as an error (redacted) and is re-raised. The
    SDK's own exception recording is off, because it stores unredacted text and tracebacks.
    """
    attrs: dict[str, Any] = {"langfuse.observation.type": kind, **(attributes or {})}
    if tags:
        attrs["langfuse.trace.tags"] = list(tags)
    with _tracer.start_as_current_span(
        name, attributes=attrs, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except Exception as exc:
            mark_error(span, exc)
            raise


def mark_error(span: Span, exc: Exception) -> None:
    span.set_attribute("error.type", type(exc).__name__)
    span.set_status(Status(StatusCode.ERROR, REDACTOR.redact(str(exc))))


def mark_failed(span: Span, message: str) -> None:
    """An error handled by the code (for example a tool result starting with ``Error:``)."""
    span.set_status(Status(StatusCode.ERROR, REDACTOR.redact(message)))


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def set_content(span: Span, *, input: Any = None, output: Any = None) -> None:
    """Attach input/output text, unless content capture is off or the span is not recorded."""
    if not (_capture_content and span.is_recording()):
        return
    if input is not None:
        span.set_attribute("langfuse.observation.input", REDACTOR.redact(_text(input)))
    if output is not None:
        span.set_attribute("langfuse.observation.output", REDACTOR.redact(_text(output)))


def set_model(model: str) -> None:
    """Called by an LLM adapter inside the model-call span, so Langfuse can price it."""
    trace.get_current_span().set_attribute("gen_ai.request.model", model)


def answer_json(answer: Any) -> dict[str, Any]:
    """The parts of an ``Answer`` worth reading in a trace (references by ID only)."""
    return {
        "claims": [
            {"kind": c.kind, "text": c.text, "sources": list(c.sources)} for c in answer.claims
        ],
        "references": [w.id for w in answer.references],
        "data_references": [d.id for d in answer.data_references],
        "limitations": list(answer.limitations),
        "rejected_citations": list(answer.rejected_citations),
        "structured": answer.structured,
    }
