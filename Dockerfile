# syntax=docker/dockerfile:1
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so this layer is cached until uv.lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra ui --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra ui --no-dev

# Config and cache paths are relative to the working directory (/app).
RUN useradd --system --uid 10001 --home-dir /app app \
    && mkdir -p .cache && chown -R app:app .cache
USER app

ENV PATH="/app/.venv/bin:$PATH"

# Cloud Run sets PORT (8080); on Azure set the ingress target port to 8080 (the default here).
# API keys come from environment variables / platform secrets, never from the image.
EXPOSE 8080
CMD ["sh", "-c", "exec research-agent ui --host 0.0.0.0 --port ${PORT:-8080}"]
