"""Thin I/O shell: fetch NOAA CPC's RONI file (cached), then hand it to the pure analysis."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

from research_agent.config import EnsoConfig
from research_agent.enso.analysis import compute_analysis, parse_roni
from research_agent.enso.errors import (
    EnsoApiError,
    EnsoError,
    EnsoNetworkError,
    InvalidEnsoQueryError,
)
from research_agent.enso.models import PHASES, EnsoAnalysis
from research_agent.retrieval.cache import ResponseCache, cache_key

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
PHASE_CHOICES = ("both", *PHASES)


def _optional_year(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidEnsoQueryError(f"{name} must be a positive integer year, got {value!r}")
    return value


class EnsoClient:
    def __init__(
        self,
        config: EnsoConfig,
        *,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._http = http_client or httpx.Client(timeout=config.timeout_seconds)
        self._owns_http = http_client is None
        self._sleep = sleep
        self._cache = ResponseCache(config.cache_dir, config.cache_ttl_days, now=now)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> EnsoClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def analyze(
        self,
        *,
        phase: str = "both",
        start_year: int | None = None,
        end_year: int | None = None,
        year: int | None = None,
    ) -> EnsoAnalysis:
        """El Nino / La Nina events that start in ``start_year..end_year`` (default: all).

        ``year`` adds what the record says about one calendar year. Invalid arguments raise
        InvalidEnsoQueryError before any request.
        """
        if phase not in PHASE_CHOICES:
            raise InvalidEnsoQueryError(f"phase must be one of {PHASE_CHOICES}, got {phase!r}")
        start_year = _optional_year(start_year, "start_year")
        end_year = _optional_year(end_year, "end_year")
        year = _optional_year(year, "year")
        if start_year is not None and end_year is not None and start_year > end_year:
            raise InvalidEnsoQueryError("start_year must not be after end_year")

        cfg = self._config
        key = cache_key(cfg.url, {})
        seasons, from_cache = None, False
        cached = self._cache.get(key)
        if isinstance(cached, str):
            try:
                seasons, from_cache = parse_roni(cached), True
            except EnsoApiError as exc:
                log.warning("ignoring cached RONI file with unexpected content: %s", exc)
        if seasons is None:
            text = self._get_text(cfg.url)
            seasons = parse_roni(text)
            self._cache.put(key, text)  # only a file that parsed is cached
        log.debug("RONI: %d seasons, cache %s", len(seasons), "hit" if from_cache else "miss")
        return compute_analysis(
            seasons,
            config=cfg,
            source_url=cfg.url,
            from_cache=from_cache,
            start_year=start_year,
            end_year=end_year,
            phase=phase,
            year=year,
        )

    def _get_text(self, url: str) -> str:
        cfg = self._config
        for attempt in range(cfg.max_retries + 1):
            error: type[EnsoError] = EnsoNetworkError
            try:
                response = self._http.get(url, timeout=cfg.timeout_seconds)
            except httpx.TransportError as exc:  # includes timeouts
                failure = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status == 200:
                    return response.text
                if status not in RETRYABLE_STATUS:
                    raise EnsoApiError(f"NOAA CPC returned HTTP {status} for the RONI file")
                failure, error = f"HTTP {status}", EnsoApiError
            if attempt == cfg.max_retries:
                raise error(f"RONI download failed after {attempt + 1} attempts: {failure}")
            wait = min(cfg.backoff_base_seconds * 2**attempt, cfg.max_backoff_seconds)
            log.warning(
                "RONI download failed (%s); retry %d/%d in %.1fs",
                failure,
                attempt + 1,
                cfg.max_retries,
                wait,
            )
            self._sleep(wait)
        raise EnsoApiError("RONI download failed")  # pragma: no cover
