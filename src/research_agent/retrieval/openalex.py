"""Deterministic OpenAlex literature search. No LLM involved.

Pure functions (``build_params``, ``parse_work``, ``parse_response``) are separated from the
I/O in ``OpenAlexClient`` so they can be tested without a network.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any

import httpx

from research_agent.config import Limits, OpenAlexConfig
from research_agent.retrieval.abstract import reconstruct_abstract, truncate_abstract
from research_agent.retrieval.cache import ResponseCache, cache_key
from research_agent.retrieval.errors import (
    ApiError,
    AuthError,
    InvalidQueryError,
    NetworkError,
    RateLimitError,
)
from research_agent.retrieval.models import SearchResult, Work

log = logging.getLogger(__name__)

API_KEY_ENV_VAR = "OPENALEX_API_KEY"
MAX_QUERY_CHARS = 500
MIN_YEAR, MAX_YEAR = 1500, 2100
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
AUTH_STATUS = frozenset({401, 403})
_SHORT_ID = re.compile(r"^W\d+$")

# Only the fields we use, to keep responses small.
SELECT_FIELDS = (
    "id,doi,title,publication_year,publication_date,cited_by_count,"
    "authorships,primary_location,type,language,abstract_inverted_index"
)


def api_key_from_env() -> str | None:
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    return key or None


def build_params(query: str, year_from: int | None, per_page: int) -> dict[str, str]:
    """Build the query parameters for a works search. Pure and order-independent."""
    params = {
        # Matches only title and abstract, i.e. text we can actually show as evidence.
        "search.title_and_abstract": query,
        "per_page": str(per_page),
        "select": SELECT_FIELDS,
    }
    if year_from is not None:
        params["filter"] = f"from_publication_date:{year_from}-01-01"
    return params


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_work(record: Any, abstract_max_chars: int) -> Work | None:
    """Convert one API record to a Work, or None if it has no usable OpenAlex ID."""
    if not isinstance(record, dict):
        return None
    raw_id = record.get("id")
    if not isinstance(raw_id, str):
        return None
    short_id = raw_id.rstrip("/").rsplit("/", 1)[-1]
    if not _SHORT_ID.match(short_id):
        return None

    authors: list[str] = []
    authorships = record.get("authorships")
    for authorship in authorships if isinstance(authorships, list) else []:
        author = authorship.get("author") if isinstance(authorship, dict) else None
        name = _str_or_none(author.get("display_name")) if isinstance(author, dict) else None
        if name:
            authors.append(name)

    location = record.get("primary_location")
    source = location.get("source") if isinstance(location, dict) else None
    venue = _str_or_none(source.get("display_name")) if isinstance(source, dict) else None

    abstract = reconstruct_abstract(record.get("abstract_inverted_index"))
    truncated = False
    if abstract is not None:
        abstract, truncated = truncate_abstract(abstract, abstract_max_chars)

    return Work(
        id=short_id,
        url=f"https://openalex.org/{short_id}",
        title=_str_or_none(record.get("title")),
        year=_int_or_none(record.get("publication_year")),
        date=_str_or_none(record.get("publication_date")),
        authors=tuple(authors),
        venue=venue,
        doi=_str_or_none(record.get("doi")),
        type=_str_or_none(record.get("type")),
        language=_str_or_none(record.get("language")),
        cited_by_count=_int_or_none(record.get("cited_by_count")),
        abstract=abstract,
        abstract_truncated=truncated,
    )


def parse_response(
    payload: Any, abstract_max_chars: int
) -> tuple[tuple[Work, ...], int, int]:
    """Parse a works response into ``(works, total_count, skipped_records)``.

    Raises ApiError if the envelope is not the expected shape. Individual unparseable records
    are counted and logged, not dropped silently.
    """
    if not isinstance(payload, dict):
        raise ApiError("unexpected response: top-level JSON is not an object")
    results = payload.get("results")
    meta = payload.get("meta")
    if not isinstance(results, list) or not isinstance(meta, dict):
        raise ApiError("unexpected response: missing 'results' list or 'meta' object")
    total = _int_or_none(meta.get("count"))
    if total is None:
        raise ApiError("unexpected response: meta.count is missing or not an integer")

    works: list[Work] = []
    skipped = 0
    for record in results:
        work = parse_work(record, abstract_max_chars)
        if work is None:
            skipped += 1
            log.warning("skipped unparseable OpenAlex record: %.120r", record)
        else:
            works.append(work)
    return tuple(works), total, skipped


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        for key in ("error", "message"):
            if isinstance(body.get(key), str):
                return body[key][:200]
    return str(body)[:200]


class OpenAlexClient:
    """Search OpenAlex works with caching, retries and typed errors.

    ``search`` is the single entry point a later agent tool layer should call. It raises
    ``RetrievalError`` subclasses and never returns partial or guessed data.
    """

    def __init__(
        self,
        config: OpenAlexConfig,
        limits: Limits,
        *,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._limits = limits
        self._http = http_client or httpx.Client(timeout=config.timeout_seconds)
        self._owns_http = http_client is None
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._sleep = sleep
        self._cache = ResponseCache(config.cache_dir, config.cache_ttl_days, now=now)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> OpenAlexClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def search(
        self, query: str, *, year_from: int | None = None, max_results: int | None = None
    ) -> SearchResult:
        query = query.strip() if isinstance(query, str) else ""
        if not query:
            raise InvalidQueryError("query must be a non-empty string")
        if len(query) > MAX_QUERY_CHARS:
            raise InvalidQueryError(f"query is longer than {MAX_QUERY_CHARS} characters")
        if year_from is not None and (
            isinstance(year_from, bool) or not MIN_YEAR <= year_from <= MAX_YEAR
        ):
            raise InvalidQueryError(f"year_from must be between {MIN_YEAR} and {MAX_YEAR}")
        if max_results is not None and (isinstance(max_results, bool) or max_results < 1):
            raise InvalidQueryError("max_results must be a positive integer")

        # The configured limit is a hard cap; callers can only ask for fewer.
        cap = self._limits.results_per_search
        per_page = cap if max_results is None else min(max_results, cap)

        url = f"{self._config.base_url.rstrip('/')}/works"
        params = build_params(query, year_from, per_page)
        key = cache_key(url, params)
        abstract_chars = self._limits.abstract_max_chars

        cached = self._cache.get(key)
        if cached is not None:
            try:
                works, total, skipped = parse_response(cached, abstract_chars)
            except ApiError as exc:
                log.warning("ignoring cached response with unexpected shape: %s", exc)
            else:
                return SearchResult(query, year_from, works, total, skipped, from_cache=True)

        payload = self._get_json(url, params)
        works, total, skipped = parse_response(payload, abstract_chars)
        self._cache.put(key, payload)  # only validated responses are cached
        return SearchResult(query, year_from, works, total, skipped, from_cache=False)

    def _get_json(self, url: str, params: dict[str, str]) -> Any:
        cfg = self._config
        last_failure = "no attempt made"
        for attempt in range(cfg.max_retries + 1):
            try:
                response = self._http.get(
                    url, params=params, headers=self._headers, timeout=cfg.timeout_seconds
                )
            except httpx.TransportError as exc:  # includes timeouts
                last_failure = f"{type(exc).__name__}: {exc}"
                wait = self._backoff(attempt)
                if attempt == cfg.max_retries:
                    raise NetworkError(
                        f"OpenAlex request failed after {attempt + 1} attempts: {last_failure}"
                    ) from exc
            else:
                status = response.status_code
                if status == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ApiError("OpenAlex returned a non-JSON 200 response") from exc
                if status in AUTH_STATUS:
                    raise AuthError(f"OpenAlex rejected the request ({status}): "
                                    f"{_error_detail(response)}")
                if status not in RETRYABLE_STATUS:
                    raise ApiError(f"OpenAlex returned {status}: {_error_detail(response)}")

                last_failure = f"HTTP {status}: {_error_detail(response)}"
                retry_after = _retry_after_seconds(response)
                if retry_after is not None and retry_after > cfg.max_backoff_seconds:
                    # E.g. daily budget exhausted: waiting is pointless, fail fast.
                    raise RateLimitError(
                        f"OpenAlex asked to wait {retry_after:.0f}s (> max "
                        f"{cfg.max_backoff_seconds:.0f}s); giving up. {last_failure}"
                    )
                wait = retry_after if retry_after is not None else self._backoff(attempt)
                if attempt == cfg.max_retries:
                    error_cls = RateLimitError if status == 429 else ApiError
                    raise error_cls(
                        f"OpenAlex request failed after {attempt + 1} attempts: {last_failure}"
                    )
            log.warning(
                "OpenAlex request failed (%s); retry %d/%d in %.1fs",
                last_failure,
                attempt + 1,
                cfg.max_retries,
                wait,
            )
            self._sleep(wait)
        raise ApiError(f"OpenAlex request failed: {last_failure}")  # pragma: no cover

    def _backoff(self, attempt: int) -> float:
        cfg = self._config
        return min(cfg.backoff_base_seconds * 2**attempt, cfg.max_backoff_seconds)
