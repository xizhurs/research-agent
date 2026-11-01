"""On-disk cache of raw API responses, keyed by a hash of the request.

Raw JSON is cached (not parsed objects), so parsing or truncation changes never need a purge.
Credentials are never part of a key.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400


def cache_key(url: str, params: dict[str, str]) -> str:
    canonical = json.dumps({"url": url, "params": params}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path, ttl_days: float, *, now=time.time) -> None:
        self._dir = directory
        self._ttl_seconds = ttl_days * SECONDS_PER_DAY
        self._now = now

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        """Return the cached payload, or None on a miss, an expired entry or a corrupt file."""
        path = self._path(key)
        try:
            age = self._now() - path.stat().st_mtime
        except FileNotFoundError:
            return None
        if age > self._ttl_seconds:
            log.info("cache entry expired: %s", path.name)
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("ignoring unreadable cache entry %s: %s", path.name, exc)
            return None

    def put(self, key: str, payload: Any) -> None:
        """Write atomically so a crash never leaves a half-written entry.

        The cache is only an optimisation, so a write failure (disk full, read-only folder,
        OneDrive lock) is logged and the run continues.
        """
        path = self._path(key)
        tmp = path.with_suffix(".json.tmp")
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not write cache entry %s: %s", path.name, exc)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)  # best effort: leave no stray temp file
