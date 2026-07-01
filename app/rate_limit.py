from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic

from fastapi import Request


@dataclass(frozen=True)
class RateLimitRule:
    scope: str
    limit: int
    window_seconds: int


class InMemoryRateLimiter:
    """Simple per-process fixed-window limiter keyed by (scope, client)."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(
        self,
        *,
        scope: str,
        client_key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        if limit <= 0 or window_seconds <= 0:
            return (False, 0)

        now = monotonic()
        cutoff = now - window_seconds
        bucket_key = (scope, client_key)

        with self._lock:
            bucket = self._events[bucket_key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                retry_after = max(1, int(bucket[0] + window_seconds - now))
                return (True, retry_after)

            bucket.append(now)
            return (False, 0)


def resolve_client_ip(request: Request, *, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for", "").strip()
        if forwarded_for:
            first_ip = forwarded_for.split(",")[0].strip()
            if first_ip:
                return first_ip

        real_ip = request.headers.get("x-real-ip", "").strip()
        if real_ip:
            return real_ip

    if request.client and request.client.host:
        return request.client.host
    return "unknown"
