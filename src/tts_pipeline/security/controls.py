"""Constant-time authentication hook and bounded in-memory rate-limiting interface."""

from __future__ import annotations

import hmac
import os
import threading
import time
from dataclasses import dataclass


class APIKeyAuthenticator:
    def __init__(self, environment_name: str) -> None:
        self.expected = os.getenv(environment_name)

    @property
    def enabled(self) -> bool:
        return bool(self.expected)

    def authorize(self, supplied: str | None) -> bool:
        return not self.expected or bool(supplied and hmac.compare_digest(supplied, self.expected))


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketRateLimiter:
    def __init__(self, rate_per_second: float = 1.0, capacity: int = 10) -> None:
        self.rate = rate_per_second
        self.capacity = capacity
        self.buckets: dict[str, _Bucket] = {}
        self.lock = threading.Lock()

    def allow(self, identity: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self.lock:
            bucket = self.buckets.setdefault(identity, _Bucket(float(self.capacity), now))
            bucket.tokens = min(self.capacity, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            if bucket.tokens < cost:
                return False
            bucket.tokens -= cost
            if len(self.buckets) > 10000:
                self.buckets = {k: v for k, v in self.buckets.items() if now - v.updated < 3600}
            return True
