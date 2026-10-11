"""Client-side pacing and quota cooldown for free-tier providers.

* ``RateLimiter`` keeps a 60-second sliding window of requests and tokens, so a
  provider with a small tokens-per-minute cap (Groq free tier: 8,000 TPM) is
  never pushed over it. The estimate made before a call is replaced by the
  real usage reported afterwards.
* ``QuotaCooldown`` remembers, per provider/model, that a long quota limit
  (per-day, or a retry-after longer than we are willing to wait) was hit, so
  later calls fail fast instead of hammering the API.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, rpm: float | None = None, tpm: int | None = None, sleep=time.sleep, clock=time.monotonic):
        self.rpm = rpm
        self.tpm = tpm
        self._events: deque[list] = deque()  # [timestamp, tokens]
        self._lock = threading.Lock()
        self._sleep = sleep
        self._clock = clock

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()

    def acquire(self, est_tokens: int) -> list:
        """Block until a request of ``est_tokens`` fits the window; returns a handle for ``settle``."""
        with self._lock:
            while True:
                now = self._clock()
                self._prune(now)
                wait = 0.0
                if self.rpm and len(self._events) >= self.rpm:
                    wait = max(wait, 60.0 - (now - self._events[0][0]))
                if self.tpm and self._events:
                    used = sum(e[1] for e in self._events)
                    if used + est_tokens > self.tpm:
                        # wait until enough old entries leave the window
                        freed, target = 0, used + est_tokens - self.tpm
                        for ts, tok in self._events:
                            freed += tok
                            if freed >= target:
                                wait = max(wait, 60.0 - (now - ts))
                                break
                        else:
                            wait = max(wait, 60.0 - (now - self._events[-1][0]))
                if wait <= 0:
                    entry = [now, est_tokens]
                    self._events.append(entry)
                    return entry
                self._sleep(min(wait + 0.05, 60.0))

    def settle(self, handle: list, actual_tokens: int | None) -> None:
        if actual_tokens is not None:
            with self._lock:
                handle[1] = actual_tokens


class QuotaCooldown:
    def __init__(self, clock=time.time):
        self._until: dict[str, float] = {}
        self._reason: dict[str, str] = {}
        self._clock = clock

    def block(self, key: str, seconds: float, reason: str) -> None:
        self._until[key] = self._clock() + seconds
        self._reason[key] = reason

    def check(self, key: str) -> str | None:
        """Reason string while the key is cooling down, else None."""
        until = self._until.get(key)
        if until is None or self._clock() >= until:
            return None
        remaining = until - self._clock()
        return f"{self._reason[key]} (cooldown, retry in {remaining / 60:.0f} min)"


COOLDOWN = QuotaCooldown()
