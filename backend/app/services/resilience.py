import time
import asyncio
import logging
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger("oculus.resilience")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Three-state circuit breaker pattern for external cloud APIs.
    CLOSED -> (consecutive failures >= threshold) -> OPEN
    OPEN -> (after cooldown_seconds) -> HALF_OPEN
    HALF_OPEN -> (success) -> CLOSED, (failure) -> OPEN with backoff.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        max_cooldown_seconds: float = 300.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_cooldown_seconds = max_cooldown_seconds

        self.state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.last_failure_time: float = 0.0
        self.current_cooldown: float = cooldown_seconds

    def can_execute(self) -> bool:
        now = time.time()
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if now - self.last_failure_time >= self.current_cooldown:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"CircuitBreaker '{self.name}' entering HALF_OPEN probe state.")
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True
        return False

    def record_success(self) -> None:
        if self.state != CircuitState.CLOSED:
            logger.info(f"CircuitBreaker '{self.name}' recovered to CLOSED state.")
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.current_cooldown = self.cooldown_seconds

    def record_failure(self, error: Optional[Exception] = None) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Probe failed, back to OPEN with doubled cooldown
            self.state = CircuitState.OPEN
            self.current_cooldown = min(self.current_cooldown * 2, self.max_cooldown_seconds)
            logger.warning(
                f"CircuitBreaker '{self.name}' probe failed ({error}). Re-opening circuit for {self.current_cooldown}s."
            )
        elif self.state == CircuitState.CLOSED and self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"CircuitBreaker '{self.name}' tripped to OPEN after {self.consecutive_failures} failures ({error})."
            )


class QuotaGuard:
    """
    Token bucket rate limiter to keep API consumption safely inside free-tier boundaries.
    """

    def __init__(self, name: str, max_tokens: int, refill_period_sec: float):
        self.name = name
        self.max_tokens = max_tokens
        self.refill_period_sec = refill_period_sec

        self.tokens = float(max_tokens)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: int = 1) -> bool:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            if elapsed > 0:
                # Refill tokens proportionally
                refill = (elapsed / self.refill_period_sec) * self.max_tokens
                self.tokens = min(float(self.max_tokens), self.tokens + refill)
                self.last_refill = now

            if self.tokens >= cost:
                self.tokens -= cost
                return True
            logger.warning(f"QuotaGuard '{self.name}' exhausted: {round(self.tokens, 1)}/{self.max_tokens} available.")
            return False
