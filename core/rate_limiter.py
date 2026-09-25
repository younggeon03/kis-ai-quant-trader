"""비동기 토큰 버킷 기반 유량 제어기.

KIS Open API는 계정 단위로 초당 요청 수를 제한한다(실전 20건/초, 모의투자 2건/초).
한도를 넘기면 `EGW00201`(초당 거래건수 초과) 오류가 발생하고, 최악의 경우 청산 주문이
거부되어 리스크 관리가 무력화된다. 따라서 **모든 REST 호출은 예외 없이 이 모듈을 통과**한다.

특징
 - 단조시계 기반 연속 리필(시스템 시간 변경에 영향받지 않음)
 - 우선순위 큐: 주문/취소(P0) > 계좌조회(P1) > 시세조회(P2)
   → 시세 폴링이 폭주해도 손절 주문이 뒤로 밀리지 않는다.
 - 적응형 백오프: 유량 초과 응답을 받으면 자동으로 레이트를 일시 감축 후 점진 복구
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

# 우선순위 상수 (작을수록 먼저)
PRIORITY_ORDER = 0      # 신규주문/정정취소 - 절대 지연 불가
PRIORITY_ACCOUNT = 1    # 잔고/주문가능 조회
PRIORITY_QUOTE = 2      # 시세 조회
PRIORITY_BACKGROUND = 3  # 사전 데이터 적재 등


@dataclass
class RateLimiterStats:
    granted: int = 0
    rejected_by_server: int = 0
    total_wait: float = 0.0
    max_wait: float = 0.0
    penalties: int = 0
    recent: Deque[float] = field(default_factory=lambda: deque(maxlen=512))

    @property
    def avg_wait(self) -> float:
        return self.total_wait / self.granted if self.granted else 0.0

    def measured_rps(self, window: float = 1.0) -> float:
        """최근 `window`초 동안 실제로 승인된 요청 수."""
        if not self.recent:
            return 0.0
        now = time.monotonic()
        return float(sum(1 for t in self.recent if now - t <= window))

    def peak_rps(self, window: float = 1.0) -> float:
        """기록된 승인 시각 전체에서 관측된 최대 순간 RPS."""
        if not self.recent:
            return 0.0
        times = sorted(self.recent)
        peak = 0
        left = 0
        for right in range(len(times)):
            while times[right] - times[left] > window:
                left += 1
            peak = max(peak, right - left + 1)
        return float(peak)

    def to_dict(self) -> dict:
        return {
            "granted": self.granted,
            "server_rejects": self.rejected_by_server,
            "avg_wait_ms": round(self.avg_wait * 1000, 2),
            "max_wait_ms": round(self.max_wait * 1000, 2),
            "penalties": self.penalties,
            "recent_rps": self.measured_rps(),
            "peak_rps": self.peak_rps(),
        }


class AsyncTokenBucket:
    """단일 토큰 버킷. 우선순위가 필요 없는 경로(WebSocket 등)에서 사용."""

    def __init__(self, rate: float, capacity: Optional[float] = None):
        self.rate = max(0.01, float(rate))
        self.capacity = float(capacity if capacity is not None else max(1.0, rate))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now

    async def acquire(self, tokens: float = 1.0) -> float:
        """토큰을 확보할 때까지 대기. 대기한 시간(초)을 반환."""
        start = time.monotonic()
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return time.monotonic() - start
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            await asyncio.sleep(min(wait, 1.0))

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens


class PriorityRateLimiter:
    """우선순위 큐가 결합된 토큰 버킷.

    사용 예::

        limiter = PriorityRateLimiter(rate=2.0, burst=2)
        async with limiter.slot(PRIORITY_ORDER):
            await send_order()
    """

    def __init__(
        self,
        rate: float,
        burst: Optional[int] = None,
        penalty_factor: float = 0.5,
        penalty_seconds: float = 5.0,
        max_penalty_seconds: float = 60.0,
    ):
        self.base_rate = max(0.05, float(rate))
        self.capacity = float(burst if burst else max(1, int(rate)))
        self.penalty_factor = penalty_factor
        self.penalty_seconds = penalty_seconds
        self.max_penalty_seconds = max_penalty_seconds

        self._tokens = self.capacity
        self._last = time.monotonic()
        self._waiters: List[Tuple[int, int, asyncio.Future]] = []
        self._counter = itertools.count()
        self._cv = asyncio.Condition()
        self._dispatcher: Optional[asyncio.Task] = None
        self._penalty_until = 0.0
        self._current_penalty = penalty_seconds
        self.stats = RateLimiterStats()

    # ------------------------------------------------------------------ 내부
    @property
    def current_rate(self) -> float:
        """페널티가 걸려 있으면 감축된 레이트를 반환."""
        if time.monotonic() < self._penalty_until:
            return self.base_rate * self.penalty_factor
        return self.base_rate

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.current_rate)
            self._last = now

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._run(), name="rate-limiter")

    async def _run(self) -> None:
        while True:
            async with self._cv:
                while not self._waiters:
                    await self._cv.wait()
                self._refill()
                if self._tokens >= 1.0:
                    granted = False
                    while self._waiters and not granted:
                        _, _, fut = heapq.heappop(self._waiters)
                        if fut.cancelled() or fut.done():
                            continue  # 취소된 대기자는 토큰을 소비하지 않는다
                        self._tokens -= 1.0
                        fut.set_result(None)
                        granted = True
                    continue
                wait = (1.0 - self._tokens) / self.current_rate
            await asyncio.sleep(max(0.0005, min(wait, 0.5)))
            async with self._cv:
                self._cv.notify_all()

    # ------------------------------------------------------------------ 공개 API
    async def acquire(self, priority: int = PRIORITY_QUOTE) -> float:
        """우선순위에 따라 토큰을 확보한다. 대기 시간(초) 반환."""
        self._ensure_dispatcher()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        start = time.monotonic()
        async with self._cv:
            heapq.heappush(self._waiters, (priority, next(self._counter), fut))
            self._cv.notify_all()
        try:
            await fut
        except asyncio.CancelledError:
            async with self._cv:
                self._waiters = [w for w in self._waiters if w[2] is not fut]
                heapq.heapify(self._waiters)
            raise
        waited = time.monotonic() - start
        self.stats.granted += 1
        self.stats.total_wait += waited
        self.stats.max_wait = max(self.stats.max_wait, waited)
        self.stats.recent.append(time.monotonic())
        return waited

    @asynccontextmanager
    async def slot(self, priority: int = PRIORITY_QUOTE):
        """`async with limiter.slot(...)` 형태로 사용."""
        await self.acquire(priority)
        try:
            yield self
        finally:
            pass  # 토큰 버킷은 반납 개념이 없다(시간 경과로만 회복)

    def notify_rate_limit_error(self) -> float:
        """서버가 유량 초과를 응답했을 때 호출. 적용된 페널티 시간을 반환."""
        self.stats.rejected_by_server += 1
        self.stats.penalties += 1
        now = time.monotonic()
        if now < self._penalty_until:
            self._current_penalty = min(self.max_penalty_seconds, self._current_penalty * 2)
        else:
            self._current_penalty = self.penalty_seconds
        self._penalty_until = now + self._current_penalty
        self._tokens = 0.0
        self._last = now
        return self._current_penalty

    def notify_success(self) -> None:
        """정상 응답이 이어지면 페널티를 조기 해제한다."""
        if self._penalty_until and time.monotonic() >= self._penalty_until:
            self._penalty_until = 0.0
            self._current_penalty = self.penalty_seconds

    @property
    def penalized(self) -> bool:
        return time.monotonic() < self._penalty_until

    @property
    def queue_depth(self) -> int:
        return len(self._waiters)

    async def close(self) -> None:
        if self._dispatcher:
            self._dispatcher.cancel()
            try:
                await self._dispatcher
            except (asyncio.CancelledError, Exception):
                pass
            self._dispatcher = None
        async with self._cv:
            for _, _, fut in self._waiters:
                if not fut.done():
                    fut.cancel()
            self._waiters.clear()

    def snapshot(self) -> dict:
        return {
            "base_rate": self.base_rate,
            "current_rate": round(self.current_rate, 2),
            "tokens": round(self._tokens, 2),
            "queue_depth": self.queue_depth,
            "penalized": self.penalized,
            **self.stats.to_dict(),
        }
