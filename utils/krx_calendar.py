"""KRX 장 운영 시간 / 휴장일 판정 유틸.

매매 루프가 "지금 무엇을 해야 하는지"(관망·진입·청산·정리)를 결정하는 기준이 된다.
휴장일 목록은 `config/holidays.txt`(한 줄에 YYYY-MM-DD)로 덮어쓸 수 있다.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Set
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

# 시간 경계 (KRX 정규장 기준)
PRE_AUCTION_START = dt.time(8, 30)
REGULAR_OPEN = dt.time(9, 0)
CLOSING_AUCTION_START = dt.time(15, 20)
REGULAR_CLOSE = dt.time(15, 30)
AFTER_HOURS_END = dt.time(18, 0)

# 기본 휴장일(2025~2026). 연초 KRX 공지로 반드시 재확인하고 필요 시
# config/holidays.txt 로 덮어쓸 것.
_DEFAULT_HOLIDAYS: Set[str] = {
    # 2025
    "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-03",
    "2025-05-01", "2025-05-05", "2025-05-06", "2025-06-03", "2025-06-06",
    "2025-08-15", "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08",
    "2025-10-09", "2025-12-25", "2025-12-31",
    # 2026
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-02",
    "2026-05-01", "2026-05-05", "2026-05-25", "2026-06-03", "2026-08-17",
    "2026-09-24", "2026-09-25", "2026-10-05", "2026-10-09", "2026-12-25",
    "2026-12-31",
}


class MarketSession(str, Enum):
    """현재 장 상태."""

    CLOSED = "CLOSED"                 # 휴장 / 장외
    PRE_AUCTION = "PRE_AUCTION"       # 08:30~09:00 장전 동시호가
    REGULAR = "REGULAR"               # 09:00~15:20 정규장
    CLOSING_AUCTION = "CLOSING_AUCTION"  # 15:20~15:30 장마감 동시호가
    AFTER_HOURS = "AFTER_HOURS"       # 15:30~18:00 시간외


class KRXCalendar:
    def __init__(self, extra_holidays: Optional[Iterable[str]] = None,
                 holiday_file: str | Path = "config/holidays.txt"):
        self._holidays: Set[dt.date] = set()
        for s in _DEFAULT_HOLIDAYS:
            self._holidays.add(dt.date.fromisoformat(s))
        path = Path(holiday_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    self._holidays.add(dt.date.fromisoformat(line))
                except ValueError:
                    continue
        for s in extra_holidays or []:
            self._holidays.add(dt.date.fromisoformat(s))

    # ---------------------------------------------------------------- 기본 판정
    @staticmethod
    def now() -> dt.datetime:
        return dt.datetime.now(tz=KST)

    def is_holiday(self, day: dt.date) -> bool:
        return day in self._holidays

    def is_trading_day(self, day: dt.date) -> bool:
        return day.weekday() < 5 and not self.is_holiday(day)

    def session(self, when: Optional[dt.datetime] = None) -> MarketSession:
        when = when or self.now()
        if when.tzinfo is None:
            when = when.replace(tzinfo=KST)
        else:
            when = when.astimezone(KST)
        if not self.is_trading_day(when.date()):
            return MarketSession.CLOSED
        t = when.time()
        if PRE_AUCTION_START <= t < REGULAR_OPEN:
            return MarketSession.PRE_AUCTION
        if REGULAR_OPEN <= t < CLOSING_AUCTION_START:
            return MarketSession.REGULAR
        if CLOSING_AUCTION_START <= t < REGULAR_CLOSE:
            return MarketSession.CLOSING_AUCTION
        if REGULAR_CLOSE <= t < AFTER_HOURS_END:
            return MarketSession.AFTER_HOURS
        return MarketSession.CLOSED

    def is_tradable(self, when: Optional[dt.datetime] = None) -> bool:
        """정규장 여부(지정가/시장가 주문 자유로운 구간)."""
        return self.session(when) is MarketSession.REGULAR

    # ---------------------------------------------------------------- 시간 계산
    def seconds_to_close(self, when: Optional[dt.datetime] = None) -> float:
        when = (when or self.now()).astimezone(KST)
        close_dt = dt.datetime.combine(when.date(), REGULAR_CLOSE, tzinfo=KST)
        return max(0.0, (close_dt - when).total_seconds())

    def minutes_to_close(self, when: Optional[dt.datetime] = None) -> float:
        return self.seconds_to_close(when) / 60.0

    def next_trading_day(self, day: Optional[dt.date] = None) -> dt.date:
        day = day or self.now().date()
        nxt = day + dt.timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(nxt):
                return nxt
            nxt += dt.timedelta(days=1)
        return nxt

    def next_open(self, when: Optional[dt.datetime] = None) -> dt.datetime:
        when = (when or self.now()).astimezone(KST)
        today_open = dt.datetime.combine(when.date(), REGULAR_OPEN, tzinfo=KST)
        if self.is_trading_day(when.date()) and when < today_open:
            return today_open
        return dt.datetime.combine(self.next_trading_day(when.date()), REGULAR_OPEN, tzinfo=KST)

    def seconds_until_next_open(self, when: Optional[dt.datetime] = None) -> float:
        when = (when or self.now()).astimezone(KST)
        return max(0.0, (self.next_open(when) - when).total_seconds())


class SimulatedClock:
    """Mock 모드용 가속 시계.

    계좌 발급 전(장 시간과 무관하게) 전체 매매 로직을 검증하기 위해,
    가상의 장중 시각을 `speed`배로 흘려보낸다.
    """

    def __init__(self, start: Optional[dt.datetime] = None, speed: float = 60.0):
        base = start or dt.datetime.combine(dt.date.today(), REGULAR_OPEN, tzinfo=KST)
        if base.tzinfo is None:
            base = base.replace(tzinfo=KST)
        self._virtual = base
        self.speed = max(1.0, float(speed))

    def now(self) -> dt.datetime:
        return self._virtual

    def advance(self, real_seconds: float) -> dt.datetime:
        """실제 경과 시간을 가상 시간으로 환산해 전진시킨다."""
        self._virtual = self._virtual + dt.timedelta(seconds=real_seconds * self.speed)
        return self._virtual

    def is_after_close(self) -> bool:
        return self._virtual.time() >= REGULAR_CLOSE
