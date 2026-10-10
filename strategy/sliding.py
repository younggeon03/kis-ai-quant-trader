"""장중 슬라이딩 윈도우 거래대금 진입 — 순수 로직.

## 무엇을 판정하는가

> 매 분, **직전 W분 거래대금**이 기준선의 k배 이상이 되는 **그날 첫 분**에 발동한다.

장 시작 구간(`strategy/opening.py`)은 09:0N에 한 번만 본다. 이것은 09:00~15:19 내내
본다 — 급증이 **시작되는 순간**을 잡으려는 것이다(사용자 제안, 2026-10-10).

## 기준선이 둘이다

| 기준선 | 정의 |
|---|---|
| `day_avg` | 평소 하루 거래대금(20봉 중앙) x W/390 |
| `same_time` | 지난 20거래일 **같은 시각의 같은 W분** 거래대금 중앙(최소 10일) |

장 시작·마감 직전은 원래 거래가 몰린다. `day_avg`로 보면 그 시간에 발동이 몰리고, 장 시작
구간 측정에서는 그쪽이 현재 규칙보다 나빴다. `same_time`은 시각마다 따로 비교한다.

## 측정과 같은 코드

`scripts/sliding_study.py`가 이 함수들을 그대로 쓴다(CLAUDE.md §11).
주문도 포트폴리오도 모른다.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

SESSION_MINUTES = 390.0
LAST_ENTRY_MIN = 379          # 15:19 — 15:20부터 장마감 동시호가라 진입하지 않는다


def minute_turnover(minute_idx: np.ndarray, close: np.ndarray, volume: np.ndarray,
                    n_minutes: int = 380) -> np.ndarray:
    """09:00부터 분 단위 거래대금 배열(길이 `n_minutes`). 봉이 없는 분은 0."""
    out = np.zeros(int(n_minutes), dtype=float)
    mi = np.asarray(minute_idx, dtype=int)
    tv = np.asarray(close, dtype=float) * np.asarray(volume, dtype=float)
    ok = (mi >= 0) & (mi < n_minutes) & np.isfinite(tv)
    np.add.at(out, mi[ok], tv[ok])
    return out


def rolling_sum(tv: np.ndarray, w: int) -> np.ndarray:
    """`out[m]` = 분 m-w+1..m 의 합(분 m이 **끝난 뒤** 알 수 있는 값). 앞쪽 w-1분은 NaN."""
    tv = np.asarray(tv, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(tv)])
    out = np.full(len(tv), np.nan)
    if len(tv) >= w:
        out[w - 1:] = c[w:] - c[:-w]
    return out


def same_time_base(history: np.ndarray, days: int = 20, min_days: int = 10) -> np.ndarray:
    """같은 시각 기준선 — `history`는 (과거 일수 x 분) 행렬(오늘 제외, 오래된 것부터).

    시각마다 마지막 `days`일의 중앙값. 유효 표본이 `min_days`보다 적은 시각은 NaN(판정 안 함).
    """
    h = np.asarray(history, dtype=float)
    if h.ndim != 2 or h.shape[0] == 0:
        return np.full(h.shape[1] if h.ndim == 2 else 0, np.nan)
    h = h[-int(days):]
    cnt = np.sum(np.isfinite(h), axis=0)
    with warnings.catch_warnings():          # 전부 NaN인 열의 경고 — 아래에서 NaN 처리한다
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(h, axis=0)
    med[(cnt < int(min_days)) | ~(med > 0)] = np.nan
    return med


def first_trigger(window_tv: np.ndarray, base: np.ndarray, mult: float,
                  last_min: int = LAST_ENTRY_MIN) -> Optional[Tuple[int, float]]:
    """(발동 분, 배수) — 그날 처음 `window_tv >= mult x base`가 되는 분. 없으면 None.

    `base`는 스칼라(day_avg)이거나 분별 배열(same_time)이다.
    """
    w = np.asarray(window_tv, dtype=float)[: last_min + 1]
    b = np.broadcast_to(np.asarray(base, dtype=float), np.asarray(window_tv).shape)[: last_min + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = w / b
    hit = np.flatnonzero(np.isfinite(ratio) & (ratio >= mult))
    if not len(hit):
        return None
    m = int(hit[0])
    return m, float(ratio[m])


class SlidingWatcher:
    """종목별 오늘 분봉을 쌓고, 그날 첫 발동을 한 번만 돌려준다. 네트워크·주문을 모른다.

    `main.py`가 당일분봉(최근 30분)을 받을 때마다 `update()`에 넘긴다. **완성된 분만**
    판정한다 — 진행 중인 분의 봉은 거래량이 아직 덜 찼다(측정은 분이 끝난 값을 쓴다).
    기준선은 하루에 한 번 `set_base()`로 넣는다(same_time 이면 분별 배열, day_avg 면 스칼라).
    """

    def __init__(self, window: int, mult: float) -> None:
        self.window = int(window)
        self.mult = float(mult)
        self.day = ""
        self.tv: dict = {}            # 종목 → 분별 거래대금(길이 380)
        self.bars: dict = {}          # 종목 → {ts: (o, h, l, c, v)} — 기록용
        self.base: dict = {}          # 종목 → 기준선(스칼라 또는 분별 배열)
        self.fired: set = set()

    def reset_day(self, day: str) -> None:
        if day != self.day:
            self.day = day
            self.tv.clear()
            self.bars.clear()
            self.fired.clear()

    def set_base(self, symbol: str, base) -> None:
        self.base[symbol] = base

    def update(self, symbol: str, ts, opn, high, low, close, volume,
               now_minute: int) -> Optional[Tuple[int, float]]:
        """분봉 묶음을 반영하고, 오늘 처음 발동하면 (발동 분, 배수)를 돌려준다.

        `ts`는 `YYYYMMDDHHMM` 배열이다. 오늘(`reset_day`로 정한 날)이 아닌 봉은 버린다.
        `now_minute` 미만(완성된 분)만 판정에 쓴다.
        """
        self._store(symbol, ts, opn, high, low, close, volume)    # 발동 뒤에도 기록은 계속한다
        if symbol in self.fired or symbol not in self.base:
            return None
        arr = self.tv[symbol]
        done = min(int(now_minute), LAST_ENTRY_MIN + 1)
        if done < self.window:
            return None
        rs = rolling_sum(arr[:done], self.window)
        hit = first_trigger(rs, np.asarray(self.base[symbol])[:done]
                            if np.ndim(self.base[symbol]) else self.base[symbol], self.mult,
                            last_min=done - 1)
        if hit is None:
            return None
        self.fired.add(symbol)
        return hit

    def _store(self, symbol, ts, opn, high, low, close, volume) -> None:
        arr = self.tv.setdefault(symbol, np.zeros(LAST_ENTRY_MIN + 1))
        book = self.bars.setdefault(symbol, {})
        for t, o, h, lo, c, v in zip(ts, opn, high, low, close, volume):
            t = str(t)
            if t[:8] != self.day:
                continue
            mi = (int(t[8:10]) - 9) * 60 + int(t[10:12])
            book[t] = (float(o), float(h), float(lo), float(c), float(v))
            if 0 <= mi <= LAST_ENTRY_MIN:
                # 같은 분이 다시 오면 **덮어쓴다** — 지난 조회 때 진행 중이던 분이 완성된 값으로 온다
                arr[mi] = float(c) * float(v)
