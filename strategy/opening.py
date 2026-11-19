"""장 시작 구간 거래대금 진입 — 순수 로직.

## 무엇을 판정하는가

> 09:00부터 N분 동안의 거래대금 ≥ k x (평소 하루 거래대금 x N/390)

"평소 하루 거래대금"은 **전일까지 20봉 거래대금 중앙값**이다(`burst` 모드의 분모와
같다 — `main.py::_prime_intraday`가 계산해 둔 값을 쓴다). 하루 평균 N분 거래대금과
비교하는 것이므로, 장 시작 직후 거래가 원래 몰리는 효과까지 배수에 들어간다.

## 왜 누적 10배가 아닌가

누적 거래대금은 하루 종일 천천히 쌓여 감지 중앙이 10:59이고 그때 이미 시가 대비
+7%다. 문턱을 낮춰도 빨라지지 않았다. 짧은 구간은 **판정 시각이 고정**(09:0N)이다.

## 측정과 같은 코드

`scripts/research/opening_study.py`가 **이 함수들을 그대로** 쓴다. 측정과 실거래가 각자
구현하면 창이 한 칸만 어긋나도 다른 규칙을 실거래로 들고 간다(CLAUDE.md §11).

## 시각 표기 (실서버 확인 필요)

분봉의 `HHMM`은 **그 분의 시작**으로 읽는다(09:00 봉 = 09:00:00~09:00:59). 토스
분봉이 그렇게 측정됐다. KIS 당일분봉(`FHKST03010200`)의 표기가 같은지는 **실서버
첫날 로그로 확인**한다 — 끝 표기라면 창이 1분 밀린다.

주문도 포트폴리오도 모른다. 셀프테스트 22번이 네트워크 없이 검증한다.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

SESSION_MINUTES = 390.0


def minute_index(hhmm: Iterable[str]) -> np.ndarray:
    """`HHMM`(또는 `YYYYMMDDHHMM`의 뒤 4자리) → 09:00 기준 경과 분."""
    vals = [str(x)[-4:] for x in hhmm]
    return np.array([(int(v[:2]) - 9) * 60 + int(v[2:]) for v in vals], dtype=int)


def window_turnover(minute_idx: np.ndarray, close: np.ndarray, volume: np.ndarray,
                    n: int) -> float:
    """09:00부터 **n분**(경과 분 0..n-1)의 거래대금 합. 정의는 `종가 x 거래량`이다
    (일봉 `turnover`와 같다)."""
    mi = np.asarray(minute_idx, dtype=int)
    m = (mi >= 0) & (mi < int(n))
    tv = np.asarray(close, dtype=float)[m] * np.asarray(volume, dtype=float)[m]
    tv = tv[np.isfinite(tv)]
    return float(tv.sum()) if len(tv) else 0.0


def day_avg_base(med20: float, n: int) -> float:
    """평소 하루 거래대금(20봉 중앙)의 N분 몫."""
    if not (np.isfinite(med20) and med20 > 0):
        return 0.0
    return float(med20) * float(n) / SESSION_MINUTES


def opening_ratio(window_tv: float, med20: float, n: int) -> float:
    """구간 거래대금 ÷ 평소 N분 몫. 계산할 수 없으면 0(= 발동 안 함)."""
    base = day_avg_base(med20, n)
    if base <= 0 or not (np.isfinite(window_tv) and window_tv > 0):
        return 0.0
    return float(window_tv) / base


def rank_triggers(ratios: Dict[str, float], mult: float) -> List[Tuple[str, float]]:
    """배수 이상인 종목을 **배수가 큰 순**으로. 측정(`opening_study`)과 같은 순서다.

    같은 시각에 발동이 슬롯보다 많을 때 무엇을 먼저 사느냐는 근거가 없는 선택이다 —
    측정과 실거래가 **같은 선택**을 하는 것만 보장한다.
    """
    hits = [(s, float(r)) for s, r in ratios.items() if np.isfinite(r) and r >= mult]
    hits.sort(key=lambda x: (-x[1], x[0]))
    return hits


def same_window_base(history: Sequence[float], days: int = 20, min_days: int = 10) -> float:
    """**같은 시간대** 기준선 — 지난 `days`거래일(데이터가 있는 날) 같은 구간 거래대금의 중앙값.

    `history`는 **오늘을 뺀** 과거 값이고 오래된 것부터다. 측정의
    `rolling(20, min_periods=10).median().shift(1)`과 같다(셀프테스트 22번이 대조한다).
    장 시작 직후는 원래 거래가 몰리므로, 하루 평균(`day_avg_base`)과 비교하면 "원래 아침에
    거래가 많은 종목"이 1등이 된다 — 측정에서 그쪽은 현재 규칙보다 낫지 않았고, 이쪽은 3~15분
    넓은 범위에서 꼬리가 높았다(2026-10-10). 표본이 `min_days`보다 적으면 0(= 판정 안 함).
    """
    vals = np.asarray(list(history)[-int(days):], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < int(min_days):
        return 0.0
    med = float(np.median(vals))
    return med if med > 0 else 0.0


def same_window_ratio(window_tv: float, history: Sequence[float]) -> float:
    """오늘 구간 거래대금 ÷ 같은 시간대 기준선. 계산할 수 없으면 0."""
    base = same_window_base(history)
    if base <= 0 or not (np.isfinite(window_tv) and window_tv > 0):
        return 0.0
    return float(window_tv) / base
