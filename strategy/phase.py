"""대회 국면 — 날짜에 따라 전략의 공격성을 바꾼다.

## 왜 국면을 나누는가

제16회 뱅키스 대회는 **2026-10-12 ~ 11-20, 거래일 30일**이고 심사는 절대수익률
top10이다. 그리고 규정에 **10/12~10/16 중 단 1회 수익률 초기화**가 있다.

초기화는 공짜 콜옵션이다. 1주차에 크게 베팅해서 지면 0으로 되돌리고, 이기면 그대로
들고 간다. 하방이 없다. 실데이터로 옵션 가치 `E[max(R, 0)]`를 계산하면 1종목 집중
시 +2.72%, 5종목 분산 시 +1.89%였다 — **집중할수록 옵션이 비싸진다.**

그래서 대회를 두 단계로 나눈다.

| | 1단계 (10/12~10/16) | 2단계 (10/19~11/20) |
|---|---|---|
| 보유 종목 | 1~2 | 3~5 |
| 손절 | 없음 (초기화가 대신한다) | 12% |
| MDD 게이트 | 해제 | DERISK 15% / HALT 25% |

**1단계에서 리스크 관리를 끄는 근거는 초기화 옵션뿐이다.** 10/16이 지나면 그 근거가
사라지므로 전환이 자동으로, 날짜만 보고 일어나야 한다. 사람의 판단이나 성과에 따라
전환을 미루면 하방이 열린 채로 남는다.

**끄는 것은 손절과 MDD 게이트뿐이다.** 감쇠 청산(`EXIT_RULES`, 지금은 `vol_alive`)은
리스크 손절이 아니라 전략의 기본 청산이라 1단계에도 돈다. 예전에는 함께 꺼져서
1주차가 "사서 30봉 상한까지 보유"가 됐다(2026-10-09 수정, `risk_manager.evaluate_exits`).
위 표의 보유 종목 수·손절 폭은 초기 설계값이고, 실제 값은 `.env`가 정한다.

## 이 모듈이 하지 않는 것

**초기화를 실행하지 않는다.** 되돌릴 수 없는 1회성 계좌 설정 변경이므로 코드가
대신 누르지 않는다. `reset_recommendation()`으로 권고만 만들고, 대시보드와 로그에
띄운 뒤 사용자가 KIS 화면에서 직접 수행한다.

## 영업일 계산의 한계

`pandas.bdate_range`로 주말만 제외한다. 2026-10-12~11-20 구간에는 국내 공휴일이
없어(개천절 10/3, 한글날 10/9는 개시 전) 이 단순화가 정확하다. 다른 기간에 쓰려면
`COMP_HOLIDAYS`로 공휴일을 넣어야 한다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from config.settings import Settings, get_settings
from core.events import Portfolio

PHASE_OFF = "OFF"
PHASE1 = "PHASE1"
PHASE2 = "PHASE2"


@dataclass(frozen=True)
class PhaseConfig:
    """국면별 전략 설정. 리스크 매니저와 선정 로직이 이 값을 읽는다."""

    phase: str
    top_k: int
    use_stop_loss: bool
    use_mdd_gate: bool
    label: str

    @property
    def is_competition(self) -> bool:
        return self.phase != PHASE_OFF


@dataclass(frozen=True)
class ResetAdvice:
    """수익률 초기화 권고. 실행은 사람이 한다."""

    should_reset: bool
    window_open: bool
    total_return: float
    deadline: str
    reason: str


def _parse(value: str) -> Optional[dt.date]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _as_date(now: Optional[dt.datetime | dt.date]) -> dt.date:
    if now is None:
        return dt.date.today()
    return now.date() if isinstance(now, dt.datetime) else now


def business_days(start: dt.date, end: dt.date,
                  holidays: Optional[List[str]] = None) -> int:
    """[start, end] 양끝 포함 영업일 수. 주말만 제외한다(위 한계 참고)."""
    if end < start:
        return 0
    days = pd.bdate_range(start, end)
    if holidays:
        skip = {_parse(h) for h in holidays} - {None}
        days = [d for d in days if d.date() not in skip]
    return len(days)


def current_phase(now: Optional[dt.datetime | dt.date] = None,
                  settings: Optional[Settings] = None) -> str:
    """오늘이 어느 국면인지. 날짜만 보고 판단한다.

    대회 날짜가 설정되지 않았거나 기간 밖이면 `OFF`를 돌려준다. 이때 호출자는
    평소 동작을 유지해야 한다 — 백테스트는 과거 데이터를 쓰므로 항상 `OFF`다.
    """
    s = settings or get_settings()
    today = _as_date(now)
    start, end, p1_end = _parse(s.COMP_START), _parse(s.COMP_END), _parse(s.PHASE1_END)
    if start is None or end is None:
        return PHASE_OFF
    if today < start or today > end:
        return PHASE_OFF
    if p1_end is not None and today <= p1_end:
        return PHASE1
    return PHASE2


def phase_config(phase: Optional[str] = None,
                 settings: Optional[Settings] = None,
                 now: Optional[dt.datetime | dt.date] = None) -> PhaseConfig:
    """국면에 맞는 설정을 만든다."""
    s = settings or get_settings()
    phase = phase or current_phase(now, s)

    if phase == PHASE1:
        disable = bool(s.PHASE1_DISABLE_RISK)
        return PhaseConfig(
            phase=PHASE1,
            top_k=max(1, int(s.PHASE1_TOP_K)),
            use_stop_loss=not disable,
            use_mdd_gate=not disable,
            label="1단계 꼬리 베팅 — 초기화 옵션이 하방을 덮는다",
        )
    if phase == PHASE2:
        return PhaseConfig(
            phase=PHASE2,
            top_k=max(1, int(s.PHASE2_TOP_K)),
            use_stop_loss=True,
            use_mdd_gate=True,
            label="2단계 본 전략 — 손절·MDD 게이트 적용",
        )
    return PhaseConfig(
        phase=PHASE_OFF,
        top_k=int(s.MAX_POSITIONS),
        use_stop_loss=True,
        use_mdd_gate=True,
        label="대회 기간 외 — 평소 설정",
    )


def trading_days_remaining(now: Optional[dt.datetime | dt.date] = None,
                           settings: Optional[Settings] = None) -> int:
    """오늘 포함 대회 종료까지 남은 영업일. 기간 밖이면 0."""
    s = settings or get_settings()
    today = _as_date(now)
    start, end = _parse(s.COMP_START), _parse(s.COMP_END)
    if start is None or end is None or today > end:
        return 0
    return business_days(max(today, start), end,
                         getattr(s, "COMP_HOLIDAYS", None))


def reset_recommendation(portfolio: Portfolio,
                         now: Optional[dt.datetime | dt.date] = None,
                         settings: Optional[Settings] = None) -> ResetAdvice:
    """초기화 권고를 만든다.

    초기화는 수익률을 0으로 되돌린다. 따라서 **누적수익이 0 이하이면 초기화하는 것이
    이득이고, 0을 넘으면 그대로 두는 것이 이득이다.** 1회만 가능하므로 창이 닫히는
    마지막 날(`PHASE1_END`)에 판단한다. 그 전에 미리 쓰면 남은 날의 상승을 포기한다.
    """
    s = settings or get_settings()
    today = _as_date(now)
    start, p1_end = _parse(s.COMP_START), _parse(s.PHASE1_END)
    ret = float(portfolio.total_return)
    deadline = s.PHASE1_END or "(미설정)"

    if start is None or p1_end is None:
        return ResetAdvice(False, False, ret, deadline, "대회 날짜가 설정되지 않았다")
    if today < start:
        return ResetAdvice(False, False, ret, deadline, "대회 시작 전")
    if today > p1_end:
        return ResetAdvice(False, False, ret, deadline,
                           f"초기화 창이 닫혔다({deadline} 종료). 이제 되돌릴 수 없다")

    last_day = today == p1_end
    if not last_day:
        left = business_days(today, p1_end, getattr(s, "COMP_HOLIDAYS", None)) - 1
        return ResetAdvice(
            False, True, ret, deadline,
            f"판단 보류 — 초기화 창이 {left}영업일 남았다. "
            f"현재 누적 {ret:+.2%}. 마지막 날에 결정한다")

    if ret <= 0:
        return ResetAdvice(
            True, True, ret, deadline,
            f"초기화 권고 — 누적 {ret:+.2%}. 오늘이 마지막 기회이고, "
            f"초기화하면 이 손실이 사라진다")
    return ResetAdvice(
        False, True, ret, deadline,
        f"초기화하지 않는다 — 누적 {ret:+.2%}. 초기화하면 이 이익을 버린다")
