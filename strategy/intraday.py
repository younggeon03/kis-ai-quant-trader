"""장중 돌파 감시와 청산 — 분봉 전략의 실거래 집행 레이어.

## 왜 별도 레이어가 필요한가

`main.py::TradingEngine`은 **일봉 기준**으로 만들어졌다. 하루 한 번 신호를 만들고,
Mock 가상 시계가 "봉 1개 = 1거래일"이다. 분봉 단독 전략은 장중에 조건이 충족되는
**순간** 진입하고 그날 안에 청산하므로, 판단 주기가 하루에서 초 단위로 바뀐다.

이 모듈은 **순수 로직만** 담는다 — 주문도, 네트워크도, 파일도 건드리지 않는다.
시세를 받아 "지금 사라/팔라"를 돌려주고, 실제 주문은 `TradingEngine`이
`RiskManager`를 통과시킨 뒤 낸다. 리스크 매니저가 모든 주문의 최종 관문이라는
기존 구조를 깨지 않기 위해서다. 덕분에 셀프테스트에서 네트워크 없이 검증된다.

## 트리거 — 백테스트와 같은 조건

```
① 현재가 ≥ 전일종가 × (1 + INTRADAY_TRIGGER_PCT)
② 이번 분의 거래량 ≥ INTRADAY_VOL_MULT × (전일 거래량 ÷ 391)
```

`Quote.volume`은 KIS `acml_vol`(누적거래량)이라 **차분**으로 분당 거래량을 만든다.
분이 바뀔 때 기준값을 갱신하고, 그 뒤 들어온 누적값에서 빼면 이번 분의 거래량이다.

## 백테스트보다 보수적인 지점 둘

**거래량 조건**: 백테스트는 1분봉이 닫힌 뒤 그 분의 **전체** 거래량으로 판정한다.
실거래는 분 중간에 **누적된 만큼**으로 판정하므로, 같은 분이라도 더 늦게(또는 아예
안) 발동한다. 부분 ≤ 전체이므로 실거래가 더 엄격하다.

**진입 시점**: 백테스트는 돌파 봉의 **다음 봉 시가**에 진입한다(최대 60초 지연).
실거래는 조건 충족을 감지한 즉시 주문하므로 더 빠르다. 실측 진입 슬리피지가
**+0.785%**(왕복비용의 2.8배)였으므로 이 차이는 실거래에 유리하다.

두 방향 모두 **백테스트가 실거래를 과소평가**하는 쪽이다. 반대 방향이면 위험하다.

## 청산 — 측정으로 확정할 자리

부분 데이터 격자에서는 **종가 청산**(규칙 없음)이 꼬리가 가장 두꺼웠다. 청산 규칙이
분산을 깎고, 분산이 꼬리를 만들기 때문이다. 그래서 기본값은 규칙 없음이고
`INTRADAY_*`로 켠다. 전 표본 격자 결과가 나오면 그 값을 기본값으로 올린다.

**장 마감 전 정리는 규칙과 무관하게 항상 한다**(`INTRADAY_FLATTEN_HHMM`).
오버나이트를 들고 가면 그건 분봉 전략이 아니라 스윙이고, 측정한 분포와 달라진다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SESSION_MINUTES = 391          # 09:00~15:30
_EPS = 1e-9


@dataclass
class SymbolState:
    """종목 하나의 장중 상태. 날이 바뀌면 `reset_day()`로 비운다."""

    prev_close: float = 0.0
    prev_volume: float = 0.0
    minute_key: str = ""           # 현재 분 (YYYYMMDDHHMM)
    minute_vol_base: float = 0.0   # 그 분이 시작될 때의 누적거래량
    cum_volume: float = 0.0        # 마지막으로 본 누적거래량
    last_price: float = 0.0
    session_high: float = 0.0
    triggered: bool = False        # 오늘 이미 발동했는가 (하루 한 번)
    half_minute: int = -1          # 돌파선 절반을 처음 지난 분 (급등 속도의 분모)
    minute_index: int = 0          # 09:00부터 경과 분

    @property
    def minute_volume(self) -> float:
        """이번 분에 쌓인 거래량. 누적값의 차분이다."""
        return max(0.0, self.cum_volume - self.minute_vol_base)


@dataclass(frozen=True)
class TriggerSignal:
    """돌파 발동. `TradingEngine`이 리스크 관문을 통과시킨 뒤 주문한다."""

    symbol: str
    price: float
    prev_close: float
    minute_volume: float
    vol_floor: float
    ts: dt.datetime
    trig_minute: int = 0           # 09:00부터 경과 분
    speed: int = 0                 # 절반 돌파에서 완전 돌파까지 걸린 분
    turnover_ratio: float = 0.0    # 오늘 누적 거래량 / 전일 거래량
    burst: float = 0.0             # 이번 분 거래량 / 전일 분당 평균
    freshness: float = 0.0         # 0~1, 높을수록 '덜 돌고 빠른' 급등

    @property
    def level_pct(self) -> float:
        return self.price / max(self.prev_close, _EPS) - 1.0

    def describe(self) -> str:
        return (f"장중 돌파 {self.level_pct:+.2%} "
                f"(거래량 {self.minute_volume:,.0f} ≥ {self.vol_floor:,.0f} · "
                f"{self.trig_minute}분 · 속도 {self.speed}분 · "
                f"누적 {self.turnover_ratio:.2f}배 · 신선도 {self.freshness:.3f})")


@dataclass
class PositionWatch:
    """보유 중인 장중 포지션의 청산 감시 상태."""

    symbol: str
    entry: float
    peak: float = 0.0
    armed: bool = False            # 트레일링 발동 조건을 넘었는가

    def gain(self, price: float) -> float:
        return price / max(self.entry, _EPS) - 1.0


class IntradayWatcher:
    """장중 돌파 감시기.

    `prime()`으로 전일 종가·거래량을 넣고, 시세가 올 때마다 `on_quote()`를 부른다.
    발동하면 `TriggerSignal`을 돌려주고, 같은 날 같은 종목은 다시 발동하지 않는다.
    """

    def __init__(self, trigger_pct: float = 0.03, vol_mult: float = 1.5,
                 target_pct: float = 0.0, stop_pct: float = 0.0,
                 trail_pct: float = 0.0, trail_activate_pct: float = 0.0,
                 flatten_hhmm: str = "1515"):
        self.trigger_pct = float(trigger_pct)
        self.vol_mult = float(vol_mult)
        self.target_pct = float(target_pct)
        self.stop_pct = float(stop_pct)
        self.trail_pct = float(trail_pct)
        self.trail_activate_pct = float(trail_activate_pct)
        self.flatten_hhmm = str(flatten_hhmm)
        self.states: Dict[str, SymbolState] = {}
        self.watches: Dict[str, PositionWatch] = {}
        self._day: str = ""

    # ------------------------------------------------------------ 준비
    def prime(self, symbol: str, prev_close: float, prev_volume: float) -> None:
        """일봉에서 전일 종가·거래량을 넣는다. 둘 다 트리거 판정의 기준이다."""
        st = self.states.setdefault(symbol, SymbolState())
        st.prev_close = float(prev_close or 0.0)
        st.prev_volume = float(prev_volume or 0.0)

    def reset_day(self, day: str) -> None:
        """날짜 롤오버. **발동 이력과 분봉 누적을 반드시 비운다** — 비우지 않으면
        어제 발동한 종목이 오늘 영원히 발동하지 않는다."""
        if day == self._day:
            return
        self._day = day
        for st in self.states.values():
            st.minute_key = ""
            st.minute_vol_base = 0.0
            st.cum_volume = 0.0
            st.session_high = 0.0
            st.triggered = False
            st.half_minute = -1
            st.minute_index = 0
        self.watches.clear()

    def vol_floor(self, symbol: str) -> float:
        """분당 거래량 하한. 전일 거래량을 모르면 0(조건 없음)이다."""
        st = self.states.get(symbol)
        if st is None or st.prev_volume <= 0:
            return 0.0
        return self.vol_mult * (st.prev_volume / SESSION_MINUTES)

    # ------------------------------------------------------------ 감시
    def on_quote(self, symbol: str, price: float, cum_volume: float,
                 ts: dt.datetime) -> Optional[TriggerSignal]:
        """시세 한 건을 반영하고, 트리거가 충족되면 신호를 돌려준다."""
        if price <= 0:
            return None
        day = ts.strftime("%Y%m%d")
        self.reset_day(day)
        st = self.states.get(symbol)
        if st is None or st.prev_close <= 0:
            return None

        key = ts.strftime("%Y%m%d%H%M")
        if key != st.minute_key:
            # 분이 바뀌었다. 이번 분의 기준 누적값을 지금 값으로 잡는다.
            st.minute_key = key
            st.minute_vol_base = max(st.cum_volume, 0.0)
        st.minute_index = max(0, (ts.hour - 9) * 60 + ts.minute)
        # 돌파선 **절반**을 처음 지난 분. 거기서 완전 돌파까지 걸린 시간이 급등
        # 속도이고, 실측에서 빠른 급등이 좋았다(37분 vs 64분).
        if st.half_minute < 0 and price >= st.prev_close * (1.0 + self.trigger_pct / 2.0):
            st.half_minute = st.minute_index
        # 누적거래량은 단조증가다. 역행하면(장 리셋·데이터 오류) 기준을 다시 잡는다.
        if cum_volume + _EPS < st.cum_volume:
            st.minute_vol_base = 0.0
        st.cum_volume = max(cum_volume, 0.0)
        st.last_price = price
        st.session_high = max(st.session_high, price)

        if st.triggered:
            return None
        if price < st.prev_close * (1.0 + self.trigger_pct):
            return None
        floor = self.vol_floor(symbol)
        if floor > 0 and st.minute_volume < floor:
            return None

        st.triggered = True
        speed = max(0, st.minute_index - (st.half_minute if st.half_minute >= 0
                                          else st.minute_index))
        turnover_ratio = (st.cum_volume / st.prev_volume) if st.prev_volume > 0 else 0.0
        per_min = (st.prev_volume / SESSION_MINUTES) if st.prev_volume > 0 else 0.0
        burst = (st.minute_volume / per_min) if per_min > 0 else 0.0
        return TriggerSignal(
            symbol=symbol, price=price, prev_close=st.prev_close,
            minute_volume=st.minute_volume, vol_floor=floor, ts=ts,
            trig_minute=st.minute_index, speed=speed,
            turnover_ratio=turnover_ratio, burst=burst,
            freshness=self.freshness(st.minute_index, speed, turnover_ratio, burst))

    # ------------------------------------------------------------ 신선도
    @staticmethod
    def freshness(trig_minute: int, speed: int, turnover_ratio: float,
                  burst: float) -> float:
        """0~1. 높을수록 "덜 돌고 빠른" 급등이다.

        ## 왜 돌파율이 아니라 신선도로 고르는가

        하루에 돌파 사건이 여러 건 나오는데 보유 한도는 2종목이다. 무엇을 고르느냐가
        `tail_optimizer` 기준 `P(+100%)`를 1.04%(무작위)에서 6.17%(스킬 20%)까지
        바꾼다 — 구조보다 큰 레버다.

        처음에는 돌파율이 큰 순서로 골랐다. 그런데 실측은 그 방향이 **반대**라고
        말한다. 종가 수익률 상위 25% 사건과 나머지를 비교하면:

        | 피처 | 상위군 | 하위군 | 방향 |
        |---|---|---|---|
        | 돌파 시각 | 62분 | 106분 | 이를수록 좋다 |
        | +5%→돌파 속도 | 37분 | 64분 | 빠를수록 좋다 |
        | 누적/전일 거래대금 | 0.99 | 1.30 | **덜 돌았을수록 좋다** |
        | 폭증 배수 | 10.7 | 15.6 | **과하지 않을수록 좋다** |

        뒤 두 개가 직관과 반대다. 폭증이 과하고 전일 거래대금을 이미 넘겼다면
        소재가 터진 직후가 아니라 **터지고 한참 지난 뒤**다. 네 피처가 같은 축을
        가리킨다 — 신선도다.

        ## 왜 모델이 아니라 고정 공식인가

        분봉 모델은 부분 데이터(97종목)에서 AUC 0.5435, 스킬 0%였고 선택이 무작위
        보다 나빴다. 정보 없는 피처 6개가 중요도의 35%를 먹으면서 노이즈를 넣은
        것이 원인으로 보인다. 전 표본 재측정 전에 모델을 쓰면 그 상태를 그대로
        실거래에 들고 간다. 고정 공식은 과적합 여지가 없고 지금 적용할 수 있다.

        네 항목에 같은 가중(0.25)을 준다. 가중을 데이터로 맞추면 그 순간 모델이
        되고 과적합 여지가 생긴다. 정규화 분모는 실측 분포의 하위군 수준으로
        잡았다 — 60분, 2.0배, 20배.
        """
        early = 1.0 - min(1.0, max(0, trig_minute) / float(SESSION_MINUTES))
        fast = 1.0 - min(1.0, max(0, speed) / 60.0)
        quiet = 1.0 - min(1.0, max(0.0, turnover_ratio) / 2.0)
        calm = 1.0 - min(1.0, max(0.0, burst) / 20.0)
        return float(0.25 * (early + fast + quiet + calm))

    # ------------------------------------------------------------ 청산
    def open_watch(self, symbol: str, entry: float) -> None:
        """체결된 포지션의 청산 감시를 시작한다. **접수가 아니라 체결 시점**에
        부른다 — 접수 시점에는 평단가가 없어 목표·손절 기준이 어긋난다
        (`main.py::_on_fill_event`가 손절선을 체결 시점에 잡는 것과 같은 이유다)."""
        if entry > 0:
            self.watches[symbol] = PositionWatch(symbol=symbol, entry=float(entry),
                                                 peak=float(entry))

    def close_watch(self, symbol: str) -> None:
        self.watches.pop(symbol, None)

    def exit_reason(self, symbol: str, price: float,
                    ts: Optional[dt.datetime] = None) -> Optional[str]:
        """청산해야 하면 사유, 아니면 `None`.

        우선순위는 **손절 > 트레일링 > 목표 > 마감정리**다. 손절을 먼저 보는 것은
        백테스트(`scripts/minute_exit_grid.py::simulate_rule`)와 같은 선택이고,
        같은 틱에서 양쪽이 충족될 때 유리한 쪽을 가정하면 결과가 부풀려진다.
        """
        w = self.watches.get(symbol)
        if w is None or price <= 0:
            return None
        w.peak = max(w.peak, price)
        g = w.gain(price)

        if self.stop_pct > 0 and g <= -self.stop_pct:
            return f"장중 손절 {g:+.2%}"
        if self.trail_pct > 0:
            peak_gain = w.peak / max(w.entry, _EPS) - 1.0
            if peak_gain >= self.trail_activate_pct:
                w.armed = True
            if w.armed and price <= w.peak * (1.0 - self.trail_pct):
                return f"장중 트레일링 (고점 {peak_gain:+.2%} → {g:+.2%})"
        if self.target_pct > 0 and g >= self.target_pct:
            return f"장중 익절 {g:+.2%}"
        if ts is not None and self.should_flatten(ts):
            return f"마감 정리 {g:+.2%}"
        return None

    def should_flatten(self, ts: dt.datetime) -> bool:
        """마감 정리 시각을 지났는가. 규칙과 무관하게 오버나이트를 만들지 않는다."""
        return ts.strftime("%H%M") >= self.flatten_hhmm

    # ------------------------------------------------------------ 관측
    def near_trigger(self, symbol: str) -> float:
        """돌파선까지 남은 거리(비율). 시세 관심목록을 고르는 기준이다.

        WebSocket 구독 한도가 41건이라 284종목을 전부 볼 수 없다. 일봉 전략은
        모델 점수 상위를 구독했지만, 분봉 전략에서 중요한 것은 **오늘 돌파할
        가능성**이므로 돌파선에 가까운 종목을 우선한다.
        """
        st = self.states.get(symbol)
        if st is None or st.prev_close <= 0 or st.last_price <= 0:
            return float("inf")
        level = st.prev_close * (1.0 + self.trigger_pct)
        return (level - st.last_price) / max(level, _EPS)

    def snapshot(self) -> Dict[str, object]:
        """대시보드용 요약."""
        return {
            "trigger_pct": self.trigger_pct,
            "vol_mult": self.vol_mult,
            "flatten_hhmm": self.flatten_hhmm,
            "primed": sum(1 for s in self.states.values() if s.prev_close > 0),
            "triggered_today": sorted(k for k, s in self.states.items() if s.triggered),
            "watching": sorted(self.watches),
        }
