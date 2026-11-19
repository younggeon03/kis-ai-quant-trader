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
    turnover_med20: float = 0.0    # 전일까지의 20봉 거래대금 중앙값 (burst 모드 분모)
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

    @property
    def burst_ratio(self) -> float:
        """지금까지 쌓인 **당일 거래대금** / 20봉 중앙값(전일까지).

        **일봉 조건과 분모가 한 칸 다르다.** `catalyst_features.turnover_burst_20`은
        `rolling(20).median()`으로 **당일을 포함**한다 — 인과적이지만 09:00에는 알
        수 없는 값이다. 그래서 여기서는 전일까지의 20봉으로 잡는다
        (`main.py::_prime_intraday`가 `hist.tail(20)`로 계산해 넘긴다).
        폭증일 자신이 중앙값에 들어가지 않으므로 **같은 배수에서 이쪽이 약간 더
        쉽게 발동한다.** 두 집단을 비교할 때 이 차이를 감안해야 한다.

        분자가 **장중 누적**이라 하루가 끝나기 전에 조건 성립을 알 수 있고,
        그것이 이 모드의 목적이다.

        거래대금 = 현재가 x 누적거래량으로 근사한다. 정확한 값은 체결 단가의
        가중합이지만 틱 테이프가 없다. 장중 가격 변동폭 안에서의 오차이고,
        **분모와 같은 정의(종가 x 거래량)를 쓰므로 체계적 편향은 작다.**
        """
        if self.turnover_med20 <= 0 or self.cum_volume <= 0 or self.last_price <= 0:
            return 0.0
        return (self.last_price * self.cum_volume) / self.turnover_med20


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
    score: float = 0.0             # 진입 시점 점수 (교체 판정의 기준)

    def gain(self, price: float) -> float:
        return price / max(self.entry, _EPS) - 1.0


class IntradayWatcher:
    """장중 돌파 감시기.

    `prime()`으로 전일 종가·거래량을 넣고, 시세가 올 때마다 `on_quote()`를 부른다.
    발동하면 `TriggerSignal`을 돌려주고, 같은 날 같은 종목은 다시 발동하지 않는다.
    """

    def __init__(self, trigger_pct: float = 0.03, vol_mult: float = 1.5,
                 entry_mode: str = "breakout", burst_mult: float = 10.0,
                 flatten: bool = True,
                 target_pct: float = 0.0, stop_pct: float = 0.0,
                 trail_pct: float = 0.0, trail_activate_pct: float = 0.0,
                 flatten_hhmm: str = "1515",
                 candidate_ttl: int = 30,
                 swap_margin: float = 0.10,
                 max_swaps: int = 2):
        self.trigger_pct = float(trigger_pct)
        self.vol_mult = float(vol_mult)
        # 진입 판정 방식.
        #   breakout : 현재가 >= 전일종가 x (1+trigger) AND 분당 거래량 >= 하한
        #   burst    : 당일 누적 거래대금 >= 20봉 중앙값 x burst_mult
        #
        # `burst`를 넣은 이유: 일봉 catalyst 조건이 `turnover = 종가 x 거래량`이라
        # **장 마감 후에야 확정된다.** 그래서 T+1 시가에 들어가고 폭증일을 통째로
        # 버린다 — 실측 4,902건에서 폭증일 시가→종가가 평균 **+5.29%**(승률 71.8%)
        # 인데 현재 먹는 구간은 중앙 **−1.50%**다. 같은 조건을 장중 누적으로 보면
        # 하루 안에 감지할 수 있다.
        self.entry_mode = str(entry_mode)
        self.burst_mult = float(burst_mult)
        # 15:15 마감 정리를 끌 수 있다. 끄면 포지션이 **일봉 청산 경로**
        # (`risk_manager`: 손절 + 감쇠 3종 + 시간손절)로 넘어가 다일 보유가 된다.
        # 당일 청산은 변동을 하루로 묶고 꼬리는 변동에서 나온다 — 분봉 단타가
        # `P(+100%)` 0.00%였던 원인이 진입이 아니라 이쪽일 수 있다.
        self.flatten = bool(flatten)
        self.target_pct = float(target_pct)
        self.stop_pct = float(stop_pct)
        self.trail_pct = float(trail_pct)
        self.trail_activate_pct = float(trail_activate_pct)
        self.flatten_hhmm = str(flatten_hhmm)
        self.candidate_ttl = int(candidate_ttl)
        self.swap_margin = float(swap_margin)
        self.max_swaps = int(max_swaps)
        self.states: Dict[str, SymbolState] = {}
        self.watches: Dict[str, PositionWatch] = {}
        # 후보 명부 — {종목: (신호, 점수)}. **매 분 비우지 않는다.**
        self.book: Dict[str, tuple] = {}
        self.swaps_today: int = 0
        self._day: str = ""

    # ------------------------------------------------------------ 준비
    def prime(self, symbol: str, prev_close: float, prev_volume: float,
              turnover_med20: float = 0.0) -> None:
        """일봉에서 전일 기준값을 넣는다.

        `prev_close`/`prev_volume`은 `breakout` 모드의 기준, `turnover_med20`은
        `burst` 모드의 분모다. **`burst` 모드인데 이 값이 0이면 그 종목은 영원히
        발동하지 않는다** — 조용한 미작동이므로 호출부가 반드시 채워야 한다.
        """
        st = self.states.setdefault(symbol, SymbolState())
        st.prev_close = float(prev_close or 0.0)
        st.prev_volume = float(prev_volume or 0.0)
        if turnover_med20:
            st.turnover_med20 = float(turnover_med20)

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
        self.book.clear()
        self.swaps_today = 0

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
        # opening 모드는 시세 한 건 단위로 판정하지 않는다 — 09:0N에 분봉으로 한 번
        # 판정한다(`main.py::_opening_scan`, `strategy/opening.py`). 여기서 막지 않으면
        # 아래 breakout 분기로 떨어져 가격 돌파로 진입한다.
        if self.entry_mode in ("opening", "sliding"):
            return None
        # `floor`는 두 분기 밖에서 쓰인다(신호의 `vol_floor`). breakout 분기에서만
        # 정의하면 burst 모드에서 `UnboundLocalError`가 난다 — 셀프테스트 17번이
        # 잡았다. burst 모드에서는 거래량 하한을 쓰지 않으므로 관측값으로만 싣는다.
        floor = self.vol_floor(symbol)
        if self.entry_mode == "burst":
            # 당일 누적 거래대금만 본다. 가격 조건을 걸지 않는 것은 의도다 —
            # "거래대금은 10배인데 가격은 안 오른 날"이 분별 대상의 일부이고,
            # 가격 조건을 같이 걸면 그 집단이 사라져 수집 편향과 같은 문제가 된다.
            if st.turnover_med20 <= 0 or st.burst_ratio < self.burst_mult:
                return None
        else:
            if price < st.prev_close * (1.0 + self.trigger_pct):
                return None
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

    # ------------------------------------------------------- 후보 명부·교체
    def add_candidate(self, sig: TriggerSignal, score: float) -> None:
        """발동한 사건을 명부에 올린다. 점수는 **호출자가 정한다.**

        점수 산정을 감시기 밖에 두는 이유는, 모델 점수와 고정 공식(신선도)을
        같은 구조에서 바꿔 끼워 비교해야 하기 때문이다. 감시기가 점수까지
        계산하면 그 비교가 코드 수정 없이는 불가능해진다.
        """
        self.book[sig.symbol] = (sig, float(score))

    def prune_book(self, now: dt.datetime) -> None:
        """오래된 후보를 버린다.

        돌파 후 시간이 지나면 그 정보는 낡는다 — 실측에서 `trig_minute`가
        36.8분(상위 25%) vs 74.9분(하위 75%)으로 **이른 돌파가 좋다**. 몇 시간 전
        돌파를 지금 사는 것은 그 신호를 거스른다. `candidate_ttl`분만 보관한다.
        """
        if not self.book:
            return
        cutoff = now - dt.timedelta(minutes=self.candidate_ttl)
        for sym in [s for s, (sig, _) in self.book.items() if sig.ts < cutoff]:
            self.book.pop(sym, None)

    def pick_entries(self, now: dt.datetime, free_slots: int,
                     exclude: Optional[set] = None) -> List[TriggerSignal]:
        """빈 슬롯을 명부의 상위 점수로 채운다.

        ## 왜 명부가 필요한가 (겪은 설계 결함)

        처음에는 발동 신호를 **매 분 비웠다**(`pending = []`). 그래서 같은 분에
        동시 발동한 것들 사이에서만 골랐다. 실측에서 하루 돌파가 중앙 14건인데
        경과분이 중앙 15분 · p75 82분 · p90 253분으로 **시간에 흩어져 있고**,
        2번째 돌파가 5분 안에 나는 날이 **76.9%**였다. 즉 슬롯 2개가 09:00~09:01에
        차고 나머지 12건은 **선택 기회조차 없이 버려졌다.**

        결과적으로 "하루의 첫 2건"을 사는 전략이었고, 모델도 신선도도 쓰이지
        않았다. 점수를 아무리 잘 만들어도 개입할 자리가 없었다.

        명부에 `candidate_ttl`분 보관하면 선택이 하루 종일 작동한다.
        """
        self.prune_book(now)
        skip = set(exclude or set()) | set(self.watches)
        cand = [(sig, sc) for sym, (sig, sc) in self.book.items() if sym not in skip]
        cand.sort(key=lambda x: -x[1])
        picked = [sig for sig, _ in cand[: max(0, free_slots)]]
        for sig in picked:
            self.book.pop(sig.symbol, None)
        return picked

    def swap_target(self, now: dt.datetime,
                    exclude: Optional[set] = None
                    ) -> Optional[Tuple[str, TriggerSignal]]:
        """슬롯이 꽉 찼을 때 **갈아탈 대상**. (내보낼 종목, 들일 신호) 또는 `None`.

        점수가 가장 낮은 보유 종목과 명부 최고 점수를 비교해, 차이가
        `swap_margin`을 넘을 때만 교체한다.

        ## 여유(margin)가 필요한 이유

        교체는 왕복비용 0.278%를 즉시 지불한다. 점수가 조금만 높은 후보로 계속
        갈아타면 비용만 쌓인다(일봉 전략에서 `MIN_HOLDING_BARS`를 둔 것과 같은
        이유다). 그리고 하루 교체 횟수를 `max_swaps`로 제한한다 — 명부가 계속
        갱신되므로 제한이 없으면 이론상 하루 수십 번 갈아탈 수 있다.

        보유 종목의 평가손익은 보지 않는다. 점수는 "이 급등이 계속 갈 확률"이고
        그 질문은 이미 올랐든 안 올랐든 같다. 손익으로 판단하면 손절·익절 규칙과
        역할이 겹쳐 어느 쪽이 작동했는지 알 수 없게 된다.
        """
        if self.swaps_today >= self.max_swaps or not self.watches:
            return None
        self.prune_book(now)
        skip = set(exclude or set()) | set(self.watches)
        cand = [(sig, sc) for sym, (sig, sc) in self.book.items() if sym not in skip]
        if not cand:
            return None
        best_sig, best_sc = max(cand, key=lambda x: x[1])
        worst = min(self.watches.values(), key=lambda w: w.score)
        if best_sc - worst.score <= self.swap_margin:
            return None
        return worst.symbol, best_sig

    def note_swap(self, sig: TriggerSignal) -> None:
        """교체를 실행했음을 기록한다. 호출자가 **매도 주문을 낸 뒤** 부른다.

        후보를 명부에서 **빼지 않는다.** 매도가 체결돼 슬롯이 비면 다음 루프의
        `pick_entries()`가 그 후보를 집어 매수한다. 매도와 매수를 같은 루프에서
        내면 브로커 가용금액이 `cash - reserved`라 매도 대금이 들어오기 전이어서
        "주문가능금액 부족"으로 거부된다(비중 45%로 실제 겪었다).

        매도가 실패하면 교체 횟수만 소모된다. 하루 한도를 넘지 않게 하는 쪽이
        안전하므로 그대로 둔다.
        """
        self.swaps_today += 1

    # ------------------------------------------------------------ 청산
    def open_watch(self, symbol: str, entry: float, score: float = 0.0) -> None:
        """체결된 포지션의 청산 감시를 시작한다. **접수가 아니라 체결 시점**에
        부른다 — 접수 시점에는 평단가가 없어 목표·손절 기준이 어긋난다
        (`main.py::_on_fill_event`가 손절선을 체결 시점에 잡는 것과 같은 이유다)."""
        if entry > 0:
            self.watches[symbol] = PositionWatch(symbol=symbol, entry=float(entry),
                                                 peak=float(entry), score=float(score))

    def close_watch(self, symbol: str) -> None:
        self.watches.pop(symbol, None)

    def exit_reason(self, symbol: str, price: float,
                    ts: Optional[dt.datetime] = None) -> Optional[str]:
        """청산해야 하면 사유, 아니면 `None`.

        우선순위는 **손절 > 트레일링 > 목표 > 마감정리**다. 손절을 먼저 보는 것은
        백테스트(`scripts/research/minute_exit_grid.py::simulate_rule`)와 같은 선택이고,
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
        """마감 정리 시각을 지났는가.

        `flatten=False`면 **항상 False**다. 그 경우 포지션은 일봉 청산 경로로
        넘어가 다일 보유가 된다(오버나이트가 생긴다 — 의도된 것이다).
        """
        if not self.flatten:
            return False
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
