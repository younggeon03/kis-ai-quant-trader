"""감쇠 기반 청산 규칙 — 거래량 감소 · 상승 감쇠 · 급격한 하락.

## 왜 모듈로 분리했나

청산 규칙이 `risk_manager.evaluate_exits()` 안에 하드코딩돼 있어서, 규칙 하나를
바꾸려면 **모든 주문의 최종 관문인 코드를 건드려야** 했다. 파인튜닝 대상인데
가장 조심해야 하는 곳에 있었다.

여기에는 **순수 로직만** 둔다 — 주문도 포트폴리오도 모른다. 가격·거래대금·ATR을
받아 "청산해야 하면 사유, 아니면 `None`"만 돌려준다. 덕분에 네트워크 없이
셀프테스트로 검증된다(`strategy/intraday.py`와 같은 설계다).

관문은 그대로 `RiskManager`다. 이 모듈은 조언만 하고 수량을 정하지 않는다.

## 세 규칙과 근거

| 규칙 | 무엇을 보는가 | 근거 |
|---|---|---|
| `crash` | 최근 `CRASH_LOOKBACK`봉 고점 대비 `CRASH_DROP_PCT` 하락 | 급등주의 붕괴는 단발로 온다 |
| `trailing` | 고점 − `TRAILING_ATR_MULT` x ATR | 절제 실험에서 **트레일링 제거가 두 실행 모두 악화**였다(−0.255 / −0.089) |
| `vol_decay` | 거래대금이 보유 중 고점의 `EXIT_VOL_DECAY_RATIO` 아래 | 신규. 폭증이 식으면 상승 동력이 사라진다 |

`vol_decay`가 **거래대금 고점 대비**인 이유: 진입일 거래대금을 기준으로 두면
진입 시점 배관이 필요하고, 폭증이 진입 다음 날 더 커지는 경우(이월 재료)를
놓친다. 고점 대비는 트레일링이 가격 고점을 쓰는 것과 같은 꼴이라 해석이 일관된다.

## 순서와 on/off는 설정으로 바꾼다

`EXIT_RULES`가 쉼표로 구분한 **순서 있는 목록**이다. 빈 문자열이면 전부 끈다.

```
EXIT_RULES=crash,trailing,vol_decay     # 세 규칙 전부, 이 우선순위로
EXIT_RULES=vol_decay                    # 거래량만
EXIT_RULES=                             # 감쇠 청산 없음(시간·손절만)
```

**우선순위가 결과를 바꾼다.** 같은 봉에서 둘이 동시에 충족될 때 어느 쪽으로
기록되느냐가 달라지고, 사유별 통계가 달라진다. 그래서 순서를 설정으로 노출한다.

## 임계값이 0이면 그 규칙은 꺼진다

`EXIT_VOL_DECAY_RATIO=0` / `TRAILING_ACTIVATE_PCT=0` / `CRASH_DROP_PCT>=1`이면
목록에 있어도 발동하지 않는다. **단 `STOP_LOSS_PCT=0`은 다르다** — 그것은 손절을
끄는 것이 아니라 0% 손절이 되며, 이 모듈이 아니라 `risk_manager`가 다룬다.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

_EPS = 1e-12

# 규칙 이름 — `EXIT_RULES` 설정이 이 이름을 쓴다.
RULE_CRASH = "crash"
RULE_TRAILING = "trailing"
RULE_VOL_DECAY = "vol_decay"
ALL_RULES: List[str] = [RULE_CRASH, RULE_TRAILING, RULE_VOL_DECAY]


@dataclass
class DecayState:
    """보유 포지션 하나의 감쇠 추적 상태.

    `peak_price`와 `turnover_peak`은 **단방향으로만 올라간다.** 내려가면 되돌림을
    감지할 기준이 사라진다.
    """

    symbol: str
    entry: float
    peak_price: float = 0.0
    turnover_peak: float = 0.0
    bars_seen: int = 0
    armed: bool = False
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def observe(self, price: float, turnover: float) -> None:
        if price > 0:
            self.peak_price = max(self.peak_price, price)
            self.prices.append(float(price))
        if turnover > 0:
            self.turnover_peak = max(self.turnover_peak, float(turnover))
        self.bars_seen += 1

    def gain(self, price: float) -> float:
        return price / max(self.entry, _EPS) - 1.0


@dataclass
class DecayExit:
    """청산 판정 결과. `stop_price`는 트레일링이 올려 둔 손절선(있을 때만)."""

    rule: str
    reason: str
    stop_price: float = 0.0


class DecayExitTracker:
    """종목별 감쇠 상태를 들고 규칙을 평가한다.

    `RiskManager`가 하나만 들고 쓴다. 상태를 들고 있으므로 **매 봉 `observe()`를
    한 번만** 불러야 한다 — 두 번 부르면 `bars_seen`이 어긋나고 룩백 창이 짧아진다.
    """

    def __init__(self, settings) -> None:
        self.s = settings
        self.states: Dict[str, DecayState] = {}

    # ------------------------------------------------------------ 상태 관리
    def order(self) -> List[str]:
        """평가 순서. 모르는 이름은 조용히 버리지 않고 제외하되 목록에 남기지 않는다."""
        raw = str(getattr(self.s, "EXIT_RULES", "") or "")
        names = [n.strip() for n in raw.split(",") if n.strip()]
        return [n for n in names if n in ALL_RULES]

    def open(self, symbol: str, entry: float) -> None:
        """체결 시점에 감시를 시작한다. **접수가 아니라 체결**이다 — 접수 시점에는
        평단가가 없어 되돌림 기준이 어긋난다(`main.py::_on_fill_event`와 같은 이유).
        """
        if entry <= 0:
            return
        lookback = max(2, int(getattr(self.s, "CRASH_LOOKBACK", 10)))
        st = DecayState(symbol=symbol, entry=float(entry),
                        peak_price=float(entry))
        st.prices = deque(maxlen=lookback)
        self.states[symbol] = st

    def close(self, symbol: str) -> None:
        self.states.pop(symbol, None)

    def reset(self) -> None:
        self.states.clear()

    # ------------------------------------------------------------ 평가
    def check(self, symbol: str, price: float, turnover: float = 0.0,
              atr: float = 0.0, entry: Optional[float] = None,
              ts: Optional[dt.datetime] = None) -> Optional[DecayExit]:
        """청산해야 하면 `DecayExit`, 아니면 `None`.

        감시가 없으면 `entry`로 즉석 시작한다 — 재시작·잔고 대조로 포지션이 먼저
        생기는 경로가 있어 감시 누락이 조용한 미작동이 되지 않게 한다.
        """
        if price <= 0:
            return None
        st = self.states.get(symbol)
        if st is None:
            if not entry or entry <= 0:
                return None
            self.open(symbol, entry)
            st = self.states[symbol]
        st.observe(price, turnover)

        for name in self.order():
            hit = self._rule(name, st, price, turnover, atr)
            if hit is not None:
                return hit
        return None

    def _rule(self, name: str, st: DecayState, price: float,
              turnover: float, atr: float) -> Optional[DecayExit]:
        if name == RULE_CRASH:
            return self._crash(st, price)
        if name == RULE_TRAILING:
            return self._trailing(st, price, atr)
        if name == RULE_VOL_DECAY:
            return self._vol_decay(st, turnover)
        return None

    # --- 급격한 하락 ---
    def _crash(self, st: DecayState, price: float) -> Optional[DecayExit]:
        drop = float(getattr(self.s, "CRASH_DROP_PCT", 1.0))
        if drop <= 0 or drop >= 1.0 or len(st.prices) < 3:
            return None
        peak = max(st.prices)
        if peak <= 0:
            return None
        fell = (peak - price) / peak
        if fell >= drop:
            return DecayExit(RULE_CRASH,
                             f"급락 {fell:.2%} (최근 {len(st.prices)}봉 고점 대비)")
        return None

    # --- 상승 감쇠 ---
    def _trailing(self, st: DecayState, price: float,
                  atr: float) -> Optional[DecayExit]:
        activate = float(getattr(self.s, "TRAILING_ACTIVATE_PCT", 0.0))
        if activate <= 0:
            return None
        peak_gain = st.peak_price / max(st.entry, _EPS) - 1.0
        if peak_gain >= activate:
            st.armed = True
        if not st.armed:
            return None
        mult = float(getattr(self.s, "TRAILING_ATR_MULT", 2.2))
        dist = mult * atr if atr > 0 else st.peak_price * 0.03
        stop = st.peak_price - dist
        if price <= stop:
            return DecayExit(RULE_TRAILING,
                             f"상승 감쇠 (고점 {peak_gain:+.2%} → {st.gain(price):+.2%})",
                             stop_price=float(stop))
        # 청산은 아니지만 손절선을 올려 둔다(단방향 상향).
        return None

    # --- 거래량 감소 ---
    def _vol_decay(self, st: DecayState, turnover: float) -> Optional[DecayExit]:
        ratio = float(getattr(self.s, "EXIT_VOL_DECAY_RATIO", 0.0))
        if ratio <= 0 or st.turnover_peak <= 0 or turnover <= 0:
            return None
        min_bars = max(1, int(getattr(self.s, "EXIT_VOL_DECAY_MIN_BARS", 2)))
        if st.bars_seen < min_bars:
            return None   # 진입 당일 거래가 적다고 바로 털지 않는다
        cur = turnover / st.turnover_peak
        if cur < ratio:
            return DecayExit(RULE_VOL_DECAY,
                             f"거래 감소 (고점의 {cur:.0%} < 기준 {ratio:.0%})")
        return None

    # ------------------------------------------------------------ 관측
    def trail_stop(self, symbol: str, atr: float = 0.0) -> float:
        """트레일링이 제시하는 손절선. `RiskManager`가 단방향 상향에 쓴다.

        무장 전이거나 트레일링이 꺼져 있으면 0을 돌려준다 — 0은 "제안 없음"이며
        손절선을 내리라는 뜻이 아니다.
        """
        st = self.states.get(symbol)
        if st is None or not st.armed:
            return 0.0
        if float(getattr(self.s, "TRAILING_ACTIVATE_PCT", 0.0)) <= 0:
            return 0.0
        mult = float(getattr(self.s, "TRAILING_ATR_MULT", 2.2))
        dist = mult * atr if atr > 0 else st.peak_price * 0.03
        return float(max(0.0, st.peak_price - dist))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            sym: {"entry": st.entry, "peak_price": st.peak_price,
                  "turnover_peak": st.turnover_peak,
                  "bars_seen": float(st.bars_seen), "armed": float(st.armed)}
            for sym, st in self.states.items()
        }
