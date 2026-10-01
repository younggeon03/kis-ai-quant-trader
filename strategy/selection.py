"""종목 선정 — 절대 임계값이 아니라 상대 순위로 목표 보유 종목을 정한다.

## 왜 이 모듈이 필요한가

기존 진입 로직은 `sig.score < SIGNAL_BUY_THRESHOLD`(0.22)면 건너뛰는 절대 임계값
방식이었다. 이 방식의 구조적 문제는 **임계값을 넘는 종목이 없으면 현금을 든다**는 것이다.
실데이터 워크포워드에서 평균 총 익스포저가 30%에 머물렀고, 시장이 +81% 오른 구간에서
전략은 −12% 손실을 냈다. 현금 70%를 들고 상승장을 이길 방법은 없다.

여기서는 **점수 상위 K개를 뽑아 항상 목표 익스포저를 채운다.** 모델이 해야 할 일이
"이 종목의 절대 상승확률이 0.6을 넘는가"에서 "지금 후보 중 어느 것이 가장 좋은가"로
바뀐다. 후자가 훨씬 쉬운 문제다. 확률 캘리브레이션이 틀려도 순위가 맞으면 동작하고,
이미 만들어 둔 횡단면 피처(`cs_*`)와 수급 피처가 정확히 순위를 매기는 도구다.

## 백테스트와 실거래가 같은 코드를 쓴다

`backtest/engine.py`와 `main.py`가 이 모듈의 같은 함수를 호출한다. 두 경로의 선정
로직이 갈라지면 백테스트 결과는 실거래를 예측하지 못하고, 그 순간 백테스트는
의미를 잃는다.

## 회전율 통제

순위는 매 봉 흔들린다. 순위가 바뀔 때마다 교체하면 왕복비용 0.278%가 알파를 먹는다.
그래서 `MIN_HOLDING_BARS`만큼은 순위가 밀려도 보유를 유지한다(잠금). 잠긴 포지션은
목표 슬롯을 차지하므로, 빈 슬롯만 새 후보로 채운다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from core.events import Portfolio, Signal
from config.settings import Settings, get_settings

# 이 값 아래의 레짐에서만 보유 종목 수를 절반으로 줄인다. 실측 `regime.scale` 분포는
# 중앙값 0.80 · 범위 0.17~1.00이고, 0.3 미만은 BEAR_HIGH_VOL 등 극단 구간에 해당한다.
_REGIME_CUT = 0.30


def target_count(settings: Settings, top_k: Optional[int] = None,
                 regime_scale: float = 1.0) -> int:
    """이번 봉에 보유하고자 하는 종목 수.

    ## 레짐을 곱하지 않는 이유 (겪은 버그)

    처음에는 `k * regime_scale`로 종목 수를 줄였다. 그런데 레짐은
    `RiskManager.evaluate_entry()`가 이미 **비중**에 곱한다
    (`target_w = vol_w × kelly_mult × signal_scale × signal.regime`).
    종목 수에서 또 곱하면 **이중 축소**가 된다.

    실측 `regime.scale` 중앙값이 0.80이라 5종목 의도가 4종목이 되고, 19.9% 구간에서는
    2종목 이하로 줄었다. 그 결과 실데이터 21구간 측정에서 평균 익스포저가 37.3% →
    **12.3%로 붕괴**하고 Sharpe가 −0.813이 됐다. 올리려고 만든 기능이 반대로 작동했다.

    그래서 종목 수는 레짐과 무관하게 유지하고, **아주 나쁜 레짐에서만 문턱값으로**
    절반으로 줄인다. 연속 곱셈은 쓰지 않는다.
    """
    k = top_k if top_k is not None else settings.MAX_POSITIONS
    k = min(int(k), int(settings.MAX_POSITIONS))
    if regime_scale < _REGIME_CUT:
        k = max(1, k // 2)
    return max(1, int(k))


def locked_symbols(portfolio: Portfolio, settings: Settings) -> Set[str]:
    """최소 보유 기간을 아직 채우지 못해 교체 대상에서 제외되는 종목."""
    floor = int(getattr(settings, "MIN_HOLDING_BARS", 0))
    if floor <= 0:
        return set()
    return {
        symbol for symbol, pos in portfolio.open_positions().items()
        if pos.bars_held < floor
    }


def desired_holdings(
    signals: Sequence[Signal],
    portfolio: Portfolio,
    settings: Optional[Settings] = None,
    top_k: Optional[int] = None,
    regime_scale: float = 1.0,
) -> List[str]:
    """이번 봉의 목표 보유 종목 목록(순서 = 선호도).

    두 단계로 정한다. 먼저 최소 보유 기간을 못 채운 종목이 슬롯을 차지하고, 남은
    슬롯을 점수 상위 후보로 채운다. 잠긴 종목 수가 목표 수를 넘으면 신규 진입은
    없다 — 강제로 팔 수 없는 포지션을 팔았다고 가정하면 백테스트가 거짓이 된다.

    `signals`가 점수 내림차순이라고 가정하지 않고 여기서 다시 정렬한다. 호출자가
    필터링하다 순서를 깨뜨릴 수 있다.
    """
    s = settings or get_settings()
    want = target_count(s, top_k, regime_scale)
    locked = locked_symbols(portfolio, s)
    ranked = sorted(signals, key=lambda x: x.score, reverse=True)

    # 1) 잠긴 포지션 — 점수 순서대로 나열하되, 이번 봉 신호가 없는 종목도 포함한다.
    chosen: List[str] = [sig.symbol for sig in ranked if sig.symbol in locked]
    chosen += [sym for sym in locked if sym not in chosen]

    # 2) 남은 슬롯을 점수 상위 후보로 채운다.
    for sig in ranked:
        if len(chosen) >= want:
            break
        if sig.symbol in chosen:
            continue
        # 매도 신호가 명확하면 순위가 높아도 담지 않는다.
        if sig.score <= s.SIGNAL_SELL_THRESHOLD:
            continue
        chosen.append(sig.symbol)

    return chosen


def select_entries(
    signals: Sequence[Signal],
    portfolio: Portfolio,
    settings: Optional[Settings] = None,
    top_k: Optional[int] = None,
    regime_scale: float = 1.0,
    pending: Optional[Set[str]] = None,
) -> List[Signal]:
    """신규 진입 대상 신호. 이미 보유 중이거나 주문이 떠 있는 종목은 제외한다."""
    s = settings or get_settings()
    want = desired_holdings(signals, portfolio, s, top_k, regime_scale)
    held = set(portfolio.open_positions())
    inflight = pending or set()

    by_symbol: Dict[str, Signal] = {sig.symbol: sig for sig in signals}
    out: List[Signal] = []
    for sym in want:
        if sym in held or sym in inflight:
            continue
        sig = by_symbol.get(sym)
        if sig is not None:
            out.append(sig)
    return out


def select_rotation_exits(
    signals: Sequence[Signal],
    portfolio: Portfolio,
    settings: Optional[Settings] = None,
    top_k: Optional[int] = None,
    regime_scale: float = 1.0,
) -> List[str]:
    """목표 보유 목록에서 밀려난 종목 — 순위 교체로 청산할 대상.

    손절·급락·시간손절과 달리 이것은 **더 좋은 후보가 나타났다**는 이유의 청산이다.
    `RiskManager.evaluate_exits()`가 만드는 청산과 중복되지 않도록, 호출자는 이미
    청산 예정인 종목을 걸러야 한다.
    """
    s = settings or get_settings()
    want = set(desired_holdings(signals, portfolio, s, top_k, regime_scale))
    locked = locked_symbols(portfolio, s)
    return [
        sym for sym in portfolio.open_positions()
        if sym not in want and sym not in locked
    ]
