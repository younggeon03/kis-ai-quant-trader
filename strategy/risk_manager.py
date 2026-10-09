"""리스크 관리 모듈 — 모든 주문의 단일 관문(Single Gate).

전략이 아무리 강한 신호를 내도 **최종 수량은 이 모듈이 결정**하며, 거부권(veto)을
가진다. 대회 심사의 Sharpe / MDD 항목은 사실상 이 모듈의 품질로 결정된다.

방어 계층
 1. 종목 단위 : 고정 손절, ATR 트레일링 스톱, 익절, 시간 손절, 급락 감지
 2. 포트폴리오: 일일 손실 한도, MDD 2단계(소프트 디리스킹 → 하드 킬스위치),
                총 익스포저 상한, 최대 보유 종목 수
 3. 분산      : 단일 종목 비중 상한, 섹터 비중 상한
 4. 사이징    : 변동성 타게팅 ∩ Fractional Kelly ∩ 비중 상한 ∩ 가용현금
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from config.settings import Settings, get_settings
from core.events import ExitReason, Fill, Portfolio, Position, RiskDecision, Side, Signal
from strategy.exit_rules import (RULE_CRASH, RULE_TRAILING, RULE_VOL_ALIVE,
                                 RULE_VOL_DECAY, DecayExitTracker)
from utils.krx_calendar import KST
from utils.logger import logger
from utils.market_rules import affordable_qty
from strategy.eligibility import is_buyable, position_weight_cap


class RiskState:
    NORMAL = "NORMAL"      # 정상 매매
    DERISK = "DERISK"      # 소프트 트리거: 신규 진입 금지 + 익스포저 축소
    HALT = "HALT"          # 하드 트리거: 전량 청산 + 쿨다운


@dataclass
class ExitOrder:
    symbol: str
    qty: int
    reason: ExitReason
    price: float = 0.0
    detail: str = ""


@dataclass
class RiskSnapshot:
    state: str = RiskState.NORMAL
    drawdown: float = 0.0
    daily_pnl: float = 0.0
    gross_exposure: float = 0.0
    positions: int = 0
    cooldown_until: Optional[str] = None
    blocked_count: int = 0
    kill_switch_count: int = 0
    kelly_weight: float = 0.0
    win_rate: float = 0.0
    payoff: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "drawdown_pct": round(self.drawdown * 100, 2),
            "daily_pnl_pct": round(self.daily_pnl * 100, 2),
            "gross_exposure": round(self.gross_exposure, 3),
            "positions": self.positions,
            "cooldown_until": self.cooldown_until,
            "blocked": self.blocked_count,
            "kill_switches": self.kill_switch_count,
            "kelly_weight": round(self.kelly_weight, 4),
            "win_rate": round(self.win_rate * 100, 1),
            "payoff": round(self.payoff, 2),
            "recent_reasons": self.reasons[-8:],
        }


# 감쇠 규칙 이름 → 청산 사유. `exit_rules`는 순수 로직이라 `ExitReason`을 모른다.
_DECAY_REASON = {
    RULE_CRASH: ExitReason.CRASH_GUARD,
    RULE_TRAILING: ExitReason.TRAILING_STOP,
    RULE_VOL_DECAY: ExitReason.VOL_DECAY,
    RULE_VOL_ALIVE: ExitReason.VOL_DECAY,   # 같은 성격(거래 소멸)이라 사유를 공유한다
}


class RiskManager:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or get_settings()
        self.state: str = RiskState.NORMAL
        self.cooldown_until: Optional[dt.datetime] = None
        self.blocked_count = 0
        self.kill_switch_count = 0
        self._recent_trades: Deque[float] = deque(maxlen=60)  # 수익률 기록(Kelly 추정)
        self._reasons: Deque[str] = deque(maxlen=50)
        # 감쇠 청산(거래량 감소·상승 감쇠·급락)은 `strategy/exit_rules.py`가 맡는다.
        # 규칙을 관문 코드 안에 두면 파인튜닝 때마다 모든 주문의 최종
        # 결정권자를 건드려야 한다. 상태(고점·거래대금 고점·가격 버퍼)도
        # 그쪽이 들고 있어 여기서 중복 추적하지 않는다.
        self.decay = DecayExitTracker(self.s)
        self._last_day: Optional[dt.date] = None
        self._halt_equity: float = 0.0
        self._phase = None   # 대회 국면(strategy/phase.PhaseConfig). None이면 평소 규칙

    # ================================================================== 상태 갱신
    def set_phase(self, phase) -> None:
        """대회 국면을 지정한다(`strategy/phase.PhaseConfig` 또는 None).

        1단계(초기화 창이 열린 5영업일)에는 손절과 MDD 게이트를 끈다. 근거는
        초기화 옵션이 하방을 완전히 덮는다는 것뿐이고, 창이 닫히면 그 근거가
        사라지므로 `phase.current_phase()`가 날짜만 보고 전환한다.
        """
        self._phase = phase

    @property
    def use_stop_loss(self) -> bool:
        return self._phase is None or bool(self._phase.use_stop_loss)

    @property
    def use_mdd_gate(self) -> bool:
        return self._phase is None or bool(self._phase.use_mdd_gate)

    def update(self, portfolio: Portfolio, now: Optional[dt.datetime] = None) -> str:
        """포트폴리오 상태를 반영해 리스크 상태를 갱신한다. 매 루프 1회 호출."""
        now = now or dt.datetime.now(tz=KST)

        # 날짜가 바뀌면 일일 기준선 리셋
        if self._last_day is None:
            self._last_day = now.date()
        elif now.date() != self._last_day:
            self._last_day = now.date()
            portfolio.start_new_day()
            # 새 거래일은 '일일 손실 한도'만 리셋한다. 낙폭은 여러 날에 걸친 지표이므로
            # 여전히 한도를 넘는 상태에서 날짜가 바뀌었다는 이유로 매매를 재개하면
            # HALT → NORMAL → HALT 가 매 봉 반복되며 킬스위치가 무력화된다.
            if self.state == RiskState.HALT and (self.cooldown_until is None or now >= self.cooldown_until):
                dd_now = portfolio.drawdown
                if dd_now < self.s.MDD_SOFT_LIMIT:
                    self._transition(RiskState.NORMAL, "새 거래일 시작 · 낙폭 해소로 매매 재개")
                elif dd_now < self.s.MDD_HARD_LIMIT:
                    self._transition(RiskState.DERISK, f"새 거래일이지만 낙폭 {dd_now:.2%} 잔존 → 축소 운용")
                # 하드 리밋 이상이면 HALT 유지

        # 1단계에서는 MDD 게이트를 쓰지 않는다. 날짜 기준선만 갱신하고 NORMAL을 유지한다.
        if not self.use_mdd_gate:
            if self.state != RiskState.NORMAL:
                self._transition(RiskState.NORMAL, "1단계 — MDD 게이트 해제(초기화 옵션이 하방을 덮는다)")
            return self.state

        dd = portfolio.drawdown
        daily = portfolio.daily_pnl_pct

        # 쿨다운 만료 처리
        if self.state == RiskState.HALT and self.cooldown_until and now >= self.cooldown_until:
            if dd < self.s.MDD_SOFT_LIMIT:
                self._transition(RiskState.NORMAL, f"쿨다운 종료(MDD {dd:.2%})")
            else:
                self._transition(RiskState.DERISK, f"쿨다운 종료했으나 MDD 잔존({dd:.2%})")

        # 하드 트리거
        if dd >= self.s.MDD_HARD_LIMIT and self.state != RiskState.HALT:
            self.kill_switch_count += 1
            self.cooldown_until = now + dt.timedelta(minutes=self.s.COOLDOWN_MINUTES)
            self._halt_equity = portfolio.equity
            self._transition(
                RiskState.HALT,
                f"MDD 하드 리밋 돌파 {dd:.2%} ≥ {self.s.MDD_HARD_LIMIT:.2%} → 전량 청산 + "
                f"{self.s.COOLDOWN_MINUTES}분 쿨다운",
            )
            return self.state

        # 일일 손실 한도 → 하드 트리거와 동일 취급(당일 매매 중단)
        if daily <= -self.s.DAILY_LOSS_LIMIT and self.state != RiskState.HALT:
            self.kill_switch_count += 1
            self.cooldown_until = now + dt.timedelta(minutes=self.s.COOLDOWN_MINUTES)
            self._transition(
                RiskState.HALT,
                f"일일 손실 한도 도달 {daily:.2%} ≤ -{self.s.DAILY_LOSS_LIMIT:.2%} → 당일 매매 중단",
            )
            return self.state

        # 소프트 트리거
        if self.state == RiskState.NORMAL and dd >= self.s.MDD_SOFT_LIMIT:
            self._transition(
                RiskState.DERISK,
                f"MDD 소프트 리밋 {dd:.2%} ≥ {self.s.MDD_SOFT_LIMIT:.2%} → 신규 진입 차단 + 익스포저 축소",
            )
        elif self.state == RiskState.DERISK and dd < self.s.MDD_SOFT_LIMIT * 0.6:
            self._transition(RiskState.NORMAL, f"낙폭 회복({dd:.2%}) → 정상 매매 재개")

        return self.state

    def _transition(self, new_state: str, reason: str) -> None:
        if new_state == self.state:
            return
        logger.warning(f"[Risk] 상태 전환 {self.state} → {new_state} :: {reason}")
        self.state = new_state
        self._reasons.append(f"{new_state}: {reason}")

    def track_price(self, symbol: str, price: float) -> None:
        """하위 호환용 — 가격 버퍼는 이제 `exit_rules`가 `check()`에서 직접 채운다.

        **여기서 또 채우면 안 된다.** 룩백 창이 두 배로 빨리 흘러 급락 감지가
        실제보다 짧은 구간만 보게 된다. 호출부를 지우지 않고 남겨 둔 것은
        외부에서 부르는 경로가 있을 때 조용히 깨지지 않게 하기 위해서다.
        """
        return None

    # ================================================================== 포지션 사이징
    _MIN_KELLY_SAMPLES = 12
    # 표본 부족 시 쓰는 배수. 설정(`KELLY_COLD_START_MULT`)으로 뺐다 — 대회처럼
    # 기간이 짧으면 Kelly를 추정할 거래 이력이 모이지 않으므로, 낮은 고정값은
    # 안전이 아니라 노이즈에 맞추는 것과 같다.
    _COLD_START_MULT = 0.40   # 설정이 없을 때의 폴백
    _MIN_KELLY_MULT = 0.15
    _MIN_BET = 0.12        # 베팅 사이즈 바닥값 — 신호가 임계값을 넘었으면 최소한은 잡는다

    def _kelly_stats(self) -> Tuple[Optional[float], float, float]:
        """(fractional Kelly 비중, 승률, 손익비). 표본이 적으면 첫 값이 None."""
        arr = np.array(self._recent_trades, dtype=float)
        if len(arr) < self._MIN_KELLY_SAMPLES:
            return None, 0.0, 0.0
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        p = float(len(wins) / len(arr))
        if len(wins) == 0 or len(losses) == 0:
            return None, p, 0.0
        avg_win = float(wins.mean())
        avg_loss = float(-losses.mean())
        b = avg_win / avg_loss if avg_loss > 1e-9 else 1.0
        kelly = (p - (1.0 - p) / b) if b > 0 else 0.0
        return float(max(0.0, kelly) * self.s.KELLY_FRACTION), p, float(b)

    def _kelly_multiplier(self) -> Tuple[float, float, float, float]:
        """(사이징 배수 0~1, 승률, 손익비, 켈리비중).

        Kelly를 '비중 상한'이 아니라 **full-size 대비 배수**로 쓴다. 이렇게 해야
        변동성 타게팅이 가려지지 않고 두 기준이 함께 작동한다.
        (상한으로 쓰면 표본이 적을 때 상수값이 항상 이겨 변동성 조절이 무력화된다.)
        """
        kelly, p, b = self._kelly_stats()
        if kelly is None:
            cold = float(getattr(self.s, "KELLY_COLD_START_MULT",
                                 self._COLD_START_MULT))
            return cold, p, b, 0.0
        mult = float(np.clip(kelly / max(1e-9, self.s.KELLY_MAX_WEIGHT), self._MIN_KELLY_MULT, 1.0))
        return mult, p, b, kelly

    def _volatility_weight(self, annual_vol: float) -> float:
        """변동성 타게팅 비중: 목표변동성 / 자산변동성."""
        vol = max(0.08, float(annual_vol) if np.isfinite(annual_vol) else 0.30)
        return float(self.s.TARGET_VOLATILITY / vol)

    # ================================================================== 진입 심사
    def evaluate_entry(
        self,
        signal: Signal,
        portfolio: Portfolio,
        now: Optional[dt.datetime] = None,
        pending: Optional[Dict[str, float]] = None,
    ) -> RiskDecision:
        """신규/추가 진입 심사. 거부 사유를 반드시 남긴다.

        `pending`은 아직 체결되지 않은 매수 주문의 평가금액({종목코드: 원})이다.
        이를 한도 계산에 넣지 않으면, 체결 전에 같은 섹터로 주문이 연달아 나가
        섹터·종목 비중 상한이 사후적으로 무너진다.
        """
        now = now or dt.datetime.now(tz=KST)
        pending = pending or {}

        if self.state != RiskState.NORMAL:
            return self._reject(f"리스크 상태 {self.state} - 신규 진입 차단")
        if self.cooldown_until and now < self.cooldown_until:
            left = (self.cooldown_until - now).total_seconds() / 60
            return self._reject(f"쿨다운 {left:.0f}분 남음")
        if signal.score < self.s.SIGNAL_BUY_THRESHOLD:
            return self._reject(f"신호 미달 {signal.score:.3f} < {self.s.SIGNAL_BUY_THRESHOLD}")

        # 대회 규정 관문 — 현재가 1,000원 미만, 발행주식수 10만주 미만, 관리종목,
        # KONEX 등은 주문 자체가 거부되거나 실격 사유가 된다. 전략 점수가 아무리
        # 높아도 여기를 통과하지 못하면 주문을 내지 않는다.
        ok, why = is_buyable(signal.symbol, signal.price, self.s)
        if not ok:
            return self._reject(f"대회 규정 — {why}")

        equity = portfolio.equity
        if equity <= 0:
            return self._reject("자산 0 이하")

        open_positions = portfolio.open_positions()
        already = signal.symbol in open_positions
        # 미체결 주문이 걸린 종목도 '보유 예정'으로 세어 한도를 지킨다
        slots_used = len(set(open_positions) | set(pending))
        if not already and signal.symbol not in pending and slots_used >= self.s.MAX_POSITIONS:
            return self._reject(f"최대 보유 종목 수 도달 ({slots_used}/{self.s.MAX_POSITIONS})")

        pending_total = sum(pending.values())

        # --- 익스포저 여유 (미체결 포함) ---
        gross = (portfolio.market_value + pending_total) / equity
        gross_room = self.s.MAX_GROSS_EXPOSURE - gross
        if gross_room <= 0.01:
            return self._reject(f"총 익스포저 상한 도달 ({gross:.1%}, 미체결 포함)")

        # --- 섹터 여유 (미체결 포함) ---
        sector = self.s.sector_of(signal.symbol)
        sector_w = portfolio.sector_weights().get(sector, 0.0)
        sector_w += sum(v for s_, v in pending.items() if self.s.sector_of(s_) == sector) / equity
        # 고정 비중(꼬리) 모드에서는 섹터 상한을 적용하지 않는다.
        #
        # 두 가지 이유다. 첫째, **대회 규정에 섹터 제한이 없다** — 단일 종목 50%만
        # 있다. 섹터 상한은 우리가 넣은 리스크 통제이고, 꼬리 전략에서는 그 통제가
        # 목표와 충돌한다.
        #
        # 둘째, **지금 섹터 분류는 신뢰할 수 없다.** `sector_of()`는 `SECTOR_MAP`에
        # 없는 종목을 시장명으로 폴백하는데, 284종목 유니버스에서 **242종목(85.2%)이
        # "KOSPI"/"KOSDAQ" 하나로 묶인다.** 상한 35%를 적용하면 그 242종목 합계가
        # 35%에서 잘려 익스포저가 구조적으로 35%를 못 넘는다. 그 상태로 측정하면
        # "유니버스 확장이 효과 없다"는 틀린 결론이 나온다.
        #
        # 대가는 분명하다 — 같은 테마 두 종목에 90%가 들어가면 테마가 꺾일 때
        # 전량이 함께 맞는다. 30거래일에 회복 시간은 없다. 그것이 꼬리를 사는 값이다.
        # (업종 정보를 `universe.json`에 채우는 것은 대회 후 작업으로 남겼다.)
        if self.s.SIZING_MODE == "fixed":
            sector_room = float("inf")
        else:
            sector_room = self.s.MAX_SECTOR_WEIGHT - sector_w
            if sector_room <= 0.01:
                return self._reject(f"섹터({sector}) 비중 상한 도달 {sector_w:.1%}")

        # --- 단일 종목 여유 (미체결 포함) ---
        current_w = (open_positions[signal.symbol].market_value / equity) if already else 0.0
        current_w += pending.get(signal.symbol, 0.0) / equity
        # 대회 규정 단일 종목 한도(50%)와 설정값 중 더 작은 쪽을 쓴다.
        # 설정을 실수로 올려도 규정을 넘지 않게 하는 안전장치다.
        weight_cap = position_weight_cap(self.s)
        symbol_room = weight_cap - current_w
        if symbol_room <= 0.01:
            return self._reject(f"종목 비중 상한 도달 {current_w:.1%} (미체결 포함)")

        # --- 사이징: 변동성 타게팅 × 켈리 배수 × 베팅 사이즈 × 레짐, 그리고 모든 상한 ---
        vol_w = self._volatility_weight(signal.volatility)
        kelly_mult, win_rate, payoff, kelly_w = self._kelly_multiplier()

        # 베팅 사이즈는 캘리브레이션된 상승확률에서 나온다.
        #   z = (p − 0.5)/sqrt(p(1−p)),  size = 2·Φ(z) − 1
        # 확률 0.51과 0.95를 같게 취급하던 이전 선형 스케일을 대체한다.
        # 신호가 임계값을 넘었는데 베팅이 0에 가까우면 최소 바닥값을 적용해
        # "진입은 하되 아주 작게" 가 되도록 한다.
        if signal.bet == 0.0:
            # 베팅 사이즈 0은 "들어가지 말라"는 뜻이다. 바닥값으로 덮으면 차단이 무력해진다.
            return self._reject("베팅 사이즈 0 — 모델이 진입을 차단")
        if signal.bet > 0.0:
            signal_scale = float(np.clip(signal.bet, self._MIN_BET, 1.0))
        else:
            # 모델 미로딩(bet=-1): 기존 점수 기반 선형 스케일로 폴백한다
            strength = float(np.clip((signal.score - self.s.SIGNAL_BUY_THRESHOLD) /
                                     max(1e-6, 1.0 - self.s.SIGNAL_BUY_THRESHOLD), 0.0, 1.0))
            signal_scale = 0.6 + 0.4 * strength

        # --- 고정 비중 모드 (꼬리 전략) ---
        #
        # 변동성 타게팅은 **위험을 고르게 나누는** 설계다. 대회 심사는 절대수익
        # top10이라 보상이 계단함수이고, 그때 최적화 대상은 `E[R]`이 아니라
        # `P(R > 임계)`다. 두 목적의 최적해는 다르다.
        #
        # 실측으로 드러난 문제: Top-K 2종목 설정에서 평균 익스포저가 **5.8%**였다.
        # 역산하면 `vol_w(0.34) × kelly(0.40) × 신호(~0.3) × 레짐(~0.8) ≈ 3.3%`,
        # 두 종목이면 6.6%로 관측값과 맞는다. **상한 18%·섹터 35%는 닿지도 않았다.**
        # 즉 "2종목에 50%씩"을 `TARGET_VOLATILITY`를 올려 만들려면 그 값이 2.5 같은
        # 비현실적 숫자가 되어야 한다. 공식과 싸우는 대신 모드를 따로 둔다.
        #
        # 고정 모드는 종목당 `FIXED_POSITION_WEIGHT`를 목표로 하고, 규정·섹터·총
        # 익스포저·가용현금 상한만 적용한다. 변동성·켈리·레짐을 곱하지 않는다 —
        # 그것들이 곱해지는 순간 다시 3%가 된다.
        if self.s.SIZING_MODE == "fixed":
            target_w = float(self.s.FIXED_POSITION_WEIGHT)
            # 베팅 사이즈 0(= 모델의 진입 차단)은 위에서 이미 걸렀으므로 여기서는
            # 신호 강도를 비중에 반영하지 않는다. 꼬리를 노릴 때 "조금 덜 확신하니
            # 조금만 담는다"는 중간 지대가 기대값은 지키지만 꼬리를 깎는다.
        else:
            target_w = vol_w * kelly_mult * signal_scale * signal.regime
        target_w = min(target_w, weight_cap, symbol_room, sector_room, gross_room)
        if target_w <= 0.005:
            return self._reject(f"산출 비중 과소 {target_w:.3%}")

        price = signal.price
        if price <= 0:
            return self._reject("가격 정보 없음")

        budget = equity * target_w
        free_cash = max(0.0, portfolio.cash - pending_total)
        qty = int(min(budget // price, affordable_qty(free_cash, price, self.s.FEE_BPS)))
        if qty < 1:
            return self._reject(
                f"수량 0 (예산 {budget:,.0f}원, 현금 {portfolio.cash:,.0f}원, 가격 {price:,.0f}원)"
            )
        # 최소 주문금액(자투리 방지)
        if qty * price < max(100_000.0, equity * 0.005):
            return self._reject(f"주문금액 과소 {qty * price:,.0f}원")

        stop_price = self.initial_stop(price, signal.atr)
        return RiskDecision(
            approved=True,
            qty=qty,
            weight=target_w,
            stop_price=stop_price,
            reason=(
                f"vol_w={vol_w:.3f} kelly_mult={kelly_mult:.2f} bet={signal_scale:.2f} "
                f"→ w={target_w:.3f} qty={qty}"
            ),
            detail={
                "vol_weight": round(vol_w, 4),
                "kelly_mult": round(kelly_mult, 4),
                "bet_size": round(signal_scale, 4),
                "model_proba": round(signal.model_proba, 4),
                "kelly_weight": round(kelly_w, 4),
                "win_rate": round(win_rate, 3),
                "payoff": round(payoff, 3),
                "sector": sector,
                "sector_weight": round(sector_w, 4),
                "target_weight": round(target_w, 4),
                "stop_price": stop_price,
            },
        )

    def initial_stop(self, price: float, atr: float) -> float:
        """고정 손절과 ATR 손절 중 합리적인 쪽을 선택(과도하게 좁은 손절 방지)."""
        pct_stop = price * (1 - self.s.STOP_LOSS_PCT)
        atr_stop = price - 1.2 * atr if atr > 0 else pct_stop
        stop = min(pct_stop, atr_stop)
        floor = price * (1 - self.s.STOP_LOSS_PCT * 2.0)  # 너무 넓어지지 않도록 하한
        return float(max(stop, floor))

    def _reject(self, reason: str) -> RiskDecision:
        self.blocked_count += 1
        self._reasons.append(f"REJECT: {reason}")
        return RiskDecision(approved=False, qty=0, reason=reason)

    # ================================================================== 청산 심사
    def evaluate_exits(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
        atr_map: Optional[Dict[str, float]] = None,
        signals: Optional[Dict[str, float]] = None,
        now: Optional[dt.datetime] = None,
        turnover_map: Optional[Dict[str, float]] = None,
        burst_map: Optional[Dict[str, float]] = None,
    ) -> List[ExitOrder]:
        """보유 포지션에 대한 청산 주문 목록을 생성한다.

        우선순위: **킬스위치 > 손절 > 감쇠(`EXIT_RULES` 순서) > 익절 > 시간손절
        > 신호청산 > 디리스킹**

        감쇠 규칙(급락·상승 감쇠·거래량 감소)은 `strategy/exit_rules.py`가 맡고
        순서는 `EXIT_RULES` 설정이 정한다. 분리 전에는 급락이 손절보다 먼저였는데
        **지금은 손절이 먼저다.** 급락 임계값(기본 4.5%, catalyst 30%)이 손절
        (12%)보다 크면 손절이 먼저 걸리므로 사유 라벨이 CRASH_GUARD에서
        STOP_LOSS로 바뀐다. 청산 자체는 어느 쪽이든 일어나므로 수익에는 영향이
        없지만 **사유별 통계는 달라진다.**

        `turnover_map`은 거래량 감소 규칙(`vol_decay`)의 입력이다. 넘기지 않으면
        그 규칙만 조용히 비활성화된다 — 백테스트와 실거래 **양쪽 호출부에서
        같이 넘겨야 한다**(한쪽만 넘기면 두 경로가 갈린다).
        """
        atr_map = atr_map or {}
        signals = signals or {}
        turnover_map = turnover_map or {}
        # `vol_alive` 입력 — 거래대금 / 20봉 중앙값. **넘기지 않으면 그 규칙만
        # 조용히 비활성화된다**(배수 0은 "모른다"로 처리한다). 백테스트와 실거래
        # 양쪽 호출부에서 같이 넘겨야 한다.
        burst_map = burst_map or {}
        exits: List[ExitOrder] = []
        open_positions = portfolio.open_positions()

        # **보유하지 않는 종목의 감쇠 상태를 버린다.** 남겨 두면 같은 종목에 다시
        # 진입할 때 묵은 거래대금 고점이 그대로 살아 있어 `vol_decay`가 첫 봉에서
        # 바로 걸린다(고점은 과거 폭증일, 현재는 평상 거래량이므로 비율이 낮다).
        for sym in list(self.decay.states):
            if sym not in open_positions:
                self.decay.close(sym)

        # --- 하드 킬스위치: 전량 청산 ---
        if self.state == RiskState.HALT:
            for symbol, pos in open_positions.items():
                exits.append(
                    ExitOrder(symbol, pos.qty, ExitReason.KILL_SWITCH,
                              prices.get(symbol, pos.last_price),
                              f"킬스위치 발동(MDD {portfolio.drawdown:.2%})")
                )
            if exits:
                logger.error(f"[Risk] 킬스위치 - 전 종목 청산 {len(exits)}건")
            return exits

        for symbol, pos in open_positions.items():
            price = prices.get(symbol, pos.last_price)
            if price <= 0:
                continue
            pos.mark(price)
            atr = atr_map.get(symbol, 0.0)

            # 감쇠 규칙 평가. **판정보다 먼저, 매 봉 한 번만** 부른다 —
            # `check()`가 고점·거래대금 고점·가격 버퍼를 갱신하므로 건너뛰면
            # 되돌림 기준이 멈추고, 두 번 부르면 룩백 창이 두 배로 빨리 흐른다.
            decay = self.decay.check(symbol, price,
                                     turnover=turnover_map.get(symbol, 0.0),
                                     atr=atr, entry=pos.avg_price,
                                     burst=burst_map.get(symbol, 0.0))
            # 트레일링이 제시한 손절선을 단방향으로만 올린다(내리지 않는다).
            trail = self.decay.trail_stop(symbol, atr)
            if trail > 0:
                pos.stop_price = max(pos.stop_price, trail)
                if not pos.trailing_active:
                    pos.trailing_active = True
                    self._reasons.append(f"TRAIL_ON: {symbol} @{pos.pnl_pct:.2%}")

            # 1단계(초기화 옵션이 하방을 덮는 구간)에서는 **손절만** 건너뛴다.
            #
            # 감쇠 청산(`EXIT_RULES`)은 리스크 손절이 아니라 **전략의 기본 청산**이라
            # 1단계에도 돈다. 예전에는 이 분기가 감쇠까지 건너뛰어 1주차가
            # "2종목을 사서 30봉 상한까지 보유"가 됐다 — 측정한 전략(`vol_alive`
            # 상태 기반 보유)과 다른 것이고, 백테스트에는 대회 날짜가 없어 어떤
            # 측정에도 나타나지 않았다.
            if not self.use_stop_loss:
                if decay is not None:
                    exits.append(ExitOrder(symbol, pos.qty, _DECAY_REASON[decay.rule],
                                           price, decay.reason))
                    continue
                if pos.bars_held >= self.s.MAX_HOLDING_BARS:
                    exits.append(ExitOrder(symbol, pos.qty, ExitReason.TIME_STOP, price,
                                           f"보유 {pos.bars_held}봉 초과"))
                continue

            # 1) 고정 손절
            if pos.stop_price > 0 and price <= pos.stop_price:
                exits.append(ExitOrder(symbol, pos.qty, ExitReason.STOP_LOSS, price,
                                       f"손절선 {pos.stop_price:,.0f} 이탈"))
                continue
            if pos.pnl_pct <= -self.s.STOP_LOSS_PCT:
                exits.append(ExitOrder(symbol, pos.qty, ExitReason.STOP_LOSS, price,
                                       f"손실률 {pos.pnl_pct:.2%}"))
                continue

            # 2) 감쇠 청산 — 거래량 감소 · 상승 감쇠 · 급격한 하락
            if decay is not None:
                exits.append(ExitOrder(symbol, pos.qty, _DECAY_REASON[decay.rule],
                                       price, decay.reason))
                continue

            # 4) 익절
            if self.s.TAKE_PROFIT_PCT > 0 and pos.pnl_pct >= self.s.TAKE_PROFIT_PCT:
                exits.append(ExitOrder(symbol, pos.qty, ExitReason.TAKE_PROFIT, price,
                                       f"목표 수익 {pos.pnl_pct:.2%}"))
                continue

            # 5) 시간 손절
            if pos.bars_held >= self.s.MAX_HOLDING_BARS:
                exits.append(ExitOrder(symbol, pos.qty, ExitReason.TIME_STOP, price,
                                       f"보유 {pos.bars_held}봉 초과"))
                continue

            # 6) 신호 반전
            score = signals.get(symbol)
            if score is not None and score <= self.s.SIGNAL_SELL_THRESHOLD:
                exits.append(ExitOrder(symbol, pos.qty, ExitReason.SIGNAL, price,
                                       f"신호 반전 {score:.3f}"))

        # --- 소프트 디리스킹: 익스포저를 목표 비율까지 축소 ---
        if self.state == RiskState.DERISK:
            exits.extend(self._derisk_orders(portfolio, prices, {e.symbol for e in exits}))

        return exits

    def _derisk_orders(
        self, portfolio: Portfolio, prices: Dict[str, float], already: set
    ) -> List[ExitOrder]:
        """손실이 큰 순서로 익스포저를 `DERISK_RATIO`까지 줄인다."""
        equity = portfolio.equity
        if equity <= 0:
            return []
        target_mv = equity * self.s.MAX_GROSS_EXPOSURE * self.s.DERISK_RATIO
        current_mv = portfolio.market_value
        to_reduce = current_mv - target_mv
        if to_reduce <= equity * 0.01:
            return []

        orders: List[ExitOrder] = []
        candidates = sorted(
            (p for s, p in portfolio.open_positions().items() if s not in already),
            key=lambda p: p.pnl_pct,
        )
        for pos in candidates:
            if to_reduce <= 0:
                break
            price = prices.get(pos.symbol, pos.last_price)
            if price <= 0:
                continue
            qty = min(pos.qty, max(1, int(math.ceil(to_reduce / price))))
            orders.append(ExitOrder(pos.symbol, qty, ExitReason.DERISK, price,
                                    f"디리스킹(현재 익스포저 {portfolio.gross_exposure:.1%})"))
            to_reduce -= qty * price
        if orders:
            logger.warning(f"[Risk] 디리스킹 청산 {len(orders)}건 (목표 익스포저 "
                           f"{self.s.MAX_GROSS_EXPOSURE * self.s.DERISK_RATIO:.0%})")
        return orders

    def liquidate_all(self, portfolio: Portfolio, prices: Dict[str, float],
                      reason: ExitReason = ExitReason.EOD) -> List[ExitOrder]:
        """전 종목 청산 주문(장 마감/시스템 종료 시퀀스)."""
        return [
            ExitOrder(s, p.qty, reason, prices.get(s, p.last_price), "전량 청산")
            for s, p in portfolio.open_positions().items()
        ]

    # ================================================================== 체결 반영
    def on_fill(self, fill: Fill, realized_return: Optional[float] = None) -> None:
        """매도 체결의 수익률을 Kelly 추정 표본에 반영한다."""
        if fill.side is Side.SELL and realized_return is not None:
            self._recent_trades.append(float(realized_return))

    def apply_decision_to_position(self, position: Position, decision: RiskDecision) -> None:
        """진입 승인 시 계산된 손절선을 포지션에 기록."""
        if decision.approved and decision.stop_price > 0:
            position.stop_price = max(position.stop_price, decision.stop_price) \
                if position.stop_price > 0 else decision.stop_price

    def force_halt(self, reason: str, minutes: Optional[int] = None,
                   now: Optional[dt.datetime] = None) -> None:
        """외부(엔진/운영자)에서 강제로 킬스위치를 발동한다."""
        now = now or dt.datetime.now(tz=KST)
        self.cooldown_until = now + dt.timedelta(minutes=minutes or self.s.COOLDOWN_MINUTES)
        self.kill_switch_count += 1
        self._transition(RiskState.HALT, f"강제 중단: {reason}")

    def reset(self) -> None:
        self.state = RiskState.NORMAL
        self.cooldown_until = None
        self.decay.reset()

    # ================================================================== 스냅샷
    def snapshot(self, portfolio: Optional[Portfolio] = None) -> Dict[str, Any]:
        _, win_rate, payoff, kelly_w = self._kelly_multiplier()
        snap = RiskSnapshot(
            state=self.state,
            drawdown=portfolio.drawdown if portfolio else 0.0,
            daily_pnl=portfolio.daily_pnl_pct if portfolio else 0.0,
            gross_exposure=portfolio.gross_exposure if portfolio else 0.0,
            positions=len(portfolio.open_positions()) if portfolio else 0,
            cooldown_until=self.cooldown_until.isoformat() if self.cooldown_until else None,
            blocked_count=self.blocked_count,
            kill_switch_count=self.kill_switch_count,
            kelly_weight=kelly_w,
            win_rate=win_rate,
            payoff=payoff,
            reasons=list(self._reasons),
        )
        return snap.to_dict()
