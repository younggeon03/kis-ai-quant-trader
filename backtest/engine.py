"""이벤트 기반 백테스트 엔진.

원칙
 1. **실매매와 같은 객체를 쓴다.** 전략(`EnsembleAlphaStrategy`)과 리스크
    매니저(`RiskManager`)를 그대로 주입하므로 "백테스트에선 되는데 실전에선
    다른" 불일치가 없다.
 2. **룩어헤드 차단.** 봉 t의 종가까지로 판단하고 체결은 봉 t+1의 시가에서 이뤄진다.
 3. **비용 반영.** 수수료·증권거래세·슬리피지를 모두 차감한다.
 4. **갭 인식 손절.** 봉 내부 저가가 손절선을 깨면 그 봉에서 청산한다
    (갭하락은 손절가가 아니라 시가에 체결).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.metrics import PerformanceMetrics, compute_metrics, equity_curve_frame
from config.settings import Settings, get_settings
from core.events import ExitReason, Fill, Portfolio, Side
from strategy.ai_model import EnsembleAlphaStrategy, RegimeState, regime_series
from strategy.base import AbstractStrategy
from strategy.risk_manager import ExitOrder, RiskManager
from strategy.selection import select_entries, select_rotation_exits
from utils.logger import logger
from utils.market_rules import buy_cost, round_to_tick, sell_cost


def align_panel(panel: Dict[str, pd.DataFrame],
                min_bars: int = 0) -> Dict[str, pd.DataFrame]:
    """모든 종목을 공통 `ts`로 정렬해 인덱스를 일치시킨다.

    ## `min_bars`가 필요한 이유

    이 함수는 공통 날짜의 **교집합**을 잡는다. 따라서 상장이 늦은 종목 하나가
    전체 패널을 그 종목의 이력만큼으로 잘라낸다. 실제로 40종목에서 LG에너지솔루션
    (2022년 상장) 때문에 1,499봉이 1,138봉으로 줄었고, 2026년 신규 상장 종목이
    섞이면 패널이 수십 봉으로 붕괴한다.

    `min_bars` 미만인 종목은 교집합 계산 **전에** 제외한다. 학습창이 500봉이므로
    그보다 짧은 이력은 어차피 쓸 수 없다.

    ## 하한을 데이터 길이에 맞춰 낮추는 이유

    `min_bars`를 그대로 적용하면 짧은 패널(합성 데이터 400봉, `--days 300` 백테스트)에서
    **모든 종목이 걸러져 "데이터가 없다"로 끝난다.** 그래서 실제 적용 하한은
    `min(min_bars, 중앙값 × 0.9)`로 둔다. 이렇게 하면

    - 실데이터 1,499봉 + `min_bars=750` → 750봉 미만 제외 (의도대로)
    - 합성 400봉 + `min_bars=750` → 360봉 미만 제외 (아무것도 안 걸러짐)

    즉 "다른 종목보다 유독 짧은 종목"만 걸러내고, 패널 전체가 짧은 경우는 건드리지 않는다.
    """
    frames = {s: df for s, df in panel.items() if df is not None and len(df) > 0}
    if not frames:
        return {}

    if min_bars > 0 and len(frames) > 1:
        lengths = sorted(len(df) for df in frames.values())
        median_bars = lengths[len(lengths) // 2]
        floor = min(int(min_bars), int(median_bars * 0.9))
        short = {s: len(df) for s, df in frames.items() if len(df) < floor}
        if short:
            frames = {s: df for s, df in frames.items() if s not in short}
            sample = ", ".join(f"{s}({n}봉)" for s, n in
                               sorted(short.items(), key=lambda kv: kv[1])[:5])
            logger.warning(
                f"[Panel] 이력 {floor}봉 미만 {len(short)}종목 제외 "
                f"(설정 {min_bars} · 중앙값 {median_bars}) — {sample}"
                + (f" 외 {len(short) - 5}종목" if len(short) > 5 else ""))
        if not frames:
            return {}

    common: Optional[set] = None
    for df in frames.values():
        ts = set(df["ts"].astype(str))
        common = ts if common is None else (common & ts)
    if not common:
        return {}
    ordered = sorted(common)
    aligned: Dict[str, pd.DataFrame] = {}
    for symbol, df in frames.items():
        sub = df.copy()
        sub["ts"] = sub["ts"].astype(str)
        sub = sub[sub["ts"].isin(common)].drop_duplicates(subset="ts")
        sub = sub.set_index("ts").reindex(ordered).reset_index()
        aligned[symbol] = sub

    # 교집합이 개별 종목 이력보다 크게 짧아지면 조용히 넘기지 않는다.
    # 원인을 모르면 "왜 백테스트 구간이 이것밖에 안 되지"로 시간을 잃는다.
    longest = max(len(df) for df in frames.values())
    if len(ordered) < longest * 0.6:
        culprit = min(frames.items(), key=lambda kv: kv[1]["ts"].astype(str).min())
        logger.warning(
            f"[Panel] 공통 구간 {len(ordered)}봉이 최장 종목 {longest}봉의 "
            f"{len(ordered) / longest:.0%}에 불과합니다. 상장이 늦은 종목이 구간을 "
            f"자르고 있습니다(가장 늦은 시작: {culprit[0]} "
            f"{culprit[1]['ts'].astype(str).min()}). min_bars를 올려 보십시오")
    return aligned


@dataclass
class TradeRecord:
    ts: str
    symbol: str
    side: str
    qty: int
    price: float
    fee: float
    tax: float
    reason: str
    pnl: float = 0.0
    equity_after: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts, "symbol": self.symbol, "side": self.side, "qty": self.qty,
            "price": round(self.price, 1), "fee": round(self.fee, 1), "tax": round(self.tax, 1),
            "reason": self.reason, "pnl": round(self.pnl, 0), "equity": round(self.equity_after, 0),
        }


@dataclass
class BacktestResult:
    metrics: PerformanceMetrics
    curve: pd.DataFrame
    trades: List[TradeRecord] = field(default_factory=list)
    regime_counts: Dict[str, int] = field(default_factory=dict)
    risk_events: List[str] = field(default_factory=list)
    rejected: int = 0

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def summary(self) -> str:
        body = self.metrics.pretty()
        regimes = ", ".join(f"{k}:{v}" for k, v in sorted(self.regime_counts.items(),
                                                          key=lambda kv: -kv[1]))
        extra = [
            f" 레짐 분포       : {regimes}",
            f" 리스크 거부     : {self.rejected:,}건",
        ]
        if self.risk_events:
            extra.append(f" 리스크 이벤트   : {len(self.risk_events)}건 (최근: {self.risk_events[-1]})")
        return body + "\n" + "\n".join(extra)


class BacktestEngine:
    def __init__(
        self,
        panel: Dict[str, pd.DataFrame],
        strategy: Optional[AbstractStrategy] = None,
        risk_manager: Optional[RiskManager] = None,
        settings: Optional[Settings] = None,
        initial_capital: Optional[float] = None,
        flow_panel: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.s = settings or get_settings()
        self.panel = align_panel(panel, self.s.MIN_HISTORY_BARS)
        self.flow_panel = flow_panel or {}
        self.strategy = strategy or EnsembleAlphaStrategy(self.s)
        self.risk = risk_manager or RiskManager(self.s)
        self.initial_capital = float(initial_capital or self.s.INITIAL_CAPITAL)
        self.portfolio = Portfolio(cash=self.initial_capital, initial_capital=self.initial_capital)
        self.trades: List[TradeRecord] = []
        self.turnover = 0.0
        self.rejected = 0

    # ------------------------------------------------------------------ 체결
    def _buy(self, symbol: str, qty: int, raw_price: float, ts: str, reason: str) -> Optional[Fill]:
        price = round_to_tick(raw_price * (1 + self.s.SLIPPAGE_BPS / 10_000.0))
        if price <= 0 or qty <= 0:
            return None
        fee = buy_cost(qty, price, self.s.FEE_BPS)
        need = qty * price + fee
        if need > self.portfolio.cash:
            qty = int((self.portfolio.cash - 1) // (price * (1 + self.s.FEE_BPS / 10_000.0)))
            if qty <= 0:
                return None
            fee = buy_cost(qty, price, self.s.FEE_BPS)
        fill = Fill(symbol=symbol, side=Side.BUY, qty=qty, price=float(price), fee=fee,
                    tax=0.0, reason=reason)
        self.portfolio.apply_fill(fill, sector=self.s.sector_of(symbol))
        self.turnover += qty * price
        self.trades.append(TradeRecord(ts, symbol, "BUY", qty, float(price), fee, 0.0, reason,
                                       0.0, self.portfolio.equity))
        return fill

    def _sell(self, symbol: str, qty: int, raw_price: float, ts: str, reason: str) -> Optional[Fill]:
        pos = self.portfolio.positions.get(symbol)
        if pos is None or pos.qty <= 0:
            return None
        qty = min(qty, pos.qty)
        price = round_to_tick(raw_price * (1 - self.s.SLIPPAGE_BPS / 10_000.0))
        if price <= 0 or qty <= 0:
            return None
        fee, tax = sell_cost(qty, price, self.s.FEE_BPS, self.s.TAX_BPS)
        entry = pos.avg_price
        fill = Fill(symbol=symbol, side=Side.SELL, qty=qty, price=float(price), fee=fee,
                    tax=tax, reason=reason)
        realized = self.portfolio.apply_fill(fill, sector=self.s.sector_of(symbol))
        self.turnover += qty * price
        if entry > 0:
            self.risk.on_fill(fill, (price - entry) / entry)
        self.trades.append(TradeRecord(ts, symbol, "SELL", qty, float(price), fee, tax, reason,
                                       realized, self.portfolio.equity))
        return fill

    # ------------------------------------------------------------------ 실행
    def _rotation_exits(self, signals, prices, regime, already) -> List[ExitOrder]:
        """순위에서 밀려난 보유 종목을 교체 청산한다.

        손절·급락과 달리 이것은 "더 좋은 후보가 나타났다"는 이유의 청산이다.
        이미 다른 사유로 청산 예정인 종목은 중복해서 팔지 않는다.
        """
        taken = {order.symbol for order in already}
        out: List[ExitOrder] = []
        for symbol in select_rotation_exits(signals, self.portfolio, self.s,
                                            top_k=self.s.MAX_POSITIONS,
                                            regime_scale=regime.scale):
            if symbol in taken:
                continue
            pos = self.portfolio.positions.get(symbol)
            if pos is None or pos.qty <= 0:
                continue
            out.append(ExitOrder(symbol, pos.qty, ExitReason.SIGNAL,
                                 prices.get(symbol, pos.last_price),
                                 "순위 이탈 — 더 좋은 후보로 교체"))
        return out

    def run(self, warmup: Optional[int] = None, verbose: bool = True) -> BacktestResult:
        if not self.panel:
            raise ValueError("백테스트할 데이터가 없습니다.")

        warmup = warmup or max(self.strategy.warmup_bars(), 100)
        n = min(len(df) for df in self.panel.values())
        if n <= warmup + 10:
            raise ValueError(f"데이터 길이 부족: {n}봉 (워밍업 {warmup}봉 필요)")

        if verbose:
            logger.info(f"[Backtest] 피처 사전계산 시작 ({len(self.panel)}종목 × {n}봉)")
        featured = (
            self.strategy.precompute(self.panel, self.flow_panel)
            if hasattr(self.strategy, "precompute")
            else {}
        )
        regimes: List[RegimeState] = regime_series(self.panel)
        if not regimes:
            regimes = [RegimeState()] * n

        symbols = list(self.panel.keys())
        closes = {s: self.panel[s]["close"].to_numpy(dtype=float) for s in symbols}
        opens = {s: self.panel[s]["open"].to_numpy(dtype=float) for s in symbols}
        lows = {s: self.panel[s]["low"].to_numpy(dtype=float) for s in symbols}
        ts_list = self.panel[symbols[0]]["ts"].astype(str).tolist()
        atrs = {
            s: (featured[s]["atr_abs"].to_numpy(dtype=float)
                if s in featured and "atr_abs" in featured[s].columns
                else np.zeros(n))
            for s in symbols
        }

        equity_curve: List[float] = []
        cash_curve: List[float] = []
        exposure_curve: List[float] = []
        curve_ts: List[str] = []
        regime_counts: Dict[str, int] = {}

        base_date = dt.date(2024, 1, 1)

        for i in range(warmup, n - 1):
            nxt = i + 1
            now = dt.datetime.combine(base_date + dt.timedelta(days=i), dt.time(15, 30))

            # --- 현재 봉 종가로 평가 ---
            prices = {s: float(closes[s][i]) for s in symbols if np.isfinite(closes[s][i])}
            self.portfolio.mark_prices(prices)
            for s, p in prices.items():
                self.risk.track_price(s, p)

            regime = regimes[i] if i < len(regimes) else RegimeState()
            regime_counts[regime.label] = regime_counts.get(regime.label, 0) + 1

            # --- 리스크 상태 갱신(날짜 변경 시 bars_held/일일기준 자동 갱신) ---
            self.risk.update(self.portfolio, now)

            # --- 신호 생성 ---
            signals = (
                self.strategy.signals_at(featured, i, regime=regime, ts=now)
                if hasattr(self.strategy, "signals_at") and featured
                else self.strategy.generate_signals(
                    {s: df.iloc[: i + 1] for s, df in self.panel.items()}, ts=now
                )
            )
            score_map = {sig.symbol: sig.score for sig in signals}
            atr_map = {s: float(atrs[s][i]) if np.isfinite(atrs[s][i]) else 0.0 for s in symbols}

            # --- 청산 심사 → 다음 봉 시가 체결 ---
            exits = self.risk.evaluate_exits(self.portfolio, prices, atr_map, score_map, now)
            if self.s.USE_TOP_K:
                exits.extend(self._rotation_exits(signals, prices, regime, exits))
            for order in exits:
                open_px = float(opens[order.symbol][nxt])
                if np.isfinite(open_px) and open_px > 0:
                    self._sell(order.symbol, order.qty, open_px, ts_list[nxt],
                               f"{order.reason.value}|{order.detail}")

            # --- 진입 심사 → 다음 봉 시가 체결 ---
            # Top-K 모드는 절대 임계값 대신 점수 순위로 후보를 고른다. 임계값 방식은
            # 넘는 종목이 없으면 현금을 들기 때문에 상승장 참여율이 구조적으로 낮다.
            if self.s.USE_TOP_K:
                candidates = select_entries(signals, self.portfolio, self.s,
                                            top_k=self.s.MAX_POSITIONS,
                                            regime_scale=regime.scale)
            else:
                candidates = [x for x in signals
                              if x.score >= self.s.SIGNAL_BUY_THRESHOLD]
            for sig in candidates:
                decision = self.risk.evaluate_entry(sig, self.portfolio, now)
                if not decision.approved:
                    self.rejected += 1
                    continue
                open_px = float(opens[sig.symbol][nxt])
                if not np.isfinite(open_px) or open_px <= 0:
                    continue
                fill = self._buy(sig.symbol, decision.qty, open_px, ts_list[nxt],
                                 f"ENTRY|score={sig.score:.3f}")
                if fill:
                    pos = self.portfolio.positions.get(sig.symbol)
                    if pos is not None:
                        # 손절선은 실제 체결가 기준으로 재산정
                        pos.stop_price = self.risk.initial_stop(fill.price, sig.atr)

            # --- 갭/장중 손절: 다음 봉 저가가 손절선을 깼는지 확인 ---
            for symbol, pos in list(self.portfolio.open_positions().items()):
                if pos.stop_price <= 0:
                    continue
                low_px = float(lows[symbol][nxt])
                if np.isfinite(low_px) and low_px <= pos.stop_price:
                    open_px = float(opens[symbol][nxt])
                    exec_px = min(open_px, pos.stop_price) if np.isfinite(open_px) else pos.stop_price
                    self._sell(symbol, pos.qty, exec_px, ts_list[nxt],
                               f"{ExitReason.STOP_LOSS.value}|intrabar")

            # --- 다음 봉 종가로 마킹 후 자산 기록 ---
            next_prices = {s: float(closes[s][nxt]) for s in symbols if np.isfinite(closes[s][nxt])}
            self.portfolio.mark_prices(next_prices)
            equity_curve.append(self.portfolio.equity)
            cash_curve.append(self.portfolio.cash)
            exposure_curve.append(self.portfolio.gross_exposure)
            curve_ts.append(ts_list[nxt])

            if verbose and (i - warmup) % 100 == 0:
                logger.info(
                    f"[Backtest] {ts_list[i]} equity={self.portfolio.equity:,.0f} "
                    f"dd={self.portfolio.drawdown:.2%} pos={len(self.portfolio.open_positions())} "
                    f"regime={regime.label}"
                )

        # --- 종료 시 전량 청산 ---
        last = n - 1
        for symbol, pos in list(self.portfolio.open_positions().items()):
            self._sell(symbol, pos.qty, float(closes[symbol][last]), ts_list[last], "EOD|final")
        if equity_curve:
            equity_curve[-1] = self.portfolio.equity

        curve = equity_curve_frame(curve_ts, equity_curve, cash_curve, exposure_curve)
        metrics = compute_metrics(
            equity_curve,
            trade_returns=self.portfolio.trade_returns,
            exposures=exposure_curve,
            total_fees=self.portfolio.total_fees,
            turnover=self.turnover,
            extras={
                "strategy": getattr(self.strategy, "name", "unknown"),
                "symbols": len(symbols),
                "warmup": warmup,
            },
        )
        risk_snapshot = self.risk.snapshot(self.portfolio)
        return BacktestResult(
            metrics=metrics,
            curve=curve,
            trades=self.trades,
            regime_counts=regime_counts,
            risk_events=risk_snapshot.get("recent_reasons", []),
            rejected=self.rejected,
        )
