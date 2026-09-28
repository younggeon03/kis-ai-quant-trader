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
from strategy.risk_manager import RiskManager
from utils.logger import logger
from utils.market_rules import buy_cost, round_to_tick, sell_cost


def align_panel(panel: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """모든 종목을 공통 `ts`로 정렬해 인덱스를 일치시킨다."""
    frames = {s: df for s, df in panel.items() if df is not None and len(df) > 0}
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
        self.panel = align_panel(panel)
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
            for order in exits:
                open_px = float(opens[order.symbol][nxt])
                if np.isfinite(open_px) and open_px > 0:
                    self._sell(order.symbol, order.qty, open_px, ts_list[nxt],
                               f"{order.reason.value}|{order.detail}")

            # --- 진입 심사 → 다음 봉 시가 체결 ---
            for sig in signals:
                if sig.score < self.s.SIGNAL_BUY_THRESHOLD:
                    break  # 점수 내림차순이므로 이후는 볼 필요 없음
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
