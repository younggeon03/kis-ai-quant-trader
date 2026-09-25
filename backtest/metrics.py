"""성과 지표 계산.

대회 심사 항목(위험조정수익률·MDD)에 직접 대응하는 지표를 산출한다.
Sharpe는 연율화하며, 무위험수익률은 국내 단기금리를 가정해 파라미터로 둔다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0
    trades: int = 0
    exposure: float = 0.0
    turnover: float = 0.0
    total_fees: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    var_95: float = 0.0
    periods: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def pretty(self) -> str:
        lines = [
            "════════════════ 백테스트 성과 ════════════════",
            f" 기간(봉)        : {self.periods:,}",
            f" 초기자본        : {self.initial_capital:>15,.0f} 원",
            f" 최종자산        : {self.final_equity:>15,.0f} 원",
            f" 총수익률        : {self.total_return * 100:>14.2f} %",
            f" CAGR            : {self.cagr * 100:>14.2f} %",
            "───────────────── 위험조정 ─────────────────",
            f" Sharpe          : {self.sharpe:>14.3f}",
            f" Sortino         : {self.sortino:>14.3f}",
            f" Calmar          : {self.calmar:>14.3f}",
            f" 연변동성        : {self.volatility * 100:>14.2f} %",
            f" MDD             : {self.max_drawdown * 100:>14.2f} %  ({self.max_drawdown_duration}봉 지속)",
            f" VaR(95%)        : {self.var_95 * 100:>14.2f} %",
            "───────────────── 매매통계 ─────────────────",
            f" 거래 횟수       : {self.trades:>14,}",
            f" 승률            : {self.win_rate * 100:>14.2f} %",
            f" Profit Factor   : {self.profit_factor:>14.3f}",
            f" 손익비          : {self.payoff_ratio:>14.3f}",
            f" 평균이익/손실   : {self.avg_win * 100:>7.2f}% / {self.avg_loss * 100:.2f}%",
            f" 평균 익스포저   : {self.exposure * 100:>14.2f} %",
            f" 총 비용         : {self.total_fees:>15,.0f} 원",
            f" 최고/최악 일간  : {self.best_day * 100:>7.2f}% / {self.worst_day * 100:.2f}%",
            "══════════════════════════════════════════════",
        ]
        return "\n".join(lines)


def max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    """(최대낙폭, 최대낙폭 지속 기간)."""
    if len(equity) == 0:
        return 0.0, 0
    peak = np.maximum.accumulate(equity)
    dd = np.where(peak > 0, (peak - equity) / peak, 0.0)
    mdd = float(dd.max()) if len(dd) else 0.0

    longest = 0
    current = 0
    for i in range(len(equity)):
        if equity[i] < peak[i]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return mdd, longest


def compute_metrics(
    equity: Sequence[float],
    trade_returns: Optional[List[float]] = None,
    exposures: Optional[Sequence[float]] = None,
    total_fees: float = 0.0,
    turnover: float = 0.0,
    risk_free: float = 0.03,
    periods_per_year: int = TRADING_DAYS,
    extras: Optional[Dict[str, Any]] = None,
) -> PerformanceMetrics:
    eq = np.asarray(list(equity), dtype=float)
    eq = eq[np.isfinite(eq)]
    if len(eq) < 2:
        return PerformanceMetrics(extras=extras or {})

    rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
    rets = rets[np.isfinite(rets)]
    n = len(rets)

    total_return = float(eq[-1] / eq[0] - 1.0) if eq[0] > 0 else 0.0
    years = max(n / periods_per_year, 1e-9)
    cagr = float((eq[-1] / eq[0]) ** (1 / years) - 1.0) if eq[0] > 0 and eq[-1] > 0 else -1.0

    vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    rf_per_period = risk_free / periods_per_year
    excess = rets - rf_per_period
    sharpe = float(excess.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year)) \
        if n > 1 and rets.std(ddof=1) > 1e-12 else 0.0

    downside = rets[rets < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(excess.mean() / dstd * np.sqrt(periods_per_year)) if dstd > 1e-12 else 0.0

    mdd, mdd_dur = max_drawdown(eq)
    calmar = float(cagr / mdd) if mdd > 1e-9 else 0.0

    tr = np.asarray(trade_returns or [], dtype=float)
    wins = tr[tr > 0]
    losses = tr[tr <= 0]
    win_rate = float(len(wins) / len(tr)) if len(tr) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (
        float("inf") if len(wins) else 0.0
    )
    payoff = float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0

    exposure = float(np.mean(list(exposures))) if exposures is not None and len(list(exposures)) else 0.0

    return PerformanceMetrics(
        initial_capital=float(eq[0]),
        final_equity=float(eq[-1]),
        total_return=total_return,
        cagr=cagr,
        volatility=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        max_drawdown_duration=mdd_dur,
        calmar=calmar,
        win_rate=win_rate,
        profit_factor=min(profit_factor, 999.0),
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff,
        trades=len(tr),
        exposure=exposure,
        turnover=float(turnover),
        total_fees=float(total_fees),
        best_day=float(rets.max()) if n else 0.0,
        worst_day=float(rets.min()) if n else 0.0,
        var_95=float(np.percentile(rets, 5)) if n >= 20 else 0.0,
        periods=len(eq),
        extras=extras or {},
    )


def equity_curve_frame(timestamps: Sequence[str], equity: Sequence[float],
                       cash: Optional[Sequence[float]] = None,
                       exposure: Optional[Sequence[float]] = None) -> pd.DataFrame:
    data = {"ts": list(timestamps), "equity": list(equity)}
    if cash is not None:
        data["cash"] = list(cash)
    if exposure is not None:
        data["exposure"] = list(exposure)
    df = pd.DataFrame(data)
    eq = df["equity"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    df["drawdown"] = np.where(peak > 0, (eq - peak) / peak, 0.0)
    return df


def plot_report(curve: pd.DataFrame, out_path: str, title: str = "KIS AI Quant Backtest") -> Optional[str]:
    """자산곡선 + 낙폭 차트를 PNG로 저장한다(matplotlib 없으면 건너뜀)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    x = range(len(curve))
    axes[0].plot(x, curve["equity"], linewidth=1.4, color="#1f77b4", label="Equity")
    axes[0].set_title(title)
    axes[0].set_ylabel("Equity (KRW)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].fill_between(x, curve["drawdown"] * 100, 0, color="#d62728", alpha=0.45)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Bars")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
