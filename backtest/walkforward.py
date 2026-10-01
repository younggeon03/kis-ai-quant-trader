"""워크포워드 재학습 — 대회 기간 중 실제로 어떻게 돌아갈지에 가장 가까운 시뮬레이션.

## 일반 백테스트와 무엇이 다른가

일반 백테스트는 전 구간을 한 번 학습한 모델로 평가한다. 학습에 쓴 기간과 평가 기간이
분리되어 있어도, **모델은 딱 한 번 만들어진 것**이다. 실전은 그렇지 않다. 대회가 진행되는
동안 새 데이터가 쌓이고, 주기적으로 재학습하게 된다.

워크포워드는 그 과정을 그대로 재현한다.

    [── 학습 500봉 ──][평가 63봉]
             [── 학습 500봉 ──][평가 63봉]
                      [── 학습 500봉 ──][평가 63봉]  ...

각 구간에서 모델은 **그 시점까지의 과거만** 본다. 평가 구간을 이어 붙이면 전 기간에 걸친
하나의 out-of-sample 자산 곡선이 나온다. 여기서 나온 Sharpe가 실전에 가장 가까운 숫자다.

덤으로 얻는 것이 하나 더 있다. 재학습할 때마다 피처 중요도가 어떻게 바뀌는지 보면
**모델이 무엇에 의존하고 있고 그것이 얼마나 안정적인지** 알 수 있다. 중요도가 구간마다
크게 출렁이면 그 모델은 시장 국면에 과하게 반응하고 있다는 뜻이다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.validation import max_drawdown_from_returns, sharpe_ratio
from strategy.ai_model import AlphaModel, EnsembleAlphaStrategy
from strategy.features import build_panel_features
from strategy.risk_manager import RiskManager
from utils.logger import logger

MIN_WARMUP = 120      # 피처가 안정되는 최소 봉 수
MIN_TRAIN_BARS = 200  # 종목당 최소 학습 봉 수


# ====================================================== 공유 헬퍼 (검증 스크립트와 공용)
def slice_panel(panel: Dict[str, pd.DataFrame], start: int = 0,
                end: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    return {s: df.iloc[start:end].reset_index(drop=True) for s, df in panel.items()}


def train_on_bars(panel: Dict[str, pd.DataFrame], bar_index: np.ndarray,
                  settings,
                  flow_panel: Optional[Dict[str, pd.DataFrame]] = None) -> AlphaModel:
    """주어진 봉 인덱스만으로 모델을 학습한다.

    purge된 구간을 인덱스에서 빼고 넘기면 그 봉들이 학습에서 제외되므로,
    테스트 구간의 정보가 새지 않는다.

    ## `flow_panel`을 반드시 넘겨야 하는 이유

    예전에는 `build_panel_features(sub)`를 수급 없이 호출했다. 그러면 모델이 수급
    피처 16종을 **전부 NaN으로 보고 학습**한다. 그 상태로 백테스트에서만 수급을
    채우면 학습과 추론의 입력 분포가 달라져(train/serve mismatch) 도움이 되기는커녕
    해롭다. 즉 **수급 피처 전체가 워크포워드 경로에서 죽어 있었다.**

    수급을 쓰려면 학습과 평가에 **같은 수급 패널**을 넘겨야 한다.
    """
    keep = set(int(i) for i in bar_index)
    sub: Dict[str, pd.DataFrame] = {}
    for symbol, df in panel.items():
        mask = pd.Series([i in keep for i in range(len(df))], index=df.index)
        trimmed = df[mask].reset_index(drop=True)
        if len(trimmed) >= MIN_TRAIN_BARS:
            sub[symbol] = trimmed
    if not sub:
        raise ValueError("학습 구간이 너무 짧습니다")

    model = AlphaModel(settings)
    model.fit(build_panel_features(sub, flow_panel), verbose=False)
    return model


def segment_returns(panel: Dict[str, pd.DataFrame], model: AlphaModel, settings,
                    start: int, end: int,
                    flow_panel: Optional[Dict[str, pd.DataFrame]] = None
                    ) -> Optional[np.ndarray]:
    """[start, end) 구간만 평가한 백테스트의 일별 수익률.

    `flow_panel`은 **학습에 쓴 것과 같아야 한다**(`train_on_bars` 참고).
    한쪽만 수급을 쓰면 학습/추론 입력 분포가 달라진다.
    """
    if end - start < 10:
        return None
    strategy = EnsembleAlphaStrategy(settings, model=model)
    engine = BacktestEngine(slice_panel(panel, 0, end), strategy=strategy,
                            risk_manager=RiskManager(settings), settings=settings,
                            initial_capital=settings.INITIAL_CAPITAL,
                            flow_panel=flow_panel)
    try:
        result = engine.run(warmup=max(start, MIN_WARMUP), verbose=False)
    except ValueError:
        return None
    equity = result.curve["equity"].to_numpy(dtype=float)
    if len(equity) < 5:
        return None
    return np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1.0)


# ====================================================================== 결과
@dataclass
class WalkForwardSegment:
    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    ts_from: str = ""
    ts_to: str = ""
    sharpe: float = 0.0
    total_return: float = 0.0
    mdd: float = 0.0
    bars: int = 0
    top_features: List[str] = field(default_factory=list)
    cv_auc: float = 0.0


@dataclass
class WalkForwardResult:
    segments: List[WalkForwardSegment] = field(default_factory=list)
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    train_window: int = 0
    step: int = 0

    # ------------------------------------------------------------------ 지표
    @property
    def sharpe(self) -> float:
        return sharpe_ratio(self.returns)

    @property
    def mdd(self) -> float:
        return max_drawdown_from_returns(self.returns)

    @property
    def total_return(self) -> float:
        if len(self.returns) == 0:
            return 0.0
        return float(np.prod(1.0 + self.returns) - 1.0)

    @property
    def calmar(self) -> float:
        mdd = self.mdd
        if mdd <= 1e-9 or len(self.returns) == 0:
            return 0.0
        years = len(self.returns) / 252.0
        cagr = (1.0 + self.total_return) ** (1 / max(years, 1e-9)) - 1.0
        return float(cagr / mdd)

    @property
    def feature_stability(self) -> float:
        """구간마다 상위 피처가 얼마나 유지되는가 (0~1).

        연속한 두 구간의 상위 10개 피처가 겹치는 비율의 평균이다.
        1에 가까우면 모델이 같은 것을 계속 보고 있다는 뜻이고,
        낮으면 국면마다 다른 것에 의존한다는 뜻이다.
        """
        tops = [set(s.top_features) for s in self.segments if s.top_features]
        if len(tops) < 2:
            return 1.0
        overlaps = [len(a & b) / max(1, len(a | b)) for a, b in zip(tops, tops[1:])]
        return float(np.mean(overlaps))

    def summary(self) -> str:
        s = np.array([x.sharpe for x in self.segments], dtype=float)
        positive = int((s > 0).sum())
        lines = [
            "════════════════ 워크포워드 결과 ════════════════",
            f" 설정          : 학습 {self.train_window}봉 → 평가 {self.step}봉, "
            f"재학습 {len(self.segments)}회",
            f" 평가 봉 수    : {len(self.returns):,} (전부 out-of-sample)",
            "───────────────── 통합 성과 ─────────────────",
            f" Sharpe        : {self.sharpe:>8.3f}",
            f" 총수익률      : {self.total_return * 100:>7.2f} %",
            f" MDD           : {self.mdd * 100:>7.2f} %",
            f" Calmar        : {self.calmar:>8.3f}",
            "───────────────── 구간 분포 ─────────────────",
            f" 구간 Sharpe   : 중앙값 {np.median(s):.3f} · "
            f"범위 {s.min():.3f}~{s.max():.3f}",
            f" 양수 구간     : {positive} / {len(s)}",
            f" 피처 안정성   : {self.feature_stability:>8.3f}  (상위 피처 유지 비율)",
            "════════════════════════════════════════════",
        ]
        return "\n".join(lines)

    def segment_table(self) -> str:
        lines = [
            f" {'#':>3} {'기간':<20} {'봉':>5} {'Sharpe':>8} {'수익%':>8} {'MDD%':>7} "
            f"{'AUC':>6}  주요 피처",
            "-" * 100,
        ]
        for seg in self.segments:
            period = f"{seg.ts_from}~{seg.ts_to}" if seg.ts_from else \
                     f"{seg.test_start}-{seg.test_end}"
            lines.append(
                f" {seg.index:>3} {period:<20} {seg.bars:>5} {seg.sharpe:>8.3f}"
                f" {seg.total_return * 100:>8.2f} {seg.mdd * 100:>7.2f} {seg.cv_auc:>6.3f}"
                f"  {', '.join(seg.top_features[:3])}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_window": self.train_window, "step": self.step,
            "n_retrains": len(self.segments),
            "sharpe": round(self.sharpe, 4), "mdd": round(self.mdd, 5),
            "total_return": round(self.total_return, 5), "calmar": round(self.calmar, 4),
            "feature_stability": round(self.feature_stability, 4),
            "segments": [
                {"index": s.index, "from": s.ts_from, "to": s.ts_to, "bars": s.bars,
                 "sharpe": round(s.sharpe, 4), "return": round(s.total_return, 5),
                 "mdd": round(s.mdd, 5), "cv_auc": round(s.cv_auc, 4),
                 "top_features": s.top_features[:5]}
                for s in self.segments
            ],
        }


# ====================================================================== 실행
def run_walkforward(panel: Dict[str, pd.DataFrame], settings,
                    train_window: int = 500, step: int = 63,
                    verbose: bool = True,
                    flow_panel: Optional[Dict[str, pd.DataFrame]] = None
                    ) -> WalkForwardResult:
    """롤링 윈도우로 재학습하며 전 구간을 out-of-sample로 평가한다."""
    n_bars = min(len(df) for df in panel.values())
    if n_bars < train_window + step + MIN_WARMUP:
        raise ValueError(
            f"데이터가 부족합니다: {n_bars}봉 "
            f"(학습 {train_window} + 평가 {step} + 워밍업 {MIN_WARMUP} 필요)")

    symbols = list(panel.keys())
    ts_series = panel[symbols[0]]["ts"].astype(str).tolist()

    segments: List[WalkForwardSegment] = []
    pieces: List[np.ndarray] = []
    started = time.monotonic()
    index = 0

    test_start = max(train_window, MIN_WARMUP)
    while test_start + step <= n_bars:
        test_end = min(test_start + step, n_bars)
        train_start = max(0, test_start - train_window)
        # 라벨이 미래 H봉을 보므로 학습 끝을 그만큼 앞당긴다(누수 차단)
        train_end = max(train_start + 1, test_start - settings.LABEL_HORIZON - settings.CV_EMBARGO)

        index += 1
        try:
            model = train_on_bars(panel, np.arange(train_start, train_end),
                                  settings, flow_panel)
        except Exception as exc:
            if verbose:
                logger.warning(f"[WalkForward] 구간 {index} 학습 실패: {exc}")
            test_start += step
            continue

        returns = segment_returns(panel, model, settings, test_start, test_end,
                                  flow_panel)
        if returns is None or len(returns) == 0:
            test_start += step
            continue

        importance = (model.report.feature_importance if model.report else {}) or {}
        top = [k for k, _ in sorted(importance.items(), key=lambda kv: -kv[1])[:10]]

        seg = WalkForwardSegment(
            index=index, train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            ts_from=ts_series[test_start] if test_start < len(ts_series) else "",
            ts_to=ts_series[min(test_end, len(ts_series)) - 1] if ts_series else "",
            sharpe=sharpe_ratio(returns),
            total_return=float(np.prod(1.0 + returns) - 1.0),
            mdd=max_drawdown_from_returns(returns),
            bars=len(returns),
            top_features=top,
            cv_auc=float(model.report.cv_auc) if model.report else 0.0,
        )
        segments.append(seg)
        pieces.append(returns)

        if verbose:
            logger.info(
                f"[WalkForward] 구간 {index} {seg.ts_from}~{seg.ts_to} "
                f"Sharpe={seg.sharpe:.3f} 수익={seg.total_return * 100:+.2f}% "
                f"AUC={seg.cv_auc:.3f} ({time.monotonic() - started:.0f}s)")

        test_start += step

    result = WalkForwardResult(
        segments=segments,
        returns=np.concatenate(pieces) if pieces else np.array([]),
        train_window=train_window, step=step,
    )
    return result
