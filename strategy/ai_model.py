"""AI 알파 모델 & 앙상블 전략.

구성
 1. 삼중배리어(Triple-Barrier) 라벨링
    단순 "다음 봉 상승/하락"이 아니라, **거래비용을 넘는 익절/손절 배리어에
    먼저 닿았는가**를 라벨로 삼는다. 실제 손익 구조와 일치하므로 정확도가
    그대로 수익성으로 이어진다.
 2. PurgedKFold (+ embargo)
    라벨이 미래 H봉을 참조하므로 학습/검증 구간이 겹치면 정보가 누수된다.
    겹치는 구간을 제거(purge)하고 경계에 금지구간(embargo)을 둔다.
 3. GBDT 듀얼 백엔드
    lightgbm이 있으면 사용, 없으면 sklearn HistGradientBoosting으로 자동 대체.
 4. 앙상블 전략
    score = w1·(2p-1) + w2·기술적팩터 + w3·횡단면랭크, 전체를 레짐 스케일로 조절.
    모델 파일이 없어도 기술적 팩터만으로 동작한다(콜드 스타트 안전).
"""

from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings, get_settings
from core.events import Signal
from strategy.base import AbstractStrategy
from strategy.features import (
    FEATURE_COLUMNS,
    MIN_HISTORY,
    build_panel_features,
    stack_training_matrix,
)
from utils.indicators import realized_volatility, sma
from utils.logger import logger

try:  # pragma: no cover
    import lightgbm as lgb

    HAS_LGB = True
except Exception:  # pragma: no cover
    lgb = None
    HAS_LGB = False


# ======================================================================== 라벨링
def make_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    pt_atr: float = 1.5,
    sl_atr: float = 1.0,
    cost: float = 0.003,
) -> pd.Series:
    """삼중배리어 라벨.

    각 시점 t에서 다음 봉 시가에 진입했다고 가정하고, t+1..t+H 구간에서
      · 상단배리어(+pt_atr·ATR) 선도달 → 1
      · 하단배리어(-sl_atr·ATR) 선도달 → 0
      · 둘 다 미도달 → 최종 수익률이 비용을 넘으면 1, 아니면 0
    """
    n = len(df)
    labels = np.full(n, np.nan)
    if n < horizon + 2:
        return pd.Series(labels, index=df.index)

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    atr_abs = df["atr_abs"].to_numpy(dtype=float) if "atr_abs" in df.columns else None
    if atr_abs is None:
        from utils.indicators import atr as atr_fn

        atr_abs = atr_fn(high, low, close, 14)

    for t in range(n - horizon - 1):
        entry = open_[t + 1]
        vol = atr_abs[t]
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(vol) or vol <= 0:
            continue
        upper = entry * (1 + cost) + pt_atr * vol
        lower = entry * (1 - cost) - sl_atr * vol
        outcome = None
        for k in range(t + 1, min(t + 1 + horizon, n)):
            hit_up = high[k] >= upper
            hit_dn = low[k] <= lower
            if hit_up and hit_dn:
                outcome = 0  # 같은 봉에서 양쪽 터치 시 보수적으로 손절 가정
                break
            if hit_up:
                outcome = 1
                break
            if hit_dn:
                outcome = 0
                break
        if outcome is None:
            exit_idx = min(t + horizon, n - 1)
            ret = (close[exit_idx] - entry) / entry
            outcome = 1 if ret > cost else 0
        labels[t] = outcome
    return pd.Series(labels, index=df.index)


def make_magnitude_labels(
    df: pd.DataFrame,
    horizon: int = 30,
    threshold: float = 0.50,
    cost: float = 0.003,
) -> pd.Series:
    """크기 라벨 — "`horizon`봉 안에 `threshold` 이상 오르는가".

    ## 왜 방향 라벨로는 안 되는가

    삼중배리어 라벨은 **방향**을 가르친다(상단 먼저 닿으면 1). 그래서 모델은 상승확률
    0.7인 +5% 종목을 0.5인 +150% 종목보다 높게 평가한다. 대회 심사가 절대수익 top10
    (지난 대회 +100%)이면 그 선택이 정확히 반대다.

    실측이 이를 뒷받침한다. 30봉 창마다 '그 창의 최고 종목'을 100% 들었을 때 확장
    284종목의 최고 종목 중앙값이 **+124%**였다. 재료는 있고, 문제는 고르는 일이다.
    그리고 급등 포착 연구(`scripts/surge_study.py`)에서 모든 소재 조건의 **중앙값이
    음수**인데 `P(+100%)`는 1.6~3.1배였다 — 평균을 기준으로 배우면 전부 버린다.

    ## 종점 수익률을 쓰는 이유

    구간 내 최대값이 아니라 **봉 t+horizon 종가**를 쓴다. 대회 성적은 11/20 시점의
    계좌 수익률이고, 꼬리 모드에서는 익절·트레일링을 끄기로 했으므로(사용자 결정)
    중간 고점은 실현되지 않는다. 익절 규칙을 다시 넣는다면 라벨도 최대값으로
    바꿔야 한다 — **라벨과 청산 규칙은 같은 것을 가정해야 한다.**

    체결 가정은 삼중배리어와 같다(봉 t+1 시가 진입). 비용을 차감한 뒤 문턱과 비교한다.

    ## 표본 불균형

    `threshold=0.50`, `horizon=30`의 양성비율은 실측 **약 3.9%**다(삼중배리어 43.2%).
    `bet_size()`의 중립점이 `AlphaModel.base_rate`이므로 사이징은 자동으로 맞지만,
    `SIGNAL_BUY_THRESHOLD`는 0.5 기준으로 잡힌 값이라 다시 정해야 한다.
    """
    n = len(df)
    labels = np.full(n, np.nan)
    if n < horizon + 2:
        return pd.Series(labels, index=df.index)

    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    for t in range(n - horizon - 1):
        entry = open_[t + 1]
        exit_idx = t + 1 + horizon
        if exit_idx >= n:
            break
        exit_px = close[exit_idx]
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_px):
            continue
        ret = exit_px / entry - 1.0 - cost
        labels[t] = 1.0 if ret >= threshold else 0.0
    return pd.Series(labels, index=df.index)


def label_panel(
    featured: Dict[str, pd.DataFrame], settings: Optional[Settings] = None,
    with_weights: bool = True,
) -> Dict[str, pd.DataFrame]:
    """삼중배리어 라벨과 표본 가중치를 붙인다.

    라벨이 미래 H봉을 공유하므로 연속 샘플은 서로 독립이 아니다. 가중치를 붙이지 않으면
    중첩이 심한 구간이 과대 대표되고 검증 점수도 부풀려진다(strategy/sample_weights.py 참고).
    """
    settings = settings or get_settings()
    magnitude = str(getattr(settings, "LABEL_MODE", "triple_barrier")) == "magnitude"
    # 크기 라벨은 대회 길이(30봉)를 지평으로 쓴다. 방향 라벨의 5봉과 다르므로
    # 지평을 라벨 방식에 묶는다 — 5봉 안에 +50%는 사실상 일어나지 않는다.
    horizon = (int(settings.LABEL_MAGNITUDE_HORIZON) if magnitude
               else int(settings.LABEL_HORIZON))
    for symbol, df in featured.items():
        if magnitude:
            df["label"] = make_magnitude_labels(
                df,
                horizon=horizon,
                threshold=float(settings.LABEL_MAGNITUDE_THRESHOLD),
                cost=settings.round_trip_cost,
            )
        else:
            df["label"] = make_triple_barrier_labels(
                df,
                horizon=horizon,
                pt_atr=settings.LABEL_PT_ATR,
                sl_atr=settings.LABEL_SL_ATR,
                cost=settings.round_trip_cost,
            )
        if with_weights:
            from strategy.sample_weights import compute_sample_weights

            n = len(df)
            close = df["close"].to_numpy(dtype=float)
            logret = np.zeros(n, dtype=float)
            logret[1:] = np.log(np.maximum(close[1:], 1e-9) / np.maximum(close[:-1], 1e-9))
            t_start = np.arange(n, dtype=np.int64)
            t_end = np.minimum(t_start + horizon, n - 1)
            weights, _ = compute_sample_weights(t_start, t_end, logret, n_bars=n)
            df["sample_w"] = weights
        featured[symbol] = df
    return featured


# ======================================================================== 교차검증
class PurgedKFold:
    """시계열 정보 누수를 차단하는 K-Fold.

    라벨이 미래 `horizon` 봉을 사용하므로, 검증 구간과 겹치는 학습 샘플을 제거하고
    경계에 `embargo` 구간을 추가로 비운다.
    """

    def __init__(self, n_splits: int = 5, horizon: int = 5, embargo: int = 5):
        self.n_splits = max(2, n_splits)
        self.horizon = max(1, horizon)
        self.embargo = max(0, embargo)

    def split(self, t_index: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """`t_index`는 샘플별 시점 순번(정수, 오름차순 정렬 가정)."""
        n = len(t_index)
        if n == 0:
            return
        unique_t = np.unique(t_index)
        folds = np.array_split(unique_t, self.n_splits)
        for fold in folds:
            if len(fold) == 0:
                continue
            lo, hi = fold[0], fold[-1]
            test_mask = (t_index >= lo) & (t_index <= hi)
            # purge: 라벨 관측구간이 테스트 구간과 겹치는 학습 샘플 제거
            purge_lo = lo - self.horizon - self.embargo
            purge_hi = hi + self.horizon + self.embargo
            train_mask = (t_index < purge_lo) | (t_index > purge_hi)
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) < 50 or len(test_idx) < 10:
                continue
            yield train_idx, test_idx


# ======================================================================== 베팅 사이징
def bet_size(proba, threshold: float = 0.5):
    """상승확률을 0~1 베팅 사이즈로 바꾼다 (López de Prado 방식).

        z = (p − 0.5) / sqrt(p(1−p))
        size = 2·Φ(z) − 1

    확률이 0.5면 0, 확신이 커질수록 1에 접근한다. 분모의 sqrt(p(1−p))는 확률 자체의
    표준오차라, **같은 0.1 차이라도 확률이 극단에 가까울수록 더 크게 반영**된다.

    기존 방식(임계값 대비 점수의 선형 스케일)은 0.51과 0.95를 거의 같게 취급했다.
    확률을 수량에 직접 연결하는 것이 이 변경의 핵심이다.
    """
    from scipy.stats import norm

    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    denom = np.sqrt(np.maximum(p * (1.0 - p), 1e-12))
    z = (p - threshold) / denom
    size = 2.0 * norm.cdf(z) - 1.0
    return float(size) if np.isscalar(proba) or np.ndim(proba) == 0 else size


class _PurgedSplitAdapter:
    """PurgedKFold를 sklearn의 cv 인터페이스로 감싼다.

    `CalibratedClassifierCV`가 요구하는 `split(X, y, groups)` 시그니처를 맞춰 주면,
    **폴드 안에서만** 캘리브레이터를 학습해 누수를 막을 수 있다.
    """

    def __init__(self, splits: List[Tuple[np.ndarray, np.ndarray]]):
        self._splits = splits

    def split(self, X=None, y=None, groups=None):
        for train_idx, test_idx in self._splits:
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._splits)


# ======================================================================== 모델
def _build_estimator(seed: int = 42):
    """GBDT 듀얼 백엔드. lightgbm > sklearn 순으로 선택."""
    if HAS_LGB:  # pragma: no cover
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=6,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=400,
        learning_rate=0.035,
        max_leaf_nodes=31,
        max_depth=6,
        min_samples_leaf=40,
        l2_regularization=0.5,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=seed,
    )


@dataclass
class TrainReport:
    backend: str
    samples: int
    features: int
    positive_rate: float
    cv_auc: float
    cv_acc: float
    cv_f1: float
    fold_aucs: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    trained_at: str = ""
    brier: float = 0.0            # 확률 예측의 평균제곱오차. 낮을수록 잘 맞는 확률
    brier_uncal: float = 0.0      # 캘리브레이션 전
    effective_n: float = 0.0      # 중첩 라벨을 감안한 유효 표본 수
    mean_uniqueness: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend, "samples": self.samples, "features": self.features,
            "positive_rate": round(self.positive_rate, 4), "cv_auc": round(self.cv_auc, 4),
            "cv_acc": round(self.cv_acc, 4), "cv_f1": round(self.cv_f1, 4),
            "fold_aucs": [round(a, 4) for a in self.fold_aucs],
            "brier": round(self.brier, 5), "brier_uncal": round(self.brier_uncal, 5),
            "effective_n": round(self.effective_n, 1),
            "mean_uniqueness": round(self.mean_uniqueness, 4),
            "top_features": dict(sorted(self.feature_importance.items(),
                                        key=lambda kv: -kv[1])[:15]),
            "trained_at": self.trained_at,
        }

    def pretty(self) -> str:
        lines = [
            "──────────── 모델 학습 리포트 ────────────",
            f" 백엔드        : {self.backend}",
            f" 학습 샘플     : {self.samples:,} / 피처 {self.features}개",
            f" 양성 비율     : {self.positive_rate:.2%}",
            f" CV AUC        : {self.cv_auc:.4f}   (fold: "
            + ", ".join(f"{a:.3f}" for a in self.fold_aucs) + ")",
            f" CV 정확도     : {self.cv_acc:.4f}",
            f" CV F1         : {self.cv_f1:.4f}",
            f" Brier (보정후) : {self.brier:.5f}  (보정전 {self.brier_uncal:.5f})",
            f" 유효 표본     : {self.effective_n:,.0f} / {self.samples:,}"
            f"  (평균 고유도 {self.mean_uniqueness:.3f})",
            " 상위 피처     :",
        ]
        for name, imp in sorted(self.feature_importance.items(), key=lambda kv: -kv[1])[:12]:
            lines.append(f"   - {name:<14} {imp:.4f}")
        lines.append("──────────────────────────────────────────")
        return "\n".join(lines)


class AlphaModel:
    """가격 상승확률 예측기 (분류)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model = None
        self.calibrated = False
        # 라벨의 기저확률(양성 비율). 캘리브레이션된 확률은 이 값을 중심으로 분포하므로
        # 점수와 베팅 사이즈의 중립점도 0.5가 아니라 이 값이어야 한다.
        self.base_rate: float = 0.5
        # 학습에 쓴 기간 (첫 ts, 마지막 ts). 백테스트가 이 구간과 겹치면 in-sample이다.
        self.train_range: Tuple[str, str] = ("", "")
        self.columns: List[str] = list(FEATURE_COLUMNS)
        self.report: Optional[TrainReport] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ 경로
    @property
    def model_path(self) -> Path:
        return Path(self.settings.MODEL_DIR) / f"{self.settings.MODEL_NAME}.pkl"

    @property
    def report_path(self) -> Path:
        return Path(self.settings.MODEL_DIR) / f"{self.settings.MODEL_NAME}_report.json"

    # ------------------------------------------------------------------ 학습
    def fit(self, featured: Dict[str, pd.DataFrame], verbose: bool = True,
            calibrate: bool = False) -> TrainReport:
        """표본 가중치 + purged 폴드 내 확률 캘리브레이션으로 학습한다.

        `calibrate=True`면 purged 분할 안에서 isotonic 캘리브레이션을 학습한다(전체 데이터로
        보정하면 누수다). 다만 **기본값은 False**다. 측정해 보면 캘리브레이션이 확률 스프레드를
        기저확률 주위로 압축해, 확률 차이를 수량에 반영하는 베팅 사이징과 정면으로 충돌한다.

            가중치 O · 보정 X · 베팅 O → Sharpe 1.744  (익스포저 42%)
            가중치 O · 보정 isotonic  → Sharpe 0.105  (익스포저 18%)
            가중치 O · 보정 sigmoid   → Sharpe 1.003  (익스포저 33%)

        확률의 정직성이 필요하면(예: 리포트) 켜서 Brier를 보고, 사이징에는 쓰지 않는다.
        """
        from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

        from strategy.sample_weights import effective_sample_size

        labeled = label_panel(featured, self.settings)
        X, y, t_index, columns, w = stack_training_matrix(labeled)
        if len(X) < 300:
            raise ValueError(f"학습 데이터가 부족합니다 (샘플 {len(X)}개). 기간을 늘리세요.")

        self.columns = columns
        self.base_rate = float(np.clip(y.mean(), 0.05, 0.95))
        all_ts = sorted({str(t) for df in labeled.values() for t in df["ts"]})
        self.train_range = (all_ts[0], all_ts[-1]) if all_ts else ("", "")
        self._mean = X.mean(axis=0)
        self._std = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Xs = (X - self._mean) / self._std

        cv = PurgedKFold(
            n_splits=self.settings.CV_SPLITS,
            horizon=self.settings.LABEL_HORIZON,
            embargo=self.settings.CV_EMBARGO,
        )
        splits = [(tr, te) for tr, te in cv.split(t_index)
                  if len(np.unique(y[tr])) >= 2 and len(np.unique(y[te])) >= 2]
        if not splits:
            raise ValueError("PurgedKFold가 유효한 분할을 만들지 못했습니다. 기간을 늘리세요.")

        aucs, accs, f1s, briers_uncal = [], [], [], []
        for fold, (tr, te) in enumerate(splits, 1):
            est = _build_estimator(seed=42 + fold)
            est.fit(Xs[tr], y[tr], sample_weight=w[tr])
            raw = est.predict_proba(Xs[te])[:, 1]
            pred = (raw >= 0.5).astype(int)
            aucs.append(roc_auc_score(y[te], raw, sample_weight=w[te]))
            accs.append(accuracy_score(y[te], pred, sample_weight=w[te]))
            f1s.append(f1_score(y[te], pred, zero_division=0, sample_weight=w[te]))
            briers_uncal.append(brier_score_loss(y[te], raw, sample_weight=w[te]))
            if verbose:
                logger.info(f"[AlphaModel] fold{fold} AUC={aucs[-1]:.4f} "
                            f"ACC={accs[-1]:.4f} Brier={briers_uncal[-1]:.4f}")

        # --- 최종 모델: purged 분할 안에서만 isotonic 캘리브레이션 ---
        base = _build_estimator(seed=7)
        self.calibrated = False
        if calibrate and len(splits) >= 2:
            from sklearn.calibration import CalibratedClassifierCV

            try:
                model = CalibratedClassifierCV(
                    base, method="isotonic", cv=_PurgedSplitAdapter(splits), ensemble=True
                )
                model.fit(Xs, y, sample_weight=w)
                self.model = model
                self.calibrated = True
            except Exception as exc:
                logger.warning(f"[AlphaModel] 캘리브레이션 실패, 원본 모델 사용: {exc}")
                base.fit(Xs, y, sample_weight=w)
                self.model = base
        else:
            base.fit(Xs, y, sample_weight=w)
            self.model = base

        # 캘리브레이션 효과: 마지막 폴드 검증 구간에서 Brier 재측정
        brier_after = float(np.mean(briers_uncal)) if briers_uncal else 0.0
        if self.calibrated:
            _, te = splits[-1]
            try:
                brier_after = float(brier_score_loss(
                    y[te], self.model.predict_proba(Xs[te])[:, 1], sample_weight=w[te]))
            except Exception:
                pass

        importance = self._extract_importance(Xs, y)
        self.report = TrainReport(
            backend=("lightgbm" if HAS_LGB else "sklearn-HistGBDT")
                    + (" + isotonic" if self.calibrated else ""),
            samples=len(X),
            features=len(columns),
            positive_rate=float(y.mean()),
            cv_auc=float(np.mean(aucs)) if aucs else 0.5,
            cv_acc=float(np.mean(accs)) if accs else 0.0,
            cv_f1=float(np.mean(f1s)) if f1s else 0.0,
            fold_aucs=[float(a) for a in aucs],
            feature_importance=importance,
            trained_at=dt.datetime.now().isoformat(timespec="seconds"),
            brier=brier_after,
            brier_uncal=float(np.mean(briers_uncal)) if briers_uncal else 0.0,
            effective_n=effective_sample_size(w),
            mean_uniqueness=float(np.median(w)) if len(w) else 1.0,
        )
        return self.report

    def _extract_importance(self, Xs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """피처 중요도. 백엔드가 제공하지 않으면 순열 중요도로 대체."""
        if hasattr(self.model, "feature_importances_"):
            raw = np.asarray(self.model.feature_importances_, dtype=float)
            total = raw.sum()
            if total > 0:
                return {c: float(v / total) for c, v in zip(self.columns, raw)}
        try:
            from sklearn.inspection import permutation_importance

            sample = min(4000, len(Xs))
            idx = np.random.default_rng(0).choice(len(Xs), size=sample, replace=False)
            result = permutation_importance(
                self.model, Xs[idx], y[idx], n_repeats=3, random_state=0, scoring="roc_auc"
            )
            raw = np.clip(result.importances_mean, 0, None)
            total = raw.sum()
            if total > 0:
                return {c: float(v / total) for c, v in zip(self.columns, raw)}
        except Exception as exc:  # pragma: no cover
            logger.debug(f"[AlphaModel] 중요도 계산 실패: {exc}")
        return {}

    # ------------------------------------------------------------------ 추론
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if self._mean is not None:
            X = (X - self._mean) / self._std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[AlphaModel] 추론 실패: {exc}")
            return np.full(len(X), 0.5)

    def predict_frame(self, df: pd.DataFrame) -> float:
        """가장 최근 봉에 대한 상승확률."""
        cols = [c for c in self.columns if c in df.columns]
        if self.model is None or not cols or df.empty:
            return 0.5
        row = df.iloc[-1][cols].to_numpy(dtype=float)
        if np.isnan(row).mean() > 0.3:
            return 0.5
        vec = np.zeros(len(self.columns), dtype=np.float32)
        lookup = {c: i for i, c in enumerate(self.columns)}
        for c, v in zip(cols, np.nan_to_num(row, nan=0.0)):
            vec[lookup[c]] = v
        return float(self.predict_proba(vec.reshape(1, -1))[0])

    # ------------------------------------------------------------------ 저장/로드
    def save(self) -> Path:
        path = self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fp:
            pickle.dump(
                {
                    "model": self.model,
                    "calibrated": self.calibrated,
                    "base_rate": self.base_rate,
                    "train_range": self.train_range,
                    "columns": self.columns,
                    "mean": self._mean,
                    "std": self._std,
                    "report": self.report.to_dict() if self.report else None,
                },
                fp,
            )
        if self.report:
            self.report_path.write_text(
                json.dumps(self.report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        logger.success(f"[AlphaModel] 모델 저장 완료 → {path}")
        return path

    def load(self) -> bool:
        path = self.model_path
        if not path.exists():
            logger.warning(f"[AlphaModel] 모델 파일이 없습니다({path}). 기술적 팩터만으로 동작합니다.")
            return False
        try:
            with path.open("rb") as fp:
                blob = pickle.load(fp)
            self.model = blob["model"]
            self.calibrated = bool(blob.get("calibrated", False))
            self.base_rate = float(blob.get("base_rate", 0.5))
            self.train_range = tuple(blob.get("train_range", ("", "")))
            self.columns = blob["columns"]
            self._mean = blob.get("mean")
            self._std = blob.get("std")
            meta = blob.get("report") or {}
            logger.success(
                f"[AlphaModel] 모델 로드 완료 (backend={meta.get('backend')}, "
                f"CV AUC={meta.get('cv_auc')})"
            )
            return True
        except Exception as exc:
            logger.error(f"[AlphaModel] 모델 로드 실패: {exc}")
            self.model = None
            return False


# ======================================================================== 전략
def technical_score(row: pd.Series) -> float:
    """기술적 팩터 종합 점수(-1 ~ 1).

    추세추종(모멘텀)과 평균회귀를 함께 보되, 거래량 확인이 없으면 가중을 줄인다.
    """

    def g(name: str, default: float = 0.0) -> float:
        v = row.get(name, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return default if not np.isfinite(v) else v

    trend = (
        0.45 * np.tanh(g("ma_cross") * 25.0)
        + 0.30 * np.tanh(g("donchian_20") * 3.0)
        + 0.25 * np.tanh(g("trend_20") * 1.5)
    )
    momentum = (
        0.40 * np.tanh(g("macd_hist_n") * 1.2)
        + 0.35 * np.tanh(g("mom_60_5") * 6.0)
        + 0.25 * np.tanh(-g("rsi_14") * 1.1 + g("rsi_slope") * 2.0)
    )
    reversion = (
        0.55 * np.tanh(-g("bb_pctb") * 3.0)
        + 0.45 * np.tanh(-g("stoch_k") * 1.5)
    )
    flow = (
        0.40 * np.tanh(g("obv_slope") * 5.0)
        + 0.30 * np.tanh(g("turnover_z") * 0.8)
        + 0.30 * np.tanh(g("mfi_14") * 1.2)
    )
    # 추세가 강할수록(ADX↑) 추세추종 비중을 높이고 평균회귀 비중을 낮춘다
    adx_strength = float(np.clip(g("adx_14") * 2.5, 0.0, 1.0))
    trend_w = 0.30 + 0.25 * adx_strength
    rev_w = 0.30 - 0.20 * adx_strength

    score = trend_w * trend + 0.25 * momentum + rev_w * reversion + 0.15 * flow
    # 거래량 확인 필터: 거래대금이 평소보다 현저히 적으면 신뢰도 하향
    if g("turnover_z") < -1.0:
        score *= 0.6
    return float(np.clip(score, -1.0, 1.0))


def cross_section_score(row: pd.Series) -> float:
    """횡단면 랭크 기반 점수(-1 ~ 1)."""
    vals = []
    weights = {"cs_mom": 0.40, "cs_rev": 0.25, "cs_turnover": 0.15, "cs_vol": 0.20}
    for name, w in weights.items():
        v = row.get(name, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        if np.isfinite(v):
            vals.append(w * v * 2.0)  # -0.5~0.5 → -1~1
    return float(np.clip(sum(vals), -1.0, 1.0))


@dataclass
class RegimeState:
    scale: float = 1.0
    trend: float = 0.0
    volatility: float = 0.0
    breadth: float = 0.0
    label: str = "NORMAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": round(self.scale, 3), "trend": round(self.trend, 4),
            "volatility": round(self.volatility, 4), "breadth": round(self.breadth, 3),
            "label": self.label,
        }


def detect_regime(panel: Dict[str, pd.DataFrame]) -> RegimeState:
    """시장 레짐 판정.

    유니버스 동일가중 지수를 만들어 추세(20일 이격도)와 변동성을 본다.
    약세 + 고변동 국면에서는 신호 전체를 축소해 MDD를 방어한다.
    """
    closes = []
    for df in panel.values():
        if df is None or len(df) < 60:
            continue
        s = df["close"].to_numpy(dtype=float)
        closes.append(s[-120:] / s[-120])
    if not closes:
        return RegimeState()

    min_len = min(len(c) for c in closes)
    index = np.mean([c[-min_len:] for c in closes], axis=0)
    ma20 = sma(index, 20)
    last_ma = float(ma20[-1])
    trend = float((index[-1] - last_ma) / last_ma) if np.isfinite(last_ma) and last_ma > 0 else 0.0
    vol = realized_volatility(index, 20)
    volatility = float(vol[-1]) if np.isfinite(vol[-1]) else 0.2

    # 상승 종목 비율(breadth)
    ups = 0
    total = 0
    for df in panel.values():
        if df is None or len(df) < 21:
            continue
        total += 1
        if df["close"].iloc[-1] > df["close"].iloc[-21]:
            ups += 1
    breadth = ups / total if total else 0.5

    return _classify_regime(trend, volatility, breadth)


def _classify_regime(trend: float, volatility: float, breadth: float) -> RegimeState:
    if trend < -0.02 and volatility > 0.30:
        scale, label = 0.25, "BEAR_HIGH_VOL"
    elif trend < -0.01:
        scale, label = 0.55, "BEAR"
    elif volatility > 0.38:
        scale, label = 0.50, "HIGH_VOL"
    elif trend > 0.015 and breadth > 0.55:
        scale, label = 1.0, "BULL"
    else:
        scale, label = 0.80, "NEUTRAL"
    if breadth < 0.30:
        scale *= 0.7
    return RegimeState(scale=float(np.clip(scale, 0.15, 1.0)), trend=float(trend),
                       volatility=float(volatility), breadth=float(breadth), label=label)


def regime_series(panel: Dict[str, pd.DataFrame]) -> List[RegimeState]:
    """봉 인덱스별 레짐 상태(백테스트용). 패널은 길이가 정렬되어 있어야 한다."""
    frames = [df for df in panel.values() if df is not None and len(df) > 0]
    if not frames:
        return []
    n = min(len(df) for df in frames)
    closes = np.vstack([df["close"].to_numpy(dtype=float)[:n] for df in frames])
    normed = closes / closes[:, [0]]
    index = normed.mean(axis=0)

    ma20 = sma(index, 20)
    trend = np.where(np.isfinite(ma20) & (ma20 > 0), (index - ma20) / np.where(ma20 > 0, ma20, 1.0), 0.0)
    vol = realized_volatility(index, 20)
    vol = np.where(np.isfinite(vol), vol, 0.25)

    breadth = np.full(n, 0.5)
    if n > 21:
        up = closes[:, 21:] > closes[:, :-21]
        breadth[21:] = up.mean(axis=0)

    return [_classify_regime(float(trend[i]), float(vol[i]), float(breadth[i])) for i in range(n)]


class EnsembleAlphaStrategy(AbstractStrategy):
    """AI 모델 + 기술적 팩터 + 횡단면 랭크 앙상블 전략."""

    name = "ensemble_alpha"

    def __init__(self, settings: Optional[Settings] = None, model: Optional[AlphaModel] = None,
                 meta=None):
        self.settings = settings or get_settings()
        self.model = model or AlphaModel(self.settings)
        # 2차 모델(메타라벨링). 없으면 1차 확률을 그대로 쓴다.
        self.meta = meta
        self.regime = RegimeState()
        self._last_featured: Dict[str, pd.DataFrame] = {}
        self._signal_log: List[Dict[str, Any]] = []

    def warmup_bars(self) -> int:
        return max(MIN_HISTORY, 80)

    def load_model(self, with_meta: bool = True) -> bool:
        loaded = self.model.load()
        if loaded and with_meta and self.meta is None and self.settings.USE_META_MODEL:
            from strategy.meta_model import MetaModel

            meta = MetaModel(self.settings)
            if meta.load():
                self.meta = meta
        return loaded

    # ------------------------------------------------------------------ 가중치
    def _weights(self) -> Tuple[float, float, float]:
        w_m, w_t, w_c = self.settings.W_MODEL, self.settings.W_TECHNICAL, self.settings.W_CROSS_SECTION
        if self.model.model is None:
            # 모델 미로딩 시 가중치를 재분배(콜드 스타트)
            total = max(1e-9, w_t + w_c)
            return 0.0, w_t / total, w_c / total
        return w_m, w_t, w_c

    # ------------------------------------------------------------------ 사전계산
    def precompute(self, panel: Dict[str, pd.DataFrame],
                   flow_panel: Optional[Dict[str, pd.DataFrame]] = None
                   ) -> Dict[str, pd.DataFrame]:
        """패널 전체의 피처 + 모델확률 + 팩터점수를 한 번에 계산한다.

        모든 지표는 인과적(causal)이므로 전체 구간에서 한 번 계산한 값과
        매 시점 잘라서 계산한 값이 동일하다. 백테스트를 O(n²)에서 O(n)으로 낮춘다.
        """
        featured = build_panel_features(panel, flow_panel)
        if not featured:
            return {}
        w_m, _, _ = self._weights()
        for symbol, df in featured.items():
            if w_m > 0:
                cols = [c for c in self.model.columns if c in df.columns]
                X = np.zeros((len(df), len(self.model.columns)), dtype=np.float32)
                lookup = {c: i for i, c in enumerate(self.model.columns)}
                for c in cols:
                    X[:, lookup[c]] = np.nan_to_num(
                        df[c].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
                    )
                df["proba"] = self.model.predict_proba(X)
            else:
                df["proba"] = 0.5
            df["tech"] = df.apply(technical_score, axis=1)
            df["cs"] = df.apply(cross_section_score, axis=1)

            # 2차 모델: 1차 신호를 따를지 말지. 없으면 전부 통과(1.0)
            if self.meta is not None and self.meta.ready:
                neutral = getattr(self.model, "base_rate", 0.5)
                df["primary_proba"] = df["proba"]
                df["primary_score"] = np.clip(
                    (df["proba"] - neutral) / max(neutral, 1.0 - neutral), -1.0, 1.0)
                df["regime_scale"] = 1.0
                cols = [c for c in self.meta.columns if c in df.columns]
                M = np.zeros((len(df), len(self.meta.columns)), dtype=np.float32)
                lookup = {c: i for i, c in enumerate(self.meta.columns)}
                for c in cols:
                    M[:, lookup[c]] = np.nan_to_num(
                        df[c].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                df["meta_proba"] = self.meta.predict(M)
            else:
                df["meta_proba"] = 1.0

            featured[symbol] = df
        self._last_featured = featured
        return featured

    def _make_signal(self, symbol: str, row: pd.Series, regime: RegimeState,
                     ts: Optional[dt.datetime]) -> Optional[Signal]:
        price = float(row["close"])
        if price <= 0 or not np.isfinite(price):
            return None
        w_m, w_t, w_c = self._weights()
        proba = float(row.get("proba", 0.5))
        tech = float(row.get("tech", 0.0))
        cs = float(row.get("cs", 0.0))
        # 모델 기여도의 중립점 = 라벨의 기저확률(양성 비율).
        # 양성 비율이 42%면 보정 여부와 무관하게 예측 확률이 그 주위에 분포한다.
        # 중립점을 0.5로 두면 정상 신호가 전부 음수로 읽혀 진입이 거의 사라진다
        #   (측정: 중립점 0.5 → Sharpe −1.47 · 익스포저 4.3%,
        #          중립점 기저확률 → Sharpe 1.74 · 익스포저 42%)
        neutral = getattr(self.model, "base_rate", 0.5)
        model_score = (proba - neutral) / max(neutral, 1.0 - neutral)
        raw = w_m * float(np.clip(model_score, -1.0, 1.0)) + w_t * tech + w_c * cs
        score = float(np.clip(raw * regime.scale, -1.0, 1.0))

        atr_abs = float(row.get("atr_abs", 0.0) or 0.0)
        if not np.isfinite(atr_abs) or atr_abs <= 0:
            atr_abs = price * 0.02
        vol20 = float(row.get("vol_20", 0.0) or 0.0)
        if not np.isfinite(vol20) or vol20 <= 0:
            vol20 = 0.25

        # 베팅 사이즈: 모델이 있으면 캘리브레이션된 확률에서, 없으면 점수에서 유도한다.
        # 확률을 수량에 직접 연결하는 경로다(strategy/risk_manager.py가 이 값을 쓴다).
        if w_m > 0:
            bet = float(np.clip(bet_size(proba, threshold=neutral), 0.0, 1.0))
            # 메타라벨링: 2차 모델이 "이 신호는 따르지 마라"고 하면 크기를 0으로 만든다.
            # 방향은 1차가, 크기는 2차가 정한다.
            meta_p = float(row.get("meta_proba", 1.0))
            if self.meta is not None and self.meta.ready:
                if meta_p < self.meta.threshold:
                    bet = 0.0   # 2차가 막으면 진입 자체를 하지 않는다
                else:
                    bet *= float(np.clip(meta_p, 0.0, 1.0))
        else:
            bet = -1.0   # 콜드 스타트: 리스크 매니저가 점수 기반 선형 스케일로 폴백한다

        return Signal(
            symbol=symbol, score=score, price=price, ts=ts or dt.datetime.now(),
            model_proba=proba, tech_score=tech, cs_score=cs, regime=regime.scale,
            atr=atr_abs, volatility=vol20, bet=bet, reason=f"regime={regime.label}",
        )

    def signals_at(
        self,
        featured: Dict[str, pd.DataFrame],
        index: int,
        regime: Optional[RegimeState] = None,
        ts: Optional[dt.datetime] = None,
    ) -> List[Signal]:
        """사전계산된 피처에서 특정 봉 인덱스의 신호를 생성한다(백테스트용)."""
        regime = regime or self.regime
        signals: List[Signal] = []
        for symbol, df in featured.items():
            if index < 0 or index >= len(df):
                continue
            sig = self._make_signal(symbol, df.iloc[index], regime, ts)
            if sig is not None:
                signals.append(sig)
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    # ------------------------------------------------------------------ 신호 생성
    def generate_signals(
        self,
        panel: Dict[str, pd.DataFrame],
        holdings: Optional[Dict[str, Any]] = None,
        ts: Optional[dt.datetime] = None,
        flow_panel: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[Signal]:
        usable = {s: df for s, df in panel.items() if df is not None and len(df) >= self.warmup_bars()}
        if not usable:
            return []

        featured = self.precompute(usable, flow_panel)
        if not featured:
            return []
        self.regime = detect_regime(usable)

        signals: List[Signal] = []
        for symbol, df in featured.items():
            sig = self._make_signal(symbol, df.iloc[-1], self.regime, ts)
            if sig is not None:
                signals.append(sig)

        signals.sort(key=lambda s: s.score, reverse=True)
        self._signal_log = [s.to_dict() for s in signals[:10]]
        return signals

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_loaded": self.model.model is not None,
            "backend": "lightgbm" if HAS_LGB else "sklearn-HistGBDT",
            "regime": self.regime.to_dict(),
            "top_signals": self._signal_log,
        }
