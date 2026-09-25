"""메타라벨링 — 방향은 1차 모델이, 크기는 2차 모델이 정한다.

## 문제

1차 모델은 "오를 것인가"를 예측한다. 그런데 이 모델이 매수 신호를 낸 경우만 모아 보면,
그중 상당수는 애초에 들어가지 말았어야 할 신호다. 임계값을 올리면 나쁜 신호가 줄지만
좋은 신호도 같이 줄어든다.

## 메타라벨링

문제를 둘로 쪼갠다.

  1차 모델 — 방향(side). 오를 것 같은가?
  2차 모델 — 크기(size). **1차 신호를 실제로 따랐을 때 이겼는가?**

2차 모델의 학습 데이터는 1차가 매수 신호를 낸 표본만이다. 라벨은 "그 거래가 실제로
수익을 냈는가". 입력은 원래 피처에 1차 확률·신호 강도·레짐을 더한 것이다.

이렇게 하면 2차 모델은 **1차가 언제 틀리는지**를 배운다. 방향 예측보다 쉬운 문제다.
예를 들어 "모멘텀 신호는 고변동 레짐에서 잘 틀린다" 같은 패턴을 잡아낸다.

최종 베팅 사이즈는 2차 확률에서 나온다. 방향이 틀렸다고 판단되면 크기를 0으로 만들어
진입 자체를 막는다. 거래 횟수는 줄고 정밀도는 올라간다.

2차 모델이 없으면 1차 확률을 그대로 쓴다(하위 호환).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings, get_settings
from utils.logger import logger

# 2차 모델이 추가로 보는 입력
META_EXTRA = ["primary_proba", "primary_score", "regime_scale", "tech", "cs"]


@dataclass
class MetaReport:
    samples: int = 0
    positive_rate: float = 0.0
    cv_auc: float = 0.5
    precision_primary: float = 0.0   # 1차 단독 정밀도
    precision_meta: float = 0.0      # 2차 필터 적용 후 정밀도
    coverage: float = 1.0            # 2차가 통과시킨 비율
    trained_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "positive_rate": round(self.positive_rate, 4),
            "cv_auc": round(self.cv_auc, 4),
            "precision_primary": round(self.precision_primary, 4),
            "precision_meta": round(self.precision_meta, 4),
            "coverage": round(self.coverage, 4),
            "trained_at": self.trained_at,
        }

    def pretty(self) -> str:
        gain = self.precision_meta - self.precision_primary
        return "\n".join([
            "──────────── 메타라벨링 리포트 ────────────",
            f" 학습 샘플     : {self.samples:,}  (1차가 매수 신호를 낸 경우만)",
            f" 양성 비율     : {self.positive_rate:.2%}",
            f" CV AUC        : {self.cv_auc:.4f}",
            f" 정밀도 1차    : {self.precision_primary:.4f}",
            f" 정밀도 2차    : {self.precision_meta:.4f}  ({gain:+.4f})",
            f" 통과 비율     : {self.coverage:.2%}  (나머지는 진입을 막는다)",
            "──────────────────────────────────────────",
        ])


class MetaModel:
    """1차 신호를 따를지 말지 판단하는 2차 분류기."""

    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or get_settings()
        self.model = None
        self.columns: List[str] = []
        self.threshold: float = 0.5
        self.report: Optional[MetaReport] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def path(self) -> Path:
        return Path(self.s.MODEL_DIR) / f"{self.s.MODEL_NAME}_meta.pkl"

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------ 학습 데이터
    @staticmethod
    def build_dataset(
        featured: Dict[str, pd.DataFrame],
        primary_proba: Dict[str, np.ndarray],
        base_columns: List[str],
        buy_threshold: float,
        neutral: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """1차가 매수 신호를 낸 표본만 모아 2차 학습셋을 만든다.

        Returns:
            X, y, 시점 인덱스, 컬럼명
            y = 1차 신호를 따랐을 때 실제로 이겼는가 (삼중배리어 라벨을 그대로 쓴다)
        """
        rows: List[pd.DataFrame] = []
        for symbol, df in featured.items():
            proba = primary_proba.get(symbol)
            if proba is None or "label" not in df.columns:
                continue
            n = min(len(df), len(proba))
            sub = df.iloc[:n].copy()
            sub["primary_proba"] = proba[:n]
            # 1차 점수: 중립점 기준 정규화 (ai_model._make_signal과 같은 규칙)
            sub["primary_score"] = np.clip(
                (sub["primary_proba"] - neutral) / max(neutral, 1.0 - neutral), -1.0, 1.0)
            sub["regime_scale"] = sub.get("regime_scale", 1.0)
            sub["tech"] = sub.get("tech", 0.0)
            sub["cs"] = sub.get("cs", 0.0)
            sub["symbol"] = symbol
            rows.append(sub)

        if not rows:
            return np.empty((0, 0)), np.empty(0), np.empty(0), []

        data = pd.concat(rows, ignore_index=True)
        # 1차가 매수 신호를 낸 구간만 — 이것이 메타라벨링의 핵심이다
        data = data[data["primary_score"] >= buy_threshold]
        data = data.dropna(subset=["label"])
        if data.empty:
            return np.empty((0, 0)), np.empty(0), np.empty(0), []

        columns = [c for c in base_columns if c in data.columns] + \
                  [c for c in META_EXTRA if c in data.columns]
        data[columns] = data[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        data = data.sort_values("ts").reset_index(drop=True)

        X = data[columns].to_numpy(dtype=np.float32)
        y = data["label"].to_numpy(dtype=np.int8)
        t = pd.factorize(data["ts"], sort=True)[0].astype(np.int64)
        return X, y, t, columns

    # ------------------------------------------------------------------ 학습
    def fit(self, X: np.ndarray, y: np.ndarray, t_index: np.ndarray,
            columns: List[str], verbose: bool = True) -> MetaReport:
        import datetime as dt

        from sklearn.metrics import precision_score, roc_auc_score

        from strategy.ai_model import PurgedKFold, _build_estimator

        if len(X) < 200:
            raise ValueError(f"메타 학습 데이터가 부족합니다 (샘플 {len(X)}개)")

        self.columns = columns
        self._mean = X.mean(axis=0)
        self._std = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Xs = (X - self._mean) / self._std

        cv = PurgedKFold(n_splits=self.s.CV_SPLITS, horizon=self.s.LABEL_HORIZON,
                         embargo=self.s.CV_EMBARGO)
        splits = [(tr, te) for tr, te in cv.split(t_index)
                  if len(np.unique(y[tr])) >= 2 and len(np.unique(y[te])) >= 2]

        aucs: List[float] = []
        oof_proba = np.full(len(y), np.nan)
        for fold, (tr, te) in enumerate(splits, 1):
            est = _build_estimator(seed=101 + fold)
            est.fit(Xs[tr], y[tr])
            proba = est.predict_proba(Xs[te])[:, 1]
            oof_proba[te] = proba
            aucs.append(roc_auc_score(y[te], proba))
            if verbose:
                logger.info(f"[MetaModel] fold{fold} AUC={aucs[-1]:.4f}")

        self.model = _build_estimator(seed=11)
        self.model.fit(Xs, y)

        # 임계값: 검증 구간에서 정밀도를 최대화하되 통과 비율이 25% 밑으로 떨어지지 않게
        valid = ~np.isnan(oof_proba)
        precision_primary = float(y[valid].mean()) if valid.any() else 0.0
        best_threshold, best_precision, best_coverage = 0.5, precision_primary, 1.0
        if valid.sum() > 50:
            for threshold in np.arange(0.40, 0.75, 0.025):
                picked = valid & (oof_proba >= threshold)
                coverage = picked.sum() / valid.sum()
                if coverage < 0.25:
                    break
                precision = float(y[picked].mean()) if picked.any() else 0.0
                if precision > best_precision:
                    best_threshold, best_precision, best_coverage = \
                        float(threshold), precision, float(coverage)
        self.threshold = best_threshold

        self.report = MetaReport(
            samples=len(X),
            positive_rate=float(y.mean()),
            cv_auc=float(np.mean(aucs)) if aucs else 0.5,
            precision_primary=precision_primary,
            precision_meta=best_precision,
            coverage=best_coverage,
            trained_at=dt.datetime.now().isoformat(timespec="seconds"),
        )
        return self.report

    # ------------------------------------------------------------------ 추론
    def predict(self, features: np.ndarray) -> np.ndarray:
        """2차 확률. 모델이 없으면 1.0(통과)을 돌려준다."""
        if self.model is None:
            return np.ones(len(np.atleast_2d(features)))
        X = np.atleast_2d(np.asarray(features, dtype=np.float32))
        if self._mean is not None:
            X = (X - self._mean) / self._std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception as exc:
            logger.warning(f"[MetaModel] 추론 실패: {exc}")
            return np.ones(len(X))

    # ------------------------------------------------------------------ 저장/로드
    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fp:
            pickle.dump({"model": self.model, "columns": self.columns,
                         "threshold": self.threshold, "mean": self._mean,
                         "std": self._std,
                         "report": self.report.to_dict() if self.report else None}, fp)
        logger.success(f"[MetaModel] 저장 완료 → {self.path}")
        return self.path

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with self.path.open("rb") as fp:
                blob = pickle.load(fp)
            self.model = blob["model"]
            self.columns = blob["columns"]
            self.threshold = float(blob.get("threshold", 0.5))
            self._mean = blob.get("mean")
            self._std = blob.get("std")
            logger.success(f"[MetaModel] 로드 완료 (임계값 {self.threshold:.3f})")
            return True
        except Exception as exc:
            logger.error(f"[MetaModel] 로드 실패: {exc}")
            self.model = None
            return False
