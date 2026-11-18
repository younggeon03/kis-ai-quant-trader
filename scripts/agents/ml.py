"""ML 에이전트(11/18) — 워크포워드 학습 순위(hit10 분류) 후보 + 매도 규칙 후보.

후보 함수 candidate_N(g) 는 그날 day_df 를 받아 종목코드(또는 None)를 돌려준다.
모델은 rows.pkl 전체를 읽지만 **학습에는 그날보다 최소 2거래일 앞(인덱스 ≤ i-2)까지의 행만** 쓴다
(d-2 행의 라벨은 d-1 15:10 에 확정 → d 09:04 판단에 쓸 수 있다). 재학습 = 20거래일 블록, 블록 시작 i0 기준 cutoff = i0-2.
첫 60거래일(학습 부족)은 엔진 판정 · ATR10 순으로 대체한다.

하이퍼파라미터는 미리 정한 값(튜닝 없음):
  LR   = 표준화 + L2 로지스틱 C=0.1, class_weight=None
  GBDT = lightgbm(max_depth=3, num_leaves=7, learning_rate=0.05, n_estimators=200, min_child_samples=40,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, seed=0) — 없으면 sklearn HistGradientBoosting 동등값
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports" / "agent_data"
WARMUP = 60
RETRAIN = 20
PURGE = 2
SEED = 0

FEATS = ["l_ratio", "atr10", "chg", "gap", "ret4", "range4", "vwap_dev", "mom20", "run3", "l_mcap", "l_tv4",
         "kosdaq", "breadth"]

_ROWS: Optional[pd.DataFrame] = None
_DAYS: Optional[list] = None
_CACHE: Dict[Tuple, object] = {}


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["l_ratio"] = np.log(df["ratio"].clip(lower=1e-3))
    X["l_mcap"] = np.log(df["mcap"].clip(lower=1e8))
    X["l_tv4"] = np.log(df["tv4"].clip(lower=1e6))
    for c in FEATS:
        if c not in X:
            X[c] = df[c].astype(float)
    return X[FEATS]


def _rows() -> pd.DataFrame:
    global _ROWS, _DAYS
    if _ROWS is None:
        r = pd.read_pickle(DATA / "rows.pkl")
        r = r[np.isfinite(r["rB"])].copy()
        _DAYS = sorted(r["day"].unique())
        r["_di"] = r["day"].map({d: i for i, d in enumerate(_DAYS)})
        _ROWS = r
    return _ROWS


def _day_index(day: str) -> int:
    _rows()
    import bisect
    return bisect.bisect_left(_DAYS, day)          # 모르는(미래) 날짜면 끝 인덱스


def _fit(kind: str, train: pd.DataFrame, label: str):
    X = make_X(train)
    y = (train["up_max"] >= 0.103).values if label == "up" else train["hit10"].astype(bool).values
    if kind == "lr":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=2000))
        m.fit(X.values, y)
        return m
    if kind == "gbdt3":                              # 시드 0·1·2 평균(분산 감소용, 하이퍼파라미터는 같다)
        global SEED
        ms, s0 = [], SEED
        for s in (0, 1, 2):
            SEED = s
            ms.append(_fit("gbdt", train, label))
        SEED = s0
        return _Avg(ms)
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(max_depth=3, num_leaves=7, learning_rate=0.05, n_estimators=200, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                           random_state=0)
    m.fit(X, y)
    return m


class _Avg:
    def __init__(self, ms):
        self.ms = ms

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


def get_model(kind: str, pop: str, di: int, label: str = "hit"):
    """di = 예측할 날의 인덱스. None 이면 워밍업(학습 부족)."""
    if di < WARMUP:
        return None
    b = (di - WARMUP) // RETRAIN
    key = (kind, pop, label, b)
    if key not in _CACHE:
        i0 = WARMUP + RETRAIN * b
        cut = i0 - PURGE
        r = _rows()
        tr = r[r["_di"] <= cut]
        if pop == "pass":
            tr = tr[tr["engine_pass"]]
        _CACHE[key] = _fit(kind, tr, label)
    return _CACHE[key]


def _atr_pick(g: pd.DataFrame) -> Optional[str]:
    c = g[g["engine_pass"]].copy()
    if c.empty:
        return None
    c["_a"] = c["atr10"].fillna(-1.0)
    return c.sort_values(["_a", "ratio", "sym"], ascending=[False, False, True])["sym"].iloc[0]


def predict_day(g: pd.DataFrame, kind: str, train_pop: str, label: str = "hit") -> Optional[pd.Series]:
    di = _day_index(str(g["day"].iloc[0]))
    m = get_model(kind, train_pop, di, label)
    if m is None:
        return None
    X = make_X(g)
    p = m.predict_proba(X.values if kind == "lr" else X)[:, 1]
    return pd.Series(p, index=g.index)


def pick_model(g: pd.DataFrame, kind: str, train_pop: str, cand: str, label: str = "hit") -> Optional[str]:
    p = predict_day(g, kind, train_pop, label)
    if p is None:
        return _atr_pick(g)
    c = g[g["engine_pass"]] if cand == "pass" else g
    if c.empty:
        return None
    c = c.assign(_p=p.loc[c.index])
    return c.sort_values(["_p", "sym"], ascending=[False, True])["sym"].iloc[0]


# ---- 최종 후보(보고서 참고) ----
def candidate_1(g: pd.DataFrame) -> Optional[str]:
    """GBDT(시드 3개 평균) · 전체 지수 종목으로 학습 · 전체 지수 종목 중 확률 1위(판정 통과 아니어도)."""
    return pick_model(g, "gbdt3", "all", "all")


def candidate_2(g: pd.DataFrame) -> Optional[str]:
    """GBDT(시드 3개 평균) · 전체 지수 종목으로 학습 · 판정 통과 중 확률 1위(지금 엔진 후보 집합 그대로)."""
    return pick_model(g, "gbdt3", "all", "pass")


def candidate_3(g: pd.DataFrame) -> Optional[str]:
    """로지스틱(L2, C=0.1) · 전체 지수 종목으로 학습 · 전체 지수 종목 중 확률 1위."""
    return pick_model(g, "lr", "all", "all")


# ---- 매도 규칙 후보 ----
TP_MULT = 1.10 / (1 - 0.00278)


def exit_baseline(o, h, l, c, entry, day_idx, minute):
    tp = entry * TP_MULT
    if h >= tp:
        return max(o, tp)
    return None


def exit_candidate_1(o, h, l, c, entry, day_idx, minute):
    """기준 익절 + 2일차 09:04 에 아직 안 팔렸으면 그 시가에 매도(→ 그날 판정 통과 ATR10 1위로 갈아타기,
    목표 = 누적 +10% 에 필요한 만큼, 15:10 청산). 갈아타기 쪽은 simulate_exit 로 표현할 수 없어 ml_exit_study.run_switch(thr=-1)."""
    px = exit_baseline(o, h, l, c, entry, day_idx, minute)
    if px is not None:
        return px
    if day_idx == 1 and minute == 4:
        return o
    return None
