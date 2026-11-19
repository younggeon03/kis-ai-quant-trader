"""선택적 예측(기각 옵션) 하루 매매 후보 — day_ml 에이전트(11/19).

09:30(m=30, 첫 30분 정보)에 전 종목의 "15:10 매도 순수익 > 0" 확률을 워크포워드로 학습한 모델로 예측하고,
그날 최고 확률 종목의 보정 확률이 임계값 이상일 때만 매수한다(Chow 1970 기각 옵션 · Geifman·El-Yaniv 2017).

룩어헤드 차단
- 예측 날짜 i 의 피처는 ctx.bars(30)(0..29분)과 ctx.daily(전일까지)만 쓴다.
- 모델은 20거래일 블록 단위로 재학습한다. 블록 시작 s 의 모델은 날짜 ≤ s-2 의 표본만 쓴다(라벨 = 그날 09:30 시가 매수 →
  15:10 시가 매도, scripts.research.day_eval.simulate(i, j, 30, {}) 와 같은 값 — 벡터 구현, study 에서 일치 확인).
- 첫 60거래일은 학습만(예측 안 함).
- 봉인 구간에서도 같은 함수가 그 날짜 이전 데이터로만 학습한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.day_eval import COST, END, Ctx  # noqa: E402

M = 30            # 결정 시각(09:30 시가 매수)
BLOCK = 20        # 재학습 주기(거래일)
WARM = 60         # 첫 학습 구간
FEATS = ["r30", "gap", "tv_ratio", "vwap_pos", "hl30", "pos30", "ret1", "ret5", "atr10",
         "mkt30", "ind30", "kosdaq"]

_FEAT: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}   # i -> (X[S,F], valid[S])
_LAB: Dict[int, np.ndarray] = {}                       # i -> ret[S] (NaN = 체결 불가)
_MODELS: Dict[tuple, object] = {}

# 속도용: study 가 같은 함수로 만든 개발 구간(2026-07-01 전) 피처·라벨 캐시가 있으면 올린다(없으면 그때 계산)
try:
    import pickle as _pk
    _c = ROOT / "reports" / "agents" / "day_ml_cache.pkl"
    if _c.exists():
        _f, _l = _pk.load(open(_c, "rb"))
        _FEAT.update(_f)
        _LAB.update(_l)
except Exception:
    pass


# ── 피처 ─────────────────────────────────────────────────────────────
def _daily_feats(ctx: Ctx, j: int):
    d = ctx.daily(j, 12)
    if len(d) < 11:
        return np.nan, np.nan, np.nan, np.nan
    c = d["close"].to_numpy(float)
    h = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float)
    v = d["volume"].to_numpy(float)
    ret1 = c[-1] / c[-2] - 1
    ret5 = c[-1] / c[-6] - 1
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(lo[1:], c[:-1])
    atr10 = np.mean(tr[-10:]) / c[-1]
    tv = np.median((c * v)[-10:])
    return ret1, ret5, atr10, tv


def features(ctx: Ctx) -> Tuple[np.ndarray, np.ndarray]:
    """ctx 날짜의 09:30 피처 [S,F] 와 유효 마스크. 0..29분 + 전일까지 일봉만 쓴다."""
    if ctx.i in _FEAT:
        return _FEAT[ctx.i]
    b = ctx.bars(M)                     # [S,5,30]
    o, h, lo, c, v = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    S = b.shape[0]
    with np.errstate(all="ignore"):
        first = np.full(S, np.nan)
        last = np.full(S, np.nan)
        for s in range(S):
            fo = np.flatnonzero(np.isfinite(o[s]) & (o[s] > 0))
            lc = np.flatnonzero(np.isfinite(c[s]) & (c[s] > 0))
            if len(fo):
                first[s] = o[s, fo[0]]
            if len(lc):
                last[s] = c[s, lc[-1]]
        pc = ctx.prev_close.astype(float)
        typ = (h + lo + c) / 3
        vv = np.nan_to_num(v)
        tv30 = np.nansum(np.nan_to_num(typ) * vv, axis=1)
        vwap = tv30 / np.where(vv.sum(1) > 0, vv.sum(1), np.nan)
        hi30, lo30 = np.nanmax(h, axis=1), np.nanmin(lo, axis=1)
        r30 = last / first - 1
        gap = first / pc - 1
        vwap_pos = last / vwap - 1
        hl30 = (hi30 - lo30) / first
        pos30 = (last - lo30) / np.where(hi30 > lo30, hi30 - lo30, np.nan)
        dd = np.array([_daily_feats(ctx, j) for j in range(S)])
        tv_ratio = np.log((tv30 + 1) / (dd[:, 3] + 1))       # 첫 30분 거래대금 / 하루 거래대금 중앙(10일)
        valid = np.isfinite(r30) & np.isfinite(gap) & np.isfinite(dd[:, 2]) & (tv30 > 0)
        mkt30 = np.nanmedian(np.where(valid, r30, np.nan))
        ind = np.asarray(ctx.ind)
        ind30 = np.full(S, np.nan)
        for g in np.unique(ind):
            k = (ind == g) & valid
            if k.sum():
                ind30[ind == g] = np.nanmean(r30[k])
        X = np.column_stack([r30, gap, tv_ratio, vwap_pos, hl30, pos30, dd[:, 0], dd[:, 1], dd[:, 2],
                             np.full(S, mkt30), ind30, np.asarray(ctx.kosdaq, float)])
    X = np.where(np.isfinite(X), X, 0.0)
    _FEAT[ctx.i] = (X, valid)
    return X, valid


def label(i: int) -> np.ndarray:
    """날짜 i 의 09:30 매수 → 15:10 매도 순수익 [S] — day_eval.simulate(i, j, 30, {}) 의 벡터 판. 학습용(과거 날짜)만."""
    if i in _LAB:
        return _LAB[i]
    p = Ctx(i).bars(END + 1)            # [S,5,371]
    S = p.shape[0]
    out = np.full(S, np.nan)
    for s in range(S):
        entry = np.nan
        for k in range(M, M + 3):
            if np.isfinite(p[s, 0, k]) and p[s, 0, k] > 0:
                entry = float(p[s, 0, k])
                break
        if not np.isfinite(entry):
            continue
        px = p[s, 0, END] if np.isfinite(p[s, 0, END]) else p[s, 3, END - 1]
        if not np.isfinite(px):
            lc = np.flatnonzero(np.isfinite(p[s, 3, :END]))
            if not len(lc):
                continue
            px = p[s, 3, lc[-1]]
        out[s] = px / entry * (1 - COST) - 1
    _LAB[i] = out
    return out


# ── 모델 ─────────────────────────────────────────────────────────────
class _Platt:
    def __init__(self, a=1.0, b=0.0):
        self.a, self.b = a, b

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        lr = LogisticRegression(C=1e6).fit(z, y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, p):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return 1 / (1 + np.exp(-(self.a * z + self.b)))


def _make(kind: str):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(max_depth=3, num_leaves=7, n_estimators=200, learning_rate=0.05,
                                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, verbose=-1, random_state=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                              min_samples_leaf=200, random_state=0)


def _train_set(days):
    Xs, Rs = [], []
    for t in days:
        X, valid = features(Ctx(t))
        r = label(t)
        k = valid & np.isfinite(r)
        Xs.append(X[k])
        Rs.append(r[k])
    return Xs, Rs


def _clip_fit(X):
    lo, hi = np.quantile(X, 0.01, axis=0), np.quantile(X, 0.99, axis=0)
    return lo, hi


def model_for(i: int, kind: str, target: float):
    """날짜 i 를 예측할 모델(블록 캐시). 학습 = 날짜 ≤ s-2 (s = i 가 속한 블록 시작)."""
    s = WARM + ((i - WARM) // BLOCK) * BLOCK
    key = (s, kind, target)
    if key in _MODELS:
        return _MODELS[key]
    tdays = list(range(0, s - 1))       # ≤ s-2
    Xs, Rs = _train_set(tdays)
    n = len(tdays)
    cut = int(n * 0.75)
    Xa, Ra = np.vstack(Xs), np.concatenate(Rs)
    lo, hi = _clip_fit(Xa)
    ya = (Ra >= target) if target > 0 else (Ra > 0)
    # 보정: 앞 75% 날짜로 학습 → 뒤 25% 날짜로 Platt → 전체 재학습 후 같은 보정 적용
    X1, y1 = np.clip(np.vstack(Xs[:cut]), lo, hi), ya[:sum(len(x) for x in Xs[:cut])]
    X2, y2 = np.clip(np.vstack(Xs[cut:]), lo, hi), ya[sum(len(x) for x in Xs[:cut]):]
    m1 = _make(kind).fit(X1, y1)
    cal = _Platt().fit(m1.predict_proba(X2)[:, 1], y2)
    m = _make(kind).fit(np.clip(Xa, lo, hi), ya)
    _MODELS[key] = (m, cal, lo, hi)
    return _MODELS[key]


def predict(ctx: Ctx, kind: str, target: float):
    if ctx.i < WARM:
        return None
    X, valid = features(ctx)
    if not valid.any():
        return None
    m, cal, lo, hi = model_for(ctx.i, kind, target)
    p = cal(m.predict_proba(np.clip(X, lo, hi))[:, 1])
    return np.where(valid, p, -1.0)


def make_rule(kind: str, target: float, thr: float):
    def decide(ctx: Ctx):
        p = predict(ctx, kind, target)
        if p is None:
            return None
        j = int(np.argmax(p))
        if p[j] < thr:
            return None
        return j, M, {}
    decide.__name__ = f"{kind}_{target}_{thr}"
    return decide


# ── 최종 후보(study 결과로 고름 — reports/agents/day_ml.md) ──────────────
# 1: 로지스틱(라벨 >0) 보정 확률 ≥ 0.55 — 미리 정한 임계값 중 매매 40건 이상인 유일한 칸
# 2: 얕은 GBDT(라벨 >0) ≥ 0.50 — 곡선 점(사전 등록 0.55 는 24건으로 후보 불가)
# 3: 로지스틱(라벨 ≥+5%) ≥ 0.30 — +5% 확률 쪽
candidate_1 = make_rule("lr", 0.0, 0.55)
candidate_2 = make_rule("gbdt", 0.0, 0.50)
candidate_3 = make_rule("lr", 0.05, 0.30)
