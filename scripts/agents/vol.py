"""변동성 예측 이론 에이전트(11/18) — 장벽 도달 확률 P(max ≥ b) ≈ 2(1-Φ(b/(σ√T))) 이므로 이틀 σ 예측으로 종목을 고른다.

추정량은 모두 **그날 전일까지의 일봉, 최근 ≤90봉**으로만 계산한다(daily_feats 의 shift(1)).
    Parkinson(1980) · Garman-Klass(1980) · Rogers-Satchell(1991) · Yang-Zhang(2000) 10·20일,
    EWMA(RiskMetrics λ=0.94, 90봉 절단) · GARCH(1,1)(Bollerslev 1986, 최근 90봉 적합, 분산 타게팅) · HAR(Corsi 2009, 1·5·20 동일가중)
장중 결합(Clark 1973): σ_mix = σ_daily x ratio^0.5 등. 드리프트 장벽 확률(μ = 20일 로그수익/20).

후보 함수: candidate_1/2/3(g) → 종목코드 | None. g = agent_eval 이 넘기는 그날 day_df.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DAILY = ROOT / "data" / "cache"
COST = 0.00278
B = math.log(1.10 / (1 - COST))      # 순수익 +10% 장벽(로그)
T = 2.0                               # 이틀(09:04~d2 15:10 ≈ 1.97일, 둥글게 2)
LAM = 0.94
WIN_MAX = 90


def _yz(o, h, l, c, n):
    lo = np.log(o / c.shift(1))
    lc = np.log(c / o)
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v = lo.rolling(n).var() + k * lc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()
    return np.sqrt(v.clip(lower=0))


@lru_cache(maxsize=None)
def daily_feats(sym: str) -> Optional[pd.DataFrame]:
    p = DAILY / f"{sym}.csv"
    if not p.exists():
        return None
    dd = pd.read_csv(p, dtype={"ts": str}).sort_values("ts").reset_index(drop=True)
    dd["ts"] = dd["ts"].str[:8]
    o, h, l, c = (dd[k].astype(float).where(lambda x: x > 0) for k in ("open", "high", "low", "close"))
    r = np.log(c / c.shift(1))
    hl2 = np.log(h / l) ** 2
    park_d = hl2 / (4 * math.log(2))
    gk_d = 0.5 * hl2 - (2 * math.log(2) - 1) * np.log(c / o) ** 2
    rs_d = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    f = pd.DataFrame({"ts": dd["ts"]})
    for n in (10, 20):
        f[f"park{n}"] = np.sqrt(park_d.rolling(n).mean())
        f[f"gk{n}"] = np.sqrt(gk_d.rolling(n).mean().clip(lower=0))
        f[f"rs{n}"] = np.sqrt(rs_d.rolling(n).mean().clip(lower=0))
        f[f"yz{n}"] = _yz(o, h, l, c, n)
        f[f"cc{n}"] = r.rolling(n).std()
    # EWMA(λ=0.94) — 90봉 절단 가중(가중합 정규화)
    w = LAM ** np.arange(WIN_MAX)[::-1]
    r2 = (r ** 2).to_numpy()
    ew = np.full(len(r2), np.nan)
    for i in range(WIN_MAX, len(r2)):
        x = r2[i - WIN_MAX + 1:i + 1]
        if np.isfinite(x).all():
            ew[i] = float(np.dot(w, x) / w.sum())
    f["ewma"] = np.sqrt(ew)
    # HAR(Corsi 2009) — Parkinson 일분산의 1·5·20일 평균을 동일가중(문헌 계수 대신 둥근 값)
    f["har"] = np.sqrt((park_d + park_d.rolling(5).mean() + park_d.rolling(20).mean()) / 3.0)
    f["mu20"] = (np.log(c / c.shift(20))) / 20.0          # 일 드리프트 추정
    f["_r"] = r
    sh = f.drop(columns=["ts"]).shift(1)                  # 그날 전일까지
    sh["ts"] = f["ts"]
    return sh.set_index("ts")


def _garch_fit(x: np.ndarray):
    """GARCH(1,1) 가우시안 QML, 분산 타게팅(ω = s²(1-α-β)). 2일 분산 예측 h1+h2 을 돌려준다."""
    from scipy.optimize import minimize
    x = x - x.mean()
    s2 = float(x.var())
    if not np.isfinite(s2) or s2 <= 0:
        return np.nan

    def filt(a, b):
        om = s2 * (1 - a - b)
        h = np.empty(len(x) + 1)
        h[0] = s2
        for t in range(len(x)):
            h[t + 1] = om + a * x[t] ** 2 + b * h[t]
        return h

    def nll(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        h = filt(a, b)[:-1]
        return 0.5 * float(np.sum(np.log(h) + x ** 2 / h))

    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.80)):
        res = minimize(nll, [a0, b0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 300})
        if best is None or res.fun < best.fun:
            best = res
    a, b = best.x
    if a < 0 or b < 0 or a + b >= 0.999:
        return np.nan
    h1 = filt(a, b)[-1]
    om = s2 * (1 - a - b)
    h2 = om + (a + b) * h1
    return float(math.sqrt((h1 + h2) / 2.0))           # 일 σ 단위로 환산


_GARCH: Dict[tuple, float] = {}


def garch_sigma(sym: str, day: str) -> float:
    key = (sym, day)
    if key in _GARCH:
        return _GARCH[key]
    f = daily_feats(sym)
    v = np.nan
    if f is not None and day in f.index:
        i = f.index.get_loc(day)
        x = f["_r"].to_numpy()[max(0, i - WIN_MAX + 1):i + 1]     # 이미 shift(1) — 전일까지 90봉
        x = x[np.isfinite(x)]
        if len(x) >= 60:
            v = _garch_fit(x)
    _GARCH[key] = v
    return v


EST = ["park10", "park20", "gk10", "gk20", "rs10", "rs20", "yz10", "yz20", "cc10", "cc20", "ewma", "har", "mu20"]


def attach(g: pd.DataFrame, garch: bool = False) -> pd.DataFrame:
    g = g.copy()
    day = g["day"].iloc[0]
    vals = {k: [] for k in EST}
    for s in g["sym"]:
        f = daily_feats(s)
        row = f.loc[day] if f is not None and day in f.index else None
        for k in EST:
            vals[k].append(float(row[k]) if row is not None else np.nan)
    for k in EST:
        g[k] = vals[k]
    if garch:
        g["garch"] = [garch_sigma(s, day) for s in g["sym"]]
    return g


def barrier_p(sig, mu=0.0, b=B, t=T):
    """브라운 운동(드리프트 μ, 확산 σ)이 [0,t]에 b 이상 닿을 확률 — 반사 원리 일반형."""
    from scipy.stats import norm
    sig = np.asarray(sig, float)
    mu = np.asarray(mu, float)
    st = sig * math.sqrt(t)
    with np.errstate(all="ignore"):
        p = (1 - norm.cdf((b - mu * t) / st)) + np.exp(np.clip(2 * mu * b / sig ** 2, -50, 50)) * norm.cdf((-b - mu * t) / st)
    return np.where(np.isfinite(p), p, np.nan)


def _top(c: pd.DataFrame, col: str) -> Optional[str]:
    c = c.copy()
    c["_s"] = c[col].fillna(-1e9)
    if c.empty:
        return None
    return c.sort_values(["_s", "ratio", "sym"], ascending=[False, False, True])["sym"].iloc[0]


def _pass(g: pd.DataFrame) -> pd.DataFrame:
    return g[g["engine_pass"]]


# ── 최종 후보 ─────────────────────────────────────────────────────────────
def _mix(c: pd.DataFrame, col: str, alpha: float = 0.5) -> pd.Series:
    """Clark(1973) 혼합분포: 분산 ∝ 정보 유입(거래량) → σ_eff = σ_daily x ratio^0.5."""
    return c[col] * c["ratio"].clip(lower=0).fillna(0) ** alpha


def candidate_1(g: pd.DataFrame) -> Optional[str]:
    """전체 지수 종목(엔진 판정 없음) 중 σ_EWMA(λ=0.94, 전일까지 90봉) x √ratio 최대.
    ratio = 09:00~09:03 거래대금 / 지난 20일 같은 4분 중앙(엔진 판정과 같은 값). 판정 통과 여부·전일 대비는 보지 않는다."""
    if g.empty:
        return None
    c = attach(g)
    c["_m"] = _mix(c, "ewma")
    return _top(c, "_m")


def candidate_2(g: pd.DataFrame) -> Optional[str]:
    """엔진 판정 통과 종목 중 ATR10 x √ratio 최대 — 지금 서버(ATR10 순)에 Clark 결합만 더한 것."""
    c = _pass(g).copy()
    if c.empty:
        return None
    c["_m"] = _mix(c, "atr10")
    return _top(c, "_m")


def candidate_3(g: pd.DataFrame) -> Optional[str]:
    """엔진 판정 통과 종목 중 GARCH(1,1)(전일까지 90봉 QML·분산 타게팅) 이틀 평균 σ 예측 최대."""
    c = _pass(g)
    if c.empty:
        return None
    c = attach(c, garch=True)
    return _top(c, "garch")
