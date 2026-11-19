"""평균 회귀·유동성 공급 후보(11/19, 에이전트 day_revert). day_eval 의 ctx 메서드로만 데이터를 본다.

공통 구조(Nagel 2012 — 시장 전체가 빠진 날 유동성 공급의 반전 수익):
  09:30(m=30) 판정. 시장(유니버스 전 종목 전일 대비 중앙값)이 빠진 날만,
  전일까지 20일 평균 거래대금 상위 100 중 전일 대비 −3% 이하이면서 당일 저점 대비 +1% 이상 반등한 종목의
  거래대금 1위를 09:30 시가에 사서, 작은 익절(순수익 0.5% 또는 1%) 아니면 15:10 매도. 손절 없음.
조건이 안 맞는 날은 사지 않는다(현금).
"""
from __future__ import annotations
import numpy as np

M = 30
_ADV = {}


def _adv20(ctx, j):
    key = (ctx.day, j)
    if key not in _ADV:
        d = ctx.daily(j, 20)
        _ADV[key] = float((d["close"] * d["volume"]).mean()) if len(d) >= 20 else np.nan
    return _ADV[key]


def _pick(ctx, mkt_max, drop=0.03, bounce_min=0.01, top=100):
    b = ctx.bars(M)                                   # 09:00~09:29 분봉만
    c, l = b[:, 3], b[:, 2]
    S = c.shape[0]
    last = np.full(S, np.nan)
    lo = np.full(S, np.nan)
    for j in range(S):
        f = np.isfinite(c[j])
        if f.any():
            last[j] = c[j][f][-1]
            lo[j] = np.nanmin(l[j])
    with np.errstate(all="ignore"):
        r = last / ctx.prev_close - 1
        bo = last / lo - 1
    if not np.isfinite(r).any():
        return None
    if np.nanmedian(r) > mkt_max:                     # 시장이 충분히 빠지지 않은 날 → 사지 않음
        return None
    adv = np.array([_adv20(ctx, j) for j in range(S)])
    rank = np.full(S, np.inf)
    ok = np.isfinite(adv)
    rank[ok] = (-adv[ok]).argsort().argsort() + 1
    cand = np.flatnonzero((rank <= top) & (r <= -drop) & (bo >= bounce_min))
    if not len(cand):
        return None
    return int(cand[np.argmax(adv[cand])])


def candidate_1(ctx):
    """시장 중앙 ≤ −0.5% · 종목 ≤ −3% · 반등 ≥ 1% → 익절 0.5%(순) / 15:10."""
    j = _pick(ctx, -0.005)
    return None if j is None else (j, M, {"tp": 0.005})


def candidate_2(ctx):
    """시장 중앙 ≤ −1% · 종목 ≤ −3% · 반등 ≥ 1% → 익절 1%(순) / 15:10."""
    j = _pick(ctx, -0.01)
    return None if j is None else (j, M, {"tp": 0.01})


def candidate_3(ctx):
    """시장 중앙 ≤ −1% · 종목 ≤ −3% · 반등 ≥ 1% → 익절 0.5%(순) / 15:10."""
    j = _pick(ctx, -0.01)
    return None if j is None else (j, M, {"tp": 0.005})


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.research.day_eval import evaluate, baseline_engine
    evaluate(baseline_engine, "기준: 엔진 09:04 ATR10 1위 → 15:10")
    for f in (candidate_1, candidate_2, candidate_3):
        evaluate(f, f.__name__ + " " + f.__doc__.split("→")[0].strip())
