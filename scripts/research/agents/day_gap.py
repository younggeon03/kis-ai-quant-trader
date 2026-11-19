"""day_gap 후보 — 밤·낮 수익 분리·갭·주의 집중 연구(11/19, 개발 구간만).

공통 골격(Berkman 외 2012 "Paying attention" 의 반대편 — 주의 때문이 아닌 개별 갭 하락 뒤 장 초반 회복):
  09:00 시가가 전일 종가보다 1% 이상 낮게 열렸는데(종목 고유 갭 하락), 09:30 직전까지 가격이 당일 VWAP 위로 올라온 종목.
  시장 전체는 갭 하락이 아니어야 한다(시장 갭 중앙 ≥ 0) — 시장이 같이 빠진 날은 회복이 이어지지 않았다.
  여럿이면 시가 대비 회복이 가장 큰 종목 1개. 09:30 시가 매수 → 15:10 매도(후보별 익절 차이).
  조건을 만족하는 종목이 없으면 그날은 사지 않는다(현금).

ctx 메서드로만 데이터를 본다. 매수 분 m 에서 ctx.bars(m) 까지만 쓴다.
"""
from __future__ import annotations

import numpy as np

LIQ_KRW = 5e9          # 지난 20일 거래대금 중앙 50억 이상(체결 가능성)


def _snap(ctx, m):
    b = ctx.bars(m)                                   # [S,5,m] — 0..m-1 분
    o, c, v = b[:, 0], b[:, 3], b[:, 4]
    S = b.shape[0]
    o0 = np.full(S, np.nan)
    for t in range(min(3, m) - 1, -1, -1):
        x = o[:, t]
        ok = np.isfinite(x) & (x > 0)
        o0[ok] = x[ok]
    last = np.full(S, np.nan)
    for t in range(m):
        x = c[:, t]
        ok = np.isfinite(x)
        last[ok] = x[ok]
    tov = np.nansum(np.where(np.isfinite(c * v), c * v, 0.0), axis=1)
    vol = np.nansum(np.where(np.isfinite(v), v, 0.0), axis=1)
    vwap = np.where(vol > 0, tov / np.maximum(vol, 1), np.nan)
    pc = np.asarray(ctx.prev_close, dtype=float)
    with np.errstate(all="ignore"):
        gap = o0 / pc - 1
        ret = last / o0 - 1
    return gap, ret, last, vwap


def _liquid(ctx, j):
    d = ctx.daily(j, 20)
    if len(d) < 10:
        return False
    return float(np.median(d["close"].to_numpy(float) * d["volume"].to_numpy(float))) > LIQ_KRW


def gap_recover(ctx, m=30, gap_max=-0.01, need_mkt_up=True, rule=None, rel=False):
    gap, ret, last, vwap = _snap(ctx, m)
    with np.errstate(all="ignore"):
        mgap = np.nanmedian(gap) if np.isfinite(gap).any() else np.nan
        if need_mkt_up and not (np.isfinite(mgap) and mgap >= 0):
            return None
        g = gap - mgap if rel else gap
        ok = np.isfinite(g) & (g <= gap_max) & np.isfinite(vwap) & (last > vwap) & np.isfinite(ret)
    order = np.argsort(-np.where(ok, ret, -np.inf))
    for j in order[: int(ok.sum())]:
        if _liquid(ctx, int(j)):
            return int(j), m, dict(rule or {})
    return None


# ---- 최종 후보 (매수 09:30 시가 고정 — 1라운드 캐시에서 미리 고른 시각, 이웃 09:25~09:40 도 같은 방향) ----------------
def candidate_1(ctx):
    """손실 확률 최소: 종목 갭 ≤ −1% · 09:30 직전 VWAP 위 · 시장 갭 중앙 ≥ 0 → 09:30 매수, 순수익 +1% 익절, 아니면 15:10."""
    return gap_recover(ctx, m=30, gap_max=-0.01, need_mkt_up=True, rule={"tp": 0.01})


def candidate_2(ctx):
    """균형: candidate_1 과 같은 진입, 순수익 +2% 익절, 아니면 15:10."""
    return gap_recover(ctx, m=30, gap_max=-0.01, need_mkt_up=True, rule={"tp": 0.02})


def candidate_3(ctx):
    """+5% 확률: candidate_1 과 같은 진입, 순수익 +5% 익절, 아니면 15:10."""
    return gap_recover(ctx, m=30, gap_max=-0.01, need_mkt_up=True, rule={"tp": 0.05})
