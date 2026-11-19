"""시간대 효과 1차 연구(표 기반, 손절·익절 없음 — 시가 체결·시가 매도). 시험 설정 수를 센다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COST = 0.00278
T = np.load(ROOT / "reports" / "agents" / "day_time_tab.npz")
CP = list(T["cp"])
O, C, HI, LO, TV, VW, PC, T20 = (T[k] for k in ("O", "C", "HI", "LO", "TV", "VW", "PC", "T20"))
KQ = T["kosdaq"].astype(bool)
D, S, _ = O.shape
ci = {m: k for k, m in enumerate(CP)}
N_CFG = [0]


def ret(a, b):
    return O[:, :, ci[b]] / O[:, :, ci[a]] * (1 - COST) - 1


def st(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return "n=0"
    return f"n={len(r):5d} 승 {np.mean(r>0):.0%} 손 {np.mean(r<0):.0%} +5↑ {np.mean(r>=.05):.1%} 평균 {r.mean():+.2%}"


def fold(r_by_day):
    idx = np.arange(D)
    edges = np.linspace(0, D, 6).astype(int)
    out = []
    for f in range(5):
        x = r_by_day[edges[f]:edges[f + 1]]
        x = x[np.isfinite(x)]
        out.append(f"{np.mean(x>0):.0%}" if len(x) else "-")
    return " ".join(out)


def pick(score, r, mask=None, day_ok=None, name=""):
    """날마다 score 최대 종목 1개(mask 안) → r. day_ok False면 쉼."""
    N_CFG[0] += 1
    sc = np.where(np.isfinite(score) & np.isfinite(r) & (mask if mask is not None else True), score, -np.inf)
    j = np.argmax(sc, axis=1)
    ok = np.isfinite(sc[np.arange(D), j])
    if day_ok is not None:
        ok &= day_ok
    rr = np.where(ok, r[np.arange(D), j], np.nan)
    print(f"{name:<52} {st(rr)} · 조각 {fold(rr)}")
    return rr


if __name__ == "__main__":
    liq = np.isfinite(T20) & (T20 > 0)
    big = T20 >= np.nanmedian(T20, axis=1, keepdims=True)
    print("== 1. 매수·매도 시각 격자(전 종목 동일가중, 종목-일 단위) ==")
    buys = [0, 30, 60, 90, 180, 240, 330, 360]
    sells = [30, 60, 90, 180, 240, 330, 360, 370]
    for a in buys:
        for b in sells:
            if b <= a:
                continue
            r = ret(a, b)
            N_CFG[0] += 1
            line = f"매수 {a:3d} → 매도 {b:3d}: 전체 {st(r)}"
            print(line)
    print("-- 군별(매수 → 370) --")
    for a in buys:
        r = ret(a, 370)
        for gname, g in (("코스피", ~KQ[None, :] & liq), ("코스닥", KQ[None, :] & liq), ("대형", big), ("소형", ~big & liq)):
            N_CFG[0] += 1
            print(f"  {a:3d}→370 {gname}: {st(r[np.broadcast_to(g, r.shape)])}")

    print("\n== 2. 시장 장중 모멘텀(Gao 외 2018) ==")
    r30 = C[:, :, ci[30]] / PC - 1
    M30 = np.nanmean(r30, axis=1)
    for a in (330, 340, 350, 360):
        r = ret(a, 370)
        rd = np.nanmean(r, axis=1)  # 시장(동일가중) 마지막 구간 수익
        print(f"  시장 첫30분 vs 시장 {a}→370: corr {np.corrcoef(M30[np.isfinite(rd)], rd[np.isfinite(rd)])[0,1]:+.3f}")
        for cond, nm in ((M30 > 0, "M30>0"), (M30 <= 0, "M30<=0"), (M30 > 0.005, "M30>0.5%")):
            N_CFG[0] += 1
            print(f"   {a}→370 {nm} 종목-일 {st(r[cond])}")
    rbuy = {a: C[:, :, ci[a]] / PC - 1 for a in (330, 360)}
    for a in (330, 360):
        r = ret(a, 370)
        pick(r30, r, liq, None, f"G1 {a}→370 첫30분 1위")
        pick(r30, r, liq, M30 > 0, f"G2 {a}→370 첫30분 1위 · M30>0")
        pick(rbuy[a], r, liq, None, f"G3 {a}→370 당일 상승 1위")
        pick(rbuy[a], r, liq, M30 > 0, f"G4 {a}→370 당일 상승 1위 · M30>0")
        pick(r30, r, big, M30 > 0, f"G5 {a}→370 대형 첫30분 1위 · M30>0")
        rv = C[:, :, ci[a]] / VW[:, :, ci[a]] - 1
        pick(r30, r, liq & (rv > 0) & (r30 > 0), M30 > 0, f"G6 {a}→370 첫30분1위·VWAP위 · M30>0")
        # 마지막 구간 직전 30분 모멘텀(14:00→14:30)
        rlast = C[:, :, ci[a]] / C[:, :, ci[a - 30]] - 1
        pick(rlast, r, liq, M30 > 0, f"G7 {a}→370 직전30분 상승 1위 · M30>0")
        pick(-r30, r, big, M30 < 0, f"G8 {a}→370 대형 첫30분 꼴찌 · M30<0(반전)")

    print("\n== 3. 같은 시간대 지속성(Heston 외 2010) ==")
    for a, b in ((0, 30), (30, 60), (330, 370), (360, 370), (0, 370)):
        r = ret(a, b)
        for L in (1, 5):
            lag = np.full_like(r, np.nan)
            for t in range(L, D):
                lag[t] = np.nanmean(r[t - L:t], axis=0)
            N_CFG[0] += 1
            # 종목 간 순위 상관(일내 평균)
            cs = []
            for t in range(L, D):
                x, y = lag[t], r[t]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() > 30:
                    cs.append(np.corrcoef(np.argsort(np.argsort(x[m])), np.argsort(np.argsort(y[m])))[0, 1])
            print(f"  {a}→{b} 지연 평균 {L}일: 일내 순위상관 {np.mean(cs):+.3f}")
            pick(lag, r, liq, None, f"  H {a}→{b} 지난{L}일 같은 시간대 1위")
            pick(lag, r, big, None, f"  H {a}→{b} 대형 지난{L}일 같은 시간대 1위")
    print(f"\n시험 설정 수(표 단계): {N_CFG[0]}")
