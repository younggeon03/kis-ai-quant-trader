"""day_gap 1차 탐색 — 캐시(개발 구간)에서 15:10 매도 라벨로 규칙을 빠르게 잰다. 시험 설정 수를 센다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

C = np.load(Path(__file__).resolve().parent / "_day_gap_cache.npz")
D = {k: C[k] for k in C.files}
COST = 0.00278
ND = len(D["days"])
K = 5
N_TESTED = [0]


def lab(m):
    return D["exit"] / D[f"ent{m}"] * (1 - COST) - 1


def pick(score, elig, m, name, verbose=True, count=True):
    """하루 1종목: elig 중 score 최대. score nan 은 제외. 후보 없으면 현금."""
    if count:
        N_TESTED[0] += 1
    y = lab(m)
    s = np.where(elig & np.isfinite(score) & np.isfinite(y), score, -np.inf)
    tr, fold = [], [[] for _ in range(K)]
    edges = np.linspace(0, ND, K + 1).astype(int)
    for n in range(ND):
        if not np.isfinite(s[n]).any() or s[n].max() == -np.inf:
            continue
        j = int(np.argmax(s[n]))
        tr.append(y[n, j])
        fold[min(int(np.searchsorted(edges, n, side="right") - 1), K - 1)].append(y[n, j])
    a = np.array(tr)
    if verbose:
        if len(a):
            fw = " ".join(f"{(np.array(f) > 0).mean():.0%}" if f else "--" for f in fold)
            print(f"{name:<52} n={len(a):3d} 승 {np.mean(a>0):.0%} 손 {np.mean(a<0):.0%} +5 {np.mean(a>=.05):.1%} "
                  f"평균 {a.mean():+.2%} 중앙 {np.median(a):+.2%} q05 {np.quantile(a,.05):+.1%} | {fw}")
        else:
            print(f"{name:<52} n=0")
    return a


def xs(arr):  # 횡단면 중앙
    return np.nanmedian(arr, axis=1)


if __name__ == "__main__":
    np.seterr(all="ignore")
    gap = D["o0"] / D["pc"] - 1
    base = np.isfinite(gap) & (D["to20"] > 5e9)   # 거래대금 중앙 50억 이상(체결 가능성)
    print("== 무조건(전 종목 평균) 15:10 수익 ==")
    for m in [1, 5, 15, 30, 60]:
        y = lab(m)
        v = y[base & np.isfinite(y)]
        print(f"m={m:2d} 승 {np.mean(v>0):.1%} 평균 {v.mean():+.3%} +5 {np.mean(v>=.05):.1%}")
    # 횡단면 IC(스피어만 근사: 순위 상관) — 피처별
    from scipy.stats import spearmanr  # noqa
    print("== 일별 순위상관 평균(15:10 수익, m=5) ==")
    y = lab(5)
    feats = {"gap": gap, "dr1": D["dr1"], "dr3": D["dr3"], "dr5": D["dr5"], "dr10": D["dr10"],
             "on1": D["on1"], "on5": D["on5"], "on10": D["on10"], "r1": D["r1"], "r5": D["r5"],
             "atr10": D["atr10"], "ret5": D["ret5"], "dr5-on5": D["dr5"] - D["on5"],
             "dr_pos5": D["dr_pos5"], "to5/to20": D["to5"] / D["to20"]}
    for k, f in feats.items():
        ics = []
        for n in range(ND):
            ok = base[n] & np.isfinite(f[n]) & np.isfinite(y[n])
            if ok.sum() > 30:
                ics.append(spearmanr(f[n][ok], y[n][ok])[0])
        ics = np.array(ics)
        print(f"{k:<10} IC {ics.mean():+.4f}  t {ics.mean()/ics.std()*np.sqrt(len(ics)):+.2f}")
