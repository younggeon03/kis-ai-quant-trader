"""day_ml 측정 — 워크포워드 확률 → 그날 최고 확률 종목 → 임계값 이상만 매매. 커버리지-승률 곡선."""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import day_eval as de  # noqa: E402
from scripts.agents import day_ml as dm  # noqa: E402

CACHE = ROOT / "reports" / "agents" / "day_ml_cache.pkl"
OUT = ROOT / "reports" / "agents" / "day_ml_results.pkl"


def main():
    days = de.dev_days()
    t0 = time.time()
    if CACHE.exists():
        f, l = pickle.load(open(CACHE, "rb"))
        dm._FEAT.update(f)
        dm._LAB.update(l)
    for i in days:
        dm.features(de.Ctx(i))
        dm.label(i)
    pickle.dump(({i: dm._FEAT[i] for i in days}, {i: dm._LAB[i] for i in days}), open(CACHE, "wb"))
    print(f"피처·라벨 {len(days)}일 {time.time() - t0:.0f}s")

    # 라벨 = simulate 일치 확인(무작위 300개)
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(300):
        i, j = int(rng.choice(days)), int(rng.integers(0, 286))
        a, b = de.simulate(i, j, 30, {}), dm._LAB[i][j]
        if (a is None) != (not np.isfinite(b)) or (a is not None and abs(a - b) > 1e-9):
            bad += 1
    print(f"라벨 불일치 {bad}/300")

    # 모집단 기저율(진단: 전체 표본)
    R = np.concatenate([dm._LAB[i][dm._FEAT[i][1] & np.isfinite(dm._LAB[i])] for i in days if i >= dm.WARM])
    print(f"모집단(예측 구간 전 종목): 표본 {len(R)} · 승률 {(R > 0).mean():.1%} · +5%↑ {(R >= .05).mean():.1%} · 평균 {R.mean():+.2%}")

    res = {}
    for kind in ("lr", "gbdt"):
        for target in (0.0, 0.05):
            t1 = time.time()
            rows = []   # (day, top_p, ret_top, ret_all_mean, auc-ish)
            for i in days:
                if i < dm.WARM:
                    continue
                ctx = de.Ctx(i)
                p = dm.predict(ctx, kind, target)
                if p is None:
                    continue
                j = int(np.argmax(p))
                r = de.simulate(i, j, 30, {})
                rows.append((i, float(p[j]), r))
            res[(kind, target)] = rows
            print(f"{kind} target={target}: {len(rows)}일 {time.time() - t1:.0f}s")
    pickle.dump(res, open(OUT, "wb"))

    thrs = {0.0: [0.0, 0.50, 0.55, 0.60, 0.65, 0.70], 0.05: [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]}
    n_days = len([i for i in days if i >= dm.WARM])
    print(f"\n예측 구간 {n_days}일 (첫 {dm.WARM}일 학습만)")
    for (kind, target), rows in res.items():
        print(f"\n== {kind} · 라벨 {'>0' if target == 0 else '>=+5%'} ==")
        for th in thrs[target]:
            tr = [r for (_, p, r) in rows if p >= th and r is not None]
            n = len(tr)
            if not n:
                print(f"  thr {th:.2f}: 매매 0")
                continue
            a = np.array(tr)
            ds = [i for (i, p, r) in rows if p >= th and r is not None]
            edges = np.linspace(min(days[dm.WARM:]), max(days) + 1, 6)
            fw = []
            for k in range(5):
                f = [r for (i, p, r) in rows if p >= th and r is not None and edges[k] <= i < edges[k + 1]]
                fw.append(f"{(np.array(f) > 0).mean():.0%}({len(f)})" if f else "-(0)")
            print(f"  thr {th:.2f}: 매매 {n:3d}/{n_days} ({n / n_days:.0%}) · 승률 {(a > 0).mean():.0%} · 손실 {(a < 0).mean():.0%}"
                  f" · +5%↑ {(a >= .05).mean():.0%} · 평균 {a.mean():+.2%} · 중앙 {np.median(a):+.2%} · −3%↓ {(a <= -.03).mean():.0%}"
                  f" · 조각 {' '.join(fw)}")
        # 보정 진단: 최고 확률 버킷별 실현 승률
        ps = np.array([p for (_, p, r) in rows if r is not None])
        rs = np.array([r for (_, p, r) in rows if r is not None])
        y = rs >= target if target > 0 else rs > 0
        qs = np.quantile(ps, [0, .25, .5, .75, 1])
        print("  보정(최고확률 사분위): " + " | ".join(
            f"p {ps[(ps >= qs[k]) & (ps <= qs[k + 1])].mean():.2f}→실현 {y[(ps >= qs[k]) & (ps <= qs[k + 1])].mean():.0%}"
            for k in range(4)))


if __name__ == "__main__":
    de.enable = None
    main()
