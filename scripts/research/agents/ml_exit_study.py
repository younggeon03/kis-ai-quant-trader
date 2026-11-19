"""ML 에이전트 — 보조 측정: 전체 후보 단순 기준선 + 매도 규칙(1일차 종가 부진 → 2일차 09:04 갈아타기). 개발 구간만."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.agent_eval import load_dev, evaluate, load_paths, simulate_exit, pick_engine_atr10, COST, K  # noqa: E402
import scripts.research.agents.ml as ml  # noqa: E402

TPM = 1.10 / (1 - COST)


def pick_all_atr10(g):
    c = g.copy()
    c["_a"] = c["atr10"].fillna(-1.0)
    return c.sort_values(["_a", "ratio", "sym"], ascending=[False, False, True])["sym"].iloc[0]


def last_close(p, dd):
    c = p[dd, 3, :]
    f = np.where(np.isfinite(c))[0]
    return float(c[f[-1]]) if len(f) else np.nan


def run_switch(dev, P, pick1, pick2, thr):
    """thr=None 이면 기준(갈아타기 없음). thr=x: 1일차 종가 < 매수가 x (1-x) 이고 아직 미달이면 2일차 09:04 에 새 1위로."""
    by_day = {d: g for d, g in dev.groupby("day", sort=True)}
    days = sorted(by_day)
    res = np.zeros(len(days))
    nsw = 0
    for i, d in enumerate(days):
        g = by_day[d]
        s1 = pick1(g)
        if s1 is None:
            continue
        row = g[g["sym"] == s1].iloc[0]
        base = float(row["rB"])
        res[i] = base
        if thr is None or int(row["path_idx"]) < 0 or base >= 0.10 - 1e-9:
            continue
        p = P[int(row["path_idx"])]
        entry = float(row["entry"])
        c1 = last_close(p, 0)
        if not (c1 < entry * (1 - thr)):
            continue
        d2 = row["next_day"]
        if d2 not in by_day:
            continue
        g2 = by_day[d2]
        s2 = pick2(g2)
        if s2 is None or s2 == s1:
            continue
        r2row = g2[g2["sym"] == s2].iloc[0]
        if int(r2row["path_idx"]) < 0:
            continue
        # 1일차 TP 미달이 확정(base<10%)이지만 2일차 09:00~09:03 에 닿았을 수 있다
        hit_early = False
        for m in range(0, 4):
            o, h = float(p[1, 0, m]), float(p[1, 1, m])
            if np.isfinite(h) and h >= entry * TPM:
                hit_early = True
                break
        if hit_early:
            continue                                  # 기준과 같은 결과(익절)
        sell = float(p[1, 0, 4])
        if not np.isfinite(sell):
            continue
        w1 = sell / entry * (1 - COST)
        q = P[int(r2row["path_idx"])]
        e2 = float(r2row["entry"])
        tgt = e2 * (1.10 / w1) / (1 - COST)
        out = None
        for m in range(4, 370):
            o, h = float(q[0, 0, m]), float(q[0, 1, m])
            if m == 4:
                o = e2
            if np.isfinite(h) and h >= tgt:
                out = max(o, tgt)
                break
        if out is None:
            out = float(q[0, 0, 370]) if np.isfinite(q[0, 0, 370]) else float(q[0, 3, 369])
        w = w1 * (out / e2 * (1 - COST))
        res[i] = w - 1.0
        nsw += 1
    edges = np.linspace(0, len(days), K + 1).astype(int)
    folds = [float((res[edges[f]:edges[f + 1]] >= 0.10 - 1e-9).mean()) for f in range(K)]
    return {"p10": float((res >= 0.10 - 1e-9).mean()), "folds": folds, "mean": float(res.mean()),
            "p_loss": float((res < 0).mean()), "switch": nsw, "fold_min": min(folds)}


def main():
    dev = load_dev()
    days = sorted(dev["day"].unique())
    post = dev[dev["day"] >= days[ml.WARMUP]]
    print("--- 전체 후보 단순 기준선(워밍업 뒤 299일) ---")
    evaluate(pick_all_atr10, "전체 지수 ATR10 1위", df=post)
    evaluate(pick_all_atr10, "전체 지수 ATR10 1위(359일)")
    # 전체 후보 GBDT 가 무엇을 고르나
    picks = []
    for d, g in post.groupby("day", sort=True):
        s = ml.pick_model(g.drop(columns=["rB", "hit10"]), "gbdt", "all", "all")
        r = g[g["sym"] == s].iloc[0]
        a = pick_all_atr10(g)
        picks.append({"day": d, "sym": s, "pass": bool(r["engine_pass"]), "same_atr": s == a,
                      "atr10": r["atr10"], "chg": r["chg"], "ratio": r["ratio"], "kosdaq": r["kosdaq"], "hit": r["hit10"]})
    pk = pd.DataFrame(picks)
    print("전체 GBDT 선택: 판정통과 비율 {:.0%} · 전체ATR10 1위와 같음 {:.0%} · 코스닥 {:.0%} · 중앙 atr10 {:.3f} · 중앙 chg {:+.2%} · 중앙 배수 {:.2f}".format(
        pk["pass"].mean(), pk["same_atr"].mean(), pk["kosdaq"].mean(), pk["atr10"].median(), pk["chg"].median(), pk["ratio"].median()))
    print("  판정통과 선택의 적중 {:.1%} · 비통과 선택의 적중 {:.1%} · 고유 종목 {}개 · 최다 종목 {}".format(
        pk[pk["pass"]]["hit"].mean(), pk[~pk["pass"]]["hit"].mean(), pk["sym"].nunique(), pk["sym"].value_counts().head(5).to_dict()))
    P = load_paths()
    print("--- 매도 규칙: 판정 통과 + ATR10 고정, 2일차 갈아타기(새 1위 = 그날 판정 통과 ATR10) · 359일 ---")
    for thr in (None, 0.0, 0.03, 0.05, 0.08):
        r = run_switch(dev, P, pick_engine_atr10, pick_engine_atr10, thr)
        print(f"thr={thr}: P {r['p10']:.1%} | 조각 " + " · ".join(f"{x:.1%}" for x in r["folds"])
              + f" | 평균 {r['mean']:+.2%} | 손실 {r['p_loss']:.0%} | 갈아탄 날 {r['switch']}")
    print("--- 같은 규칙을 후보 3(GBDT 전체학습 → 판정통과) 선택에 · 359일 ---")
    c3 = lambda g: ml.candidate_3(g.drop(columns=["rB", "hit10", "r_d2c", "r_d1c", "up_max", "dn_min", "up_d1", "dn_d1"]))
    for thr in (None, 0.05):
        r = run_switch(dev, P, c3, c3, thr)
        print(f"thr={thr}: P {r['p10']:.1%} | 조각 " + " · ".join(f"{x:.1%}" for x in r["folds"])
              + f" | 평균 {r['mean']:+.2%} | 손실 {r['p_loss']:.0%} | 갈아탄 날 {r['switch']}")


if __name__ == "__main__":
    main()
