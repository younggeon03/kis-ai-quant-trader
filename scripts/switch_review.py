"""갈아타기 규칙 재측정(11/18, ML 팀 제안 — 미래 정보 누출을 고친 판) · 개발 · 봉인 구간.

규칙: d1 09:04 1위 매수 → 순수익 +10% 지정가. d2 09:03까지(1일차 전체 + 2일차 09:00~09:03) 못 닿았으면 d2 09:04 시가에 팔고
그날 1위를 사서 목표 = 누적 +10%에 필요한 만큼, d2 15:10 청산. 2일차 1위가 같은 종목이거나 없으면 그대로 보유(익절 유지).
ML 팀 판은 "rB ≥ 10%면 갈아타지 않음"을 썼다 — rB 에는 2일차 09:04 뒤에 닿은 것도 들어 있어 미래 정보다.

    PYTHONIOENCODING=utf-8 python scripts/switch_review.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.agent_eval as ae  # noqa: E402

COST = ae.COST
TPM = 1.10 / (1 - COST)


def _hit_before(p, entry):
    """1일차 09:04~15:09 또는 2일차 09:00~09:03 에 익절가에 닿았나 → (닿음, 체결가)."""
    for dd, m0, m1 in ((0, 4, 370), (1, 0, 4)):
        for m in range(m0, m1):
            o, h = float(p[dd, 0, m]), float(p[dd, 1, m])
            if dd == 0 and m == 4:
                o = entry
            if np.isfinite(h) and h >= entry * TPM:
                return True, (o if np.isfinite(o) and o >= entry * TPM else entry * TPM)
    return False, None


def run(df, P, pick1, pick2, switch=True):
    by_day = {d: g for d, g in df.groupby("day", sort=True)}
    days = sorted(by_day)
    res = np.zeros(len(days))
    nsw = 0
    for i, d in enumerate(days):
        g = by_day[d]
        s1 = pick1(g.drop(columns=ae.OUTCOME_COLS))
        if s1 is None:
            continue
        row = g[g["sym"] == s1].iloc[0]
        res[i] = float(row["rB"])
        if not switch or int(row["path_idx"]) < 0:
            continue
        p = P[int(row["path_idx"])]
        entry = float(row["entry"])
        hit, _ = _hit_before(p, entry)
        if hit:
            continue                                   # 기준과 같은 결과(이미 익절)
        d2 = row["next_day"]
        if d2 not in by_day:
            continue
        g2 = by_day[d2]
        s2 = pick2(g2.drop(columns=ae.OUTCOME_COLS))
        if s2 is None or s2 == s1:
            continue                                   # 그대로 보유 — rB 가 그 결과(2일차 익절 포함, 정당)
        r2row = g2[g2["sym"] == s2].iloc[0]
        if int(r2row["path_idx"]) < 0:
            continue
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
                out = o if np.isfinite(o) and o >= tgt else tgt
                break
        if out is None:
            out = float(q[0, 0, 370]) if np.isfinite(q[0, 0, 370]) else float(q[0, 3, 369])
        res[i] = w1 * (out / e2 * (1 - COST)) - 1.0
        nsw += 1
    k = ae.K
    edges = np.linspace(0, len(days), k + 1).astype(int)
    folds = [float((res[edges[f]:edges[f + 1]] >= 0.10 - 1e-9).mean()) for f in range(k)]
    return {"p10": float((res >= 0.10 - 1e-9).mean()), "folds": folds, "mean": float(res.mean()),
            "p_loss": float((res < 0).mean()), "switch": nsw, "days": len(days)}


def main() -> int:
    P = ae.load_paths()
    dev, hold = ae.load_dev(), ae._load_holdout_for_final_review_only()
    picks = {"ATR10": ae.pick_engine_atr10}
    try:
        import scripts.agents.vol as vol
        picks["ATR10x√배수(vol.c2)"] = vol.candidate_2
    except Exception as exc:  # noqa: BLE001
        print("vol import 실패", exc)
    for name, fn in picks.items():
        for part, df in (("개발", dev), ("봉인", hold)):
            for sw in (False, True):
                r = run(df, P, fn, fn, sw)
                print(f"{name:<22} [{part}] {'갈아타기' if sw else '기준    '} | P(≥+10%) {r['p10']:.1%} | 조각 "
                      + " · ".join(f"{x:.1%}" for x in r["folds"]) + f" | 평균 {r['mean']:+.2%} | 손실 {r['p_loss']:.0%}"
                      + f" | 갈아탄 날 {r['switch']}/{r['days']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
