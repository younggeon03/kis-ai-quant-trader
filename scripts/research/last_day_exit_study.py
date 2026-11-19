"""마지막 하루 매도 타이밍(11/19 사용자 요청) — 09:04 장 시작 판정 1위(ATR10 순)를 사서 그날 안에 판다.

데이터: `scripts/research/build_agent_data.py`가 만든 rows.pkl·paths.npz(지수 286종목 · 430거래일, 판정 통과 종목의 분 경로).
선택: `agent_eval.pick_engine_atr10`(서버와 같은 규칙). 매수 = 09:04 시가. 왕복 비용 0.278%.
체결: 익절은 분봉 고가가 닿으면 그 가격(시가가 이미 위면 시가), 손절은 저가가 닿으면 그 가격(시가가 아래면 시가),
같은 분에 둘 다면 손절 먼저(보수적). 정해진 시각 매도는 그 분 시가. 다 안 걸리면 15:10 시가.

    PYTHONIOENCODING=utf-8 python scripts/research/last_day_exit_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.research.agent_eval as ae  # noqa: E402

COST = ae.COST
END = 370          # 15:10


def m_of(hhmm: str) -> int:
    return (int(hhmm[:2]) - 9) * 60 + int(hhmm[2:])


def run(p, entry, tp=None, sl=None, t_exit=None, trail=None):
    """p = 하루 경로 [4(시·고·저·종), 391분]. 결과 = 순수익."""
    tpx = entry * (1 + tp) / (1 - COST) if tp is not None else np.inf       # 순수익 기준
    slx = entry * (1 - sl) if sl is not None else -np.inf
    last = END if t_exit is None else t_exit
    peak = entry
    for m in range(4, last):
        o, h, lo = float(p[0, m]), float(p[1, m]), float(p[2, m])
        if m == 4:
            o = entry
        if not np.isfinite(h):
            continue
        if trail is not None:
            act, dd = trail
            if peak >= entry * (1 + act) and lo <= peak * (1 - dd):
                px = o if np.isfinite(o) and o <= peak * (1 - dd) else peak * (1 - dd)
                return px / entry * (1 - COST) - 1
        if np.isfinite(o) and o <= slx:
            return o / entry * (1 - COST) - 1
        if lo <= slx:
            return slx / entry * (1 - COST) - 1
        if np.isfinite(o) and o >= tpx:
            return o / entry * (1 - COST) - 1
        if h >= tpx:
            return tpx / entry * (1 - COST) - 1
        peak = max(peak, h)
    px = float(p[0, last]) if np.isfinite(p[0, last]) else float(p[3, last - 1])
    return px / entry * (1 - COST) - 1


def main() -> int:
    df = ae._rows()
    P = ae.load_paths()
    picks = []
    for d, g in df.groupby("day", sort=True):
        s = ae.pick_engine_atr10(g.drop(columns=ae.OUTCOME_COLS))
        if s is None:
            continue
        r = g[g["sym"] == s].iloc[0]
        if int(r["path_idx"]) < 0:
            continue
        picks.append((d, P[int(r["path_idx"])][0], float(r["entry"])))
    print(f"매매일 {len(picks)}일 (판정 통과 종목이 있던 날, 2025-01~2026-10)")
    rules = {}
    for t in ("0930", "1000", "1030", "1100", "1200", "1300", "1400", "1500"):
        rules[f"{t[:2]}:{t[2:]} 매도"] = dict(t_exit=m_of(t))
    rules["15:10 매도(종가 근처)"] = {}
    for tp in (0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
        rules[f"익절 +{tp:.0%} · 아니면 15:10"] = dict(tp=tp)
    for tp, sl in ((0.03, 0.02), (0.04, 0.03), (0.05, 0.03), (0.05, 0.05), (0.10, 0.05)):
        rules[f"익절 +{tp:.0%} · 손절 −{sl:.0%}"] = dict(tp=tp, sl=sl)
    for act, dd in ((0.03, 0.02), (0.05, 0.03), (0.03, 0.03)):
        rules[f"트레일링(+{act:.0%} 뒤 고점 −{dd:.0%})"] = dict(trail=(act, dd))
    rules["익절 +4% · 트레일링(+3% 뒤 −2%)"] = dict(tp=0.04, trail=(0.03, 0.02))

    print("\n매도 방식 | 평균 | 중앙 | 손실 확률 | +2%↑ | +4%↑ | +5%↑ | +10%↑ | −5%↓ | 2025 평균 · 2026 평균")
    out = []
    for name, kw in rules.items():
        r = np.array([run(p, e, **kw) for _, p, e in picks])
        y = np.array([d[:4] for d, _, _ in picks])
        out.append((name, r))
        print(f"{name} | {r.mean():+.2%} | {np.median(r):+.2%} | {(r < 0).mean():.0%} | {(r >= .02).mean():.0%} | "
              f"{(r >= .04).mean():.0%} | {(r >= .05).mean():.0%} | {(r >= .10).mean():.1%} | {(r <= -.05).mean():.0%} | "
              f"{r[y == '2025'].mean():+.2%} · {r[y == '2026'].mean():+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
