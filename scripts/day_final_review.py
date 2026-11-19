"""하루 매매 후보 최종 심사(11/19) — 봉인 구간(2026-07-01~)에서 **한 번만** 돌린다.

미리 정한 채택 조건(봉인 구간, 매매 1건 기준): 승률 ≥ 60% · 평균 > 0 · 손실 확률 < 40% · 매매 ≥ 15건.
통과한 것 중 봉인 승률이 가장 높은 것(같으면 +5%↑)을 고른다. 없으면 내일은 현금.

    PYTHONIOENCODING=utf-8 python scripts/day_final_review.py
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.day_eval as de  # noqa: E402

TEAMS = ["day_time", "day_gap", "day_revert", "day_ml"]


def ok(st):
    return (st.get("trades", 0) >= 15 and st.get("win", 0) >= 0.60 and st.get("mean", -1) > 0
            and st.get("loss", 1) < 0.40)


def main() -> int:
    hold = de._holdout_days()
    print(f"봉인 구간 {len(hold)}일")
    de.evaluate(de.baseline_engine, "기준 엔진 09:04 → 15:10 [개발]")
    de.evaluate(de.baseline_engine, "기준 엔진 09:04 → 15:10 [봉인]", days=hold)
    rows = []
    for team in TEAMS:
        try:
            mod = importlib.import_module(f"scripts.agents.{team}")
        except Exception:
            print(f"[{team}] import 실패\n{traceback.format_exc()}")
            continue
        for k in (1, 2, 3):
            fn = getattr(mod, f"candidate_{k}", None)
            if fn is None:
                continue
            try:
                a = de.evaluate(fn, f"{team}.c{k} [개발]")
                b = de.evaluate(fn, f"{team}.c{k} [봉인]", days=hold)
            except Exception:
                print(f"[{team}.c{k}] 실패\n{traceback.format_exc()}")
                continue
            rows.append((team, k, a, b, ok(b)))
    print("\n[요약] 후보 | 개발 승률·손실·평균·매매 | 봉인 승률·손실·+5%↑·평균·매매 | 채택 조건")
    for team, k, a, b, good in sorted(rows, key=lambda r: (-r[4], -r[3].get("win", 0))):
        fa = f"{a.get('win', 0):.0%}·{a.get('loss', 0):.0%}·{a.get('mean', 0):+.2%}·{a['trades']}"
        fb = f"{b.get('win', 0):.0%}·{b.get('loss', 0):.0%}·{b.get('p5', 0):.0%}·{b.get('mean', 0):+.2%}·{b['trades']}"
        print(f"  {team}.c{k} | {fa} | {fb} | {'통과' if good else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
