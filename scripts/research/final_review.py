"""에이전트 팀 후보 최종 심사(11/18) — 봉인 구간(2026-07-01~)에서 **한 번만** 돌린다.

각 팀 파일(scripts/research/agents/{vol,micro,behav,ml}.py)의 candidate_1..3 을 개발 구간 · 봉인 구간 · 전체에서 평가하고
지금 서버 기준(ATR10 순)과 비교한다. 채택 기준(미리 정함):
  1) 봉인 구간 P(≥+10%)가 기준(ATR10)보다 높고
  2) 개발 구간 5조각 중 4조각 이상에서 기준 이상이며
  3) 엔진에서 오늘 밤 구현할 수 있다(리더가 따로 판단).
봉인 구간은 약 70일이라 1건 = 약 1.4%p 다 — 차이가 작으면 채택하지 않는다(기준보다 +3%p 이상일 때만).

    PYTHONIOENCODING=utf-8 python scripts/research/final_review.py
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.research.agent_eval as ae  # noqa: E402

TEAMS = ["vol", "micro", "behav", "ml"]


def main() -> int:
    dev = ae.load_dev()
    hold = ae._load_holdout_for_final_review_only()
    print(f"개발 {dev['day'].nunique()}일 · 봉인 {hold['day'].nunique()}일 ({hold['day'].min()}~{hold['day'].max()})")
    rows = []
    base_dev = ae.evaluate(ae.pick_engine_atr10, "기준 ATR10 [개발]", df=dev)
    base_hold = ae.evaluate(ae.pick_engine_atr10, "기준 ATR10 [봉인]", df=hold)
    ae.evaluate(ae.pick_engine_ratio, "참고 배수 순 [봉인]", df=hold)
    for team in TEAMS:
        try:
            mod = importlib.import_module(f"scripts.research.agents.{team}")
        except Exception:
            print(f"[{team}] import 실패\n{traceback.format_exc()}")
            continue
        for k in (1, 2, 3):
            fn = getattr(mod, f"candidate_{k}", None)
            if fn is None:
                continue
            try:
                a = ae.evaluate(fn, f"{team}.candidate_{k} [개발]", df=dev)
                b = ae.evaluate(fn, f"{team}.candidate_{k} [봉인]", df=hold)
            except Exception:
                print(f"[{team}.candidate_{k}] 실패\n{traceback.format_exc()}")
                continue
            wins = sum(x >= y for x, y in zip(a["folds"], base_dev["folds"]))
            ok = b["p10"] >= base_hold["p10"] + 0.03 and wins >= 4
            rows.append((team, k, a, b, wins, ok))
    print("\n[요약] 후보 | 개발 P | 개발 조각 ≥ 기준 | 봉인 P (기준 {:.1%}) | 봉인 평균 | 채택 기준 충족".format(base_hold["p10"]))
    for team, k, a, b, wins, ok in sorted(rows, key=lambda r: -r[3]["p10"]):
        print(f"  {team}.candidate_{k} | {a['p10']:.1%} | {wins}/5 | {b['p10']:.1%} | {b['mean']:+.2%} | {'예' if ok else '아니오'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
