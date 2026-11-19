"""ML 에이전트 측정 — 미리 정한 설정만 돌린다(개발 구간). PYTHONIOENCODING=utf-8 python scripts/research/agents/ml_study.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.agent_eval import load_dev, evaluate, pick_engine_atr10, pick_engine_ratio  # noqa: E402
import scripts.research.agents.ml as ml  # noqa: E402


def main():
    dev = load_dev()
    days = sorted(dev["day"].unique())
    post = dev[dev["day"] >= days[ml.WARMUP]]
    print(f"개발 {len(days)}일 · 평가(워밍업 뒤) {post['day'].nunique()}일 {days[ml.WARMUP]}~")
    res = []
    for nm, f in [("기준 배수", pick_engine_ratio), ("기준 ATR10", pick_engine_atr10)]:
        res.append(evaluate(f, nm, df=post))
    # 설정 (kind, train_pop, cand, label) — 미리 정한 8개
    cfgs = [("lr", "pass", "pass", "hit"), ("gbdt", "pass", "pass", "hit"),
            ("lr", "all", "pass", "hit"), ("gbdt", "all", "pass", "hit"),
            ("lr", "all", "all", "hit"), ("gbdt", "all", "all", "hit"),
            ("lr", "pass", "pass", "up"), ("gbdt", "pass", "pass", "up")]
    for k, tp, cd, lb in cfgs:
        t0 = time.time()
        nm = f"{k} 학습={tp} 후보={cd} 라벨={lb}"
        r = evaluate(lambda g: ml.pick_model(g, k, tp, cd, lb), nm, df=post)
        r["sec"] = time.time() - t0
        res.append(r)
    pd.DataFrame(res).to_pickle(ROOT / "reports" / "agents" / "ml_results.pkl")
    # 전체 359일(워밍업 60일은 ATR10 대체) — 후보 함수 그대로
    print("--- 전체 개발 359일(워밍업은 ATR10) ---")
    for nm, f in [("cand1 gbdt pass", ml.candidate_1), ("cand2 lr pass", ml.candidate_2),
                  ("cand3 gbdt all→pass", ml.candidate_3)]:
        evaluate(f, nm)
    # 중요도(마지막 블록 모델)
    di = len(days) - 1
    for k, tp in [("lr", "pass"), ("gbdt", "pass"), ("gbdt", "all")]:
        m = ml.get_model(k, tp, di)
        if k == "lr":
            coef = m[-1].coef_[0]
            print(f"[{k} {tp}] 표준화 계수:", ", ".join(f"{f} {c:+.2f}" for f, c in sorted(zip(ml.FEATS, coef), key=lambda x: -abs(x[1]))))
        else:
            imp = m.booster_.feature_importance("gain")
            imp = imp / imp.sum()
            print(f"[{k} {tp}] gain:", ", ".join(f"{f} {v:.0%}" for f, v in sorted(zip(ml.FEATS, imp), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
