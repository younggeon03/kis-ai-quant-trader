"""2차(사후) — 1차 결과를 보고 정한 설정이라 과적합 위험 표시. 짝지은 비교(B2 대비 날짜별 차이)도 찍는다."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scripts.research.agent_eval import load_dev, evaluate, pick_engine_atr10, OUTCOME_COLS
from scripts.research.agents.behav_study import rank_pick, specs as S1

A = [("atr10", 1)]
def idio(c):
    return c.assign(idio=c["chg"] - c["ind_chg"].fillna(0))
def with_idio(cols):
    base = rank_pick(cols)
    return lambda g: base(idio(g))
specs = {
 "R2a ATR10 + |r1|": rank_pick(A + [("absr1", 1)]),
 "R2b ATR10 + 회전율": rank_pick(A + [("turn1", 1)]),
 "R2c ATR10 + 고유 상승(chg-ind_chg)": with_idio(A + [("idio", 1)]),
 "R2d ATR10 + 업종 약세(ind_chg 작은)": rank_pick(A + [("ind_chg", -1)]),
 "R2e ATR10 + |r1| + 회전율": rank_pick(A + [("absr1", 1), ("turn1", 1)]),
}

def daily(f, df):
    out = []
    for d, g in df.groupby("day", sort=True):
        s = f(g.drop(columns=OUTCOME_COLS))
        h = g[g["sym"] == s]
        out.append(float(h["rB"].iloc[0]) if s is not None and not h.empty else 0.0)
    return np.array(out) >= 0.1 - 1e-9

if __name__ == "__main__":
    df = load_dev()
    base = daily(pick_engine_atr10, df)
    allspecs = {**{k: v for k, v in S1.items() if k[:3] in ("H1b", "H1c", "H6b", "H7a", "H8b", "H5c")}, **specs}
    rng = np.random.default_rng(0)
    for k, f in allspecs.items():
        if k in specs:
            evaluate(f, k, df=df)
        x = daily(f, df)
        d = x.astype(int) - base.astype(int)
        # 블록 부트스트랩(10일 블록)으로 차이의 90% 구간
        n = len(d); bs = []
        for _ in range(2000):
            st = rng.integers(0, n - 10, n // 10)
            bs.append(np.concatenate([d[s:s + 10] for s in st]).mean())
        lo, hi = np.percentile(bs, [5, 95])
        print(f"   {k:<36} B2 대비 {d.mean():+.1%} (이김 {int((d>0).sum())}일·짐 {int((d<0).sum())}일) 90% [{lo:+.1%}, {hi:+.1%}]")
