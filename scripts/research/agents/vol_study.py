"""변동성 예측 이론 측정(11/18) — 개발 구간만. 결과 열은 진단(순위상관)에만 쓴다."""
from __future__ import annotations
import math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.agent_eval import load_dev, evaluate  # noqa
import scripts.research.agents.vol as V  # noqa

t0 = time.time()
dev = load_dev()
parts = []
for d, g in dev.groupby("day", sort=True):
    parts.append(V.attach(g, garch=False))
df = pd.concat(parts, ignore_index=True)
ep = df["engine_pass"].to_numpy()
df["garch"] = np.nan
df.loc[ep, "garch"] = [V.garch_sigma(s, d) for s, d in zip(df.loc[ep, "sym"], df.loc[ep, "day"])]
print(f"피처 {time.time()-t0:.0f}s · GARCH 결측 {df.loc[ep,'garch'].isna().mean():.1%}", flush=True)

L2 = 4 * math.log(2)
df["r4sig"] = df["range4"]                                     # 4분 고저폭(Parkinson 비례)
df["yz20_mix"] = df["yz20"] * np.sqrt(df["ratio"].clip(lower=0))      # Clark: 분산 ∝ 정보량(거래량)
df["har_mix"] = df["har"] * np.sqrt(df["ratio"].clip(lower=0))
df["atr10_mix"] = df["atr10"] * np.sqrt(df["ratio"].clip(lower=0))
df["p_drift_yz"] = V.barrier_p(df["yz20"], df["mu20"])
df["p_drift_har"] = V.barrier_p(df["har"], df["mu20"])
df["p0_yz"] = V.barrier_p(df["yz20"], 0.0)
# 순위 평균: 일봉 σ(har) + 장중 range4 (그날 판정 통과 안에서)
df["rk_har_r4"] = np.nan
sub = df[ep]
df.loc[ep, "rk_har_r4"] = (sub.groupby("day")["har"].rank(pct=True) + sub.groupby("day")["range4"].rank(pct=True)).to_numpy()
df["up_rng"] = df["up_max"] - df["dn_min"]
df.to_pickle(ROOT / "reports" / "agents" / "vol_dev_feats.pkl")

# ── 진단: 순위상관(개발 구간, 결과 열은 여기서만) ──
cols = ["atr10", "rng1", "vol20", "park10", "park20", "gk10", "gk20", "rs10", "rs20", "yz10", "yz20", "cc10", "cc20",
        "ewma", "garch", "har", "range4", "ratio", "yz20_mix", "har_mix", "p_drift_yz", "mu20"]
def within(d, col, tgt):
    v = []
    for _, g in d.groupby("day"):
        g = g[[col, tgt]].dropna()
        if len(g) >= 5:
            v.append(g[col].rank().corr(g[tgt].rank()))
    return float(np.nanmean(v)) if v else np.nan
print("\n진단 순위상관 (풀: 전체 행 Spearman · 일내: 날마다 Spearman 평균)")
print(f"{'추정량':<12}{'전체·up_max 풀':>14}{'전체·일내':>10}{'통과·풀':>10}{'통과·일내':>10}{'통과·hit10 일내':>16}")
diag = []
for c in cols:
    a = df[[c, "up_max"]].dropna(); b = df.loc[ep, [c, "up_max", "hit10"]].dropna()
    rowd = (c, a[c].rank().corr(a["up_max"].rank()), within(df, c, "up_max") if c != "garch" else np.nan,
            b[c].rank().corr(b["up_max"].rank()), within(df[ep], c, "up_max"), within(df[ep].assign(hit10=df.loc[ep,"hit10"].astype(float)), c, "hit10"))
    diag.append(rowd)
    print(f"{c:<12}" + "".join(f"{x:>12.3f}" for x in rowd[1:]))
pd.DataFrame(diag, columns=["est", "all_pool", "all_within", "pass_pool", "pass_within", "pass_within_hit10"]).to_csv(
    ROOT / "reports" / "agents" / "vol_diag.csv", index=False)

# ── 선택 규칙 평가 ──
def top_pass(col):
    def f(g):
        c = g[g["engine_pass"]]
        return V._top(c, col) if not c.empty else None
    return f
def top_all(col):
    def f(g):
        return V._top(g, col)
    return f
configs = [(f"통과·{c} 순", top_pass(c)) for c in
           ["park10", "park20", "gk10", "gk20", "rs10", "rs20", "yz10", "yz20", "cc20", "ewma", "garch", "har",
            "range4", "yz20_mix", "har_mix", "atr10_mix", "rk_har_r4", "p_drift_yz", "p_drift_har"]]
configs += [("전체 지수·har 순(판정 없음)", top_all("har")), ("전체 지수·har_mix 순(판정 없음)", top_all("har_mix"))]
print(f"\n시험 설정 {len(configs)}개 (+기준 2)")
from scripts.research.agent_eval import pick_engine_atr10, pick_engine_ratio
res = [evaluate(pick_engine_ratio, "기준: 배수 순", df=df), evaluate(pick_engine_atr10, "기준: ATR10 순", df=df)]
for n, fn in configs:
    res.append(evaluate(fn, n, df=df))
pd.DataFrame(res).to_csv(ROOT / "reports" / "agents" / "vol_results.csv", index=False)
print(f"끝 {time.time()-t0:.0f}s")
