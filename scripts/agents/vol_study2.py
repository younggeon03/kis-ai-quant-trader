"""추가 확인 — 판정 없이 전체 지수에서 σ x ratio^α (Clark 혼합분포) 가 다른 σ 에서도 서는지, α 민감도."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.agent_eval import evaluate
import scripts.agents.vol as V
df = pd.read_pickle(ROOT / "reports" / "agents" / "vol_dev_feats.pkl")
rt = df["ratio"].clip(lower=0)
for s in ("atr10", "yz20", "ewma", "park10"):
    df[f"{s}_mix"] = df[s] * np.sqrt(rt)
df["har_mix25"] = df["har"] * rt ** 0.25
df["har_mix100"] = df["har"] * rt
res = []
for col in ("atr10_mix", "yz20_mix", "ewma_mix", "park10_mix", "har_mix25", "har_mix100", "ratio"):
    res.append(evaluate(lambda g, c=col: V._top(g, c), f"전체 지수·{col} 순(판정 없음)", df=df))
# 전체 지수 har_mix 가 고른 종목의 성격(진단)
pick = []
for d, g in df.groupby("day"):
    s = V._top(g.drop(columns=["rB","hit10","r_d2c","r_d1c","up_max","dn_min","up_d1","dn_d1"]), "har_mix")
    pick.append(g[g.sym == s].iloc[0])
p = pd.DataFrame(pick)
print(f"har_mix(전체) 고른 종목: engine_pass {p.engine_pass.mean():.0%} · ratio 중앙 {p.ratio.median():.1f} · chg 중앙 {p.chg.median():+.2%} · "
      f"chg>0 {(p.chg>0).mean():.0%} · kosdaq {p.kosdaq.mean():.0%} · 고유 종목 {p.sym.nunique()}")
pd.DataFrame(res).to_csv(ROOT / "reports" / "agents" / "vol_results2.csv", index=False)
