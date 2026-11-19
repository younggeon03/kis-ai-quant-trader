"""평균 회귀 연구용 피처 표(개발 구간만). 장중 피처는 ctx.bars(m) 로만, 일봉은 전일까지만."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.day_eval import Ctx, dev_days

MS = [5, 10, 15, 30, 60]
OUT = Path(sys.argv[1])

def daily_table(syms):
    rows = {}
    for s in syms:
        d = pd.read_csv(ROOT / "data" / "cache" / f"{s}.csv", dtype={"ts": str}).sort_values("ts")
        d["ts"] = d["ts"].str[:8]
        c = d["close"].astype(float)
        tv = c * d["volume"].astype(float)
        dl = c.diff()
        up = dl.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
        dn = (-dl).clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
        rsi2 = 100 * up / (up + dn).replace(0, np.nan)
        f = pd.DataFrame({"ts": d["ts"], "adv20": tv.rolling(20).mean(), "rsi2": rsi2,
                          "ma60r": c / c.rolling(60).mean() - 1, "ret5d": c / c.shift(5) - 1,
                          "ret1d": c / c.shift(1) - 1,
                          "atr10": ((d["high"] - d["low"]) / c).rolling(10).mean()})
        # 전일까지: 그날 행에는 전일 값
        f.iloc[:, 1:] = f.iloc[:, 1:].shift(1)
        rows[s] = f.set_index("ts")
    return rows

t0 = time.time()
days = dev_days()
c0 = Ctx(days[0]); syms = [str(s) for s in c0.syms]
dt = daily_table(syms)
recs = []
for i in days:
    ctx = Ctx(i)
    B = ctx.bars(61)
    pc = ctx.prev_close.astype(float)
    dfeat = np.array([[dt[s].loc[ctx.day].values if ctx.day in dt[s].index else [np.nan]*6][0] for s in syms], dtype=float)
    for m in MS:
        b = B[:, :, :m]
        o, h, l, c, v = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
        with np.errstate(all="ignore"):
            last = np.array([c[j][np.isfinite(c[j])][-1] if np.isfinite(c[j]).any() else np.nan for j in range(len(syms))])
            op = o[:, 0]
            tp_ = (h + l + c) / 3
            vv = np.nan_to_num(v)
            pv = np.nansum(np.nan_to_num(tp_) * vv, 1); sv = vv.sum(1)
            vwap = pv / sv
            turn = np.nansum(np.nan_to_num(c) * vv, 1)
            lo = np.nanmin(l, 1)
            r_pc = last / pc - 1
        mk = np.nanmedian(r_pc)
        for j in range(len(syms)):
            recs.append((i, ctx.day, j, m, r_pc[j], last[j] / op[j] - 1, last[j] / vwap[j] - 1, turn[j],
                         last[j] / lo[j] - 1, mk, *dfeat[j]))
print("days", len(days), time.time() - t0)
df = pd.DataFrame(recs, columns=["i", "day", "j", "m", "r_pc", "r_op", "vdev", "turn", "bounce", "mkt",
                                 "adv20", "rsi2", "ma60r", "ret5d", "ret1d", "atr10"])
df["tratio"] = df["turn"] / (df["adv20"] * df["m"] / 380)
df.to_pickle(OUT)
print(df.shape, time.time() - t0)
