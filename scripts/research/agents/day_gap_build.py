"""day_gap 연구용 캐시(개발 구간만) — 분봉은 ctx.bars 로만, 일봉은 data/cache CSV(그날 전일까지).

라벨(15:10 매도 수익)은 연구 분석용이다. 후보 규칙은 day_gap.py 에서 ctx 로만 판단하고 evaluate 로 다시 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.day_eval import Ctx, dev_days, COST, END  # noqa: E402

MS = [1, 5, 15, 30, 60]
OUT = Path(__file__).resolve().parent / "_day_gap_cache.npz"


def daily_feats(syms, days):
    """[D,S] 일봉 피처 — 그날 전일까지만."""
    D, S = len(days), len(syms)
    names = ["dr1", "dr3", "dr5", "dr10", "on1", "on3", "on5", "on10", "r1", "r5", "atr10", "to20", "to1",
             "dr_pos5", "on_pos5"]
    F = {k: np.full((D, S), np.nan, np.float32) for k in names}
    dkeys = np.array([str(d) for d in days])
    for j, s in enumerate(syms):
        p = ROOT / "data" / "cache" / f"{s}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype={"ts": str}).sort_values("ts").reset_index(drop=True)
        d["ts"] = d["ts"].str[:8]
        d = d[d["ts"] < "20260701"].reset_index(drop=True)
        o, h, l, c, v = (d[k].astype(float) for k in ["open", "high", "low", "close", "volume"])
        dr = np.log(c / o)
        on = np.log(o / c.shift(1))
        r = np.log(c / c.shift(1))
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1) / c
        to = c * v
        feat = pd.DataFrame({
            "dr1": dr, "dr3": dr.rolling(3).sum(), "dr5": dr.rolling(5).sum(), "dr10": dr.rolling(10).sum(),
            "on1": on, "on3": on.rolling(3).sum(), "on5": on.rolling(5).sum(), "on10": on.rolling(10).sum(),
            "r1": r, "r5": r.rolling(5).sum(), "atr10": tr.rolling(10).mean(), "to20": to.rolling(20).median(),
            "to1": to, "dr_pos5": (dr > 0).rolling(5).mean(), "on_pos5": (on > 0).rolling(5).mean(),
        })
        # 날짜 day 의 피처 = day 보다 앞선 마지막 행
        ts = d["ts"].to_numpy()
        k = np.searchsorted(ts, dkeys) - 1
        ok = k >= 0
        for name in names:
            col = feat[name].to_numpy(np.float32)
            vals = np.full(D, np.nan, np.float32)
            vals[ok] = col[k[ok]]
            F[name][:, j] = vals
    return F


def main():
    days = dev_days()
    c0 = Ctx(days[0])
    syms = c0.syms
    D, S = len(days), len(syms)
    R = {}
    R["pc"] = np.full((D, S), np.nan, np.float32)
    R["o0"] = np.full((D, S), np.nan, np.float32)
    for m in MS:
        for k in ["ent", "ret", "px", "to", "vwap", "lo", "hi"]:
            R[f"{k}{m}"] = np.full((D, S), np.nan, np.float32)
    R["exit"] = np.full((D, S), np.nan, np.float32)
    dayl = []
    for n, i in enumerate(days):
        c = Ctx(i)
        dayl.append(c.day)
        b = c.bars(391)                      # 라벨 계산용(연구)
        o, h, l, cl, v = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
        R["pc"][n] = c.prev_close
        fo = np.where(np.isfinite(o[:, :3]) & (o[:, :3] > 0), o[:, :3], np.nan)
        R["o0"][n] = np.array([x[np.isfinite(x)][0] if np.isfinite(x).any() else np.nan for x in fo])
        ex = o[:, END].copy()
        bad = ~np.isfinite(ex)
        ex[bad] = cl[bad, END - 1]
        R["exit"][n] = ex
        tov = np.nan_to_num(cl * v)
        for m in MS:
            win = o[:, m:m + 3]
            ent = np.full(S, np.nan, np.float32)
            for t in range(2, -1, -1):
                x = win[:, t]
                okx = np.isfinite(x) & (x > 0)
                ent[okx] = x[okx]
            R[f"ent{m}"][n] = ent
            # m 전까지(0..m-1) 정보
            lastc = np.full(S, np.nan, np.float32)
            for t in range(m):
                x = cl[:, t]
                okx = np.isfinite(x)
                lastc[okx] = x[okx]
            R[f"px{m}"][n] = lastc
            R[f"ret{m}"][n] = lastc / R["o0"][n] - 1
            to_m = tov[:, :m].sum(1)
            vv = np.nan_to_num(v[:, :m]).sum(1)
            R[f"to{m}"][n] = to_m
            R[f"vwap{m}"][n] = np.where(vv > 0, to_m / np.maximum(vv, 1), np.nan)
            R[f"lo{m}"][n] = np.nanmin(np.where(np.isfinite(l[:, :m]), l[:, :m], np.inf), 1)
            R[f"hi{m}"][n] = np.nanmax(np.where(np.isfinite(h[:, :m]), h[:, :m], -np.inf), 1)
        if n % 50 == 0:
            print(n, c.day, flush=True)
    F = daily_feats(syms, dayl)
    R.update(F)
    R["days"] = np.array(dayl)
    R["kosdaq"] = np.asarray(c0.kosdaq)
    np.savez_compressed(OUT, **R)
    print("saved", OUT)


if __name__ == "__main__":
    main()
