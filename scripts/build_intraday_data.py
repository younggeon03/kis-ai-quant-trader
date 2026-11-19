"""하루 매매 연구용 공용 데이터(11/19) — 지수 종목 x 거래일 x 1분봉(09:00~15:30) 디스크 매핑 배열.

출력(reports/agent_data/):
  min1.npy       float32 [날짜 D, 종목 S, 5(시·고·저·종·거래량), 391분] — 분 인덱스 0 = 09:00, 370 = 15:10, 390 = 15:30. 없으면 NaN
  min1_meta.npz  days[D], syms[S], prev_close[D,S](전일 종가, 일봉 캐시), ind[S](KRX 업종), kosdaq[S]
np.load(..., mmap_mode="r")로 읽으면 메모리를 거의 안 쓴다(여러 에이전트가 페이지 캐시를 나눠 쓴다).

    PYTHONIOENCODING=utf-8 python scripts/build_intraday_data.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import get_settings  # noqa: E402
from strategy.opening import minute_index  # noqa: E402

FD = ROOT / "data" / "cache" / "fullday"
OUT = ROOT / "reports" / "agent_data"


def main() -> int:
    s = get_settings()
    syms = sorted({x for v in s.index_members().values() for x in v})
    syms = [x for x in syms if (FD / f"{x}.csv").exists() or (FD / f"{x}.csv.gz").exists()]
    t0 = time.time()
    # 날짜 달력 — 첫 패스로 모은다
    days = set()
    for x in syms[:40]:
        p = FD / f"{x}.csv" if (FD / f"{x}.csv").exists() else FD / f"{x}.csv.gz"
        days |= set(pd.read_csv(p, dtype={"ts": str}, usecols=["ts"])["ts"].str[:8])
    days = sorted(days)
    di = {d: i for i, d in enumerate(days)}
    D, S = len(days), len(syms)
    OUT.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(OUT / "min1.npy", mode="w+", dtype=np.float32, shape=(D, S, 5, 391))
    arr[:] = np.nan
    prev = np.full((D, S), np.nan, dtype=np.float64)
    for j, sym in enumerate(syms):
        p = FD / f"{sym}.csv" if (FD / f"{sym}.csv").exists() else FD / f"{sym}.csv.gz"
        fd = pd.read_csv(p, dtype={"ts": str})
        fd = fd.assign(mi=minute_index(fd["ts"]), day=fd["ts"].str[:8])
        fd = fd[(fd["mi"] >= 0) & (fd["mi"] <= 390) & fd["day"].isin(di)]
        for dy, g in fd.groupby("day", sort=False):
            i = di[dy]
            mi = g["mi"].to_numpy()
            for r, col in enumerate(("open", "high", "low", "close", "volume")):
                arr[i, j, r, mi] = g[col].to_numpy(np.float32)
        dd = pd.read_csv(ROOT / "data" / "cache" / f"{sym}.csv", dtype={"ts": str}).sort_values("ts")
        pc = dict(zip(dd["ts"].str[:8], dd["close"].astype(float).shift(1)))
        for dy, i in di.items():
            v = pc.get(dy)
            if v is not None and np.isfinite(v):
                prev[i, j] = v
        if (j + 1) % 50 == 0:
            arr.flush()
            print(f"  {j + 1}/{S} ({time.time() - t0:.0f}s)", flush=True)
    arr.flush()
    kind = ROOT / "data" / "krx" / "kind_corplist_20261118.csv"
    rows = list(csv.reader(open(kind, encoding="utf-8"))) if kind.exists() else []
    ind_of = {r[2].zfill(6): r[3] for r in rows[1:] if len(r) > 3}
    mkt = s._universe_blob().get("sectors") or {}
    np.savez(OUT / "min1_meta.npz", days=np.array(days), syms=np.array(syms), prev_close=prev,
             ind=np.array([ind_of.get(x, "?") for x in syms]), kosdaq=np.array([mkt.get(x) == "KOSDAQ" for x in syms]))
    print(f"저장 {OUT / 'min1.npy'} · {D}일 x {S}종목 · {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
