"""보유 기간 비교(11/18 사용자 요청) — 09:03에 동시 상위 1위를 사서 09:30 / 10:00 / 그날 15:10 / 다음 날 15:10 에 판다.

선택은 관제 화면과 같은 함수(`monitor.ranking.cross_rows`): 09:02 분봉까지의 직전 N분 거래대금(또는 누적) x 전일 대비 등락률.
매수는 다음 분 시가(09:03). 자금 전부 한 종목. 떨어질 때의 규칙을 비교한다:
  none     끝까지 보유
  hs3..10  장중 하드스톱: 분봉 저가 ≤ 매수가 x(1-k) → 그 가격(시가가 이미 아래면 시가)
  cs0/cs3  (이틀 보유만) 첫날 종가 ≤ 매수가 x(1-k) → 다음 날 시가
  be5      +5% 이상 올랐다가 매수가로 돌아오면 매수가에(본전 지키기)
하루 단위는 "이틀 연속 했을 때"(연속 두 거래일의 복리)도 같이 본다 — 남은 이틀(11/19~20)과 같은 꼴.

    PYTHONIOENCODING=utf-8 python scripts/hold_study.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import get_settings  # noqa: E402
from monitor.ranking import cross_rows  # noqa: E402
from strategy.opening import minute_index  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

FD = ROOT / "data" / "cache" / "fullday"
DAILY = ROOT / "data" / "cache"
SEL_M = 31                   # 선택에 쓰는 분(09:00~09:30)
END_MIN = 370                # 15:10
HORIZONS = {"0930": (0, 30), "1000": (0, 60), "d1_1510": (0, END_MIN), "d2_1510": (1, END_MIN)}
STOPS = ["none", "hs3", "hs5", "hs7", "hs10", "cs0", "cs3", "be5"]
PICKS = [("3분", 3, 2), ("누적", 0, 2), ("3분·09:10", 3, 9), ("3분·09:30", 3, 29), ("무작위", -1, 2)]


def _path(sym: str) -> Path:
    p = FD / f"{sym}.csv"
    return p if p.exists() else FD / f"{sym}.csv.gz"


def load_all(syms: List[str]):
    """종목별 하루 전체 분봉 [날짜 → (O,H,L,C) 381분 float32] + 전일 종가. 메모리를 아끼려고 float32."""
    data: Dict[str, Dict[str, np.ndarray]] = {}
    pcs: Dict[str, Dict[str, float]] = {}
    md5 = hashlib.md5()
    for k, sym in enumerate(syms, 1):
        p, dp = _path(sym), DAILY / f"{sym}.csv"
        if not p.exists() or not dp.exists():
            continue
        dd = pd.read_csv(dp, dtype={"ts": str}).sort_values("ts")
        pcs[sym] = dict(zip(dd["ts"].astype(str).str[:8], dd["close"].astype(float).shift(1)))
        fd = pd.read_csv(p, dtype={"ts": str})
        fd = fd.assign(mi=minute_index(fd["ts"]), day=fd["ts"].str[:8])
        fd = fd[(fd["mi"] >= 0) & (fd["mi"] <= 390)]
        md5.update(sym.encode())
        md5.update(np.ascontiguousarray(fd["close"].to_numpy(float)).tobytes())
        per: Dict[str, np.ndarray] = {}
        for dy, g in fd.groupby("day", sort=False):
            arr = np.full((5, 391), np.nan, dtype=np.float32)
            mi = g["mi"].to_numpy()
            for r, col in enumerate(("open", "high", "low", "close", "volume")):
                arr[r, mi] = g[col].to_numpy(float)
            arr[3] = pd.Series(arr[3]).ffill().to_numpy(np.float32)
            per[dy] = arr
        data[sym] = per
        if k % 50 == 0:
            print(f"  적재 {k}/{len(syms)}", flush=True)
    return data, pcs, md5.hexdigest()[:16]


def picks_for_day(data, pcs, day: str, win: int, m: int, rng) -> Optional[str]:
    """그날 m 분 종가까지로 고른 1위(win=-1 이면 무작위)."""
    rows = []
    for sym, per in data.items():
        arr = per.get(day)
        pc = pcs[sym].get(day)
        if arr is None or pc is None or not np.isfinite(pc) or pc <= 0:
            continue
        c = arr[3, :m + 1]
        if not np.isfinite(c[-1]) or c[-1] <= 0 or not np.isfinite(arr[0, m + 1]):
            continue
        tvm = np.nan_to_num(c.astype(float)) * np.nan_to_num(arr[4, :m + 1].astype(float))
        tv = tvm.sum() if win == 0 else tvm[-win:].sum() if win > 0 else 0.0
        rows.append({"symbol": sym, "chg_pct": (float(c[-1]) / pc - 1.0) * 100.0, "turnover": float(tv)})
    if len(rows) < 50:
        return None
    if win < 0:
        return rows[int(rng.integers(len(rows)))]["symbol"]
    cr = cross_rows(rows, 1)
    return cr[0]["symbol"] if cr else None


def run_trade(arrs: List[np.ndarray], entry_m: int, horizon: Tuple[int, int], stop: str, cost: float) -> Optional[float]:
    """arrs[0]=진입일, arrs[1]=다음 날. 매수 = 진입일 entry_m 분 시가."""
    d_last, m_last = horizon
    if d_last >= len(arrs):
        return None
    a0 = arrs[0]
    entry = float(a0[0, entry_m])
    if not np.isfinite(entry) or entry <= 0:
        return None
    k = {"hs3": 0.03, "hs5": 0.05, "hs7": 0.07, "hs10": 0.10}.get(stop)
    hs = entry * (1 - k) if k else -np.inf
    peak = entry
    for d in range(d_last + 1):
        a = arrs[d]
        m0 = entry_m if d == 0 else 0
        m1 = m_last if d == d_last else END_MIN
        if d == 1 and stop in ("cs0", "cs3"):
            c0 = float(arrs[0][3, 380]) if np.isfinite(arrs[0][3, 380]) else float(arrs[0][3, END_MIN])
            if c0 <= entry * (1 - (0.0 if stop == "cs0" else 0.03)):
                fin = np.flatnonzero(np.isfinite(a[0, :END_MIN]))     # 다음 날 첫 체결(09:00 분봉이 비면 그다음)
                if not len(fin):
                    return None
                return float(a[0, fin[0]]) / entry - 1.0 - cost
        for m in range(m0, m1):
            o, h, lo = float(a[0, m]), float(a[1, m]), float(a[2, m])
            if not np.isfinite(h):
                continue
            if d == 0 and m == entry_m:
                o = entry
            if np.isfinite(o) and o <= hs:
                return o / entry - 1.0 - cost
            if lo <= hs:
                return hs / entry - 1.0 - cost
            if stop == "be5":
                if peak >= entry * 1.05 and lo <= entry:
                    px = o if np.isfinite(o) and o <= entry else entry
                    return px / entry - 1.0 - cost
                peak = max(peak, h)
    a = arrs[d_last]
    px = float(a[0, m_last]) if np.isfinite(a[0, m_last]) else float(a[3, m_last - 1])
    return px / entry - 1.0 - cost


def stats(r: np.ndarray) -> Dict[str, float]:
    if not len(r):
        return {}
    return {"n": int(len(r)), "mean": float(r.mean()), "median": float(np.median(r)), "p_loss": float((r < 0).mean()),
            "p_le_m5": float((r <= -0.05).mean()), "p_le_m10": float((r <= -0.10).mean()),
            "p_ge_5": float((r >= 0.05).mean()), "p_ge_10": float((r >= 0.10).mean()), "p_ge_20": float((r >= 0.20).mean()),
            "p5": float(np.percentile(r, 5)), "p95": float(np.percentile(r, 95))}


def main() -> int:
    enable_utf8_console()
    s = get_settings()
    cost = s.round_trip_cost
    syms = sorted({x for v in s.index_members().values() for x in v})
    t0 = time.time()
    data, pcs, fp = load_all(syms)
    days = sorted({d for per in data.values() for d in per})
    print(f"지문 {fp} · 분봉 {len(data)}종목 · {len(days)}일 {days[0]}~{days[-1]} · 왕복 {cost:.4%} · 적재 {time.time() - t0:.0f}s", flush=True)

    results = []
    for plabel, win, m in PICKS:
        rng = np.random.default_rng(11)
        pick = {d: picks_for_day(data, pcs, d, win, m, rng) for d in days}
        for hz, (d_last, m_last) in HORIZONS.items():
            if m_last <= m + 1:
                continue
            for stop in STOPS:
                if stop in ("cs0", "cs3") and d_last == 0:
                    continue
                per_day: Dict[str, float] = {}
                for j, d in enumerate(days[:-1]):
                    sym = pick.get(d)
                    if not sym:
                        continue
                    arrs = [data[sym].get(d), data[sym].get(days[j + 1])]
                    if arrs[0] is None or (d_last == 1 and arrs[1] is None):
                        continue
                    r = run_trade(arrs, m + 1, (d_last, m_last), stop, cost)
                    if r is not None:
                        per_day[d] = r
                single = np.array(list(per_day.values()))
                # 이틀 연속(남은 이틀과 같은 꼴): 하루 단위는 연속 두 날 복리, 이틀 보유는 그 한 번
                if d_last == 0:
                    two = np.array([(1 + per_day[d]) * (1 + per_day[days[j + 1]]) - 1.0
                                    for j, d in enumerate(days[:-1]) if d in per_day and days[j + 1] in per_day])
                else:
                    two = single
                results.append({"pick": plabel, "horizon": hz, "stop": stop, "single": stats(single), "two_days": stats(two)})
        print(f"  {plabel} 끝 ({time.time() - t0:.0f}s)", flush=True)

    hz_name = {"0930": "09:30 매도", "1000": "10:00 매도", "d1_1510": "당일 15:10", "d2_1510": "다음날 15:10"}
    print("\n선택 | 보유 | 떨어지면 | [이틀 기준] 평균 | 중앙 | 손실 확률 | −5%↓ | −10%↓ | +10%↑ | +20%↑ | 하위5% | 상위5%")
    for r in results:
        t = r["two_days"]
        if not t:
            continue
        print(f"{r['pick']} | {hz_name[r['horizon']]} | {r['stop']} | {t['mean']:+.2%} | {t['median']:+.2%} | {t['p_loss']:.0%} | "
              f"{t['p_le_m5']:.0%} | {t['p_le_m10']:.0%} | {t['p_ge_10']:.1%} | {t['p_ge_20']:.1%} | {t['p5']:+.1%} | {t['p95']:+.1%}")
    out = ROOT / "reports" / "hold_study.json"
    out.write_text(json.dumps({"fingerprint": fp, "symbols": len(data), "days": len(days), "cost": cost, "results": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장 {out} · {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
