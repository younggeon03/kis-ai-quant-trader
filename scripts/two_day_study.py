"""이틀 매매 평가(11/18 사용자 목표: 남은 이틀 누적 +10%) — 엔진 장 시작 판정 1위, 1자리, 09:04 매수.

연속 두 거래일(d1, d2)마다 한 번의 "이틀 대회"를 시뮬레이션하고 이틀 누적 수익의 분포를 본다. 핵심 지표는 P(누적 ≥ +10%).
방식:
  A  이틀 보유(상한 없음) — d2 15:10 매도
  B  이틀 보유, 누적 +10% 닿으면 매도
  C  +5% 익절 → (d1에 익절했으면) d2 09:04 그날 1위 재매수 → +5% 익절 또는 d2 15:10
  D  C와 같고 두 번째 매매 목표 = 누적 +10%까지 남은 만큼
  E  하루 단위: 매일 09:04 매수 · +5% 익절 또는 당일 15:10 · 이틀
손절: none / hs7(장중 −7%) / cs0(d1 종가 < 매수가면 d2 시가 매도 → 재매수 가능).
체결: 익절·손절은 분봉 고가·저가가 닿으면 그 가격(시가가 이미 넘으면 시가), 같은 분이면 손절 먼저. 왕복 비용 매 매매.

    PYTHONIOENCODING=utf-8 python scripts/two_day_study.py [--filter 이름]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import get_settings  # noqa: E402
import scripts.hold_study as hs  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

ENTRY_M = 4
END_M = hs.END_MIN


def walk(arrs: List[np.ndarray], d0: int, m0: int, entry: float, last_day: int, tp: Optional[float],
         stop: str) -> Tuple[float, int, int]:
    """arrs[d] 분봉. (d0, m0)부터 보유, last_day 15:10까지. 돌려주는 값 = (매도가, 매도 날, 매도 분).
    cs0 은 d0 종가 < 매수가면 다음 날 09:00 첫 체결에 판다."""
    tpx = entry * (1 + tp) if tp is not None else np.inf
    slx = entry * 0.93 if stop == "hs7" else -np.inf
    for d in range(d0, last_day + 1):
        a = arrs[d]
        if d > d0 and stop == "cs0":
            c_prev = arrs[d - 1][3, 380] if np.isfinite(arrs[d - 1][3, 380]) else arrs[d - 1][3, END_M]
            if c_prev < entry:
                fin = np.flatnonzero(np.isfinite(a[0, :END_M]))
                if len(fin):
                    return float(a[0, fin[0]]), d, int(fin[0])
        ms = m0 if d == d0 else 0
        me = END_M
        for m in range(ms, me):
            o = entry if (d == d0 and m == m0) else float(a[0, m])
            h, lo = float(a[1, m]), float(a[2, m])
            if not np.isfinite(h):
                continue
            if np.isfinite(o) and o <= slx:
                return o, d, m
            if lo <= slx:
                return slx, d, m
            if np.isfinite(o) and o >= tpx:
                return o, d, m
            if h >= tpx:
                return tpx, d, m
    a = arrs[last_day]
    px = float(a[0, END_M]) if np.isfinite(a[0, END_M]) else float(a[3, END_M - 1])
    return px, last_day, END_M


def episode(data, ep, d1: str, d2: str, mode: str, stop: str, cost: float) -> Optional[float]:
    p1 = (ep.get(d1) or [None])[0]
    eq = 1.0
    if p1 is None:
        held_until = (0, ENTRY_M)          # d1 매매 없음
    else:
        a1, b1 = data[p1].get(d1), data[p1].get(d2)
        if a1 is None or b1 is None:
            return None
        entry = float(a1[0, ENTRY_M])
        if not np.isfinite(entry) or entry <= 0:
            return None
        arrs = [a1, b1]
        if mode == "A":
            px, xd, xm = walk(arrs, 0, ENTRY_M, entry, 1, None, stop)
        elif mode == "B":
            px, xd, xm = walk(arrs, 0, ENTRY_M, entry, 1, 1.10 / (1 - cost) - 1.0, stop)   # 비용 뺀 순수익 +10%
        elif mode in ("C", "D"):
            px, xd, xm = walk(arrs, 0, ENTRY_M, entry, 1, 0.05, stop)
        elif mode == "E":
            px, xd, xm = walk(arrs, 0, ENTRY_M, entry, 0, 0.05, "hs7" if stop == "hs7" else "none")
        else:
            raise ValueError(mode)
        eq *= px / entry * (1 - cost)
        held_until = (xd, xm)
        if mode in ("A", "B"):
            return eq - 1.0
    # 두 번째 매매: d2 09:04 에 손이 비어 있으면(d1에 팔았거나 d2 09:04 전에 팔았으면) 그날 1위를 산다
    if mode in ("A", "B"):
        return eq - 1.0 if p1 is not None else 0.0
    xd, xm = held_until
    if xd == 1 and xm > ENTRY_M:
        return eq - 1.0
    if mode in ("C", "D") and eq - 1.0 >= 0.10:
        return eq - 1.0
    p2 = (ep.get(d2) or [None])[0]
    if p2 is None:
        return eq - 1.0
    a2 = data[p2].get(d2)
    if a2 is None:
        return eq - 1.0
    entry = float(a2[0, ENTRY_M])
    if not np.isfinite(entry) or entry <= 0:
        return eq - 1.0
    if mode == "D":
        tp = max(0.005, 1.10 / (eq * (1 - cost)) - 1.0)
    else:
        tp = 0.05
    st = "hs7" if stop == "hs7" else "none"
    px, _, _ = walk([a2], 0, ENTRY_M, entry, 0, tp, st)
    eq *= px / entry * (1 - cost)
    return eq - 1.0


def summarize(r: np.ndarray) -> Dict[str, float]:
    return {"n": int(len(r)), "p_ge10": float((r >= 0.10).mean()), "p_ge5": float((r >= 0.05).mean()),
            "mean": float(r.mean()), "median": float(np.median(r)), "p_loss": float((r < 0).mean()),
            "p_le_m10": float((r <= -0.10).mean())}


def run(data, ep, days, cost, label=""):
    rows = []
    for mode in ("A", "B", "C", "D", "E"):
        for stop in ("none", "hs7", "cs0"):
            if mode == "E" and stop == "cs0":
                continue
            out = {"all": [], "2025": [], "2026": []}
            for j in range(len(days) - 1):
                r = episode(data, ep, days[j], days[j + 1], mode, stop, cost)
                if r is None:
                    continue
                out["all"].append(r)
                out[days[j][:4]].append(r)
            res = {k: summarize(np.array(v)) for k, v in out.items() if v}
            rows.append({"mode": mode, "stop": stop, **res})
            a, y5, y6 = res["all"], res.get("2025", {}), res.get("2026", {})
            print(f"{label}{mode} {stop:>4} | P(≥+10%) {a['p_ge10']:.1%} (25 {y5.get('p_ge10', 0):.1%} · 26 {y6.get('p_ge10', 0):.1%}) | "
                  f"P(≥+5%) {a['p_ge5']:.1%} | 평균 {a['mean']:+.2%} (25 {y5.get('mean', 0):+.2%} · 26 {y6.get('mean', 0):+.2%}) | "
                  f"중앙 {a['median']:+.2%} | 손실 {a['p_loss']:.0%} | −10%↓ {a['p_le_m10']:.1%}", flush=True)
    return rows


def main() -> int:
    enable_utf8_console()
    s = get_settings()
    cost = s.round_trip_cost
    syms_all = sorted({x for v in s.index_members().values() for x in v})
    t0 = time.time()
    data, pcs, fp = hs.load_all(syms_all)
    days = sorted({d for per in data.values() for d in per})
    print(f"지문 {fp} · {len(data)}종목 · {len(days)}일 · 적재 {time.time() - t0:.0f}s", flush=True)
    ep = hs.engine_picks(data, pcs, days)
    rows = run(data, ep, days, cost, "엔진 ")
    if "--filters" in sys.argv:
        # filter_study 후보(엔진 판정 통과 전 종목 + 피처)로 필터를 건 1위 — 11/18 필터 측정에서 두 해 모두 나은 것
        import pandas as pd
        cd = pd.read_pickle(ROOT / "reports" / "filter_cands.pkl")
        fsets = {
            "52주80%": lambda f: f["near52"] >= 0.80,
            "52주80%+MAX15%": lambda f: (f["near52"] >= 0.80) & (f["max20"] < 0.15),
            "52주80%+MAX15%+배수10": lambda f: (f["near52"] >= 0.80) & (f["max20"] < 0.15) & (f["ratio"] >= 10),
            "52주80%+MAX15%+장중상승": lambda f: (f["near52"] >= 0.80) & (f["max20"] < 0.15) & (f["ret4"] > 0),
        }
        for name, fn in fsets.items():
            sub = cd[fn(cd).fillna(False)].sort_values(["day", "ratio"], ascending=[True, False])
            epf = {d: g["sym"].tolist() for d, g in sub.groupby("day")}
            rows += run(data, epf, days, cost, f"[{name}] ")
    (ROOT / "reports" / ("two_day_filters.json" if "--filters" in sys.argv else "two_day_study.json")).write_text(json.dumps({"fingerprint": fp, "rows": rows}, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
    print(f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
