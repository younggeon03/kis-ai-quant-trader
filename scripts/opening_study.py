"""장 시작 구간 거래대금 진입 — 누적 10배보다 일찍, 더 낮은 문턱으로 잡을 수 있는가.

## 왜 이 측정이 필요한가

현재 진입(당일 **누적** 거래대금 ≥ 20봉 중앙 x10)은 감지 중앙이 10:59이고 그때 이미
시가 대비 +7% 올라 있다. 문턱을 낮춰도 빨라지지 않았다(누적은 하루 종일 천천히
쌓인다). 그래서 **짧은 구간의 거래대금**으로 판정한다 — 예: 09:00~09:05 거래대금이
평소보다 많으면 산다.

## 기준선이 둘이다

장 시작 직후는 원래 거래가 몰린다. 그래서 "평소"를 두 가지로 잰다.

| 기준선 | 정의 | 비고 |
|---|---|---|
| `day_avg` | 20봉 중앙 일 거래대금 x N/390 | 하루 평균 N분 (사용자 제안) |
| `same_window` | 직전 20거래일 **같은 09:00~09:0N** 거래대금 중앙(최소 10일) | 시간대 보정 |

## 실거래 가능성

09:0N에 284종목을 한꺼번에 확인하려면 REST 조회에 약 2.6분이 걸린다. 그래서 진입은
**구간이 끝나고 `--lag`분 뒤 그 분의 시가**다(기본 3분). 판정 구간이 끝나기 전의 값은
쓰지 않는다(룩어헤드 없음).

## 보유·평가

보유는 `vol_alive` x2 + 손절 25%(`slot_sim.path_exit`, 봉 t 종가 판정 → t+1 시가).
평가는 `slot_sim.run_window` — 달력 시간순 2슬롯 x 45%, 30거래일 창. 같은 날 발동이
슬롯보다 많으면 배수가 큰 순으로 채운다(근거 없는 선택이라 결과에 같이 찍는다).

비교 기준은 **같은 기간**의 현재 규칙(누적 10배, `minute/` 분봉)이다.

## 데이터

`data/cache/opening/` (`scripts/collect_opening.py`, 모든 거래일 09:00~09:30).
**수집이 끝난 뒤에 돌린다** — 도는 중에 재면 집단이 바뀐다(CLAUDE.md §10).

```bash
python scripts/opening_study.py
python scripts/opening_study.py --lag 5 --windows 5,10
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.intraday_entry_study import (COMP_BARS, _fingerprint,  # noqa: E402
                                          burst_days, detect_entry)
from scripts.slot_sim import path_exit, run_window  # noqa: E402
from scripts.tail_optimizer import MAX_GROSS, MAX_SINGLE  # noqa: E402
from utils.logger import enable_utf8_console, logger  # noqa: E402

SESSION_MIN = 390.0


def _minute_index(ts: pd.Series) -> np.ndarray:
    h = ts.str[8:10].astype(int).to_numpy()
    m = ts.str[10:12].astype(int).to_numpy()
    return (h - 9) * 60 + m


def evaluate(events: List[Dict[str, Any]], closes: Dict[str, Dict[int, float]],
             cost: float, slots: int, weight: float, first: int, last: int
             ) -> Dict[str, Any]:
    starts = list(range(first, last - COMP_BARS + 2))
    rets = np.array([run_window(events, "rule", d0, d0 + COMP_BARS - 1, closes, cost,
                                slots, weight)[0] for d0 in starts]) if events else np.array([])
    if not len(rets):
        return {"windows": 0}
    return {"windows": int(len(rets)), "median": float(np.median(rets)),
            "mean": float(rets.mean()), "p90": float(np.percentile(rets, 90)),
            "max": float(rets.max()), "p50up": float((rets >= 0.5).mean()),
            "p100up": float((rets >= 1.0).mean()), "_rets": rets}


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="장 시작 구간 거래대금 진입 측정")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--opening-dir", default="data/cache/opening")
    ap.add_argument("--minute-dir", default="data/cache/minute")
    ap.add_argument("--since", default="20250102")
    ap.add_argument("--windows", default="3,5,10,15,20", help="판정 구간 길이(분)")
    ap.add_argument("--mults-day", default="3,5,10,20,30", help="day_avg 기준 배수")
    ap.add_argument("--mults-same", default="2,3,5,10", help="same_window 기준 배수")
    ap.add_argument("--lag", type=int, default=3, help="구간 끝 → 진입까지 지연(분)")
    ap.add_argument("--hold-vol", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=30)
    ap.add_argument("--stop", type=float, default=0.25)
    ap.add_argument("--burst-mult", type=float, default=10.0, help="비교 기준(현재 규칙)")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--out", default="reports/opening_study.json")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    weight = min(MAX_SINGLE, MAX_GROSS / max(1, args.slots))
    windows = [int(x) for x in args.windows.split(",")]
    mults = {"day_avg": [float(x) for x in args.mults_day.split(",")],
             "same_window": [float(x) for x in args.mults_same.split(",")]}
    if max(windows) + args.lag >= 31:
        logger.error("[Opening] 구간 + 지연이 수집 범위(09:00~09:30)를 넘는다")
        return 1

    panel = load_panel_from_dir(args.source, list(s.UNIVERSE))
    all_days = sorted({str(t) for df in panel.values() if df is not None
                       for t in df["ts"].astype(str)})
    cal = {t: k for k, t in enumerate(all_days)}
    odir, mdir = Path(args.opening_dir), Path(args.minute_dir)

    # (방식, 기준선, N, k) → 사건 목록
    grid: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = {}
    ref_events: List[Dict[str, Any]] = []
    closes: Dict[str, Dict[int, float]] = {}
    fp_src: Dict[str, np.ndarray] = {}
    sym_days = 0
    for k_, (sym, df) in enumerate(sorted(panel.items()), 1):
        if df is None or len(df) < 60:
            continue
        d = df.reset_index(drop=True)
        ts = d["ts"].astype(str).to_numpy()
        o = d["open"].to_numpy(float)
        c = d["close"].to_numpy(float)
        tov = c * d["volume"].to_numpy(float)
        closes[sym] = {cal[t]: float(px) for t, px in zip(ts, c)}
        med_day = pd.Series(tov).rolling(20, min_periods=10).median().shift(1).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            burst = tov / np.where(med_day > 0, med_day, np.nan)
        pos = {t: i for i, t in enumerate(ts)}

        # --- 장 시작 구간 ---
        op = odir / f"{sym}.csv"
        if op.exists():
            m = pd.read_csv(op, dtype={"ts": str})
            m = m[m["ts"].str[:8] >= args.since]
            if not m.empty:
                m = m.assign(mi=_minute_index(m["ts"]), tv=m["close"] * m["volume"],
                             day=m["ts"].str[:8])
                fp_src[sym] = m["tv"].to_numpy(float)
                days = sorted(m["day"].unique())
                by_day = {dy: g.sort_values("mi") for dy, g in m.groupby("day")}
                # 구간별 거래대금 표 — 같은 구간 기준선(직전 20일 중앙)을 만든다
                wtv = {N: pd.Series([float(by_day[dy].loc[by_day[dy]["mi"] < N, "tv"].sum())
                                     for dy in days], index=days) for N in windows}
                same_base = {N: wtv[N].rolling(20, min_periods=10).median().shift(1)
                             for N in windows}
                for dy in days:
                    i = pos.get(dy)
                    if i is None or i + args.max_hold + 1 >= len(d) or i < 25:
                        continue
                    sym_days += 1
                    g = by_day[dy]
                    for N in windows:
                        w = wtv[N][dy]
                        ent = g.loc[g["mi"] == N + args.lag, "open"]
                        if ent.empty or not (float(ent.iloc[0]) > 0) or w <= 0:
                            continue
                        entry_px = float(ent.iloc[0])
                        bases = {"day_avg": med_day[i] * N / SESSION_MIN,
                                 "same_window": same_base[N][dy]}
                        exit_cache = None
                        for bname, base in bases.items():
                            if not (np.isfinite(base) and base > 0):
                                continue
                            ratio = w / base
                            for k in mults[bname]:
                                if ratio < k:
                                    continue
                                if exit_cache is None:
                                    ex_i, ex_px, when = path_exit(
                                        entry_px, i, args.max_hold, args.stop, o, c,
                                        burst, args.hold_vol)
                                    exit_cache = {"exit_d": cal[ts[min(ex_i, len(ts) - 1)]],
                                                  "when": when,
                                                  "ret": ex_px / entry_px - 1.0 - cost}
                                grid.setdefault((bname, N, k), []).append({
                                    "sym": sym, "d": cal[dy], "entry_px": entry_px,
                                    # 같은 분 동시 발동은 배수가 큰 순으로 채운다
                                    "detect_min": N + args.lag + 1.0 / (1.0 + ratio),
                                    "rule": exit_cache})

        # --- 비교 기준: 현재 규칙(누적 10배), 같은 기간 ---
        mp = mdir / f"{sym}.csv"
        idx, thresh = burst_days(d, args.burst_mult, 25)
        idx = idx[(idx + args.max_hold + 1) < len(d)]
        idx = idx[ts[idx] >= args.since]
        if len(idx) and mp.exists():
            md = pd.read_csv(mp, dtype={"ts": str})
            mby = {dy: g for dy, g in md.groupby(md["ts"].str[:8], sort=False)}
            for i in idx:
                g = mby.get(ts[i])
                if g is None or g.empty:
                    continue
                got = detect_entry(g, float(thresh[i]))
                if got is None:
                    continue
                mins, entry_px, _ = got
                ex_i, ex_px, when = path_exit(entry_px, i, args.max_hold, args.stop, o, c,
                                              burst, args.hold_vol)
                ref_events.append({"sym": sym, "d": cal[ts[i]], "entry_px": entry_px,
                                   "detect_min": mins,
                                   "rule": {"exit_d": cal[ts[min(ex_i, len(ts) - 1)]],
                                            "when": when,
                                            "ret": ex_px / entry_px - 1.0 - cost}})
        if k_ % 50 == 0:
            logger.info(f"[OpeningStudy] {k_}/{len(panel)}종목 · 종목-일 {sym_days:,}")

    if not grid:
        logger.error("[OpeningStudy] 사건이 없습니다 — opening/ 수집을 확인하십시오")
        return 1
    fp = _fingerprint(fp_src)
    first = cal[min(t for t in all_days if t >= args.since)]
    last = max(max(v) for v in closes.values())
    n_days = len([t for t in all_days if t >= args.since])

    ref = evaluate(ref_events, closes, cost, args.slots, weight, first, last)
    rows = []
    for (bname, N, k), evs in sorted(grid.items()):
        r = evaluate(evs, closes, cost, args.slots, weight, first, last)
        rets = np.array([e["rule"]["ret"] for e in evs])
        paired = (r["_rets"] - ref["_rets"]) if ref.get("windows") else None
        rows.append({"baseline": bname, "window": N, "mult": k, "events": len(evs),
                     "per_day": len(evs) / max(1, n_days),
                     "trade_mean": float(rets.mean()), "single_2x": float((rets >= 1.0).mean()),
                     **{kk: v for kk, v in r.items() if kk != "_rets"},
                     "beats_ref": (float((paired > 0).mean()) if paired is not None else None)})

    out = {"data_fingerprint": {"hash": fp, "symbols": len(fp_src), "symbol_days": sym_days},
           "config": vars(args) | {"weight": weight, "round_trip_cost": cost},
           "reference": {"events": len(ref_events),
                         **{kk: v for kk, v in ref.items() if kk != "_rets"}},
           "grid": rows}
    op_ = Path(args.out)
    op_.parent.mkdir(parents=True, exist_ok=True)
    op_.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 118)
    print(f" 장 시작 구간 거래대금 진입 — {args.since}~ · 진입 = 구간 끝 +{args.lag}분 시가 · "
          f"보유 vol_alive x{args.hold_vol:g} · 손절 {args.stop:.0%} · {args.slots}슬롯 x {weight:.0%}")
    print(f" 데이터 지문 {fp} ({len(fp_src)}종목 · {sym_days:,}종목-일) · 30거래일 창 "
          f"{ref.get('windows', 0)}개(하루씩 이동, 서로 겹친다)")
    print("=" * 118)
    print(f" {'기준선':<12}{'N분':>4}{'배수':>6}{'사건':>7}{'/일':>6}{'평균/회':>9}{'단일2배':>8}"
          f"{'중앙':>8}{'p90':>8}{'P(+50%)':>9}{'P(+100%)':>10}{'기준대비':>9}")
    print("-" * 118)
    if ref.get("windows"):
        rr = np.array([e["rule"]["ret"] for e in ref_events])
        print(f" {'현재(누적10배)':<12}{'-':>4}{'-':>6}{len(ref_events):7,d}"
              f"{len(ref_events) / max(1, n_days):6.1f}{rr.mean() * 100:+8.2f}%"
              f"{(rr >= 1.0).mean() * 100:7.2f}%{ref['median'] * 100:+7.1f}%"
              f"{ref['p90'] * 100:+7.1f}%{ref['p50up'] * 100:8.2f}%{ref['p100up'] * 100:9.2f}%"
              f"{'—':>9}")
    for r in rows:
        if not r.get("windows"):
            continue
        print(f" {r['baseline']:<12}{r['window']:4d}{r['mult']:6g}{r['events']:7,d}"
              f"{r['per_day']:6.1f}{r['trade_mean'] * 100:+8.2f}%{r['single_2x'] * 100:7.2f}%"
              f"{r['median'] * 100:+7.1f}%{r['p90'] * 100:+7.1f}%{r['p50up'] * 100:8.2f}%"
              f"{r['p100up'] * 100:9.2f}%"
              f"{(r['beats_ref'] or 0) * 100:8.1f}%")
    print("=" * 118)
    print(" 기준대비 = 같은 30일 창에서 현재 규칙보다 수익이 높은 창의 비율(겹치는 창이라 독립 표본은 약 14개)")
    print(" 격자가 크다 — 가장 좋은 칸을 고르면 과적합이다. 넓은 영역에서 일관된 방향만 믿는다.")
    print(f" 저장 → {op_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
