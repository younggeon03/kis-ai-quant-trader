"""장중 슬라이딩 윈도우 거래대금 진입 측정 — 최근 W분 거래대금이 평소의 k배가 되는 순간.

## 무엇을 비교하는가

모든 거래일의 하루 전체 분봉(`data/cache/fullday/`, `collect_opening.py --until 1530 --dir fullday`)
에서, 매 분 **직전 W분 거래대금**이 기준선의 k배 이상이 되는 그날 첫 분에 발동한다.

| 축 | 값 |
|---|---|
| 창 W | 3·5분 |
| 기준선 | `day_avg`(하루 평균 W분 몫) · `same_time`(지난 20일 같은 시각 같은 W분 중앙) |
| 배수 k | 3·5·10·20 |
| 감지 지연 L | 1분(WS 실시간) · 4분(REST 순회 — 284종목 한 바퀴 약 3분) |

진입은 **창이 끝난 분 + L분의 시가**다(창이 끝나야 그 값을 안다 — 룩어헤드 없음).
보유는 `vol_alive` x2 + 손절 25%, 평가는 달력 2슬롯(`slot_sim.run_window`) — 다른 진입
측정과 같다.

**같은 데이터로 기준 규칙 둘을 같이 계산한다** — 지문이 다른 실행의 수치는 비교하지 않는다
(CLAUDE.md §10).

- `opening`: 장 시작 10분 같은 시간대 x3 → 09:13 (현재 실거래 규칙)
- `cum10`: 당일 누적 거래대금 ≥ 20봉 중앙 x10 → 다음 분 시가 (예전 규칙)

## 메모리

슬롯이 2개라 하루에 쓰이는 사건은 많아야 몇 건이다. 칸(조합)마다 날짜별로 **가장 이른
`--keep-per-day`건만** 남기고, 청산 계산도 남은 것에만 한다(가용 메모리가 2GB 안팎인
환경에서 측정이 두 번 끊겼다).

```bash
python scripts/sliding_study.py
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
from scripts.intraday_entry_study import COMP_BARS, _fingerprint  # noqa: E402
from scripts.slot_sim import path_exit, run_window  # noqa: E402
from scripts.tail_optimizer import MAX_GROSS, MAX_SINGLE  # noqa: E402
from strategy.opening import day_avg_base, minute_index, same_window_base  # noqa: E402
from strategy.sliding import (LAST_ENTRY_MIN, first_trigger, minute_turnover,  # noqa: E402
                              rolling_sum, same_time_base)
from utils.logger import enable_utf8_console, logger  # noqa: E402

N_MIN = LAST_ENTRY_MIN + 1


def _entry_px(op_row: np.ndarray, e: int) -> float:
    """진입 분 e의 시가. 그 분에 봉이 없으면(거래 없음) 2분 안의 다음 봉."""
    for x in range(e, min(e + 3, N_MIN)):
        v = op_row[x]
        if np.isfinite(v) and v > 0:
            return float(v)
    return float("nan")


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="장중 슬라이딩 윈도우 거래대금 진입 측정")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--fullday-dir", default="data/cache/fullday")
    ap.add_argument("--since", default="20250102")
    ap.add_argument("--windows", default="3,5")
    ap.add_argument("--mults", default="3,5,10,20")
    ap.add_argument("--lags", default="1,4")
    ap.add_argument("--hold-vol", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=30)
    ap.add_argument("--stop", type=float, default=0.25)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--keep-per-day", type=int, default=12)
    ap.add_argument("--out", default="reports/sliding_study.json")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    weight = min(MAX_SINGLE, MAX_GROSS / max(1, args.slots))
    windows = [int(x) for x in args.windows.split(",")]
    mults = [float(x) for x in args.mults.split(",")]
    lags = [int(x) for x in args.lags.split(",")]

    panel = load_panel_from_dir(args.source, list(s.UNIVERSE))
    all_days = sorted({str(t) for df in panel.values() if df is not None
                       for t in df["ts"].astype(str)})
    cal = {t: k for k, t in enumerate(all_days)}
    fdir = Path(args.fullday_dir)

    # 칸 → {달력일: [(감지분+순위, 종목, 일봉 인덱스, 진입가)]}
    Cell = Tuple[str, Any, Any, Any]
    cells: Dict[Cell, Dict[int, List[tuple]]] = {}
    daily: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    closes: Dict[str, Dict[int, float]] = {}
    fp_src: Dict[str, np.ndarray] = {}
    sym_days = 0

    def add(cell: Cell, d: int, item: tuple) -> None:
        lst = cells.setdefault(cell, {}).setdefault(d, [])
        lst.append(item)
        if len(lst) > 2 * args.keep_per_day:
            lst.sort(key=lambda x: x[0])
            del lst[args.keep_per_day:]

    for k_, (sym, df) in enumerate(sorted(panel.items()), 1):
        if df is None or len(df) < 60:
            continue
        path = fdir / f"{sym}.csv"
        if not path.exists():
            continue
        d = df.reset_index(drop=True)
        ts = d["ts"].astype(str).to_numpy()
        o = d["open"].to_numpy(float)
        c = d["close"].to_numpy(float)
        tov = c * d["volume"].to_numpy(float)
        med_day = pd.Series(tov).rolling(20, min_periods=10).median().shift(1).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            burst = tov / np.where(med_day > 0, med_day, np.nan)
        daily[sym] = (o, c, burst, med_day)
        closes[sym] = {cal[t]: float(px) for t, px in zip(ts, c)}
        pos = {t: i for i, t in enumerate(ts)}

        m = pd.read_csv(path, dtype={"ts": str})
        m = m[m["ts"].str[:8] >= args.since]
        if m.empty:
            continue
        fp_src[sym] = (m["close"] * m["volume"]).to_numpy(float)
        days = sorted(m["ts"].str[:8].unique())
        TV = np.zeros((len(days), N_MIN))
        OP = np.full((len(days), N_MIN), np.nan)
        for j, (dy, g) in enumerate(m.groupby(m["ts"].str[:8], sort=True)):
            mi = minute_index(g["ts"])
            TV[j] = minute_turnover(mi, g["close"].to_numpy(float),
                                    g["volume"].to_numpy(float), N_MIN)
            ok = (mi >= 0) & (mi < N_MIN)
            OP[j, mi[ok]] = g["open"].to_numpy(float)[ok]
        S = {W: np.vstack([rolling_sum(TV[j], W) for j in range(len(days))]) for W in windows}
        W10 = TV[:, :10].sum(axis=1)                       # 장 시작 10분(기준 규칙)
        CUM = np.cumsum(TV, axis=1)                        # 당일 누적(기준 규칙)

        for j, dy in enumerate(days):
            i = pos.get(dy)
            if i is None or i < 25 or i + args.max_hold + 1 >= len(d):
                continue
            sym_days += 1
            dc = cal[dy]
            # --- 기준 1: 장 시작 10분 같은 시간대 x3 → 09:13 (현재 실거래) ---
            base10 = same_window_base(W10[:j].tolist())
            if base10 > 0 and W10[j] >= 3.0 * base10:
                px = _entry_px(OP[j], 13)
                if np.isfinite(px):
                    r = W10[j] / base10
                    add(("opening", 10, 3.0, 3), dc, (13 + 1 / (1 + r), sym, i, px))
            # --- 기준 2: 누적 10배 → 다음 분 시가 ---
            if np.isfinite(med_day[i]) and med_day[i] > 0:
                hit = np.flatnonzero(CUM[j] >= 10.0 * med_day[i])
                if len(hit) and hit[0] + 1 <= LAST_ENTRY_MIN:
                    px = _entry_px(OP[j], int(hit[0]) + 1)
                    if np.isfinite(px):
                        add(("cum10", "-", 10.0, 1), dc, (int(hit[0]) + 1.5, sym, i, px))
            # --- 슬라이딩 ---
            for W in windows:
                bases = {"day_avg": day_avg_base(med_day[i], W),
                         "same_time": same_time_base(S[W][:j])}
                for bname, base in bases.items():
                    if np.ndim(base) == 0 and not (base > 0):
                        continue
                    for k in mults:
                        trig = first_trigger(S[W][j], base, k)
                        if trig is None:
                            continue
                        mm, r = trig
                        for L in lags:
                            e = mm + L
                            if e > LAST_ENTRY_MIN:
                                continue
                            px = _entry_px(OP[j], e)
                            if np.isfinite(px):
                                add((bname, W, k, L), dc, (e + 1 / (1 + r), sym, i, px))
        if k_ % 25 == 0:
            logger.info(f"[SlidingStudy] {k_}/{len(panel)}종목 · 종목-일 {sym_days:,}")

    if not cells:
        logger.error("[SlidingStudy] 사건이 없습니다 — fullday/ 수집을 확인하십시오")
        return 1
    fp = _fingerprint(fp_src)

    # 청산은 남은 사건에만 계산한다
    def events_of(cell: Cell) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for dc, lst in cells[cell].items():
            lst.sort(key=lambda x: x[0])
            for det, sym, i, px in lst[: args.keep_per_day]:
                o, c, burst, _ = daily[sym]
                ex_i, ex_px, when = path_exit(px, i, args.max_hold, args.stop, o, c,
                                              burst, args.hold_vol)
                ts_d = list(closes[sym].keys())
                out.append({"sym": sym, "d": dc, "detect_min": det, "entry_px": px,
                            "rule": {"exit_d": ts_d[min(ex_i, len(ts_d) - 1)], "when": when,
                                     "ret": ex_px / px - 1.0 - cost}})
        return out

    first = cal[min(t for t in all_days if t >= args.since)]
    last = max(max(v) for v in closes.values())
    starts = list(range(first, last - COMP_BARS + 2))

    def evaluate(evs: List[Dict[str, Any]]) -> np.ndarray:
        return np.array([run_window(evs, "rule", d0, d0 + COMP_BARS - 1, closes, cost,
                                    args.slots, weight)[0] for d0 in starts])

    rets: Dict[Cell, np.ndarray] = {}
    per_day: Dict[Cell, float] = {}
    n_days = len([t for t in all_days if t >= args.since])
    for cell in sorted(cells, key=str):
        evs = events_of(cell)
        rets[cell] = evaluate(evs)
        per_day[cell] = len(evs) / max(1, n_days)
    ref_open = rets.get(("opening", 10, 3.0, 3))

    rows = []
    for cell, r in rets.items():
        rows.append({"rule": cell[0], "window": cell[1], "mult": cell[2], "lag": cell[3],
                     "kept_per_day": per_day[cell],
                     "median": float(np.median(r)), "mean": float(r.mean()),
                     "p90": float(np.percentile(r, 90)), "max": float(r.max()),
                     "p50up": float((r >= 0.5).mean()), "p100up": float((r >= 1.0).mean()),
                     "beats_opening": (float((r > ref_open).mean())
                                       if ref_open is not None else None)})
    out = {"data_fingerprint": {"hash": fp, "symbols": len(fp_src), "symbol_days": sym_days},
           "config": vars(args) | {"weight": weight, "round_trip_cost": cost},
           "windows": len(starts), "rows": rows}
    op_ = Path(args.out)
    op_.parent.mkdir(parents=True, exist_ok=True)
    op_.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    order = {"opening": 0, "cum10": 1, "day_avg": 2, "same_time": 3}
    rows.sort(key=lambda x: (order.get(x["rule"], 9), str(x["window"]), x["mult"], x["lag"]))
    print()
    print("=" * 112)
    print(f" 장중 슬라이딩 윈도우 거래대금 진입 — {args.since}~ · vol_alive x{args.hold_vol:g} · "
          f"손절 {args.stop:.0%} · {args.slots}슬롯 x {weight:.0%}")
    print(f" 데이터 지문 {fp} ({len(fp_src)}종목 · {sym_days:,}종목-일) · 30거래일 창 "
          f"{len(starts)}개(하루씩 이동, 겹친다 — 독립 약 {len(starts) // COMP_BARS}개)")
    print("=" * 112)
    print(f" {'규칙':<10}{'W':>4}{'배수':>6}{'지연':>5}{'중앙':>9}{'평균':>9}{'p90':>9}"
          f"{'최대':>9}{'P(+50%)':>10}{'P(+100%)':>10}{'장시작대비':>11}")
    print("-" * 112)
    for x in rows:
        print(f" {x['rule']:<10}{str(x['window']):>4}{x['mult']:6g}{x['lag']:5d}"
              f"{x['median'] * 100:+8.1f}%{x['mean'] * 100:+8.1f}%{x['p90'] * 100:+8.1f}%"
              f"{x['max'] * 100:+8.1f}%{x['p50up'] * 100:9.2f}%{x['p100up'] * 100:9.2f}%"
              f"{(x['beats_opening'] or 0) * 100:10.1f}%")
    print("=" * 112)
    print(" opening = 현재 실거래(장 시작 10분 같은 시간대 x3 → 09:13) · cum10 = 예전 누적 10배")
    print(" 지연 1분 = WS 실시간 구독 종목, 4분 = REST 순회 종목(실거래 대부분)")
    print(" 장시작대비 = 같은 30일 창에서 opening 보다 수익이 높은 창의 비율")
    print(" 격자가 크다 — 한 칸이 아니라 넓은 영역에서 일관된 방향만 믿는다.")
    print(f" 저장 → {op_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
