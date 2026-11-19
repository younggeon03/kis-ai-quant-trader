"""대회 최소 요건 충족률 — 지금 진입 규칙으로 30거래일을 돌리면 실격 요건을 채우는가.

## 왜 필요한가

요건 4개(체결 5회 · 매매금액 5억 · 매매일수 5일 · **코스피200·코스닥150 중 5종목**)를
못 채우면 수익률과 무관하게 실격이다. 그런데 엔진이 요건을 맞추려고 하는 행동은
"최소 보유기간 완화" 하나뿐이고 지금 규칙(장 시작 구간 · vol_alive)에서는 효과가 없다.
가장 위험한 것은 지수 종목 수다 — 유니버스 284 중 111이 지수 밖이고 폭증은 소형주에서
잘 난다.

## 무엇을 하는가

`opening_study`와 같은 사건(장 시작 N분 같은 시간대 x k → 구간 끝 + 지연분 시가)을
`slot_sim.run_window`와 같은 규칙(2슬롯 x 45%, 보유가 끝나야 슬롯이 빈다)으로 30거래일 창마다
돌리면서 **체결을 기록**한다. 창 안에서 난 매수·매도만 센다(창 끝에 남은 보유는 매도 체결이
없다 — 대회 마지막 날과 같다). 지수 소속은 실거래 요건 집계(`strategy/compliance.py`)와 같은
`Settings.index_members()`다.

```bash
python scripts/research/compliance_study.py
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.research.intraday_entry_study import COMP_BARS  # noqa: E402
from scripts.research.slot_sim import path_exit  # noqa: E402
from scripts.research.tail_optimizer import MAX_GROSS, MAX_SINGLE  # noqa: E402
from strategy.opening import minute_index, same_window_base, window_turnover  # noqa: E402
from utils.logger import enable_utf8_console, logger  # noqa: E402

CAPITAL = 100_000_000.0


def simulate(events: List[Dict[str, Any]], d0: int, d1: int, slots: int,
             weight: float, cost: float) -> Dict[str, Any]:
    """`slot_sim.run_window`와 같은 규칙으로 돌리며 체결을 기록한다."""
    cash = 1.0
    held: List[Dict[str, Any]] = []
    fills: List[tuple] = []                     # (달력일, 종목, 금액(원))
    by_day: Dict[int, List[Dict[str, Any]]] = {}
    for ev in events:
        if d0 <= ev["d"] <= d1:
            by_day.setdefault(ev["d"], []).append(ev)

    def release(d: int, when: str) -> None:
        nonlocal cash
        keep = []
        for h in held:
            if h["exit_d"] == d and h["when"] == when:
                back = h["alloc"] * (1.0 + h["ret"])
                cash += back
                fills.append((d, h["sym"], back * CAPITAL))
            else:
                keep.append(h)
        held[:] = keep

    for d in range(d0, d1 + 1):
        release(d, "open")
        for ev in sorted(by_day.get(d, []), key=lambda e: e["detect_min"]):
            if len(held) >= slots:
                break
            if any(h["sym"] == ev["sym"] for h in held):
                continue
            equity = cash + sum(h["alloc"] for h in held)
            alloc = min(equity * weight, cash)
            if alloc <= 0:
                break
            cash -= alloc
            held.append({"alloc": alloc, "sym": ev["sym"], **ev["rule"]})
            fills.append((d, ev["sym"], alloc * CAPITAL))
        release(d, "close")
    return {"fills": fills}


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="대회 최소 요건 충족률")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--opening-dir", default="data/cache/opening")
    ap.add_argument("--since", default="20250102")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--mult", type=float, default=3.0)
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--hold-vol", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=30)
    ap.add_argument("--stop", type=float, default=0.25)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--out", default="reports/compliance_study.json")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    weight = min(MAX_SINGLE, MAX_GROSS / max(1, args.slots))
    members = {x for lst in s.index_members().values() for x in lst}
    if not members:
        logger.error("[Compliance] 지수 멤버십 정보가 없다 — UNIVERSE_FILE 을 확인하십시오")
        return 1
    panel = load_panel_from_dir(args.source, list(s.UNIVERSE))
    all_days = sorted({str(t) for df in panel.values() if df is not None
                       for t in df["ts"].astype(str)})
    cal = {t: k for k, t in enumerate(all_days)}
    N = args.window
    events: List[Dict[str, Any]] = []
    for sym, df in sorted(panel.items()):
        p = Path(args.opening_dir) / f"{sym}.csv"
        if df is None or len(df) < 60 or not p.exists():
            continue
        d = df.reset_index(drop=True)
        ts = d["ts"].astype(str).to_numpy()
        o, c = d["open"].to_numpy(float), d["close"].to_numpy(float)
        tov = c * d["volume"].to_numpy(float)
        med = pd.Series(tov).rolling(20, min_periods=10).median().shift(1).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            burst = tov / np.where(med > 0, med, np.nan)
        pos = {t: i for i, t in enumerate(ts)}
        m = pd.read_csv(p, dtype={"ts": str})
        m = m[m["ts"].str[:8] >= args.since]
        hist: List[float] = []
        for dy, g in m.groupby(m["ts"].str[:8], sort=True):
            mi = minute_index(g["ts"])
            w = window_turnover(mi, g["close"].to_numpy(float), g["volume"].to_numpy(float), N)
            base = same_window_base(hist)
            hist.append(w)
            i = pos.get(dy)
            if i is None or i < 25 or i + args.max_hold + 1 >= len(d) or base <= 0:
                continue
            if w < args.mult * base:
                continue
            ent = g.loc[mi == N + args.lag, "open"]
            if ent.empty or not float(ent.iloc[0]) > 0:
                continue
            px = float(ent.iloc[0])
            ex_i, ex_px, when = path_exit(px, i, args.max_hold, args.stop, o, c, burst,
                                          args.hold_vol)
            events.append({"sym": sym, "d": cal[dy], "detect_min": N + args.lag + 1 / (1 + w / base),
                           "rule": {"exit_d": cal[ts[min(ex_i, len(ts) - 1)]], "when": when,
                                    "ret": ex_px / px - 1.0 - cost}})
    if not events:
        logger.error("[Compliance] 사건이 없다")
        return 1
    first = cal[min(t for t in all_days if t >= args.since)]
    last = max(e["d"] for e in events)
    starts = list(range(first, last - COMP_BARS + 2))
    rows = []
    for d0 in starts:
        f = simulate(events, d0, d0 + COMP_BARS - 1, args.slots, weight, cost)["fills"]
        syms = {x[1] for x in f}
        rows.append({"count": len(f), "amount": sum(x[2] for x in f),
                     "days": len({x[0] for x in f}), "index_syms": len(syms & members),
                     "syms": len(syms)})
    fr = pd.DataFrame(rows)
    req = {"count": s.MIN_TRADE_COUNT or 5, "amount": s.MIN_TRADE_AMOUNT or 5e8,
           "days": s.MIN_TRADE_DAYS or 5, "index_syms": s.MIN_TRADE_SYMBOLS or 5}
    ok = {k: (fr[k] >= v) for k, v in req.items()}
    all_ok = np.logical_and.reduce(list(ok.values()))
    out = {"windows": len(fr), "events": len(events), "requirements": req,
           "met_share": {k: float(v.mean()) for k, v in ok.items()},
           "all_met_share": float(all_ok.mean()),
           "quantiles": {k: {q: float(fr[k].quantile(q / 100)) for q in (5, 25, 50)}
                         for k in req}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 92)
    print(f" 대회 최소 요건 충족률 — 장 시작 {N}분 같은 시간대 x{args.mult:g} · {args.slots}슬롯 · "
          f"30거래일 창 {len(fr)}개 · {args.since}~")
    print("=" * 92)
    names = {"count": "체결 횟수", "amount": "매매금액(원)", "days": "매매일수",
             "index_syms": "지수 소속 종목수"}
    for k, v in req.items():
        q = out["quantiles"][k]
        fmt = (lambda x: f"{x / 1e8:.1f}억") if k == "amount" else (lambda x: f"{x:.0f}")
        print(f" {names[k]:<14} 기준 {fmt(v):>7} · 충족 {ok[k].mean():6.1%} · "
              f"하위5% {fmt(q[5])} · 하위25% {fmt(q[25])} · 중앙 {fmt(q[50])}")
    print("-" * 92)
    print(f" **4개 모두 충족: {all_ok.mean():.1%}** (못 채우면 수익률과 무관하게 실격)")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
