"""지금 실거래 조합 측정 — 장 시작 + 장중 추가 매수 x VWAP 청산, 분 단위로 자리 순서까지 맞춘다.

## 왜 따로 만드나

각 기능은 따로 쟀다(`opening_study`·`sliding_study`·`takeprofit_study --vwap`). 실거래에서는 VWAP 청산이
장중에 판 자리를 장중 추가 매수가 **같은 날** 채운다. 기존 평가(`slot_sim.run_window`)는 장중 청산의 현금을
그날 장 마감에 돌려주므로 이 순서를 못 본다. 여기서는 하루 안의 사건(진입 분 · 장중 청산 분)을 시간순으로
처리한다.

## 규칙(서버 설정과 같다)

- 장 시작: 09:00~09:05 거래대금 ≥ 지난 20일 같은 5분 중앙 x3 → 09:05 시가(배수 큰 순)
- 장중 추가: 직전 5분 거래대금 ≥ 지난 20일 같은 시각 중앙 x3 을 그날 **처음** 넘은 분(09:09 창부터 —
  실거래는 09:05 판정 뒤 5분 표본이 쌓여야 본다) → 다음 분 시가
- 청산: `vol_alive` x2(다음 날 시가) · 손절 25% · 30봉 상한, 그리고(VWAP 켠 칸만) 09:10 이후 가격 < 당일 VWAP
  이고 직전 10분 거래대금 < 평소 같은 시각이면 다음 분 시가
- 같은 날 판 종목은 그날 다시 사지 않는다. 2슬롯 x 45%.

```bash
UNIVERSE_FILE=config/universe_10b.json python scripts/combo_study.py
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.intraday_entry_study import COMP_BARS, _fingerprint  # noqa: E402
from scripts.slot_sim import path_exit  # noqa: E402
from scripts.tail_optimizer import MAX_GROSS, MAX_SINGLE  # noqa: E402
from scripts.takeprofit_study import vwap_exit  # noqa: E402
from strategy.opening import minute_index, same_window_base  # noqa: E402
from strategy.sliding import (LAST_ENTRY_MIN, first_trigger, minute_turnover, rolling_sum,  # noqa: E402
                              same_time_base)
from utils.logger import enable_utf8_console, logger  # noqa: E402

N_MIN = LAST_ENTRY_MIN + 1
EOD_MIN = 10_000          # 일봉 종가·시가 청산은 그날 어떤 분보다 뒤/앞으로 놓는다


def _px_at(op_row: np.ndarray, e: int) -> float:
    for x in range(e, min(e + 3, N_MIN)):
        v = op_row[x]
        if np.isfinite(v) and v > 0:
            return float(v)
    return float("nan")


def run_window_intraday(events: List[Dict[str, Any]], arm: str, d0: int, d1: int,
                        closes: Dict[str, Dict[int, float]], cost: float, slots: int,
                        weight: float) -> float:
    """하루 안의 진입·장중 청산을 분 순서로 처리하는 30일 창 수익률.

    청산 `when`: 'open'(그날 장 시작 전), 'intra'(그날 `exit_min` 분), 'close'(장 마감 뒤).
    장중에 판 종목은 그날 다시 사지 않는다.
    """
    cash = 1.0
    held: List[Dict[str, Any]] = []
    by_day: Dict[int, List[Dict[str, Any]]] = {}
    for ev in events:
        if d0 <= ev["d"] <= d1:
            by_day.setdefault(ev["d"], []).append(ev)

    def release(d: int, upto_min: float, kinds: tuple) -> set:
        nonlocal cash
        keep, sold = [], set()
        for h in held:
            if h["exit_d"] == d and h["when"] in kinds and h["exit_min"] <= upto_min:
                cash += h["alloc"] * (1.0 + h["ret"])
                sold.add(h["sym"])
            else:
                keep.append(h)
        held[:] = keep
        return sold

    for d in range(d0, d1 + 1):
        sold_today = release(d, EOD_MIN, ("open",))
        for ev in sorted(by_day.get(d, []), key=lambda e: e["detect_min"]):
            sold_today |= release(d, ev["detect_min"], ("intra",))
            if len(held) >= slots or ev["sym"] in sold_today:
                continue
            if any(h["sym"] == ev["sym"] for h in held):
                continue
            x = ev[arm]
            equity = cash + sum(h["alloc"] for h in held)
            alloc = min(equity * weight, cash)
            if alloc <= 0:
                continue
            cash -= alloc
            held.append({"alloc": alloc, "exit_d": x["exit_d"], "when": x["when"],
                         "exit_min": x["exit_min"], "ret": x["ret"], "sym": ev["sym"],
                         "entry_px": ev["entry_px"]})
        release(d, EOD_MIN, ("intra", "close"))
    for h in held:
        px = closes[h["sym"]].get(d1)
        cash += h["alloc"] * (px / h["entry_px"] * (1.0 - cost / 2.0)) if px and px > 0 else h["alloc"]
    return cash - 1.0


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="장 시작 + 장중 추가 매수 x VWAP 청산 조합 측정")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--fullday-dir", default="data/cache/fullday")
    ap.add_argument("--since", default="20250102")
    ap.add_argument("--open-window", type=int, default=5)
    ap.add_argument("--open-mult", type=float, default=3.0)
    ap.add_argument("--slide-window", type=int, default=5)
    ap.add_argument("--slide-mult", type=float, default=3.0)
    ap.add_argument("--slide-start", type=int, default=9, help="장중 추가 매수 첫 판정 분(창 끝)")
    ap.add_argument("--vwap-roll", type=int, default=10)
    ap.add_argument("--hold-vol", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=30)
    ap.add_argument("--stop", type=float, default=0.25)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--out", default="reports/combo_study.json")
    ap.add_argument("--diag", action="store_true", help="원인 진단(출처·청산별 통계 · 회전 수 · 비용 0)")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    weight = min(MAX_SINGLE, MAX_GROSS / max(1, args.slots))
    fdir = Path(args.fullday_dir)

    def _fd(x: str) -> Path:
        p_ = fdir / f"{x}.csv"
        return p_ if p_.exists() else fdir / f"{x}.csv.gz"

    symbols = [x for x in s.UNIVERSE if _fd(x).exists()]
    panel = load_panel_from_dir(args.source, symbols)
    all_days = sorted({str(t) for df in panel.values() if df is not None for t in df["ts"].astype(str)})
    cal = {t: k for k, t in enumerate(all_days)}
    ow, sw, vw = args.open_window, args.slide_window, args.vwap_roll

    open_evs: List[Dict[str, Any]] = []
    slide_evs: List[Dict[str, Any]] = []
    closes: Dict[str, Dict[int, float]] = {}
    fp_src: Dict[str, np.ndarray] = {}
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

        fd = pd.read_csv(_fd(sym), dtype={"ts": str})
        fd = fd[fd["ts"].str[:8] >= args.since]
        if fd.empty:
            continue
        fd = fd.assign(mi=minute_index(fd["ts"]), day=fd["ts"].str[:8])
        fp_src[sym] = (fd["close"] * fd["volume"]).to_numpy(float)
        intraday = {dy: g.sort_values("mi").reset_index(drop=True) for dy, g in fd.groupby("day", sort=True)}
        days = sorted(intraday)
        TV = np.zeros((len(days), N_MIN))
        OP = np.full((len(days), N_MIN), np.nan)
        for j, dy in enumerate(days):
            g = intraday[dy]
            mi = g["mi"].to_numpy()
            TV[j] = minute_turnover(mi, g["close"].to_numpy(float), g["volume"].to_numpy(float), N_MIN)
            ok = (mi >= 0) & (mi < N_MIN)
            OP[j, mi[ok]] = g["open"].to_numpy(float)[ok]
        S_sw = np.vstack([rolling_sum(TV[j], sw) for j in range(len(days))])
        S_vw = np.vstack([rolling_sum(TV[j], vw) for j in range(len(days))])
        base_vw = pd.DataFrame(S_vw).rolling(20, min_periods=10).median().shift(1).to_numpy()
        base_vw[~(base_vw > 0)] = np.nan
        same_vw = {dy: (S_vw[r], base_vw[r]) for r, dy in enumerate(days)}
        WO = TV[:, :ow].sum(axis=1)

        def exits(entry_px: float, i: int, entry_mi: int) -> Dict[str, Dict[str, Any]]:
            ex_i, ex_px, when = path_exit(entry_px, i, args.max_hold, args.stop, o, c, burst, args.hold_vol)
            base = {"exit_d": cal[ts[min(ex_i, len(ts) - 1)]], "when": when,
                    "exit_min": -1 if when == "open" else EOD_MIN, "ret": ex_px / entry_px - 1.0 - cost}
            vx_i, vx_px, vwhen, hit, _ = vwap_exit(entry_px, i, entry_mi, args.max_hold, args.stop, o, c, burst,
                                                   args.hold_vol, ts, intraday, 0.0, -1.0, same_vw)
            if hit:
                g = intraday.get(ts[vx_i])
                # 판 분 = 그날 entry 이후 처음 조건이 맞은 분 + 1 (다음 분 시가) — vwap_exit 와 같은 규칙으로 다시 찾는다
                vmin = _vwap_min(g, entry_mi if vx_i == i else -1, vx_px)
                vwap_arm = {"exit_d": cal[ts[vx_i]], "when": "intra", "exit_min": vmin,
                            "ret": vx_px / entry_px - 1.0 - cost}
            else:
                vwap_arm = {"exit_d": cal[ts[min(vx_i, len(ts) - 1)]], "when": vwhen,
                            "exit_min": -1 if vwhen == "open" else EOD_MIN,
                            "ret": vx_px / entry_px - 1.0 - cost}
            return {"alive": base, "vwap": vwap_arm}

        for j, dy in enumerate(days):
            i = pos.get(dy)
            if i is None or i < 25 or i + args.max_hold + 1 >= len(d):
                continue
            dc = cal[dy]
            # 장 시작
            b0 = same_window_base(WO[:j].tolist())
            if b0 > 0 and WO[j] >= args.open_mult * b0:
                px = _px_at(OP[j], ow)
                if np.isfinite(px):
                    r = WO[j] / b0
                    open_evs.append({"sym": sym, "d": dc, "entry_px": px, "detect_min": ow + 1 / (1 + r),
                                     "src": "open", "chase": px / c[i - 1] - 1.0, "ratio": r,
                                     **exits(px, i, ow)})
            # 장중 추가 — 그날 첫 발동(창 끝 분 ≥ slide_start)
            if j >= 10:
                base_t = same_time_base(S_sw[:j])
                row = S_sw[j].copy()
                row[: args.slide_start] = np.nan
                trig = first_trigger(row, base_t, args.slide_mult)
                if trig is not None:
                    mm, r = trig
                    e = mm + 1
                    if e <= LAST_ENTRY_MIN:
                        px = _px_at(OP[j], e)
                        if np.isfinite(px):
                            slide_evs.append({"sym": sym, "d": dc, "entry_px": px,
                                              "detect_min": e + 1 / (1 + r), "src": "slide",
                                              "chase": px / c[i - 1] - 1.0, "ratio": r, "entry_min": e,
                                              **exits(px, i, e)})
        if k_ % 50 == 0:
            logger.info(f"[Combo] {k_}/{len(panel)}종목 · 장 시작 {len(open_evs):,} · 장중 {len(slide_evs):,}")

    fp = _fingerprint(fp_src)
    first = cal[min(t for t in all_days if t >= args.since)]
    last = max(max(v) for v in closes.values())
    starts = list(range(first, last - COMP_BARS + 2))

    def evaluate(evs: List[Dict[str, Any]], arm: str) -> np.ndarray:
        return np.array([run_window_intraday(evs, arm, d0, d0 + COMP_BARS - 1, closes, cost,
                                             args.slots, weight) for d0 in starts])

    cells = {
        "장 시작 + vol_alive (10/14 규칙)": (open_evs, "alive"),
        "장 시작 + VWAP 청산": (open_evs, "vwap"),
        "장 시작·장중 추가 + vol_alive": (open_evs + slide_evs, "alive"),
        "장 시작·장중 추가 + VWAP 청산 (지금 서버)": (open_evs + slide_evs, "vwap"),
    }
    res = {name: evaluate(evs, arm) for name, (evs, arm) in cells.items()}
    if args.diag:
        _diagnose(open_evs, slide_evs, closes, cost, args, weight, starts, cal)
    ref = res["장 시작 + vol_alive (10/14 규칙)"]
    rows = []
    for name, r in res.items():
        rows.append({"rule": name, "median": float(np.median(r)), "p90": float(np.percentile(r, 90)),
                     "p50up": float((r >= 0.5).mean()), "p100up": float((r >= 1.0).mean()),
                     "loss20": float((r <= -0.2).mean()), "beats_ref": float((r > ref).mean())})
    out = {"data_fingerprint": {"hash": fp, "symbols": len(fp_src)}, "config": vars(args),
           "events": {"opening": len(open_evs), "slide": len(slide_evs)}, "windows": len(starts),
           "rows": rows}
    op_ = Path(args.out)
    op_.parent.mkdir(parents=True, exist_ok=True)
    op_.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("=" * 104)
    print(f" 조합 측정 — 데이터 지문 {fp} ({len(fp_src)}종목) · 장 시작 {len(open_evs):,} · 장중 첫 발동 "
          f"{len(slide_evs):,} · 30거래일 창 {len(starts)}개")
    print("=" * 104)
    print(f" {'규칙':<34}{'중앙':>9}{'p90':>9}{'P(+50%)':>10}{'P(+100%)':>10}{'P(-20%)':>9}{'10/14대비':>10}")
    for x in rows:
        print(f" {x['rule']:<34}{x['median'] * 100:+8.1f}%{x['p90'] * 100:+8.1f}%{x['p50up'] * 100:9.2f}%"
              f"{x['p100up'] * 100:9.2f}%{x['loss20'] * 100:8.2f}%{x['beats_ref'] * 100:9.1f}%")
    print("=" * 104)
    print(f" 저장 → {op_}")
    return 0


def _diagnose(open_evs, slide_evs, closes, cost, args, weight, starts, cal) -> None:
    """원인 진단 — 출처·청산 규칙별 사건 통계, 실제 체결된 매매의 출처·회전 수, 비용 0 비교."""
    print()
    print(" [진단 1] 사건 단위(체결 여부와 무관) — 거래당 평균 · 단일 2배 · 진입 당일 청산 · 보유일 중앙 · 진입 시 전일比")
    for src, evs in (("장 시작", open_evs), ("장중", slide_evs)):
        ch = np.array([e["chase"] for e in evs])
        for arm in ("alive", "vwap"):
            r = np.array([e[arm]["ret"] for e in evs])
            same = np.mean([e[arm]["exit_d"] == e["d"] for e in evs])
            hold = np.median([e[arm]["exit_d"] - e["d"] for e in evs])
            print(f"   {src:<5} {arm:<6} n={len(r):6,d} 평균 {r.mean() * 100:+6.2f}% · 2배 {(r >= 1).mean() * 100:5.2f}% · "
                  f"당일 청산 {same * 100:5.1f}% · 보유 중앙 {hold:4.1f}일 · 전일比 중앙 {np.median(ch) * 100:+5.1f}%")
    if slide_evs:
        em = np.array([e["entry_min"] for e in slide_evs])
        print(f"   장중 진입 분 분포(09:00 기준): 10%{np.percentile(em, 10):.0f} · 중앙 {np.median(em):.0f} · "
              f"90%{np.percentile(em, 90):.0f}")
    # 실제 체결 매매 — 지금 서버 조합에서 무엇이 얼마나 사지나
    evs = open_evs + slide_evs
    trades = {"open": 0, "slide": 0}
    rets = {"open": [], "slide": []}
    for d0 in starts[::10]:
        d1 = d0 + COMP_BARS - 1
        took = _simulate_trades(evs, "vwap", d0, d1, args.slots)
        for e in took:
            trades[e["src"]] += 1
            rets[e["src"]].append(e["vwap"]["ret"])
    n_win = len(starts[::10])
    print(f" [진단 2] 지금 서버 조합에서 실제로 사진 매매(30일 창 {n_win}개 표본): "
          f"창당 장 시작 {trades['open'] / n_win:.1f}회 · 장중 {trades['slide'] / n_win:.1f}회")
    for k in ("open", "slide"):
        if rets[k]:
            r = np.array(rets[k])
            print(f"   {k:<5} 체결 매매 평균 {r.mean() * 100:+.2f}% · 이긴 비율 {(r > 0).mean() * 100:.1f}% · "
                  f"2배 {(r >= 1).mean() * 100:.2f}%")
    # 비용 0 — 회전 비용이 얼마나 깎나
    evs0 = []
    for e in evs:
        e2 = dict(e)
        e2["vwap"] = {**e["vwap"], "ret": e["vwap"]["ret"] + cost}
        evs0.append(e2)
    r0 = np.array([run_window_intraday(evs0, "vwap", d0, d0 + COMP_BARS - 1, closes, 0.0, args.slots, weight)
                   for d0 in starts])
    print(f" [진단 3] 지금 서버 조합, 비용 0: 중앙 {np.median(r0) * 100:+.1f}% · P(+100%) {(r0 >= 1).mean() * 100:.2f}% · "
          f"P(-20%) {(r0 <= -0.2).mean() * 100:.2f}%  (왕복 비용 {cost * 100:.2f}%/회)")


def _simulate_trades(events, arm, d0, d1, slots):
    """run_window_intraday 와 같은 순서로 실제로 사진 사건 목록만 돌려준다(진단용)."""
    held, took = [], []
    by_day = {}
    for ev in events:
        if d0 <= ev["d"] <= d1:
            by_day.setdefault(ev["d"], []).append(ev)
    for d in range(d0, d1 + 1):
        sold = {h["sym"] for h in held if h["x"]["exit_d"] == d and h["x"]["when"] == "open"}
        held = [h for h in held if not (h["x"]["exit_d"] == d and h["x"]["when"] == "open")]
        for ev in sorted(by_day.get(d, []), key=lambda e: e["detect_min"]):
            gone = [h for h in held if h["x"]["exit_d"] == d and h["x"]["when"] == "intra"
                    and h["x"]["exit_min"] <= ev["detect_min"]]
            sold |= {h["sym"] for h in gone}
            held = [h for h in held if h not in gone]
            if len(held) >= slots or ev["sym"] in sold or any(h["sym"] == ev["sym"] for h in held):
                continue
            held.append({"sym": ev["sym"], "x": ev[arm]})
            took.append(ev)
        held = [h for h in held if not (h["x"]["exit_d"] == d and h["x"]["when"] in ("intra", "close"))]
    return took


def _vwap_min(g: Optional[pd.DataFrame], after_mi: int, sell_px: float) -> float:
    """VWAP 청산이 체결된 분(다음 분 시가) — 같은 날 장중 재진입 순서를 맞추는 데만 쓴다."""
    if g is None or g.empty:
        return float(EOD_MIN)
    mi = g["mi"].to_numpy()
    op = g["open"].to_numpy(float)
    hits = np.flatnonzero((mi > after_mi) & np.isclose(op, sell_px))
    return float(mi[hits[0]]) if len(hits) else float(EOD_MIN)


if __name__ == "__main__":
    raise SystemExit(main())
