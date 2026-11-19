"""장 시작 발동 종목 중 어느 2개를 사는가 — 선정 기준 비교(분봉 기반).

## 왜 필요한가

494종목에서 장 시작 5분 같은 시간대 x3 발동은 하루 평균 약 55건인데 자리는 2개다. 지금은
"배수 큰 순"으로 채우는데, 이 순서는 측정으로 고른 적이 없다(`opening_study.py` 주석: 근거 없는 선택).
CLAUDE.md 의 교훈대로 **선정이 전략의 거의 전부**이므로 09:0N 까지 알 수 있는 값만으로 순서를 바꿔 본다.

## 무엇을 재나

1. **모집단 분위표** — 발동 사건 전체를 특징값으로 5등분해 거래당 평균·단일 2배 비율을 본다
   (선택된 표본만 보면 판단이 틀린다, CLAUDE.md §11)
2. **선정 규칙 격자** — 같은 사건 집합에서 그날 채우는 순서만 바꿔 30거래일 창 P(+100%)를 잰다.
   청산은 지금 규칙(vol_alive x2 · 손절 25%)으로 고정한다

특징값(전부 진입 시점 이전에 알 수 있다)
- `ratio`  구간 거래대금 / 지난 20일 같은 구간 중앙 (현재 순서)
- `chase`  진입가 / 전일 종가 - 1  (이미 얼마나 올랐나)
- `ret_w`  구간 끝 종가 / 09:00 시가 - 1  (구간 안의 흐름)
- `gap`    09:00 시가 / 전일 종가 - 1
- `tv`     구간 거래대금 절대액

```bash
UNIVERSE_FILE=config/universe_10b.json python scripts/research/selection_study.py --window 5
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.research.intraday_entry_study import _fingerprint  # noqa: E402
from scripts.research.opening_study import evaluate  # noqa: E402
from scripts.research.slot_sim import path_exit  # noqa: E402
from scripts.research.tail_optimizer import MAX_GROSS, MAX_SINGLE  # noqa: E402
from strategy.opening import minute_index, same_window_base, window_turnover  # noqa: E402
from utils.logger import enable_utf8_console, logger  # noqa: E402

FEATURES = ("ratio", "chase", "ret_w", "gap", "tv")


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="장 시작 발동 종목 선정 기준 비교")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--opening-dir", default="data/cache/opening")
    ap.add_argument("--since", default="20250102")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--mult", type=float, default=3.0)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--hold-vol", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=30)
    ap.add_argument("--stop", type=float, default=0.25)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=20, help="무작위 선정 반복 수")
    ap.add_argument("--out", default="reports/selection_study.json")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    weight = min(MAX_SINGLE, MAX_GROSS / max(1, args.slots))
    N, entry_mi = args.window, args.window + args.lag
    odir = Path(args.opening_dir)

    panel = load_panel_from_dir(args.source, list(s.UNIVERSE))
    all_days = sorted({str(t) for df in panel.values() if df is not None
                       for t in df["ts"].astype(str)})
    cal = {t: k for k, t in enumerate(all_days)}

    events: List[Dict[str, Any]] = []
    closes: Dict[str, Dict[int, float]] = {}
    fp_src: Dict[str, np.ndarray] = {}
    for k_, (sym, df) in enumerate(sorted(panel.items()), 1):
        if df is None or len(df) < 60 or not (odir / f"{sym}.csv").exists():
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

        m = pd.read_csv(odir / f"{sym}.csv", dtype={"ts": str})
        m = m[m["ts"].str[:8] >= args.since]
        if m.empty:
            continue
        m = m.assign(mi=minute_index(m["ts"]), day=m["ts"].str[:8])
        fp_src[sym] = (m["close"] * m["volume"]).to_numpy(float)
        days = sorted(m["day"].unique())
        by_day = {dy: g.sort_values("mi") for dy, g in m.groupby("day")}
        wtv = pd.Series([window_turnover(by_day[dy]["mi"].to_numpy(),
                                         by_day[dy]["close"].to_numpy(float),
                                         by_day[dy]["volume"].to_numpy(float), N)
                         for dy in days], index=days)
        base_s = pd.Series([same_window_base(wtv.iloc[:j].tolist()) for j in range(len(days))],
                           index=days)
        for dy in days:
            i = pos.get(dy)
            if i is None or i < 25 or i + args.max_hold + 1 >= len(d):
                continue
            w, b = wtv[dy], base_s[dy]
            if not (np.isfinite(b) and b > 0 and w > 0) or w / b < args.mult:
                continue
            g = by_day[dy]
            gmi = g["mi"].to_numpy()
            ent = g.loc[gmi == entry_mi, "open"]
            first = g.loc[gmi == 0, "open"]
            last_in = g.loc[gmi == N - 1, "close"]
            prev = float(c[i - 1])
            if ent.empty or not (float(ent.iloc[0]) > 0) or not (prev > 0):
                continue
            entry_px = float(ent.iloc[0])
            o0 = float(first.iloc[0]) if not first.empty and first.iloc[0] > 0 else float(o[i])
            cw = float(last_in.iloc[0]) if not last_in.empty else entry_px
            ex_i, ex_px, when = path_exit(entry_px, i, args.max_hold, args.stop, o, c,
                                          burst, args.hold_vol)
            events.append({
                "sym": sym, "d": cal[dy], "entry_px": entry_px,
                "ratio": w / b, "chase": entry_px / prev - 1.0, "ret_w": cw / o0 - 1.0,
                "gap": o0 / prev - 1.0, "tv": w,
                "rule": {"exit_d": cal[ts[min(ex_i, len(ts) - 1)]], "when": when,
                         "ret": ex_px / entry_px - 1.0 - cost}})
        if k_ % 100 == 0:
            logger.info(f"[Selection] {k_}/{len(panel)}종목 · 사건 {len(events):,}")

    if not events:
        logger.error("[Selection] 사건이 없습니다")
        return 1
    fp = _fingerprint(fp_src)
    first_d = cal[min(t for t in all_days if t >= args.since)]
    last_d = max(max(v) for v in closes.values())
    rets = np.array([e["rule"]["ret"] for e in events])

    # --- 1. 모집단 분위표 ---
    quint: Dict[str, List[Dict[str, float]]] = {}
    for f in FEATURES:
        x = np.array([e[f] for e in events], float)
        qs = np.nanquantile(x, [0, .2, .4, .6, .8, 1.0])
        rows = []
        for q in range(5):
            lo, hi = qs[q], qs[q + 1]
            mask = (x >= lo) & ((x <= hi) if q == 4 else (x < hi))
            r = rets[mask]
            rows.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                         "mean": float(r.mean()) if len(r) else float("nan"),
                         "single_2x": float((r >= 1.0).mean()) if len(r) else float("nan")})
        quint[f] = rows

    # --- 2. 선정 규칙 격자 --- (detect_min 이 작을수록 먼저 채운다)
    def ranked(key: Callable[[Dict[str, Any]], float],
               keep: Callable[[Dict[str, Any]], bool] = lambda e: True) -> List[Dict[str, Any]]:
        out = []
        for e in events:
            if not keep(e):
                continue
            out.append({**e, "detect_min": float(key(e))})
        return out

    rules: Dict[str, List[Dict[str, Any]]] = {
        "배수 큰 순(현재)": ranked(lambda e: -e["ratio"]),
        "덜 오른 순(chase 작은)": ranked(lambda e: e["chase"]),
        "많이 오른 순(chase 큰)": ranked(lambda e: -e["chase"]),
        "구간 상승 큰 순(ret_w)": ranked(lambda e: -e["ret_w"]),
        "구간 상승 작은 순": ranked(lambda e: e["ret_w"]),
        "거래대금 큰 순(tv)": ranked(lambda e: -e["tv"]),
        "거래대금 작은 순": ranked(lambda e: e["tv"]),
        "배수 순 · 전일比 +15% 넘으면 제외": ranked(lambda e: -e["ratio"], lambda e: e["chase"] <= 0.15),
        "배수 순 · 전일比 +10% 넘으면 제외": ranked(lambda e: -e["ratio"], lambda e: e["chase"] <= 0.10),
        "배수 순 · 하락 중 제외(ret_w<0)": ranked(lambda e: -e["ratio"], lambda e: e["ret_w"] >= 0),
        "배수 순 · 0~+10% 만": ranked(lambda e: -e["ratio"], lambda e: 0 <= e["chase"] <= 0.10),
    }
    ref = evaluate(rules["배수 큰 순(현재)"], closes, cost, args.slots, weight, first_d, last_d)
    rows = []
    for name, evs in rules.items():
        r = evaluate(evs, closes, cost, args.slots, weight, first_d, last_d)
        paired = r["_rets"] - ref["_rets"]
        rows.append({"rule": name, "events": len(evs),
                     **{kk: v for kk, v in r.items() if kk != "_rets"},
                     "beats_ref": float((paired > 0).mean())})
    rng = np.random.default_rng(7)
    rnd = []
    for _ in range(args.seeds):
        perm = {id(e): float(v) for e, v in zip(events, rng.random(len(events)))}
        r = evaluate(ranked(lambda e: perm[id(e)]), closes, cost, args.slots, weight, first_d, last_d)
        rnd.append(r)
    rows.append({"rule": f"무작위(평균 {args.seeds}회)", "events": len(events),
                 "median": float(np.mean([r["median"] for r in rnd])),
                 "p90": float(np.mean([r["p90"] for r in rnd])),
                 "p50up": float(np.mean([r["p50up"] for r in rnd])),
                 "p100up": float(np.mean([r["p100up"] for r in rnd])),
                 "p100up_sd": float(np.std([r["p100up"] for r in rnd])),
                 "beats_ref": None})

    out = {"data_fingerprint": {"hash": fp, "symbols": len(fp_src)},
           "config": vars(args) | {"weight": weight, "round_trip_cost": cost},
           "events": len(events), "per_day": len(events) / max(1, len({e['d'] for e in events})),
           "quintiles": quint, "rules": rows}
    op_ = Path(args.out)
    op_.parent.mkdir(parents=True, exist_ok=True)
    op_.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 100)
    print(f" 장 시작 선정 기준 — {N}분 x{args.mult:g} 지연 {args.lag} · 청산 vol_alive x{args.hold_vol:g} · "
          f"{args.slots}슬롯 x {weight:.0%}")
    print(f" 데이터 지문 {fp} ({len(fp_src)}종목) · 사건 {len(events):,} (하루 {out['per_day']:.1f}) · "
          f"30거래일 창 {ref['windows']}개")
    print("=" * 100)
    print(" [1] 모집단 분위표 — 발동 사건 전체, 거래당 평균 / 단일 2배 비율")
    for f in FEATURES:
        cells = "  ".join(f"Q{q + 1}[{r['lo']:+.3g}~{r['hi']:+.3g}] {r['mean'] * 100:+.2f}% / "
                          f"{r['single_2x'] * 100:.2f}%" for q, r in enumerate(quint[f]))
        print(f"  {f:<6} {cells}")
    print("-" * 100)
    print(f" [2] 선정 규칙  {'사건':>7}{'중앙':>8}{'p90':>8}{'P(+50%)':>9}{'P(+100%)':>10}{'현재보다↑':>10}")
    for r in rows:
        b = "—" if r.get("beats_ref") is None else f"{r['beats_ref'] * 100:.1f}%"
        sd = f" (±{r['p100up_sd'] * 100:.1f})" if "p100up_sd" in r else ""
        print(f"  {r['rule']:<28}{r['events']:7,d}{r['median'] * 100:+7.1f}%{r['p90'] * 100:+7.1f}%"
              f"{r['p50up'] * 100:8.2f}%{r['p100up'] * 100:9.2f}%{sd}{b:>10}")
    print("=" * 100)
    print(" 현재보다↑ = 같은 30일 창에서 현재 규칙보다 수익이 높은 창 비율(겹치는 창이라 독립 표본은 약 14개)")
    print(" 한 칸만 좋은 규칙은 믿지 않는다 — 분위표와 같은 방향인지, 무작위의 흔들림(±)보다 큰지 본다.")
    print(f" 저장 → {op_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
