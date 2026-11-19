"""슬롯 시뮬레이션 — `vol_alive` 4.06%가 라운드 근사의 산물인지 확인한다.

## 왜 이 측정이 필요한가

실엔진 검증(`reports/ablate_catalyst_v2.json`)에서 고정 6봉이 `vol_alive` x2보다
꼬리·평균 모두 앞섰다(17/28구간). 사건 단위 측정(`intraday_entry_study.py --state`)은
반대로 2.21% → 4.06%였다.

사건 단위는 거래별 수익률을 뽑아 **"30일 ÷ 중앙 보유일수 = 라운드 수"** 만큼 복리로
굴린다. 고정 6봉은 모든 거래가 6일이라 근사가 정확하다. `vol_alive`는 보유일수가
제각각인데, **20~30일 들고 가서 2배가 된 꼬리 거래도 6일짜리 한 라운드**로 세고
그 위에 4라운드를 더 복리로 얹는다. 꼬리를 만드는 것이 바로 그 긴 보유 거래라
근사가 꼬리를 부풀릴 수 있다.

## 무엇을 하는가

같은 사건 집합(장중 감지 · 다음 분 시가 진입)을 **달력 위에서 시간순으로** 돌린다.

- 슬롯 2개 x 45%. 보유가 끝나야 슬롯이 빈다(라운드 가정 없음)
- 같은 날 발동이 슬롯보다 많으면 **감지가 이른 순**으로 채운다(신선도의 주성분)
- `vol_alive`: 봉 t 종가 판정 → t+1 시가 매도(측정 규약). 그날 아침 빈 슬롯은
  그날 장중 진입에 쓸 수 있다
- 고정 6봉: `close[i+6]` 매도. 그날 장중 진입에는 못 쓴다(종가에 비므로)
- 손절 25%는 두 방식에 같게(종가 판정 → 다음 시가)
- 30거래일 창을 하루씩 밀며 창 수익률을 낸다. 창 끝에 남은 보유는 종가 평가

같은 창에서 두 방식을 **대응 비교**하고, 기존 라운드 근사도 같은 표본으로 다시
계산해 4.06% · 2.21%가 재현되는지 본다(재현되면 집단이 같다는 뜻이다).

```bash
python scripts/research/slot_sim.py
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.research.intraday_entry_study import (  # noqa: E402
    COMP_BARS, _fingerprint, burst_days, detect_entry)
from scripts.research.tail_optimizer import MAX_GROSS, MAX_SINGLE, simulate  # noqa: E402
from utils.logger import enable_utf8_console, logger  # noqa: E402

ARMS = ("fixed6", "vol_alive")


def path_exit(entry_px: float, i: int, max_hold: int, stop: float,
              o: np.ndarray, c: np.ndarray, burst: Optional[np.ndarray],
              vol_n: float) -> Tuple[int, float, str]:
    """(청산 봉 인덱스, 청산가, 시점 'open'|'close').

    `intraday_entry_study.hold_by_state`와 같은 규약이다 — 판정은 봉 t 종가,
    체결은 봉 t+1 시가. `burst`가 None이면 기간만 본다(고정 보유).
    """
    for t in range(i, i + max_hold):
        px = float(c[t])
        if not (np.isfinite(px) and px > 0):
            continue
        hit = stop > 0 and px / entry_px - 1.0 <= -stop
        if not hit and burst is not None:
            b = burst[t]
            hit = not (np.isfinite(b) and b >= vol_n)
        if hit:
            nxt = float(o[t + 1]) if (t + 1) < len(o) else px
            if not (np.isfinite(nxt) and nxt > 0):
                nxt = px
            return t + 1, nxt, "open"
    return i + max_hold, float(c[i + max_hold]), "close"


def run_window(events: List[Dict[str, Any]], arm: str, d0: int, d1: int,
               closes: Dict[str, Dict[int, float]], cost: float,
               slots: int, weight: float) -> Tuple[float, int]:
    """창 [d0, d1](달력 인덱스) 수익률과 체결 수.

    자본 1에서 시작한다. 진입은 그 시점 평가자산의 `weight`(현금 한도)를 쓴다.
    평가자산은 현금 + 보유 원가로 근사한다(장중 평가 없이도 슬롯 크기가 크게
    어긋나지 않는다 — 2종목 x 45%라 차이는 진입 시점 손익의 45%분이다).
    """
    cash = 1.0
    held: List[Dict[str, Any]] = []        # {alloc, exit_d, when, ret, sym, entry_px}
    trades = 0
    by_day: Dict[int, List[Dict[str, Any]]] = {}
    for ev in events:
        if d0 <= ev["d"] <= d1:
            by_day.setdefault(ev["d"], []).append(ev)

    def release(d: int, when: str) -> None:
        nonlocal cash
        keep = []
        for h in held:
            if h["exit_d"] == d and h["when"] == when:
                cash += h["alloc"] * (1.0 + h["ret"])
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
            x = ev[arm]
            equity = cash + sum(h["alloc"] for h in held)
            alloc = min(equity * weight, cash)
            if alloc <= 0:
                break
            cash -= alloc
            held.append({"alloc": alloc, "exit_d": x["exit_d"], "when": x["when"],
                         "ret": x["ret"], "sym": ev["sym"], "entry_px": ev["entry_px"]})
            trades += 1
        release(d, "close")
    # 창 끝 — 남은 보유는 d1 종가로 평가(매도 비용 절반 반영)
    for h in held:
        px = closes[h["sym"]].get(d1)
        if px and px > 0:
            cash += h["alloc"] * (px / h["entry_px"] * (1.0 - cost / 2.0))
        else:
            cash += h["alloc"]
    return cash - 1.0, trades


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="슬롯 시뮬레이션 — 라운드 근사 검증")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--minute-dir", default="data/cache/minute")
    ap.add_argument("--burst-mult", type=float, default=10.0)
    ap.add_argument("--hold-vol", type=float, default=2.0)
    ap.add_argument("--hold", type=int, default=6)
    ap.add_argument("--max-hold", type=int, default=30)
    ap.add_argument("--stop", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--since", default="20230101")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--out", default="reports/slot_sim.json")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    weight = min(MAX_SINGLE, MAX_GROSS / max(1, args.slots))
    panel = load_panel_from_dir(args.source, list(s.UNIVERSE))
    mdir = Path(args.minute_dir)
    span = max(args.hold, args.max_hold)      # 집단 고정 — intraday_entry_study와 같다

    # 달력: 전 종목 날짜의 합집합
    all_days = sorted({str(t) for df in panel.values() if df is not None
                       for t in df["ts"].astype(str)})
    cal = {t: k for k, t in enumerate(all_days)}

    events: List[Dict[str, Any]] = []
    closes: Dict[str, Dict[int, float]] = {}
    fp_src: Dict[str, np.ndarray] = {}
    holds: Dict[str, List[int]] = {a: [] for a in ARMS}
    for k, (sym, df) in enumerate(sorted(panel.items()), 1):
        if df is None or len(df) < args.warmup + span + 5:
            continue
        d = df.reset_index(drop=True)
        fp_src[sym] = d["close"].to_numpy(float)
        ts = d["ts"].astype(str).to_numpy()
        o = d["open"].to_numpy(float)
        c = d["close"].to_numpy(float)
        closes[sym] = {cal[t]: float(px) for t, px in zip(ts, c)}
        idx, thresh = burst_days(d, args.burst_mult, args.warmup)
        idx = idx[(idx + span) < len(d)]
        idx = idx[ts[idx] >= args.since]
        if not len(idx):
            continue
        path = mdir / f"{sym}.csv"
        if not path.exists():
            continue
        try:
            md = pd.read_csv(path, dtype={"ts": str})
        except Exception:
            continue
        if md.empty:
            continue
        by_day = {day: g for day, g in md.groupby(md["ts"].str[:8], sort=False)}
        tov = c * d["volume"].to_numpy(float)
        med = pd.Series(tov).rolling(20, min_periods=10).median().shift(1).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            burst = tov / np.where(med > 0, med, np.nan)
        for i in idx:
            g = by_day.get(ts[i])
            if g is None or g.empty:
                continue
            got = detect_entry(g, float(thresh[i]))
            if got is None:
                continue
            mins, entry_px, _ = got
            if not entry_px > 0:
                continue
            ev: Dict[str, Any] = {"sym": sym, "d": cal[ts[i]], "detect_min": mins,
                                  "entry_px": entry_px}
            for arm, (mh, bl) in {"fixed6": (args.hold, None),
                                  "vol_alive": (args.max_hold, burst)}.items():
                ex_i, ex_px, when = path_exit(entry_px, i, mh, args.stop, o, c,
                                              bl, args.hold_vol)
                ev[arm] = {"exit_d": cal[ts[min(ex_i, len(ts) - 1)]], "when": when,
                           "ret": ex_px / entry_px - 1.0 - cost, "hold": ex_i - i}
                holds[arm].append(ex_i - i)
            events.append(ev)
        if k % 50 == 0:
            logger.info(f"[SlotSim] {k}/{len(panel)}종목 · 사건 {len(events):,}")

    if not events:
        logger.error("[SlotSim] 사건이 없습니다")
        return 1
    fp = _fingerprint(fp_src)

    # --- 1) 기존 라운드 근사 재현(같은 표본) ---
    rng = np.random.default_rng(16)
    approx: Dict[str, Any] = {}
    for arm in ARMS:
        r = np.array([ev[arm]["ret"] for ev in events])
        rounds = max(1, COMP_BARS // max(1, int(np.median(holds[arm]))))
        sim = simulate(r, args.slots, rounds, args.trials, 0.0, rng)
        approx[arm] = {"rounds": rounds, "p100": float((sim >= 1.0).mean()),
                       "single_2x": float((r >= 1.0).mean()),
                       "hold_mean": float(np.mean(holds[arm])),
                       "hold_median": float(np.median(holds[arm])),
                       "hold_p90": float(np.percentile(holds[arm], 90))}

    # --- 2) 달력 슬롯 시뮬레이션 ---
    first = min(ev["d"] for ev in events)
    last_close = max(max(v) for v in closes.values())
    starts = list(range(first, last_close - COMP_BARS + 2))
    res: Dict[str, np.ndarray] = {}
    ntr: Dict[str, np.ndarray] = {}
    for arm in ARMS:
        out_r, out_n = [], []
        for d0 in starts:
            r, n = run_window(events, arm, d0, d0 + COMP_BARS - 1, closes, cost,
                              args.slots, weight)
            out_r.append(r)
            out_n.append(n)
        res[arm] = np.array(out_r)
        ntr[arm] = np.array(out_n)

    diff = res["fixed6"] - res["vol_alive"]
    # 겹치지 않는 창(30일 간격)만으로 본 대응 비교 — 겹치는 창은 서로 상관된다
    nonover = diff[::COMP_BARS]
    summary: Dict[str, Any] = {}
    for arm in ARMS:
        x = res[arm]
        summary[arm] = {
            "windows": int(len(x)), "median": float(np.median(x)),
            "mean": float(x.mean()), "p75": float(np.percentile(x, 75)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max()), "p50up": float((x >= 0.5).mean()),
            "p100up": float((x >= 1.0).mean()), "trades_mean": float(ntr[arm].mean()),
        }
    out = {
        "data_fingerprint": {"hash": fp, "symbols": len(fp_src)},
        "config": vars(args) | {"weight": weight, "round_trip_cost": cost},
        "events": len(events), "approx": approx, "slot": summary,
        "paired": {"fixed6_better_share": float((diff > 0).mean()),
                   "diff_mean": float(diff.mean()), "diff_median": float(np.median(diff)),
                   "nonoverlap_n": int(len(nonover)),
                   "nonoverlap_fixed6_better": int((nonover > 0).sum())},
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 100)
    print(f" 슬롯 시뮬레이션 — 거래대금 x{args.burst_mult:g} 장중 진입 · {args.slots}슬롯 x "
          f"{weight:.0%} · 손절 {args.stop:.0%} · 왕복비용 {cost:.3%}")
    print(f" 데이터 지문 {fp} ({len(fp_src)}종목) · 사건 {len(events):,}건 · "
          f"30거래일 창 {len(starts):,}개(하루씩 이동, 서로 겹친다)")
    print("=" * 100)
    print(" [1] 기존 라운드 근사 (intraday_entry_study 방식, 같은 표본)")
    for arm in ARMS:
        a = approx[arm]
        print(f"   {arm:<10} 보유 평균 {a['hold_mean']:.1f} · 중앙 {a['hold_median']:.0f} · "
              f"p90 {a['hold_p90']:.0f}일 → {a['rounds']}라운드 · 단일 2배 "
              f"{a['single_2x']:.2%} · 포트 P(+100%) {a['p100']:.2%}")
    print("-" * 100)
    print(" [2] 달력 슬롯 시뮬레이션 (보유가 끝나야 슬롯이 빈다)")
    print(f"   {'방식':<10}{'중앙':>9}{'평균':>9}{'p90':>9}{'p99':>9}{'최대':>9}"
          f"{'P(+50%)':>10}{'P(+100%)':>10}{'체결/창':>9}")
    for arm in ARMS:
        x = summary[arm]
        print(f"   {arm:<10}{x['median'] * 100:+8.1f}%{x['mean'] * 100:+8.1f}%"
              f"{x['p90'] * 100:+8.1f}%{x['p99'] * 100:+8.1f}%{x['max'] * 100:+8.1f}%"
              f"{x['p50up'] * 100:9.2f}%{x['p100up'] * 100:9.2f}%{x['trades_mean']:9.1f}")
    p = out["paired"]
    print(f"   대응: 고정6이 나은 창 {p['fixed6_better_share']:.1%} · 차이 평균 "
          f"{p['diff_mean'] * 100:+.2f}%p · 중앙 {p['diff_median'] * 100:+.2f}%p · "
          f"겹치지 않는 창 {p['nonoverlap_fixed6_better']}/{p['nonoverlap_n']}")
    print("=" * 100)
    print(f" 저장 → {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
