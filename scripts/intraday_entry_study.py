"""장중 진입 + 다일 보유 — 미측정 조합을 잰다.

## 왜 이 측정이 필요한가

일봉 catalyst 조건은 `turnover = 종가 x 거래량`이라 **장 마감 후에야 확정된다.**
그래서 T+1 시가에 들어가고 **폭증일을 통째로 버린다.** 거래대금 10배 사건
4,902건 실측:

| 구간 | 평균 | 중앙 | 승률 |
|---|---|---|---|
| 폭증일 시가→종가 | **+5.29%** | +4.41% | **71.8%** |
| 종가→익일시가(갭) | +0.86% | +0.17% | 52.8% |
| **T+1시가→T+6종가 (현재)** | **+1.11%** | **−1.50%** | **44.0%** |

버리는 구간이 평균 **+6.2%**이고, 먹는 구간은 중앙값이 음수다.

그리고 이것은 **한 번도 측정하지 않은 조합**이다.

| 진입 | 청산 | 측정 | 결과 |
|---|---|---|---|
| 분봉 장중 (+3% 돌파) | 당일 15:15 | O | `P(+100%)` **0.00%** |
| 일봉 (종가 판정) | 6봉 후 | O | **1.67%** |
| **분봉 장중 (거래대금 폭증)** | **6봉 후** | **X** | 이 스크립트 |

기존 분봉 실패는 "장중 진입"과 "당일 청산"이 **묶인 덩어리**였고 둘을 분리한 적이
없다. 당일 청산은 변동을 하루로 묶고 꼬리는 변동에서 나온다.

## 세 가지 진입을 같은 종료일로 비교한다

| arm | 진입 | 비고 |
|---|---|---|
| `intraday` | 폭증 감지된 **다음 분의 시가** | 실현 가능 |
| `next_open` | **T+1 시가** | 현재 catalyst |
| `day_open` | 폭증일 **시가** | **완전예지 상한** — 전략이 아니다 |

청산은 셋 다 `close[i + hold]`로 같다. 그래서 차이는 **진입 시점뿐**이다.
`intraday`가 폭증일 잔여 구간만큼 더 들고 있는데, **그 구간이 바로 되찾으려는
것**이므로 제거할 교란이 아니다.

## 룩어헤드를 막는 두 가지

1. **분모는 전일까지의 20봉 중앙값**이다. 일봉 피처
   (`catalyst_features.turnover_burst_20`)는 `rolling(20)`으로 당일을 포함하는데,
   인과적이긴 해도 09:00에는 모르는 값이다. 여기서는 `.shift(1)`을 쓴다.
   폭증일 자신이 중앙값에 들어가지 않으므로 **같은 배수에서 약간 더 쉽게**
   발동한다 — 두 집단 크기를 같이 찍어 차이를 드러낸다.
2. **진입은 감지된 분의 *다음* 분 시가**다. 같은 분의 종가/시가를 쓰면 그 분이
   끝나기 전에 그 분의 누적 거래대금을 아는 셈이 된다.

```bash
python scripts/intraday_entry_study.py                    # 기본(10배 · 6봉 보유)
python scripts/intraday_entry_study.py --hold 10
python scripts/intraday_entry_study.py --burst-mult 5 --trials 80000
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.tail_optimizer import MAX_GROSS, MAX_SINGLE, simulate  # noqa: E402
from utils.logger import enable_utf8_console, logger  # noqa: E402

COMP_BARS = 30            # 대회 거래일
ARMS = ("intraday", "next_open", "day_open")


def _fingerprint(obj: Dict[str, np.ndarray]) -> str:
    h = hashlib.md5()
    for k in sorted(obj):
        h.update(k.encode())
        h.update(np.ascontiguousarray(obj[k], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def burst_days(df: pd.DataFrame, mult: float, warmup: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """(인덱스, 그날의 임계 거래대금). 분모는 **전일까지**의 20봉 중앙값이다."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    turnover = close * volume
    med20 = turnover.rolling(20, min_periods=10).median().shift(1)
    thresh = (med20 * mult).to_numpy(float)
    hit = turnover.to_numpy(float) >= thresh
    idx = np.flatnonzero(np.nan_to_num(hit, nan=0.0).astype(bool))
    idx = idx[idx >= warmup]
    return idx, thresh


def detect_entry(minute: pd.DataFrame, thresh: float
                 ) -> Optional[tuple[int, float, float]]:
    """(감지 분, 진입가, 그날 시가). 감지된 분의 **다음 분 시가**로 들어간다.

    분당 거래대금은 `종가 x 거래량`으로 누적한다. 정확한 값은 체결 단가의
    가중합이지만 틱 테이프가 없고, **분모와 같은 정의**를 쓰므로 체계적 편향은 작다.
    """
    if minute.empty or not np.isfinite(thresh) or thresh <= 0:
        return None
    o = minute["open"].to_numpy(float)
    c = minute["close"].to_numpy(float)
    v = minute["volume"].to_numpy(float)
    cum = np.cumsum(c * v)
    hits = np.flatnonzero(cum >= thresh)
    if not len(hits):
        return None
    j = int(hits[0])
    if j + 1 >= len(minute):
        return None          # 마지막 분에 닿으면 다음 분이 없다 — 버린다
    hhmm = str(minute["ts"].iloc[j])[8:12]
    mins = max(0, (int(hhmm[:2]) - 9) * 60 + int(hhmm[2:]))
    return mins, float(o[j + 1]), float(o[0])


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="장중 진입 + 다일 보유 측정")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--minute-dir", default="data/cache/minute")
    ap.add_argument("--burst-mult", type=float, default=10.0)
    ap.add_argument("--hold", type=int, default=6, help="보유 일수(일봉 기준)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--per-day", type=int, default=2, help="동시 보유 종목 수")
    ap.add_argument("--out", default="reports/intraday_entry_study.json")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    panel = load_panel_from_dir(args.source, list(s.UNIVERSE))
    if not panel:
        logger.error(f"[EntryStudy] {args.source} 에 일봉이 없습니다")
        return 1
    mdir = Path(args.minute_dir)

    rows: List[Dict[str, Any]] = []
    n_burst = n_cover = n_detect = 0
    fp_src: Dict[str, np.ndarray] = {}

    for k, (sym, df) in enumerate(sorted(panel.items()), 1):
        if df is None or len(df) < args.warmup + args.hold + 5:
            continue
        d = df.reset_index(drop=True)
        fp_src[sym] = d["close"].to_numpy(float)
        idx, thresh = burst_days(d, args.burst_mult, args.warmup)
        idx = idx[(idx + args.hold) < len(d)]
        if not len(idx):
            continue
        n_burst += len(idx)
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

        ts = d["ts"].astype(str).to_numpy()
        o = d["open"].to_numpy(float)
        c = d["close"].to_numpy(float)
        for i in idx:
            day = ts[i]
            g = by_day.get(day)
            if g is None or g.empty:
                continue
            n_cover += 1
            got = detect_entry(g, float(thresh[i]))
            if got is None:
                continue
            n_detect += 1
            mins, entry_px, day_open = got
            exit_px = c[i + args.hold]
            if not (np.isfinite(exit_px) and exit_px > 0 and entry_px > 0):
                continue
            nxt_open = o[i + 1] if (i + 1) < len(o) else np.nan
            rows.append({
                "symbol": sym, "day": day, "detect_min": mins,
                "entry_vs_open": entry_px / max(day_open, 1e-9) - 1.0,
                "intraday": exit_px / entry_px - 1.0 - cost,
                "next_open": (exit_px / nxt_open - 1.0 - cost
                              if np.isfinite(nxt_open) and nxt_open > 0 else np.nan),
                "day_open": exit_px / max(day_open, 1e-9) - 1.0 - cost,
            })
        if k % 50 == 0:
            logger.info(f"[EntryStudy] {k}/{len(panel)}종목 · 사건 {len(rows):,}")

    if not rows:
        logger.error("[EntryStudy] 사건이 없습니다 — 분봉 커버리지를 확인하십시오")
        return 1

    fr = pd.DataFrame(rows)
    fp = _fingerprint(fp_src)
    rounds = max(1, COMP_BARS // max(1, args.hold))
    w = min(MAX_SINGLE, MAX_GROSS / max(1, args.per_day))
    rng = np.random.default_rng(16)

    out: Dict[str, Any] = {
        "data_fingerprint": {"hash": fp, "symbols": len(fp_src)},
        "config": {"burst_mult": args.burst_mult, "hold": args.hold,
                   "rounds": rounds, "per_day": args.per_day, "weight": w,
                   "trials": args.trials, "round_trip_cost": cost,
                   "warmup": args.warmup},
        "coverage": {"burst_days": n_burst, "with_minute": n_cover,
                     "detected": n_detect, "usable": int(len(fr))},
        "detect_min": {q: float(np.percentile(fr["detect_min"], q))
                       for q in (10, 25, 50, 75, 90)},
        "arms": {},
    }

    for arm in ARMS:
        r = fr[arm].dropna().to_numpy(float)
        if not len(r):
            continue
        sim = simulate(r, args.per_day, rounds, args.trials, 0.0, rng)
        out["arms"][arm] = {
            "n": int(len(r)), "mean": float(r.mean()),
            "median": float(np.median(r)), "win": float((r > 0).mean()),
            "p90": float(np.percentile(r, 90)), "std": float(r.std()),
            "p100_single": float((r >= 1.0).mean()),
            "port_median": float(np.median(sim)) if len(sim) else float("nan"),
            "port_p50up": float((sim >= 0.5).mean()) if len(sim) else float("nan"),
            "port_p100up": float((sim >= 1.0).mean()) if len(sim) else float("nan"),
            "port_p50dn": float((sim <= -0.5).mean()) if len(sim) else float("nan"),
        }

    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    tmp = op.with_suffix(op.suffix + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    tmp.replace(op)

    # ---------------------------------------------------------------- 출력
    print()
    print("=" * 104)
    print(f" 장중 진입 + 다일 보유 — 거래대금 x{args.burst_mult:g} · 보유 {args.hold}봉 x "
          f"{rounds}라운드 · {args.per_day}종목 x {w:.0%} · 왕복비용 {cost:.3%}")
    print(f" 데이터 지문 {fp} ({len(fp_src)}종목) — 지문이 다른 실행과는 비교할 수 없다")
    print("=" * 104)
    print(f" 폭증일 {n_burst:,} → 분봉 보유 {n_cover:,} ({n_cover / max(1, n_burst):.1%}) "
          f"→ 감지 {n_detect:,} → 사용 {len(fr):,}")
    dm = out["detect_min"]
    print(f" 감지 시각(09:00 기준 분)  p10 {dm[10]:.0f} · p25 {dm[25]:.0f} · "
          f"중앙 {dm[50]:.0f} · p75 {dm[75]:.0f} · p90 {dm[90]:.0f}")
    print(f" 진입가 vs 당일 시가       중앙 {np.median(fr['entry_vs_open']) * 100:+.2f}% "
          f"· p90 {np.percentile(fr['entry_vs_open'], 90) * 100:+.1f}%")
    print("-" * 104)
    print(f" {'진입 방식':<28}{'표본':>8}{'평균/회':>10}{'중앙/회':>10}{'승률':>8}"
          f"{'30일중앙':>10}{'P(+50%)':>10}{'P(+100%)':>11}")
    print("-" * 104)
    label = {"intraday": "분봉 장중 감지 (실현 가능)",
             "next_open": "T+1 시가 (현재 catalyst)",
             "day_open": "폭증일 시가 (완전예지 상한)"}
    for arm in ARMS:
        a = out["arms"].get(arm)
        if not a:
            continue
        print(f" {label[arm]:<28}{a['n']:8,d}{a['mean'] * 100:+9.2f}%"
              f"{a['median'] * 100:+9.2f}%{a['win'] * 100:7.1f}%"
              f"{a['port_median'] * 100:+9.1f}%{a['port_p50up'] * 100:9.2f}%"
              f"{a['port_p100up'] * 100:10.2f}%")
    print("=" * 104)
    print(" 기준선 — 일봉 catalyst 2종목x6봉x5회전 1.67% · 스윙 8종목 0.96% · 분봉 단타 0.00%")
    print(" `day_open`은 09:00에 그날의 거래대금을 아는 셈이라 **전략이 아니라 상한**이다.")
    print(" 분모가 전일까지의 20봉 중앙값이라 일봉 피처(당일 포함)보다 약간 쉽게 발동한다.")
    print(f" 저장 → {op}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
