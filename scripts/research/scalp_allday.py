"""장 전체 1% 단타(11/18 사용자 요청) — 직전 5분 거래대금이 평소 같은 시각의 k배 이상(까다로운 기준) + 그 5분 가격 상승이면 산다.

- 평소 = 지난 20거래일(데이터 있는 날, 최소 10일) 같은 분의 직전 5분 거래대금 중앙값(엔진 `same_window_base`와 같은 정의).
- 판단 = 분봉 m 종가까지, 매수 = m+1 분 시가. 동시 발동은 배수 큰 순. 한 번에 한 종목 · 자금 전부.
- 청산 = +tp 익절(분봉 고가가 닿으면 그 가격) · 손절 sl(저가가 닿으면) · 15:10 정리. 같은 분에 둘 다면 손절 먼저(보수적).
- 진입 09:05 ~ 15:00, 하루 최대 n_max 회, 같은 종목 하루 2번까지. 판 분이 끝나면 그 뒤 새로 발동하는 종목을 산다.

    PYTHONIOENCODING=utf-8 python scripts/research/scalp_allday.py
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.settings import get_settings  # noqa: E402
from scripts.research.hold_study import load_all  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

W = 5
FIRST_M, LAST_M, END_M = 5, 360, 370       # 09:05 ~ 15:00 진입, 15:10 정리
NM = 391


def prep(data, days):
    """종목별 [날짜 x 분] 배수 배열(조건 불충족은 0)."""
    ratio: Dict[str, Dict[str, np.ndarray]] = {}
    for k, (sym, per) in enumerate(data.items(), 1):
        ds = [d for d in days if d in per]
        if len(ds) < 15:
            continue
        X = np.zeros((len(ds), NM), dtype=np.float64)
        for j, d in enumerate(ds):
            a = per[d]
            tvm = np.nan_to_num(a[3].astype(float)) * np.nan_to_num(a[4].astype(float))
            cs = np.cumsum(tvm)
            X[j, W - 1:] = cs[W - 1:] - np.concatenate([[0.0], cs[:-W]])
        out = {}
        for j, d in enumerate(ds):
            if j < 10:
                continue
            hist = X[max(0, j - 20):j]
            if hist.shape[0] < 10:
                continue
            base = np.median(hist, axis=0)          # same_window_base 와 같은 정의(지난 20일 중앙, 최소 10일)를 분마다 한 번에
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(base > 0, X[j] / base, 0.0)
            c = per[d][3].astype(float)
            up = np.zeros(NM, dtype=bool)
            up[W:] = c[W:] > c[:-W]
            r = np.where(up & np.isfinite(r), r, 0.0)
            r[:FIRST_M] = 0.0
            r[LAST_M + 1:] = 0.0
            out[d] = r.astype(np.float32)
        ratio[sym] = out
        if k % 50 == 0:
            print(f"  배수 {k}/{len(data)}", flush=True)
    return ratio


def sim_day(day, syms, R, data, k_mult, tp, sl, n_max, cost):
    """R: [S x NM] 배수. 한 종목씩 순서대로."""
    trig = R >= k_mult
    used: Dict[int, int] = {}
    trades = []
    m = FIRST_M
    while len(trades) < n_max and m <= LAST_M:
        col = np.where(trig[:, m], R[:, m], 0.0)
        for i_, c_ in used.items():
            if c_ >= 2:
                col[i_] = 0.0
        if col.max() <= 0:
            m += 1
            continue
        i = int(col.argmax())
        a = data[syms[i]][day]
        k = m + 1
        entry = float(a[0, k]) if np.isfinite(a[0, k]) else float(a[3, m])
        if not np.isfinite(entry) or entry <= 0:
            m += 1
            continue
        used[i] = used.get(i, 0) + 1
        tpx = entry * (1 + tp) if tp > 0 else np.inf
        slx = entry * (1 - sl) if sl > 0 else -np.inf
        px, why, xk = np.nan, "time", END_M
        for kk in range(k, END_M):
            o = entry if kk == k else float(a[0, kk])
            h, lo = float(a[1, kk]), float(a[2, kk])
            if not np.isfinite(h):
                continue
            if np.isfinite(o) and o <= slx:
                px, why, xk = o, "sl", kk; break
            if lo <= slx:
                px, why, xk = slx, "sl", kk; break
            if np.isfinite(o) and o >= tpx:
                px, why, xk = o, "tp", kk; break
            if h >= tpx:
                px, why, xk = tpx, "tp", kk; break
        if why == "time":
            px = float(a[0, END_M]) if np.isfinite(a[0, END_M]) else float(a[3, END_M - 1])
        trades.append({"ret": px / entry - 1.0 - cost, "why": why})
        m = xk + 1
    return trades


def main() -> int:
    enable_utf8_console()
    s = get_settings()
    cost = s.round_trip_cost
    syms_all = sorted({x for v in s.index_members().values() for x in v})
    t0 = time.time()
    data, pcs, fp = load_all(syms_all)
    days = sorted({d for per in data.values() for d in per})
    print(f"지문 {fp} · {len(data)}종목 · {len(days)}일 · 왕복 {cost:.4%} · 적재 {time.time() - t0:.0f}s", flush=True)
    ratio = prep(data, days)
    print(f"  배수 계산 끝 ({time.time() - t0:.0f}s)", flush=True)
    mats = {}
    for d in days:
        syms = [x for x in ratio if d in ratio[x]]
        if len(syms) < 50:
            continue
        mats[d] = (syms, np.stack([ratio[x][d] for x in syms]))
    # 발동 빈도(하루 몇 번 걸리나 — 종목·분 단위 첫 발동 수)
    for km in (5, 10, 20, 30, 50):
        cnt = [int(((R >= km) & ~np.concatenate([np.zeros((R.shape[0], 1), bool), (R >= km)[:, :-1]], axis=1)).sum())
               for _, R in mats.values()]
        print(f"  k={km}: 하루 발동(새로 넘는 순간) 중앙 {np.median(cnt):.0f} · 평균 {np.mean(cnt):.1f}", flush=True)

    results = []
    grid = list(product([5, 10, 20, 30, 50], [0.01], [0.0, 0.01, 0.02, 0.03], [10]))
    grid += [(km, 0.0, 0.0, 10) for km in (10, 20, 30)]         # 익절 없이 15:10까지(비교)
    grid += [(km, 0.02, 0.0, 10) for km in (10, 20, 30)]        # 익절 2%(비교)
    for km, tp, sl, n_max in grid:
        dr, tr = [], []
        for d, (syms, R) in mats.items():
            t_ = sim_day(d, syms, R, data, km, tp, sl, n_max, cost)
            tr += t_
            dr.append(float(np.prod([1 + x["ret"] for x in t_]) - 1.0))
        dr = np.array(dr)
        rets = np.array([x["ret"] for x in tr]) if tr else np.zeros(1)
        two = (1 + dr[:-1]) * (1 + dr[1:]) - 1.0
        why = {w: float(np.mean([x["why"] == w for x in tr])) if tr else 0.0 for w in ("tp", "sl", "time")}
        r = {"k": km, "tp": tp, "sl": sl, "n_max": n_max, "trades_per_day": len(tr) / len(dr),
             "trade_mean": float(rets.mean()), "trade_win": float((rets > 0).mean()), "exit_mix": why,
             "day_mean": float(dr.mean()), "day_median": float(np.median(dr)), "p_day_neg": float((dr < 0).mean()),
             "two_mean": float(two.mean()), "two_p_neg": float((two < 0).mean())}
        results.append(r)
        print(f"k={km:>2} 익절{tp:.0%} 손절{sl:.0%} | 하루 {r['trades_per_day']:.1f}회 | 거래당 {r['trade_mean']:+.3%} | 승률 {r['trade_win']:.0%} | "
              f"익절/손절/시간 {why['tp']:.0%}/{why['sl']:.0%}/{why['time']:.0%} | 하루 평균 {r['day_mean']:+.2%} 중앙 {r['day_median']:+.2%} "
              f"손실일 {r['p_day_neg']:.0%} | 이틀 평균 {r['two_mean']:+.2%} 손실 {r['two_p_neg']:.0%}", flush=True)
    out = ROOT / "reports" / "scalp_allday.json"
    out.write_text(json.dumps({"fingerprint": fp, "days": len(mats), "cost": cost, "results": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"저장 {out} · {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
