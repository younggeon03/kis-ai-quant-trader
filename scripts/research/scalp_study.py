"""9~10시 단타 측정(11/18 사용자 요청) — 거래대금·등락률 동시 상위 1위를 09:03부터 사서 +1% 익절, 하루 5~10회,
같은 종목 2번까지, 익절 때 여전히 1위면 안 팔고 +2%까지.

관제 화면 표와 **같은 함수**(`monitor.ranking.cross_rows`)로 매 분 동시 상위를 고른다. 분봉 m 이 끝난 시점의
누적 거래대금(09:00부터)·전일 대비 등락률로 순위를 매기고, m+1 분 시가에 산다(룩어헤드 없음).
보유 중에는 분봉 고가가 익절가에 닿으면 익절가(시가가 이미 위면 시가)에, 저가가 손절가에 닿으면 손절가(시가가 아래면 시가)에
판다 — 같은 분에 둘 다 닿으면 **손절 먼저**(보수적). `--end` 분(기본 10:00)이 되면 그 분 시가에 정리한다.
판 분이 끝나면 다시 고르고 다음 분에 산다. 자금은 매번 전부(한 종목), 하루 수익은 복리.

    PYTHONIOENCODING=utf-8 python scripts/research/scalp_study.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.settings import get_settings  # noqa: E402
from monitor.ranking import cross_rows  # noqa: E402
from strategy.opening import minute_index  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

N_M = 91          # 09:00 ~ 10:30 (경과 분 0..90)
TOP = 20          # 분마다 남겨 둘 동시 상위 수
WINS = (1, 3, 5, 0)   # 순위 거래대금 창(분) — 0 = 09:00부터 누적(관제 화면 표)


def load(fdir: Path, daily_dir: Path, syms: List[str], since: str):
    """종목별 09:00~10:30 분봉 → 날짜별 [종목 x 분] 배열. 전일 종가는 일봉 캐시(전 거래일 종가)."""
    per_day: Dict[str, Dict[str, np.ndarray]] = {}
    md5 = hashlib.md5()
    kept = []
    for k, sym in enumerate(syms, 1):
        p = fdir / f"{sym}.csv"
        if not p.exists():
            p = fdir / f"{sym}.csv.gz"
        dp = daily_dir / f"{sym}.csv"
        if not p.exists() or not dp.exists():
            continue
        dd = pd.read_csv(dp, dtype={"ts": str}).sort_values("ts")
        prev_close = dict(zip(dd["ts"].astype(str).str[:8], dd["close"].astype(float).shift(1)))
        fd = pd.read_csv(p, dtype={"ts": str})
        fd = fd[fd["ts"].str[:8] >= since]
        fd = fd.assign(mi=minute_index(fd["ts"]), day=fd["ts"].str[:8])
        fd = fd[(fd["mi"] >= 0) & (fd["mi"] < N_M)]
        if fd.empty:
            continue
        kept.append(sym)
        md5.update(sym.encode())
        md5.update(np.ascontiguousarray(fd["close"].to_numpy(float)).tobytes())
        for dy, g in fd.groupby("day", sort=False):
            pc = prev_close.get(dy)
            if pc is None or not np.isfinite(pc) or pc <= 0:
                continue
            arr = np.full((5, N_M), np.nan, dtype=np.float64)
            mi = g["mi"].to_numpy()
            for r, col in enumerate(("open", "high", "low", "close", "volume")):
                arr[r, mi] = g[col].to_numpy(float)
            per_day.setdefault(dy, {})[sym] = (arr, float(pc))
        if k % 50 == 0:
            print(f"  적재 {k}/{len(syms)}", flush=True)
    return per_day, kept, md5.hexdigest()[:16]


def day_matrix(day: Dict[str, Any], sym_order: List[str]):
    """하루치 → O,H,L,C(ffill),cumTV,pc 배열 [S x N_M]."""
    S = len(sym_order)
    O = np.full((S, N_M), np.nan); H = O.copy(); L = O.copy(); C = O.copy(); V = np.zeros((S, N_M)); PC = np.zeros(S)
    for i, s in enumerate(sym_order):
        arr, pc = day[s]
        O[i], H[i], L[i], C[i] = arr[0], arr[1], arr[2], arr[3]
        V[i] = np.nan_to_num(arr[4])
        PC[i] = pc
    Cf = pd.DataFrame(C.T).ffill().to_numpy().T          # 거래 없는 분은 직전 종가
    TVm = np.nan_to_num(Cf) * V                            # 분당 거래대금
    return O, H, L, Cf, TVm, PC


def cross_lists(sym_order, Cf, TVm, PC, win: int) -> List[List[int]]:
    """분 m 마다 동시 상위(관제 화면과 같은 함수) 상위 TOP 의 종목 인덱스.
    win=0 이면 09:00부터 누적 거래대금, win>0 이면 **직전 win분 거래대금**(m-win+1..m, 순간 거래대금)."""
    cs = np.cumsum(TVm, axis=1)
    if win > 0:
        sh = np.zeros_like(cs)
        sh[:, win:] = cs[:, :-win]
        cumTV = cs - sh
    else:
        cumTV = cs
    out = []
    idx_of = {s: i for i, s in enumerate(sym_order)}
    for m in range(N_M):
        rows = []
        for i, s in enumerate(sym_order):
            px = Cf[i, m]
            if not np.isfinite(px) or px <= 0:
                continue
            rows.append({"symbol": s, "chg_pct": (px / PC[i] - 1.0) * 100.0, "turnover": float(cumTV[i, m])})
        out.append([idx_of[r["symbol"]] for r in cross_rows(rows, TOP)])
    return out


def sim_day(O, H, L, Cf, lists, *, tp: float, sl: float, n_max: int, start: int, end: int,
            max_per_sym: int, extend: bool, tp_buf: float, cost: float, pick: str,
            rng: Optional[np.random.Generator]) -> List[Dict[str, Any]]:
    """11/18 사용자 규칙: 판 분이 끝나면 그때 1위를 바로 산다 · 같은 종목 하루 max_per_sym 번까지 ·
    extend 면 익절가에 닿은 분의 **직전 분 1위가 지금 종목**이고 아직 2번째가 아니면 팔지 않고 목표를 +2·tp 로 올린다
    (그 종목의 2번째 사용으로 센다)."""
    trades = []
    m = start                     # 판단 분(이 분 종가까지 안다)
    used: Dict[int, int] = {}
    S = O.shape[0]

    def top(mm: int) -> List[int]:
        if pick == "random":
            return [i for i in rng.permutation(S) if np.isfinite(Cf[i, mm])]
        return lists[mm]

    while len(trades) < n_max and m + 1 < end:
        cand = [i for i in top(m) if used.get(i, 0) < max_per_sym]
        if not cand:
            m += 1
            continue
        i = cand[0]
        k = m + 1
        entry = O[i, k] if np.isfinite(O[i, k]) else Cf[i, k - 1]
        if not np.isfinite(entry) or entry <= 0:
            m += 1
            continue
        used[i] = used.get(i, 0) + 1
        tgt = tp
        tp_px = entry * (1.0 + tgt) if tgt > 0 else np.inf
        sl_px = entry * (1.0 - sl) if sl > 0 else -np.inf
        exit_px, why, xk, ext = np.nan, "time", end, False
        kk = k
        while kk < end:
            o_ = O[i, kk] if kk > k else entry
            h_, l_ = H[i, kk], L[i, kk]
            if not np.isfinite(h_):
                kk += 1
                continue
            if np.isfinite(o_) and o_ <= sl_px:
                exit_px, why, xk = o_, "sl", kk; break
            if l_ <= sl_px:
                exit_px, why, xk = sl_px, "sl", kk; break
            if h_ >= tp_px * (1 + tp_buf):
                # 연속 같은 종목이면 팔지 않고 목표를 올린다(직전 분 순위 — 인과)
                if extend and not ext and used.get(i, 0) < max_per_sym and kk - 1 >= 0:
                    t_ = top(kk - 1)
                    if t_ and t_[0] == i:
                        used[i] += 1
                        ext = True
                        tgt = 2 * tp
                        tp_px = entry * (1.0 + tgt)
                        if h_ >= tp_px * (1 + tp_buf):      # 같은 분에 +2%까지 닿았으면 거기서 판다
                            exit_px, why, xk = max(tp_px, o_ if np.isfinite(o_) else tp_px), "tp2", kk; break
                        kk += 1
                        continue
                exit_px, why, xk = max(tp_px, o_ if np.isfinite(o_) else tp_px), ("tp2" if ext else "tp"), kk; break
            kk += 1
        if why == "time":
            exit_px = O[i, end] if end < N_M and np.isfinite(O[i, end]) else Cf[i, end - 1]
            xk = end
        trades.append({"ret": float(exit_px / entry - 1.0 - cost), "why": why, "sym": int(i),
                       "in": int(k), "out": int(xk), "ext": ext})
        m = xk                    # 판 분 종가에 다시 고른다
    return trades


def summarize(day_trades: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    days = sorted(day_trades)
    tr = [t for d in days for t in day_trades[d]]
    rets = np.array([t["ret"] for t in tr]) if tr else np.zeros(0)
    dret = np.array([np.prod([1 + t["ret"] for t in day_trades[d]]) - 1.0 for d in days])
    two = (1 + dret[:-1]) * (1 + dret[1:]) - 1.0 if len(dret) > 1 else np.zeros(0)
    why = {w: float(np.mean([t["why"] == w for t in tr])) if tr else 0.0 for w in ("tp", "tp2", "sl", "time")}
    yr = {}
    for y in ("2025", "2026"):
        sel = np.array([d.startswith(y) for d in days])
        if sel.any():
            yr[y] = {"days": int(sel.sum()), "day_mean": float(dret[sel].mean()), "p_day_neg": float((dret[sel] < 0).mean())}
    return {
        "days": len(days), "trades": len(tr), "trades_per_day": len(tr) / max(1, len(days)),
        "trade_mean": float(rets.mean()) if len(rets) else 0.0,
        "trade_win": float((rets > 0).mean()) if len(rets) else 0.0,
        "exit_mix": why,
        "day_mean": float(dret.mean()), "day_median": float(np.median(dret)),
        "p_day_neg": float((dret < 0).mean()), "p_day_le_m3": float((dret <= -0.03).mean()),
        "p_day_ge_p5": float((dret >= 0.05).mean()),
        "day_p5": float(np.percentile(dret, 5)), "day_p95": float(np.percentile(dret, 95)),
        "two_day_mean": float(two.mean()) if len(two) else 0.0,
        "two_day_p_neg": float((two < 0).mean()) if len(two) else 0.0,
        "two_day_p_ge_p10": float((two >= 0.10).mean()) if len(two) else 0.0,
        "by_year": yr,
    }


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullday-dir", default="data/cache/fullday")
    ap.add_argument("--daily-dir", default="data/cache")
    ap.add_argument("--since", default="20250101")
    ap.add_argument("--out", default="reports/scalp_study.json")
    ap.add_argument("--tp", type=float, default=0.0, help="0보다 크면 익절 폭 하나만 짧은 격자로(3분 창)")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    syms = sorted({x for v in s.index_members().values() for x in v})
    t0 = time.time()
    per_day, kept, fp = load(ROOT / args.fullday_dir, ROOT / args.daily_dir, syms, args.since)
    days = sorted(per_day)
    print(f"지문 {fp} · 지수 {len(syms)}종목 중 분봉 {len(kept)}종목 · {len(days)}일 {days[0]}~{days[-1]} · 왕복 비용 {cost:.4%}"
          f" · 적재 {time.time() - t0:.0f}s", flush=True)

    # 날마다 행렬·동시 상위 목록을 한 번만 만든다
    pre = {}
    t1 = time.time()
    for n, dy in enumerate(days, 1):
        order = sorted(per_day[dy])
        if len(order) < 50:
            continue
        O, H, L, Cf, TVm, PC = day_matrix(per_day[dy], order)
        pre[dy] = (O, H, L, Cf, {w: cross_lists(order, Cf, TVm, PC, w) for w in (WINS if args.tp <= 0 else (3,))})
        if n % 50 == 0:
            print(f"  순위 {n}/{len(days)} ({time.time() - t1:.0f}s)", flush=True)

    grid = []
    base = dict(tp=0.01, n_max=10, start=2, end=60, max_per_sym=2, extend=True, tp_buf=0.0, pick="cross", win=3)
    # 주 격자(11/18 사용자 규칙): 09:03 첫 매수 · 같은 종목 2번 · 연속 1위면 +2%까지 · 순위 = 직전 N분 거래대금 x 등락률
    for win, sl, n_max in product([1, 3, 5, 0], [0.0, 0.01, 0.015, 0.02, 0.03], [5, 10]):
        grid.append(dict(base, win=win, sl=sl, n_max=n_max))
    # 비교(3분 창): 연속 보유 없이 · 익절 체결 보수화(+0.2% 더 올라야) · 10:30까지 · 무작위 선택 · 익절 없이 1위 사서 10시 매도
    for sl in (0.0, 0.02):
        grid.append(dict(base, sl=sl, extend=False))
        grid.append(dict(base, sl=sl, tp_buf=0.002))
        grid.append(dict(base, sl=sl, end=90))
        grid.append(dict(base, sl=sl, pick="random"))
    grid.append(dict(base, tp=0.0, sl=0.0, n_max=1))
    if args.tp > 0:      # 익절 폭 하나만(11/18 0.5% 질문)
        grid = []
        for sl, n_max, ext in product([0.0, 0.01, 0.02, 0.03], [5, 10], [True, False]):
            grid.append(dict(base, tp=args.tp, sl=sl, n_max=n_max, extend=ext))
        grid.append(dict(base, tp=args.tp, sl=0.0, tp_buf=0.002))
        grid.append(dict(base, tp=args.tp, sl=0.0, pick="random"))
        grid.append(dict(base, tp=args.tp, sl=0.02, pick="random"))
    results = []
    for g in grid:
        rng = np.random.default_rng(7)
        g_ = {k: v for k, v in g.items() if k != "win"}
        dt_ = {dy: sim_day(O, H, L, Cf, lw[g["win"]], cost=cost, rng=rng, **g_) for dy, (O, H, L, Cf, lw) in pre.items()}
        r = summarize(dt_)
        r["cfg"] = g
        results.append(r)

    def lab(g):
        return (f"{'누적' if g['win'] == 0 else str(g['win']) + '분'} 익절{g['tp']:.1%} 손절{g['sl']:.1%} {g['n_max']:>2}회 첫매수09:{g['start'] + 1:02d} 끝{9 + g['end'] // 60}:{g['end'] % 60:02d}"
                f"{' 연속+2%' if g['extend'] else ' 연속보유X'}{' 체결+0.2%' if g['tp_buf'] else ''}{' 무작위' if g['pick'] == 'random' else ''}")

    print("\n설정 | 하루 거래 | 거래당 평균 | 승률 | 익절1%/연속2%/손절/시간 | 하루 평균 | 하루 중앙 | 손실일 | −3%↓일 | +5%↑일 | 2일 평균 | 2일 손실 | 2일 +10%↑")
    for r in results:
        e = r["exit_mix"]
        print(f"{lab(r['cfg'])} | {r['trades_per_day']:.1f} | {r['trade_mean']:+.3%} | {r['trade_win']:.0%} | "
              f"{e['tp']:.0%}/{e['tp2']:.0%}/{e['sl']:.0%}/{e['time']:.0%} | {r['day_mean']:+.2%} | {r['day_median']:+.2%} | {r['p_day_neg']:.0%} | "
              f"{r['p_day_le_m3']:.0%} | {r['p_day_ge_p5']:.0%} | {r['two_day_mean']:+.2%} | {r['two_day_p_neg']:.0%} | {r['two_day_p_ge_p10']:.0%}")
    out = ROOT / args.out
    out.write_text(json.dumps({"fingerprint": fp, "symbols": len(kept), "days": len(pre), "cost": cost,
                               "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장 {out} · 전체 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
