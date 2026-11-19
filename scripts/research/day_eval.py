"""하루 매매 연구 공통 평가기(11/19) — 내일(11/20) 하루 안에 사고파는 전략을 같은 기준으로 잰다.

전략 = `decide(ctx)` 함수. 그날 사지 않으면 None, 사면 (종목 인덱스, 매수 분, 매도 규칙)을 돌려준다.
  ctx.i, ctx.day          날짜 인덱스·YYYYMMDD
  ctx.bars(m)             그날 0..m-1 분의 1분봉 [S, 5(시·고·저·종·거래량), m] — **m분 시가에 사려면 m 전까지만 본다**
  ctx.prev_close          [S] 전일 종가
  ctx.daily(j, n=60)      종목 j 의 그날 **전일까지** 일봉 n개(DataFrame ts,open,high,low,close,volume)
  ctx.syms, ctx.ind, ctx.kosdaq  종목코드·KRX 업종·코스닥 여부 [S]
  매도 규칙 dict: tp(순수익 익절, 예 0.03) · sl(가격 손절, 예 0.02) · t_exit(매도 분, 기본 370 = 15:10, 최대 370)
                 · trail=(발동 이익, 고점 대비 하락) · exit_fn(o,h,l,c,v,entry,m) → 매도가 또는 None(분마다, 사후 정보 금지)
체결: 매수 = 매수 분 시가(없으면 다음 3분 안 첫 시가). 익절은 분봉 고가가 닿으면 그 가격(시가가 위면 시가), 손절은 저가가 닿으면
그 가격(시가가 아래면 시가), 같은 분이면 손절 먼저. 마지막은 t_exit 분 시가. 왕복 비용 0.278%.

**봉인:** load()는 2026-07-01 전 날짜만 준다. 그 뒤(최종 시험 구간)는 리더만 `_holdout_days()`로 한 번 본다.
실거래 제약: 장중 정보는 1분마다 찍을 수 있는 현재가·누적 거래량·누적 거래대금(=분봉에서 계산 가능한 값)과 일봉 ≤90개.

    from scripts.research.day_eval import load, evaluate
    res = evaluate(my_decide, "이름")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports" / "agent_data"
HOLDOUT_START = "20260701"
COST = 0.00278
END = 370
K = 5

_ARR = None
_META = None
_DAILY: Dict[str, pd.DataFrame] = {}


def _load_all():
    global _ARR, _META
    if _ARR is None:
        _ARR = np.load(DATA / "min1.npy", mmap_mode="r")
        m = np.load(DATA / "min1_meta.npz", allow_pickle=False)
        _META = {k: m[k] for k in m.files}
    return _ARR, _META


class Ctx:
    def __init__(self, i: int):
        arr, meta = _load_all()
        self.i = i
        self.day = str(meta["days"][i])
        self.syms = meta["syms"]
        self.ind = meta["ind"]
        self.kosdaq = meta["kosdaq"]
        self.prev_close = meta["prev_close"][i]
        self._arr = arr

    def bars(self, m: int) -> np.ndarray:
        m = int(max(0, min(m, 391)))
        return np.array(self._arr[self.i, :, :, :m])

    def daily(self, j: int, n: int = 60) -> pd.DataFrame:
        sym = str(self.syms[j])
        if sym not in _DAILY:
            _DAILY[sym] = pd.read_csv(ROOT / "data" / "cache" / f"{sym}.csv", dtype={"ts": str}).sort_values("ts")
        d = _DAILY[sym]
        k = int(np.searchsorted(d["ts"].str[:8].to_numpy(), self.day))
        return d.iloc[max(0, k - n):k].reset_index(drop=True)


def dev_days() -> List[int]:
    _, meta = _load_all()
    return [i for i, d in enumerate(meta["days"]) if str(d) < HOLDOUT_START]


def _holdout_days() -> List[int]:      # 리더 전용 — 에이전트 사용 금지
    _, meta = _load_all()
    return [i for i, d in enumerate(meta["days"]) if str(d) >= HOLDOUT_START]


def simulate(i: int, j: int, m_in: int, rule: Dict[str, Any]) -> Optional[float]:
    arr, _ = _load_all()
    p = np.array(arr[i, j])                      # [5, 391]
    t_exit = int(min(rule.get("t_exit", END), END))
    entry, k0 = np.nan, None
    for k in range(m_in, min(m_in + 3, t_exit)):
        if np.isfinite(p[0, k]) and p[0, k] > 0:
            entry, k0 = float(p[0, k]), k
            break
    if k0 is None:
        return None
    tp, sl, trail, efn = rule.get("tp"), rule.get("sl"), rule.get("trail"), rule.get("exit_fn")
    tpx = entry * (1 + tp) / (1 - COST) if tp is not None else np.inf
    slx = entry * (1 - sl) if sl is not None else -np.inf
    peak = entry
    for k in range(k0, t_exit):
        o, h, lo, c, v = (float(x) for x in p[:, k])
        if k == k0:
            o = entry
        if not np.isfinite(h):
            continue
        if trail is not None:
            act, dd = trail
            if peak >= entry * (1 + act) and lo <= peak * (1 - dd):
                px = o if np.isfinite(o) and o <= peak * (1 - dd) else peak * (1 - dd)
                return px / entry * (1 - COST) - 1
        if np.isfinite(o) and o <= slx:
            return o / entry * (1 - COST) - 1
        if lo <= slx:
            return slx / entry * (1 - COST) - 1
        if np.isfinite(o) and o >= tpx:
            return o / entry * (1 - COST) - 1
        if h >= tpx:
            return tpx / entry * (1 - COST) - 1
        if efn is not None:
            px = efn(o, h, lo, c, v, entry, k)
            if px is not None:
                return float(px) / entry * (1 - COST) - 1
        peak = max(peak, h)
    px = float(p[0, t_exit]) if np.isfinite(p[0, t_exit]) else float(p[3, t_exit - 1])
    if not np.isfinite(px):
        last = np.flatnonzero(np.isfinite(p[3, :t_exit]))
        if not len(last):
            return None
        px = float(p[3, last[-1]])
    return px / entry * (1 - COST) - 1


def stats(trades: List[float], n_days: int, folds: Optional[List[List[float]]] = None) -> Dict[str, Any]:
    a = np.array(trades, dtype=float)
    out = {"days": n_days, "trades": int(len(a)), "trade_rate": len(a) / max(1, n_days)}
    if len(a):
        out.update(win=float((a > 0).mean()), loss=float((a < 0).mean()), p5=float((a >= 0.05).mean()),
                   mean=float(a.mean()), median=float(np.median(a)), p_le_m3=float((a <= -0.03).mean()),
                   q05=float(np.quantile(a, 0.05)))
    if folds is not None:
        out["fold_win"] = [float((np.array(f) > 0).mean()) if f else float("nan") for f in folds]
        out["fold_n"] = [len(f) for f in folds]
    return out


def evaluate(decide: Callable[[Ctx], Any], name: str = "", days: Optional[List[int]] = None,
             verbose: bool = True) -> Dict[str, Any]:
    days = dev_days() if days is None else days
    edges = np.linspace(0, len(days), K + 1).astype(int)
    trades: List[float] = []
    folds: List[List[float]] = [[] for _ in range(K)]
    for n, i in enumerate(days):
        out = decide(Ctx(i))
        if out is None:
            continue
        j, m_in, rule = out
        r = simulate(i, int(j), int(m_in), rule or {})
        if r is None:
            continue
        trades.append(r)
        f = int(np.searchsorted(edges, n, side="right") - 1)
        folds[min(f, K - 1)].append(r)
    st = stats(trades, len(days), folds)
    st["name"] = name
    if verbose:
        if st["trades"]:
            print(f"{name:<44} 매매 {st['trades']}/{st['days']}일 · 승률 {st['win']:.0%} · 손실 {st['loss']:.0%} · +5%↑ {st['p5']:.0%} · "
                  f"평균 {st['mean']:+.2%} · 중앙 {st['median']:+.2%} · −3%↓ {st['p_le_m3']:.0%} · 조각 승률 "
                  + " ".join(f"{x:.0%}" for x in st["fold_win"]))
        else:
            print(f"{name:<44} 매매 0건")
    return st


# 기준선 — 엔진(09:04 판정 1위 ATR10)을 사서 15:10 매도. rows.pkl 의 판정 통과·ATR10 을 그대로 쓴다
_ROWS = None


def baseline_engine(ctx: Ctx):
    global _ROWS
    if _ROWS is None:
        _ROWS = pd.read_pickle(DATA / "rows.pkl")[["day", "sym", "engine_pass", "atr10", "ratio"]]
    g = _ROWS[(_ROWS["day"] == ctx.day) & _ROWS["engine_pass"]]
    if g.empty:
        return None
    s = g.sort_values(["atr10", "ratio"], ascending=False)["sym"].iloc[0]
    idx = np.flatnonzero(ctx.syms == s)
    return None if not len(idx) else (int(idx[0]), 4, {})


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    evaluate(baseline_engine, "기준: 엔진 09:04 ATR10 1위 → 15:10")
    evaluate(lambda c: (lambda b: (int(b), 4, {}))(np.nanargmax(np.where(np.isfinite(c.prev_close), 0, -1))) if False else None,
             "기준: 매매 안 함")
