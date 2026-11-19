"""시간대 효과·장중 모멘텀 에이전트 — 마지막 하루 매매 후보(11/19). 결정 함수는 ctx 메서드로만 데이터를 본다.

선택 규칙(종목·매수 분)과 매도 규칙(dict)을 분리했다. 후보 3개는 맨 아래 candidate_1~3.
연구 실행: python scripts/agents/day_time.py   (개발 구간만, 시험 설정 수를 센다)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.day_eval import Ctx, evaluate  # noqa: E402

_T20: Dict[str, np.ndarray] = {}
_OC5: Dict[str, np.ndarray] = {}


# ---------- 공통 피처(전부 ctx 경유) ----------

def turnover20(ctx: Ctx) -> np.ndarray:
    """전일까지 20일 평균 거래대금(일봉)."""
    if ctx.day not in _T20:
        out = np.full(len(ctx.syms), np.nan)
        for j in range(len(ctx.syms)):
            d = ctx.daily(j, 20)
            if len(d) >= 10:
                out[j] = float((d["close"] * d["volume"]).mean())
        _T20[ctx.day] = out
    return _T20[ctx.day]


def oc5(ctx: Ctx) -> np.ndarray:
    """지난 5일 같은 시간대(시가→종가) 수익률 평균 — Heston 외(2010)의 일봉 근사."""
    if ctx.day not in _OC5:
        out = np.full(len(ctx.syms), np.nan)
        for j in range(len(ctx.syms)):
            d = ctx.daily(j, 5)
            if len(d) == 5:
                out[j] = float((d["close"] / d["open"] - 1).mean())
        _OC5[ctx.day] = out
    return _OC5[ctx.day]


def snap(b: np.ndarray, pc: np.ndarray) -> Dict[str, np.ndarray]:
    """bars(m) → m 직전까지의 상태(실거래에서 1분마다 찍을 수 있는 값만)."""
    o, h, l, c, v = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    with np.errstate(all="ignore"):
        last = np.full(b.shape[0], np.nan)
        for j in range(b.shape[0]):
            k = np.flatnonzero(np.isfinite(c[j]))
            if len(k):
                last[j] = c[j, k[-1]]
        k0 = np.argmax(np.isfinite(o), axis=1)
        op = o[np.arange(b.shape[0]), k0]
        tp = (h + l + c) / 3
        tv = np.nansum(np.where(np.isfinite(tp) & np.isfinite(v), tp * v, 0), axis=1)
        vol = np.nansum(np.where(np.isfinite(v), v, 0), axis=1)
        vwap = np.where(vol > 0, tv / np.maximum(vol, 1), np.nan)
        return {"last": last, "open": op, "hi": np.nanmax(h, axis=1), "lo": np.nanmin(l, axis=1),
                "tv": tv, "vwap": vwap, "r_pc": last / pc - 1, "r_op": last / op - 1}


def _argmax(score: np.ndarray, mask: np.ndarray) -> Optional[int]:
    s = np.where(mask & np.isfinite(score), score, -np.inf)
    j = int(np.argmax(s))
    return j if np.isfinite(s[j]) else None


# ---------- 선택 규칙: ctx → (j, m) 또는 None ----------

def sel_gao_last(ctx, a=330, need_mkt=True):
    """Gao: 시장 첫 30분(전일 종가→09:30) > 0인 날, a분(14:30)에 직전 30분 상승 1위."""
    pc = ctx.prev_close
    s30 = snap(ctx.bars(30), pc)
    m30 = np.nanmean(s30["r_pc"])
    if need_mkt and not (m30 > 0):
        return None
    b = ctx.bars(a)
    sa, sb = snap(b, pc), snap(b[:, :, : a - 30], pc)
    t20 = turnover20(ctx)
    j = _argmax(sa["last"] / sb["last"] - 1, np.isfinite(t20))
    return None if j is None else (j, a)


def sel_heston_oc(ctx, big=False, mkt_m=0, need_open_up=False):
    """Heston: 지난 5일 시가→종가 평균 1위를 시가(또는 mkt_m분)에 산다. mkt_m>0이면 그때까지 시장이 시가 위인 날만."""
    sc = oc5(ctx)
    t20 = turnover20(ctx)
    mask = np.isfinite(t20)
    if big:
        mask &= t20 >= np.nanmedian(t20)
    m = 0
    if mkt_m > 0:
        s = snap(ctx.bars(mkt_m), ctx.prev_close)
        if not (np.nanmean(s["r_op"]) > 0):
            return None
        m = mkt_m
        if need_open_up:
            mask &= s["r_op"] > 0
    j = _argmax(sc, mask)
    return None if j is None else (j, m)


def sel_orb(ctx, R=30, mkt=True, last_m=240, step=5):
    """개장 범위 돌파: R분 범위 고가를 넘고 VWAP 위인 종목 중 상대 거래대금(당일 누적/20일 평균) 1위.
    첫 돌파 체크포인트에 산다. mkt=True면 그 시점 시장(동일가중 시가 대비) > 0일 때만. 손절가 = 범위 저가."""
    pc = ctx.prev_close
    t20 = turnover20(ctx)
    br = ctx.bars(R)
    orh, orl = np.nanmax(br[:, 1], axis=1), np.nanmin(br[:, 2], axis=1)
    for m in range(R + step, last_m + 1, step):
        s = snap(ctx.bars(m), pc)
        ok = np.isfinite(t20) & (s["last"] > orh) & (s["last"] > s["vwap"])
        if not ok.any():
            continue
        if mkt and not (np.nanmean(s["r_op"]) > 0):
            continue
        j = _argmax(s["tv"] / t20, ok)
        if j is None:
            continue
        sl = float(np.clip(1 - orl[j] / s["last"][j], 0.005, 0.05))
        return j, m, sl
    return None


# ---------- 연구 ----------

_CACHE: Dict[Tuple, Dict[str, Any]] = {}
N_CFG = [0]


def run(sel_name, sel, rule, label, use_orb_sl=False):
    key = sel_name
    if key not in _CACHE:
        _CACHE[key] = {}
    cache = _CACHE[key]

    def decide(ctx):
        if ctx.day not in cache:
            cache[ctx.day] = sel(ctx)
        out = cache[ctx.day]
        if out is None:
            return None
        r = dict(rule)
        if use_orb_sl:
            r["sl"] = out[2]
        return out[0], out[1], r

    N_CFG[0] += 1
    return evaluate(decide, f"{sel_name} {label}")


EXITS = {"15:10": {}, "tp1": {"tp": .01}, "tp2": {"tp": .02}, "tp3": {"tp": .03}, "tp5": {"tp": .05},
         "sl2": {"sl": .02}, "tp2sl2": {"tp": .02, "sl": .02}, "tp1sl3": {"tp": .01, "sl": .03},
         "tp3sl3": {"tp": .03, "sl": .03}, "tp5sl3": {"tp": .05, "sl": .03}}


# ---------- 최종 후보 ----------

def candidate_1(ctx):
    """ORB-15 + 시장 강세: 09:15 범위 고가 돌파·VWAP 위 종목 중 상대 거래대금(당일 누적/20일 평균) 1위를
    5분 체크포인트(09:20~13:00)의 첫 돌파 때 산다. 그 시점 시장(동일가중 시가 대비) > 0인 날만.
    손절 = 범위 저가(0.5~5%로 자름), 익절 +1%(순수익), 나머지 15:10."""
    out = sel_orb(ctx, R=15, mkt=True)
    if out is None:
        return None
    j, m, sl = out
    return j, m, {"tp": 0.01, "sl": sl}


def candidate_2(ctx):
    """ORB-30 + 시장 강세(같은 규칙, 범위 30분): 손실 꼬리(−3%↓)가 가장 작다. 익절 +1%."""
    out = sel_orb(ctx, R=30, mkt=True)
    if out is None:
        return None
    j, m, sl = out
    return j, m, {"tp": 0.01, "sl": sl}


def candidate_3(ctx):
    """Heston 같은 시간대 지속성(일봉 근사): 09:30 시장(동일가중 시가 대비) > 0인 날만, 시가 위 종목 중
    지난 5일 시가→종가 평균 1위를 09:30에 산다. 익절 +2%, 나머지 15:10. 평균이 양수인 쪽."""
    out = sel_heston_oc(ctx, mkt_m=30, need_open_up=True)
    if out is None:
        return None
    return out[0], out[1], {"tp": 0.02}


if __name__ == "__main__":
    sels = {
        "GAO14:30": lambda c: sel_gao_last(c, 330, True),
        "HES시가": lambda c: sel_heston_oc(c),
        "HES09:30·시장↑": lambda c: sel_heston_oc(c, mkt_m=30),
        "HES09:30·시장↑·종목↑": lambda c: sel_heston_oc(c, mkt_m=30, need_open_up=True),
    }
    for nm, f in sels.items():
        for en, r in EXITS.items():
            run(nm, f, r, en)
    for R in (5, 15, 30):
        for mkt in (False, True):
            nm = f"ORB{R}{'·시장↑' if mkt else ''}"
            f = (lambda R_, mk: (lambda c: sel_orb(c, R=R_, mkt=mk)))(R, mkt)
            run(nm, f, {}, "범위저가손절", use_orb_sl=True)
            for en in ("tp1", "tp2", "tp3", "tp5"):
                run(nm, f, EXITS[en], f"범위저가손절+{en}", use_orb_sl=True)
    print("시험 설정 수(평가기 단계):", N_CFG[0])
