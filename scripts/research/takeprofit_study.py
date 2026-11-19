"""조건부 장중 익절 — "어느 정도 이익이 나고 거래가 꺾이면 그날 판다"가 꼬리를 깎는가.

## 왜 이 측정이 필요한가

현재 청산(`vol_alive` x2)은 하루 한 번, 확정 일봉으로만 판다. 장중에 이익을 확정하는
규칙이 없다. 당일 일괄 정리(0.00%)와 보유 중 고점 대비 거래대금 감쇠(`vol_decay` 1.01%)는
측정에서 꼬리를 깎았지만, **이익 조건을 붙인 장중 익절**은 따로 잰 적이 없다.

## 규칙

진입은 실거래와 같다 — 장 시작 10분 같은 시간대 x3, 지연 0(09:10 시가). 보유 중 매 분:

    종가 ≥ 진입가 x (1 + tp)  AND  (decay 없음 또는 직전 W분 거래대금 < decay x 그날 W분 최대)
    → 다음 분 시가에 판다

익절이 안 걸리면 기존 규칙 그대로(봉 t 종가 `vol_alive`·손절 판정 → t+1 시가). 즉 기준선은
`opening_study.py`의 `same_window 10분 x3 지연 0`과 같은 사건·같은 청산이다.

## 데이터

장중 분봉은 `data/cache/fullday/`(하루 전체, 279종목)만 있다 — 그래서 **284 유니버스 중
fullday가 있는 종목**으로 사건을 제한하고, 기준선도 같은 사건 집합에서 다시 잰다(짝 비교).
fullday가 없는 보유일은 장중 판정을 건너뛰고 그 수를 찍는다.

```bash
python scripts/research/takeprofit_study.py
python scripts/research/takeprofit_study.py --tps 0.1,0.2 --decays none,0.5
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
from scripts.research.intraday_entry_study import _fingerprint  # noqa: E402
from scripts.research.opening_study import evaluate  # noqa: E402
from scripts.research.slot_sim import path_exit  # noqa: E402
from scripts.research.tail_optimizer import MAX_GROSS, MAX_SINGLE  # noqa: E402
from strategy.opening import minute_index, same_window_base, window_turnover  # noqa: E402
from strategy.sliding import minute_turnover, rolling_sum  # noqa: E402
from utils.logger import enable_utf8_console, logger  # noqa: E402


def tp_exit(entry_px: float, i: int, entry_mi: int, max_hold: int, stop: float,
            o: np.ndarray, c: np.ndarray, burst: np.ndarray, vol_n: float,
            ts: np.ndarray, intraday: Dict[str, pd.DataFrame], floor: Optional[float],
            decay: Optional[float], roll_w: int, peak_mode: str,
            same: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
            ) -> Tuple[int, float, str, bool, int]:
    """(청산 봉, 청산가, 'open'|'close', 장중 청산 여부, 장중 데이터 없던 보유일 수).

    봉 t마다: ① 장중 청산(분봉) → ② 종가 판정(손절·vol_alive, `path_exit`와 같다) → t+1 시가.
    장중 청산 조건(매 분, 둘 다 만족해야 한다 — None 인 쪽은 보지 않는다):
      · 가격: 종가 ≥ 진입가 x (1 + floor)
      · 꺾임: 직전 `roll_w`분 거래대금 < decay x 기준 고점
        기준 고점 `day` = 그날 roll 최대(진입일은 진입 이후만), `hold` = 진입 이후 전체 최대
        `same` = 고점이 아니라 **지난 20일 같은 시각** roll 중앙(장중 거래대금은 U자라 그날
        고점과 비교하면 점심에 항상 꺾인다 — 시간대 보정). 이때 decay 는 배수(x1 = 평소 수준)
    진입일은 진입 분부터 거래대금을 센다 — 장 시작 폭증(판정 구간)을 고점으로 쓰면 진입 직후
    바로 꺾임으로 읽힌다(`vol_decay`가 일봉에서 폭증일 자신을 고점으로 써 1.01%였던 것과 같다).
    장중 청산은 다음 분 시가 체결, 그날 안에 현금이 돌아오므로 'close'로 기록한다.
    """
    missing = 0
    target = entry_px * (1.0 + floor) if floor is not None else -np.inf
    hold_peak = -np.inf
    for t in range(i, i + max_hold):
        g = intraday.get(ts[t])
        if g is None:
            missing += 1
        else:
            mi_all = g["mi"].to_numpy()
            sel = mi_all >= entry_mi if t == i else mi_all >= 0
            if peak_mode == "same":
                rb = same.get(ts[t]) if same is not None else None
                if rb is not None:
                    R, B = rb
                    gmi = mi_all[sel]
                    gcl = g["close"].to_numpy(float)[sel]
                    gop = g["open"].to_numpy(float)[sel]
                    for j in range(len(gcl)):
                        m_ = int(gmi[j])
                        if (t == i and m_ <= entry_mi) or m_ >= len(R):
                            continue
                        if not (gcl[j] >= target):
                            continue
                        if not (np.isfinite(R[m_]) and np.isfinite(B[m_])
                                and R[m_] < decay * B[m_]):
                            continue
                        px = (float(gop[j + 1]) if j + 1 < len(gop) and gop[j + 1] > 0
                              else float(gcl[j]))
                        return t, px, "close", True, missing
                g = None  # 아래 고점 비교는 건너뛴다
        if g is not None and peak_mode != "same":
            mi_all = g["mi"].to_numpy()
            sel = mi_all >= entry_mi if t == i else mi_all >= 0
            mi = mi_all[sel]
            cl = g["close"].to_numpy(float)[sel]
            op = g["open"].to_numpy(float)[sel]
            tv = cl * g["volume"].to_numpy(float)[sel]
            roll = pd.Series(tv).rolling(roll_w, min_periods=roll_w).sum().to_numpy()
            peak = np.fmax.accumulate(np.where(np.isfinite(roll), roll, -np.inf))
            if peak_mode == "hold":
                peak = np.maximum(peak, hold_peak)
                if len(peak):
                    hold_peak = float(peak[-1])
            for j in range(len(cl)):
                if t == i and mi[j] <= entry_mi:
                    continue
                if not (cl[j] >= target):
                    continue
                if decay is not None and not (np.isfinite(roll[j]) and roll[j] < decay * peak[j]):
                    continue
                px = float(op[j + 1]) if j + 1 < len(op) and op[j + 1] > 0 else float(cl[j])
                return t, px, "close", True, missing
        px = float(c[t])
        if not (np.isfinite(px) and px > 0):
            continue
        hit = stop > 0 and px / entry_px - 1.0 <= -stop
        if not hit:
            b = burst[t]
            hit = not (np.isfinite(b) and b >= vol_n)
        if hit:
            nxt = float(o[t + 1]) if (t + 1) < len(o) else px
            if not (np.isfinite(nxt) and nxt > 0):
                nxt = px
            return t + 1, nxt, "open", False, missing
    return i + max_hold, float(c[i + max_hold]), "close", False, missing


def trail_exit(entry_px: float, i: int, entry_mi: int, max_hold: int, stop: float,
               o: np.ndarray, c: np.ndarray, burst: np.ndarray, vol_n: float,
               ts: np.ndarray, intraday: Dict[str, pd.DataFrame], act: float, giveback: float,
               press: Optional[float], roll_w: int) -> Tuple[int, float, str, bool, int]:
    """장중 고점 대비 되돌림(+ 매도 압력) 청산.

    매 분: 진입 이후 고점이 진입가 x (1+act) 이상이었고, 종가가 고점 x (1-giveback) 이하이며,
    (press 가 있으면) 직전 roll_w 분 거래대금 중 **하락봉(종가<시가) 거래대금 비중** ≥ press 이면
    다음 분 시가에 판다. 하락봉 비중은 진짜 매도 주도 체결량의 분봉 근삿값이다(틱 이력이 없다).
    아니면 종가 판정(`path_exit`와 같다) → t+1 시가.
    """
    missing = 0
    peak = entry_px
    armed_at = entry_px * (1.0 + act)
    for t in range(i, i + max_hold):
        g = intraday.get(ts[t])
        if g is None:
            missing += 1
        else:
            mi_all = g["mi"].to_numpy()
            sel = mi_all >= entry_mi if t == i else mi_all >= 0
            mi = mi_all[sel]
            hi = g["high"].to_numpy(float)[sel]
            cl = g["close"].to_numpy(float)[sel]
            op = g["open"].to_numpy(float)[sel]
            tv = cl * g["volume"].to_numpy(float)[sel]
            if press is not None:
                down = np.where(cl < op, tv, 0.0)
                rt = pd.Series(tv).rolling(roll_w, min_periods=roll_w).sum().to_numpy()
                rd = pd.Series(down).rolling(roll_w, min_periods=roll_w).sum().to_numpy()
            for j in range(len(cl)):
                if t == i and mi[j] <= entry_mi:
                    if np.isfinite(hi[j]):
                        peak = max(peak, hi[j])
                    continue
                if np.isfinite(hi[j]):
                    peak = max(peak, hi[j])
                if peak < armed_at or not (cl[j] <= peak * (1.0 - giveback)):
                    continue
                if press is not None and not (np.isfinite(rt[j]) and rt[j] > 0
                                              and rd[j] / rt[j] >= press):
                    continue
                px = float(op[j + 1]) if j + 1 < len(op) and op[j + 1] > 0 else float(cl[j])
                return t, px, "close", True, missing
        px = float(c[t])
        if not (np.isfinite(px) and px > 0):
            continue
        hit = stop > 0 and px / entry_px - 1.0 <= -stop
        if not hit:
            b = burst[t]
            hit = not (np.isfinite(b) and b >= vol_n)
        if hit:
            nxt = float(o[t + 1]) if (t + 1) < len(o) else px
            if not (np.isfinite(nxt) and nxt > 0):
                nxt = px
            return t + 1, nxt, "open", False, missing
    return i + max_hold, float(c[i + max_hold]), "close", False, missing


def vwap_exit(entry_px: float, i: int, entry_mi: int, max_hold: int, stop: float,
              o: np.ndarray, c: np.ndarray, burst: np.ndarray, vol_n: float,
              ts: np.ndarray, intraday: Dict[str, pd.DataFrame], below: Optional[float],
              vmult: Optional[float], same: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
              min_mi: int = 10, combine: str = "and") -> Tuple[int, float, str, bool, int]:
    """당일 VWAP 아래로 밀리면(+ 거래가 몰리면) 판다 — '거래대금은 큰데 매도세가 이기는' 날을 장중에 잡는다.

    VWAP = 그날 09:00 부터의 누적 거래대금 / 누적 거래량(그날 산 사람들의 평균 단가).
    매 분(09:{min_mi} 이후, 진입일은 진입 뒤): 종가 < VWAP x (1-below) 이고, vmult 가 있으면 직전 W분
    거래대금 ≥ vmult x 지난 20일 같은 시각 중앙(`same`)일 때(음수면 '|vmult| 배 미만', 즉 거래가 줄어드는
    중일 때) 다음 분 시가에 판다.
    장 시작 직후는 VWAP 가 몇 분치뿐이라 잡음이 커서 09:{min_mi} 전에는 보지 않는다.
    `combine="or"` 면 가격·거래 조건 중 **하나만** 맞아도 판다(None 인 조건은 빼고 본다).
    """
    missing = 0
    for t in range(i, i + max_hold):
        g = intraday.get(ts[t])
        if g is None:
            missing += 1
        else:
            mi = g["mi"].to_numpy()
            cl = g["close"].to_numpy(float)
            op = g["open"].to_numpy(float)
            vo = g["volume"].to_numpy(float)
            cum_v = np.cumsum(np.where(np.isfinite(vo), vo, 0.0))
            cum_tv = np.cumsum(np.where(np.isfinite(vo), cl * vo, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                vwap = cum_tv / np.where(cum_v > 0, cum_v, np.nan)
            rb = same.get(ts[t]) if (vmult is not None and same is not None) else None
            for j in range(len(cl)):
                m_ = int(mi[j])
                if m_ < min_mi or (t == i and m_ <= entry_mi):
                    continue
                conds = []
                if below is not None:
                    conds.append(bool(np.isfinite(vwap[j]) and cl[j] < vwap[j] * (1.0 - below)))
                if vmult is not None:
                    ok = False
                    if rb is not None:
                        R, B = rb
                        if m_ < len(R) and np.isfinite(R[m_]) and np.isfinite(B[m_]):
                            # 양수 = 거래 몰림(직전 W분 ≥ k x 평소), 음수 = 거래 감소(직전 W분 < |k| x 평소)
                            ok = bool(R[m_] >= vmult * B[m_]) if vmult >= 0 else bool(R[m_] < -vmult * B[m_])
                    conds.append(ok)
                if conds and not (any(conds) if combine == "or" else all(conds)):
                    continue
                px = float(op[j + 1]) if j + 1 < len(op) and op[j + 1] > 0 else float(cl[j])
                return t, px, "close", True, missing
        px = float(c[t])
        if not (np.isfinite(px) and px > 0):
            continue
        hit = stop > 0 and px / entry_px - 1.0 <= -stop
        if not hit:
            b = burst[t]
            hit = not (np.isfinite(b) and b >= vol_n)
        if hit:
            nxt = float(o[t + 1]) if (t + 1) < len(o) else px
            if not (np.isfinite(nxt) and nxt > 0):
                nxt = px
            return t + 1, nxt, "open", False, missing
    return i + max_hold, float(c[i + max_hold]), "close", False, missing


def tp_exit_daily(entry_px: float, i: int, post_entry_high: float, max_hold: int,
                  stop: float, o: np.ndarray, h: np.ndarray, c: np.ndarray,
                  burst: np.ndarray, vol_n: float, tp: float
                  ) -> Tuple[int, float, str, bool, int]:
    """일봉 근사 — 진입가 x (1+tp) **지정가 매도**로 본다(가격만, 꺾임 조건 없음).

    장중 분봉이 없는 종목(494 확대분)용. 봉 t마다: 시가가 목표 이상이면 시가 체결, 아니면
    고가가 목표에 닿으면 목표가 체결 → 아니면 종가 판정(`path_exit`와 같다).
    진입일은 09:10 이후 고가만 써야 한다 — `post_entry_high`(호출부가 계산)를 쓴다.
    """
    target = entry_px * (1.0 + tp)
    for t in range(i, i + max_hold):
        if t == i:
            if post_entry_high >= target:
                return t, target, "close", True, 0
        else:
            if np.isfinite(o[t]) and o[t] >= target:
                return t, float(o[t]), "close", True, 0
            if np.isfinite(h[t]) and h[t] >= target:
                return t, target, "close", True, 0
        px = float(c[t])
        if not (np.isfinite(px) and px > 0):
            continue
        hit = stop > 0 and px / entry_px - 1.0 <= -stop
        if not hit:
            b = burst[t]
            hit = not (np.isfinite(b) and b >= vol_n)
        if hit:
            nxt = float(o[t + 1]) if (t + 1) < len(o) else px
            if not (np.isfinite(nxt) and nxt > 0):
                nxt = px
            return t + 1, nxt, "open", False, 0
    return i + max_hold, float(c[i + max_hold]), "close", False, 0


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="조건부 장중 익절 측정")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--opening-dir", default="data/cache/opening")
    ap.add_argument("--fullday-dir", default="data/cache/fullday")
    ap.add_argument("--since", default="20250102")
    ap.add_argument("--until", default="99999999", help="이 날까지의 진입·30일 창만(기간 나눠 확인용)")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--mult", type=float, default=3.0)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--decays", default="0.5,0.3",
                    help="꺾임(메인): 직전 W분 거래대금 < 이 값 x 기준 고점이면 판다")
    ap.add_argument("--rolls", default="10,30", help="꺾임을 보는 창 W(분)")
    ap.add_argument("--peaks", default="day,hold,same",
                    help="기준: day(그날 고점) · hold(진입 이후 고점) · same(지난 20일 같은 시각 중앙)")
    ap.add_argument("--same-mults", default="1,2,3",
                    help="same 기준 배수 — 직전 W분 거래대금 < k x 평소 같은 시각이면 판다")
    ap.add_argument("--floors", default="none,0,0.1",
                    help="꺾임과 함께 볼 가격 하한(진입가 대비, none=가격 안 봄)")
    ap.add_argument("--tps", default="0.2", help="비교용 가격만 익절(꺾임 없음)")
    ap.add_argument("--hold-vol", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=30)
    ap.add_argument("--stop", type=float, default=0.25)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--trail", action="store_true",
                    help="장중 고점 대비 되돌림(+매도 압력) 격자만 잰다(다른 변형은 건너뜀)")
    ap.add_argument("--acts", default="0.1,0.2,0.3", help="되돌림 감시를 켜는 상승폭")
    ap.add_argument("--givebacks", default="0.05,0.1,0.15", help="고점 대비 하락폭")
    ap.add_argument("--presses", default="none,0.6,0.7,0.8", help="하락봉 거래대금 비중 하한")
    ap.add_argument("--press-rolls", default="10,30", help="매도 압력을 보는 창(분)")
    ap.add_argument("--vwap", action="store_true",
                    help="당일 VWAP 아래로 밀리면(+거래 몰림) 파는 격자만 잰다")
    ap.add_argument("--belows", default="none,0,0.02,0.05",
                    help="VWAP 대비 얼마나 아래(0=VWAP 아래면, none=가격 조건 없이 거래 조건만)")
    ap.add_argument("--vmults", default="none,1,2,-1,-0.5",
                    help="직전 W분 거래대금 / 평소 같은 시각 — 양수 k 는 'k배 이상(몰림)', "
                         "음수 -k 는 'k배 미만(감소)'")
    ap.add_argument("--vrolls", default="10,30", help="거래 몰림을 보는 창(분)")
    ap.add_argument("--daily-approx", action="store_true",
                    help="장중 분봉 없이 일봉 고가로 지정가 익절을 근사(494 확대분용, 꺾임 조건 없음)")
    ap.add_argument("--out", default="reports/takeprofit_study.json")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    weight = min(MAX_SINGLE, MAX_GROSS / max(1, args.slots))
    def _opt(x: str) -> Optional[float]:
        return None if x.strip() == "none" else float(x)
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    # 변형 = (가격 하한, 꺾임, 창, 고점 기준)
    variants: List[Tuple[Optional[float], Optional[float], int, str]] = [
        (tp, None, 0, "-") for tp in tps]
    if not args.daily_approx:
        peaks = args.peaks.split(",")
        variants += [(_opt(f), float(dc), int(w), pk)
                     for dc in args.decays.split(",") for w in args.rolls.split(",")
                     for pk in peaks if pk != "same" for f in args.floors.split(",")]
        if "same" in peaks:
            variants += [(_opt(f), float(k), int(w), "same")
                         for k in args.same_mults.split(",") for w in args.rolls.split(",")
                         for f in args.floors.split(",")]
    if args.trail:
        # 변형 = (상승폭 act, 되돌림 giveback, 창, "trail<압력>")
        variants = []
        for a in args.acts.split(","):
            for b in args.givebacks.split(","):
                for pr in args.presses.split(","):
                    if pr.strip() == "none":
                        variants.append((float(a), float(b), 0, "trail-"))
                    else:
                        variants += [(float(a), float(b), int(w), f"trail{pr.strip()}")
                                     for w in args.press_rolls.split(",")]
    if args.vwap:
        # 변형 = (VWAP 아래 폭, 거래 몰림 배수, 창, "vwap")
        variants = []
        for bl in args.belows.split(","):
            blv = _opt(bl)                  # none = 가격 조건 없이 거래 조건만
            for vm in args.vmults.split(","):
                if vm.strip() == "none":
                    if blv is not None:
                        variants.append((blv, None, 0, "vwap"))
                else:
                    variants += [(blv, float(vm), int(w), "vwap")
                                 for w in args.vrolls.split(",")]
    same_ws = sorted({v[2] for v in variants
                      if v[3] == "same" or (v[3] == "vwap" and v[1] is not None)})
    N, entry_mi = args.window, args.window + args.lag

    odir, fdir = Path(args.opening_dir), Path(args.fullday_dir)
    def _fd(x: str) -> Path:
        # 압축본(.csv.gz)도 읽는다 — 서버 디스크(10GB)에 하루 전체 분봉을 풀어 둘 자리가 없다
        p_ = fdir / f"{x}.csv"
        return p_ if p_.exists() else fdir / f"{x}.csv.gz"
    symbols = [x for x in s.UNIVERSE if args.daily_approx or _fd(x).exists()]
    panel = load_panel_from_dir(args.source, symbols)
    all_days = sorted({str(t) for df in panel.values() if df is not None
                       for t in df["ts"].astype(str)})
    cal = {t: k for k, t in enumerate(all_days)}

    base_events: List[Dict[str, Any]] = []
    var_events: Dict[Tuple, List[Dict[str, Any]]] = {v: [] for v in variants}
    closes: Dict[str, Dict[int, float]] = {}
    fp_src: Dict[str, np.ndarray] = {}
    tp_hits = {v: 0 for v in variants}
    missing_days = 0
    for k_, (sym, df) in enumerate(sorted(panel.items()), 1):
        if df is None or len(df) < 60 or not (odir / f"{sym}.csv").exists():
            continue
        d = df.reset_index(drop=True)
        ts = d["ts"].astype(str).to_numpy()
        o = d["open"].to_numpy(float)
        c = d["close"].to_numpy(float)
        hi = d["high"].to_numpy(float)
        tov = c * d["volume"].to_numpy(float)
        closes[sym] = {cal[t]: float(px) for t, px in zip(ts, c)}
        med_day = pd.Series(tov).rolling(20, min_periods=10).median().shift(1).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            burst = tov / np.where(med_day > 0, med_day, np.nan)
        pos = {t: i for i, t in enumerate(ts)}

        m = pd.read_csv(odir / f"{sym}.csv", dtype={"ts": str})
        m = m[(m["ts"].str[:8] >= args.since) & (m["ts"].str[:8] <= args.until)]
        if m.empty:
            continue
        m = m.assign(mi=minute_index(m["ts"]), day=m["ts"].str[:8])
        days = sorted(m["day"].unique())
        by_day = {dy: g.sort_values("mi") for dy, g in m.groupby("day")}
        wtv = pd.Series([window_turnover(by_day[dy]["mi"].to_numpy(),
                                         by_day[dy]["close"].to_numpy(float),
                                         by_day[dy]["volume"].to_numpy(float), N)
                         for dy in days], index=days)
        base_s = pd.Series([same_window_base(wtv.iloc[:j].tolist()) for j in range(len(days))],
                           index=days)

        intraday: Dict[str, pd.DataFrame] = {}
        if args.daily_approx:
            fp_src[sym] = (m["close"] * m["volume"]).to_numpy(float)
        else:
            fd = pd.read_csv(_fd(sym), dtype={"ts": str})
            fd = fd.assign(mi=minute_index(fd["ts"]), day=fd["ts"].str[:8])
            intraday = {dy: g.sort_values("mi").reset_index(drop=True)
                        for dy, g in fd.groupby("day", sort=False)}
            fp_src[sym] = (fd["close"] * fd["volume"]).to_numpy(float)
        # same 기준선 — 날짜 x 분 행렬의 직전 W분 합, 지난 20일(최소 10일) 같은 분 중앙
        same_maps: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        if same_ws and intraday:
            fdays = sorted(intraday)
            mt = [minute_turnover(intraday[dy]["mi"].to_numpy(),
                                  intraday[dy]["close"].to_numpy(float),
                                  intraday[dy]["volume"].to_numpy(float)) for dy in fdays]
            for W in same_ws:
                mat = np.vstack([rolling_sum(x, W) for x in mt])
                base = (pd.DataFrame(mat).rolling(20, min_periods=10).median()
                        .shift(1).to_numpy())
                base[~(base > 0)] = np.nan
                same_maps[W] = {dy: (mat[r], base[r]) for r, dy in enumerate(fdays)}

        for dy in days:
            i = pos.get(dy)
            if i is None or i + args.max_hold + 1 >= len(d) or i < 25:
                continue
            w, b = wtv[dy], base_s[dy]
            if not (np.isfinite(b) and b > 0 and w > 0) or w / b < args.mult:
                continue
            ent = by_day[dy].loc[by_day[dy]["mi"] == entry_mi, "open"]
            if ent.empty or not (float(ent.iloc[0]) > 0):
                continue
            entry_px = float(ent.iloc[0])
            ratio = w / b
            common = {"sym": sym, "d": cal[dy], "entry_px": entry_px,
                      "detect_min": entry_mi + 1.0 / (1.0 + ratio)}
            ex_i, ex_px, when = path_exit(entry_px, i, args.max_hold, args.stop, o, c,
                                          burst, args.hold_vol)
            base_events.append({**common, "rule": {
                "exit_d": cal[ts[min(ex_i, len(ts) - 1)]], "when": when,
                "ret": ex_px / entry_px - 1.0 - cost}})
            if args.daily_approx:
                # 진입일 고가는 09:10 이후만 — 일봉 고가가 진입 전 고가보다 높으면 그 뒤에 났다
                g0 = by_day[dy]
                gmi, ghi = g0["mi"].to_numpy(), g0["high"].to_numpy(float)
                pre = ghi[gmi < entry_mi]
                pre_hi = float(np.nanmax(pre)) if len(pre) else 0.0
                post = ghi[gmi >= entry_mi]
                post_hi = float(hi[i]) if hi[i] > pre_hi else (
                    float(np.nanmax(post)) if len(post) else 0.0)
            for v in variants:
                if args.daily_approx:
                    ex_i, ex_px, when, hit, miss = tp_exit_daily(
                        entry_px, i, post_hi, args.max_hold, args.stop, o, hi, c, burst,
                        args.hold_vol, v[0])
                elif v[3] == "vwap":
                    ex_i, ex_px, when, hit, miss = vwap_exit(
                        entry_px, i, entry_mi, args.max_hold, args.stop, o, c, burst,
                        args.hold_vol, ts, intraday, v[0], v[1], same_maps.get(v[2]))
                elif v[3].startswith("trail"):
                    pr = v[3][len("trail"):]
                    ex_i, ex_px, when, hit, miss = trail_exit(
                        entry_px, i, entry_mi, args.max_hold, args.stop, o, c, burst,
                        args.hold_vol, ts, intraday, v[0], v[1],
                        None if pr == "-" else float(pr), max(1, v[2]))
                else:
                    ex_i, ex_px, when, hit, miss = tp_exit(
                        entry_px, i, entry_mi, args.max_hold, args.stop, o, c, burst,
                        args.hold_vol, ts, intraday, v[0], v[1], max(1, v[2]), v[3],
                        same_maps.get(v[2]))
                if v == variants[0]:
                    missing_days += miss
                tp_hits[v] += int(hit)
                var_events[v].append({**common, "rule": {
                    "exit_d": cal[ts[min(ex_i, len(ts) - 1)]], "when": when,
                    "ret": ex_px / entry_px - 1.0 - cost}})
        if k_ % 50 == 0:
            logger.info(f"[TakeProfit] {k_}/{len(panel)}종목 · 사건 {len(base_events):,}")

    if not base_events:
        logger.error("[TakeProfit] 사건이 없습니다")
        return 1
    fp = _fingerprint(fp_src)
    first = cal[min(t for t in all_days if t >= args.since)]
    last = max(max(v) for v in closes.values())
    in_range = [t for t in all_days if t <= args.until]
    if in_range:
        last = min(last, cal[in_range[-1]])
    ref = evaluate(base_events, closes, cost, args.slots, weight, first, last)
    rr = np.array([e["rule"]["ret"] for e in base_events])

    rows = []
    for v in variants:
        evs = var_events[v]
        r = evaluate(evs, closes, cost, args.slots, weight, first, last)
        rets = np.array([e["rule"]["ret"] for e in evs])
        paired = r["_rets"] - ref["_rets"]
        rows.append({"floor": v[0], "decay": v[1], "roll": v[2], "peak": v[3], "tp_hit_rate": tp_hits[v] / len(evs),
                     "trade_mean": float(rets.mean()), "single_2x": float((rets >= 1.0).mean()),
                     **{kk: x for kk, x in r.items() if kk != "_rets"},
                     "beats_ref": float((paired > 0).mean()),
                     "ties_ref": float((paired == 0).mean())})

    out = {"data_fingerprint": {"hash": fp, "symbols": len(fp_src)},
           "config": vars(args) | {"weight": weight, "round_trip_cost": cost},
           "events": len(base_events), "missing_intraday_hold_days": missing_days,
           "reference": {"trade_mean": float(rr.mean()), "single_2x": float((rr >= 1.0).mean()),
                         **{kk: x for kk, x in ref.items() if kk != "_rets"}},
           "grid": rows}
    op_ = Path(args.out)
    op_.parent.mkdir(parents=True, exist_ok=True)
    op_.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 122)
    print(f" 조건부 장중 익절 — 진입 same_window {N}분 x{args.mult:g} 지연 {args.lag} · "
          f"기본 청산 vol_alive x{args.hold_vol:g} · 손절 {args.stop:.0%} · {args.slots}슬롯 x {weight:.0%}")
    print(f" 데이터 지문 {fp} ({len(fp_src)}종목) · 사건 {len(base_events):,} · "
          f"장중 데이터 없는 보유일 {missing_days:,} · 30거래일 창 {ref['windows']}개")
    print("=" * 122)
    print(f" {'가격하한':>7}{'꺾임':>6}{'창':>4}{'고점':>6}{'발동':>8}{'평균/회':>9}{'단일2배':>8}{'중앙':>8}{'p90':>8}"
          f"{'P(+50%)':>9}{'P(+100%)':>10}{'기준보다↑':>10}{'같음':>7}")
    print("-" * 122)
    print(f" {'현재':>7}{'-':>6}{'-':>4}{'-':>6}{'-':>8}{rr.mean() * 100:+8.2f}%{(rr >= 1.0).mean() * 100:7.2f}%"
          f"{ref['median'] * 100:+7.1f}%{ref['p90'] * 100:+7.1f}%{ref['p50up'] * 100:8.2f}%"
          f"{ref['p100up'] * 100:9.2f}%{'—':>10}{'—':>7}")
    for r in rows:
        dc = "-" if r["decay"] is None else f"{r['decay']:g}"
        fl = "-" if r["floor"] is None else f"{r['floor'] * 100:+.0f}%"
        w_ = "-" if not r["roll"] else str(r["roll"])
        print(f" {fl:>7}{dc:>6}{w_:>4}{r['peak']:>6}{r['tp_hit_rate'] * 100:7.1f}%"
              f"{r['trade_mean'] * 100:+8.2f}%{r['single_2x'] * 100:7.2f}%"
              f"{r['median'] * 100:+7.1f}%{r['p90'] * 100:+7.1f}%{r['p50up'] * 100:8.2f}%"
              f"{r['p100up'] * 100:9.2f}%{r['beats_ref'] * 100:9.1f}%{r['ties_ref'] * 100:6.1f}%")
    print(" 꺾임 = 직전 W분 거래대금 < 꺾임 x 고점(day=그날 최대, hold=진입 이후 최대, "
          "same=지난 20일 같은 시각 중앙의 배수) · "
          "가격하한 = 그 이상일 때만 판다")
    print("=" * 122)
    print(" 기준보다↑ = 같은 30일 창에서 현재 규칙보다 수익이 높은 창 비율(겹치는 창이라 독립 표본은 약 14개)")
    print(f" 저장 → {op_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
