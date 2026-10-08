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
from strategy.exit_rules import DecayExitTracker  # noqa: E402
from utils.indicators import atr as atr_fn  # noqa: E402
from utils.logger import enable_utf8_console, logger  # noqa: E402

COMP_BARS = 30            # 대회 거래일
LIMIT_UP = 0.295          # 상한가 판정 — 가격제한폭 30%(utils/market_rules.PRICE_LIMIT_PCT)
                          # 에서 호가단위 반올림만큼 여유를 둔다


def load_flow_net(flow_dir: Path, sym: str, ts: np.ndarray) -> np.ndarray:
    """일봉 인덱스에 맞춘 **외국인+기관 순매수**. 없으면 NaN 배열.

    **1거래일 지연시킨다.** 수급은 T일 장마감 후에 공시되므로 T일 종가로 판단할
    때는 쓸 수 없다(`strategy/flow_features.py::merge_flow`가 같은 이유로
    `FLOW_LAG_DAYS=1`을 적용한다). 지연 없이 붙이면 미래 정보다.
    """
    out = np.full(len(ts), np.nan)
    path = flow_dir / f"{sym}.csv"
    if not path.exists():
        return out
    try:
        fl = pd.read_csv(path, dtype={"ts": str})
    except Exception:
        return out
    cols = [c for c in fl.columns
            if c in ("inv_foreigner_netBuyVolume", "inv_institution_netBuyVolume")]
    if not cols or "ts" not in fl.columns:
        return out
    net = fl[cols].astype(float).sum(axis=1)
    m = dict(zip(fl["ts"].astype(str), net))
    vals = np.array([m.get(t, np.nan) for t in ts], dtype=float)
    out[1:] = vals[:-1]          # 1거래일 지연
    return out

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


class _ExitCfg:
    """`DecayExitTracker`가 읽는 설정만 흉내낸다.

    `Settings` 전체를 복제하지 않는 이유: 추적기는 `getattr`로 필요한 필드만
    읽으므로, 흉내 객체로 두면 **스윕할 값이 무엇인지 코드에 드러난다.**
    """

    def __init__(self, rules: str, crash: float, lookback: int,
                 trail_activate: float, trail_mult: float, vol_decay: float,
                 vol_min_bars: int) -> None:
        self.EXIT_RULES = rules
        self.CRASH_DROP_PCT = crash
        self.CRASH_LOOKBACK = lookback
        self.TRAILING_ACTIVATE_PCT = trail_activate
        self.TRAILING_ATR_MULT = trail_mult
        self.EXIT_VOL_DECAY_RATIO = vol_decay
        self.EXIT_VOL_DECAY_MIN_BARS = vol_min_bars


def hold_by_state(entry_px: float, i: int, max_hold: int, stop_pct: float,
                  o: np.ndarray, c: np.ndarray, burst: np.ndarray,
                  flow_net: np.ndarray, limit_up: np.ndarray,
                  w: tuple[float, float, float], thresh: float, vol_n: float,
                  cost: float) -> tuple[float, int, str]:
    """**기간이 아니라 상태가 청산을 정한다.** (수익률, 보유일수, 사유).

    "얼마나 들고 있다가 팔자"를 미리 정하지 않고, 폭증을 만든 것이 **아직 살아
    있는 동안** 들고 간다. 지금까지의 측정은 전부 6봉 시간손절 **위에 얹은**
    규칙이어서 "더 일찍 자를까"만 물었고, 그래서 전부 꼬리를 깎았다. 상태 기반은
    반대로 **더 오래 들고 갈 수도** 있다 — 보유 10봉이 28구간 중 유일하게 2배를
    넘긴 구간(+116.8%)을 만든 것이 그 방향의 증거다.

    생존 점수 = `w_vol x I(거래대금 >= 20일중앙값 x vol_n)`
              `+ w_flow x I(외국인+기관 순매수 > 0)`
              `+ w_limit x I(상한가)`

    점수가 `thresh` 아래로 떨어지면 청산한다. 가중치와 임계값으로 조합이 정해진다
    — (1,1,1)/1 은 OR, (1,1,1)/3 은 AND, (1,1,1)/2 는 "셋 중 둘"이다.

    판정은 **봉 t 종가**, 체결은 **봉 t+1 시가**다(백테스트 규약).
    `max_hold`는 무한 보유를 막는 안전장치이고 전략의 일부가 아니다.
    """
    wv, wf, wl = w
    for t in range(i, i + max_hold):
        px = float(c[t])
        if px <= 0 or not np.isfinite(px):
            continue
        reason = ""
        if stop_pct > 0 and px / entry_px - 1.0 <= -stop_pct:
            reason = "stop"
        else:
            live = 0.0
            if wv > 0 and vol_n > 0 and np.isfinite(burst[t]) and burst[t] >= vol_n:
                live += wv
            if wf > 0 and np.isfinite(flow_net[t]) and flow_net[t] > 0:
                live += wf
            if wl > 0 and limit_up[t]:
                live += wl
            if live < thresh:
                reason = "state"
        if reason:
            nxt = float(o[t + 1]) if (t + 1) < len(o) else px
            if not (np.isfinite(nxt) and nxt > 0):
                nxt = px
            return nxt / entry_px - 1.0 - cost, t - i + 1, reason
    return (float(c[i + max_hold]) / entry_px - 1.0 - cost, max_hold, "cap")


def hold_with_exits(entry_px: float, i: int, hold: int, cfg: _ExitCfg,
                    stop_pct: float, o: np.ndarray, c: np.ndarray,
                    turnover: np.ndarray, atrs: np.ndarray,
                    cost: float) -> tuple[float, str]:
    """보유 기간에 **실제 청산 규칙**을 적용한다. (수익률, 사유).

    `strategy/exit_rules.py`의 `DecayExitTracker`를 그대로 쓴다 — 재구현하면
    검증이 아니라 다른 코드를 측정하는 것이 된다.

    판정은 **봉 t 종가**로 하고 체결은 **봉 t+1 시가**다(백테스트 규약과 같다).
    아무 규칙도 걸리지 않으면 `close[i+hold]`에서 시간손절로 나간다.

    손절은 추적기가 아니라 여기서 본다 — `risk_manager`가 손절을 먼저 보고
    감쇠를 나중에 보기 때문이다(우선순위를 같게 유지한다).
    """
    tr = DecayExitTracker(cfg)
    tr.open("X", entry_px)
    for t in range(i, i + hold):
        px = float(c[t])
        if px <= 0 or not np.isfinite(px):
            continue
        gain = px / entry_px - 1.0
        reason = ""
        if stop_pct > 0 and gain <= -stop_pct:
            reason = "stop"
        else:
            hit = tr.check("X", px, turnover=float(turnover[t]),
                           atr=float(atrs[t]) if np.isfinite(atrs[t]) else 0.0,
                           entry=entry_px)
            if hit is not None:
                reason = hit.rule
        if reason:
            nxt = float(o[t + 1]) if (t + 1) < len(o) else px
            if not (np.isfinite(nxt) and nxt > 0):
                nxt = px
            return nxt / entry_px - 1.0 - cost, reason
    return float(c[i + hold]) / entry_px - 1.0 - cost, "time"


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="장중 진입 + 다일 보유 측정")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--minute-dir", default="data/cache/minute")
    ap.add_argument("--burst-mult", type=float, default=10.0)
    ap.add_argument("--hold", type=int, default=6, help="보유 일수(일봉 기준)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--since", default="20230101",
                    help="이 날짜 이후의 폭증일만 쓴다. **커버리지 편향을 막는 장치다** — "
                         "토스 분봉은 종목별 보유 시작이 문서상 한계(2022-01-03)보다 "
                         "늦어, 2021년 0% · 2022년 월별 38~65% · 2023년 이후 94.8%다. "
                         "구간을 넓히면 분봉 있는 날만 골라 보게 된다")
    ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--per-day", type=int, default=2, help="동시 보유 종목 수")
    # --- 보유 중 청산 규칙 (기본 OFF — 끄면 종료일까지 들고 간다) ---
    #
    # 기본을 OFF로 둔 것은 **이전 결과를 재현 가능하게** 두기 위해서다.
    # 켜면 `strategy/exit_rules.py`의 실제 규칙이 보유 기간에 적용된다.
    ap.add_argument("--exits", action="store_true",
                    help="보유 중 청산 규칙을 적용한다(기본은 종료일까지 보유)")
    ap.add_argument("--exit-rules", default="crash,trailing,vol_decay",
                    help="감쇠 규칙 순서. 빈 문자열이면 감쇠 없음(손절·시간만)")
    ap.add_argument("--crash", type=float, default=0.15, help="급락 임계값(0=끔)")
    ap.add_argument("--crash-lookback", type=int, default=10)
    ap.add_argument("--trail-activate", type=float, default=0.05,
                    help="트레일링 무장 수익률(0=끔)")
    ap.add_argument("--trail-mult", type=float, default=2.2, help="트레일링 ATR 배수")
    ap.add_argument("--vol-decay", type=float, default=0.30,
                    help="거래대금이 보유 중 고점의 이 비율 아래면 청산(0=끔)")
    ap.add_argument("--vol-min-bars", type=int, default=2)
    ap.add_argument("--stop", type=float, default=0.12, help="손절(0=끔)")
    # --- 상태 기반 보유 (기간을 정해두지 않는다) ---
    ap.add_argument("--state", action="store_true",
                    help="보유를 기간이 아니라 **상태**가 정한다. 켜면 --hold 는 "
                         "안전 상한(--max-hold)으로만 쓰인다")
    ap.add_argument("--hold-vol", type=float, default=2.0,
                    help="거래대금이 20일 중앙값의 이 배수 이상이면 '살아 있음'")
    ap.add_argument("--weights", default="1,1,1",
                    help="생존 점수 가중치 w_vol,w_flow,w_limit")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="점수가 이 값 아래면 청산. (1,1,1)에서 1=OR · 2=셋 중 둘 · 3=AND")
    ap.add_argument("--max-hold", type=int, default=30,
                    help="무한 보유를 막는 안전 상한(전략의 일부가 아니다)")
    ap.add_argument("--flow-dir", default="data/cache/flow")
    ap.add_argument("--out", default="reports/intraday_entry_study.json")
    args = ap.parse_args()

    s = get_settings()
    cost = s.round_trip_cost
    panel = load_panel_from_dir(args.source, list(s.UNIVERSE))
    if not panel:
        logger.error(f"[EntryStudy] {args.source} 에 일봉이 없습니다")
        return 1
    mdir = Path(args.minute_dir)
    cfg = _ExitCfg(args.exit_rules, args.crash, args.crash_lookback,
                   args.trail_activate, args.trail_mult, args.vol_decay,
                   args.vol_min_bars)
    reasons: Dict[str, int] = {}
    holds: List[int] = []
    wts = tuple(float(x) for x in args.weights.split(","))
    if len(wts) != 3:
        logger.error("[EntryStudy] --weights 는 w_vol,w_flow,w_limit 세 개다")
        return 1
    fdir = Path(args.flow_dir)
    # **집단을 모드와 무관하게 고정한다.** 길이 필터를 보유 방식에 따라 바꾸면
    # 종목·사건 집단이 달라져 지문이 바뀌고, 시간 보유와 상태 보유를 비교할 수
    # 없게 된다(실제로 그렇게 지문이 258 → 257종목으로 갈렸다).
    span = max(args.hold, args.max_hold)

    rows: List[Dict[str, Any]] = []
    n_burst = n_cover = n_detect = 0
    fp_src: Dict[str, np.ndarray] = {}

    for k, (sym, df) in enumerate(sorted(panel.items()), 1):
        if df is None or len(df) < args.warmup + span + 5:
            continue
        d = df.reset_index(drop=True)
        fp_src[sym] = d["close"].to_numpy(float)
        idx, thresh = burst_days(d, args.burst_mult, args.warmup)
        idx = idx[(idx + span) < len(d)]
        if args.since:
            days_arr = d["ts"].astype(str).to_numpy()
            idx = idx[days_arr[idx] >= args.since]
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
        tov = c * d["volume"].to_numpy(float)
        atrs = (atr_fn(d["high"].to_numpy(float), d["low"].to_numpy(float), c)
                if args.exits else np.zeros(len(d)))
        if args.state:
            # 생존 판정 입력. 분모는 **전일까지**의 20봉 중앙값으로 진입 조건과
            # 같은 기준을 쓴다 — `vol_decay` 가 "보유 중 고점"을 분모로 써서
            # 발동률 80.9%가 된 실패를 반복하지 않는다.
            med = (pd.Series(tov).rolling(20, min_periods=10).median()
                   .shift(1).to_numpy(float))
            with np.errstate(invalid="ignore", divide="ignore"):
                burst_lv = tov / np.where(med > 0, med, np.nan)
            prev_c = np.concatenate([[np.nan], c[:-1]])
            limit_up = np.nan_to_num(c >= prev_c * (1.0 + LIMIT_UP), nan=0.0).astype(bool)
            flow_net = load_flow_net(fdir, sym, ts)
        else:
            burst_lv = limit_up = flow_net = np.zeros(len(d))
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
            # **`span` 은 데이터 길이 필터 전용이다.** 청산 지평에 쓰면 `--hold` 가
            # 무시된다(실제로 그렇게 6봉 대신 30봉을 들어 숫자가 16%로 폭발했다).
            exit_px = c[i + args.hold]
            if not (np.isfinite(exit_px) and exit_px > 0 and entry_px > 0):
                continue
            nxt_open = o[i + 1] if (i + 1) < len(o) else np.nan
            if args.state:
                # **세 arm 모두 같은 생존 규칙으로 들고 간다.** 비교 arm만
                # 종료일을 고정하면 "진입 시점 비교"가 아니라 "보유 방식 비교"가
                # 섞여 숫자가 뒤집힌다(실제로 T+1 이 11%로 보였다).
                r_intra, hd, why = hold_by_state(
                    entry_px, i, args.max_hold, args.stop, o, c, burst_lv,
                    flow_net, limit_up, wts, args.threshold, args.hold_vol, cost)
                reasons[why] = reasons.get(why, 0) + 1
                holds.append(hd)
                r_next = (hold_by_state(float(nxt_open), i + 1, args.max_hold - 1,
                                        args.stop, o, c, burst_lv, flow_net,
                                        limit_up, wts, args.threshold,
                                        args.hold_vol, cost)[0]
                          if np.isfinite(nxt_open) and nxt_open > 0 else np.nan)
                r_dayopen = hold_by_state(
                    day_open, i, args.max_hold, args.stop, o, c, burst_lv,
                    flow_net, limit_up, wts, args.threshold, args.hold_vol, cost)[0]
            elif args.exits:
                r_intra, why = hold_with_exits(entry_px, i, args.hold, cfg,
                                               args.stop, o, c, tov, atrs, cost)
                reasons[why] = reasons.get(why, 0) + 1
            else:
                r_intra = exit_px / entry_px - 1.0 - cost
            rows.append({
                "symbol": sym, "day": day, "detect_min": mins,
                "entry_vs_open": entry_px / max(day_open, 1e-9) - 1.0,
                "intraday": r_intra,
                "next_open": (r_next if args.state else
                              (exit_px / nxt_open - 1.0 - cost
                               if np.isfinite(nxt_open) and nxt_open > 0 else np.nan)),
                "day_open": (r_dayopen if args.state else
                             exit_px / max(day_open, 1e-9) - 1.0 - cost),
            })
        if k % 50 == 0:
            logger.info(f"[EntryStudy] {k}/{len(panel)}종목 · 사건 {len(rows):,}")

    if not rows:
        logger.error("[EntryStudy] 사건이 없습니다 — 분봉 커버리지를 확인하십시오")
        return 1

    fr = pd.DataFrame(rows)
    fp = _fingerprint(fp_src)
    if args.state and holds:
        # **보유일수가 거래마다 다르다.** 부트스트랩은 고정 라운드를 가정하므로
        # 중앙 보유일수로 라운드를 환산한다 — **근사다.** 실제로는 짧은 거래가
        # 많은 날과 긴 거래가 몰리는 날이 섞인다. 그래서 보유 분포를 같이 찍어
        # 환산이 얼마나 거친지 드러낸다.
        med_hold = int(np.median(holds))
        rounds = max(1, COMP_BARS // max(1, med_hold))
    else:
        rounds = max(1, COMP_BARS // max(1, args.hold))
    w = min(MAX_SINGLE, MAX_GROSS / max(1, args.per_day))
    rng = np.random.default_rng(16)

    out: Dict[str, Any] = {
        "data_fingerprint": {"hash": fp, "symbols": len(fp_src)},
        "config": {"burst_mult": args.burst_mult, "hold": args.hold,
                   "rounds": rounds, "per_day": args.per_day, "weight": w,
                   "trials": args.trials, "round_trip_cost": cost,
                   "warmup": args.warmup, "since": args.since,
                   "state": args.state, "hold_vol": args.hold_vol,
                   "weights": args.weights, "threshold": args.threshold,
                   "max_hold": args.max_hold,
                   "exits": args.exits, "exit_rules": args.exit_rules,
                   "stop": args.stop, "crash": args.crash,
                   "trail_activate": args.trail_activate,
                   "trail_mult": args.trail_mult, "vol_decay": args.vol_decay},
        "exit_reasons": reasons,
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
    print(f" 구간 {args.since}~ · 폭증일 {n_burst:,} → 분봉 보유 {n_cover:,} "
          f"({n_cover / max(1, n_burst):.1%}) → 감지 {n_detect:,} → 사용 {len(fr):,}")
    if n_cover < n_burst * 0.9:
        print(f" **커버리지 {n_cover / max(1, n_burst):.1%} — 90% 미만이면 "
              f"'분봉 있는 날'만 보는 편향이 남는다. --since 를 늦춰라**")
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
    if holds:
        hq = {q: float(np.percentile(holds, q)) for q in (10, 50, 90)}
        out["hold_days"] = {"mean": float(np.mean(holds)), **{str(k): v
                                                              for k, v in hq.items()}}
        print(f" 보유일수   평균 {np.mean(holds):.1f}일 · p10 {hq[10]:.0f} · "
              f"중앙 {hq[50]:.0f} · p90 {hq[90]:.0f}  → 라운드 {rounds}회로 환산(근사)")
        print(f" 생존 조건  거래대금 >= 20일중앙x{args.hold_vol:g} · 수급 순매수 · 상한가"
              f"  가중 [{args.weights}] 임계 {args.threshold:g}")
    if reasons:
        tot = sum(reasons.values())
        parts = " · ".join(f"{k} {v / tot:.1%}"
                           for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
        print(f" 청산 사유  {parts}")
        print(f" 규칙  [{args.exit_rules or '없음'}] · 손절 {args.stop:.0%} · "
              f"급락 {args.crash:.0%} · 트레일 +{args.trail_activate:.0%}/"
              f"{args.trail_mult}xATR · 거래량 {args.vol_decay:.0%}")
    print("=" * 104)
    print(" 기준선 — 일봉 catalyst 2종목x6봉x5회전 1.67% · 스윙 8종목 0.96% · 분봉 단타 0.00%")
    print(" `day_open`은 09:00에 그날의 거래대금을 아는 셈이라 **전략이 아니라 상한**이다.")
    print(" 분모가 전일까지의 20봉 중앙값이라 일봉 피처(당일 포함)보다 약간 쉽게 발동한다.")
    print(f" 저장 → {op}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
