"""분봉 급등 피처 — 돌파 순간에 "이 급등이 계속 갈 것인가"를 판단할 재료.

## 예측 단위는 종목이 아니라 **돌파 사건**이다

일봉 전략은 "오늘 어느 종목을 살까"를 묻는다. 분봉 전략은 다르다. 장중에 어떤
종목이 전일종가 대비 +10%를 돌파하는 **순간**이 하나의 사건이고, 그 순간에 가진
정보로 "앞으로 더 갈까, 밀릴까"를 맞혀야 한다.

그래서 이 모듈은 **(종목, 날짜, 돌파 시각)** 단위로 피처를 만든다. 피처는
돌파 봉까지의 분봉만 쓰고, 라벨은 돌파 **이후** 구간에서 나온다.

## 무엇을 재는가 — 되돌림이 핵심이다

실측에서 장중 +10% 돌파 중 **54.8%가 종가까지 밀린다**(284종목 4년9개월).
그 밀림을 걸러내면 30봉 `P(+100%)`가 3.27% → 4.42%로 오른다. 즉 분봉의 가치는
싸게 사는 것(일봉 대비 상대 6%)이 아니라 **밀릴 급등을 거르는 것**에 있다.

일봉 피처로는 이걸 볼 수 없다. 일봉은 하루가 끝난 뒤의 요약이라 "돌파 후
되돌림이 얼마나 깊었는지", "거래량이 돌파 직후에도 살아 있었는지", "돌파가 장
초반인지 마감 직전인지"가 전부 사라진다. 그 세 가지가 여기서 만드는 피처의 축이다.

## 체결 가정

**돌파 봉의 다음 봉 시가에 진입한다.** 돌파 가격에 즉시 체결된다고 보면 실제보다
낙관적이다(돌파 순간을 알아차리고 주문이 나가 체결되기까지 지연이 있다). 일봉
백테스트가 "봉 t 종가 판단 → 봉 t+1 시가 체결"을 쓰는 것과 같은 원칙이다.

## 룩어헤드 차단

`build_event_features()`는 돌파 봉 인덱스까지로 잘라낸 배열만 받는다. 전체 하루를
넘기고 내부에서 자르는 방식으로 만들면 실수 한 줄로 미래가 샌다. 라벨 생성은
`label_event()`로 분리해, 피처와 라벨이 같은 함수에서 같은 배열을 보지 않게 했다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SESSION_MINUTES = 391          # 09:00~15:30
_EPS = 1e-9

MINUTE_FEATURES: List[str] = [
    # --- 돌파의 성격 ---
    "trig_minute",        # 09:00부터 돌파까지 경과 분 (장 초반 돌파와 마감 돌파는 다르다)
    "trig_close_level",   # 돌파 봉 **종가**의 전일종가 대비 상승률 (돌파선 위로 얼마나 더)
    "gap_trigger",        # 돌파가 첫 5분 안에 났는가 — 분봉 이력이 거의 없는 집단
    "gap_open",           # 시가 갭 — 갭으로 시작했는가, 장중에 올렸는가
    "run_from_open",      # 시가 대비 돌파까지 상승분
    "speed_5pct",         # +5% 통과부터 돌파까지 걸린 분 (급등 속도)
    # --- 거래량 ---
    "vol_burst",          # 돌파 봉 거래량 / 당일 분당 거래량 중앙값
    "vol_accel",          # 최근 5분 거래량 / 그 앞 15분 평균
    "turnover_vs_prev",   # 돌파까지 누적 거래대금 / 전일 전체 거래대금
    "vol_trend",          # 거래량 선형 추세(최근 30분, 부호가 중요)
    # --- 가격 구조 (되돌림) ---
    "pullback_max",       # 돌파 전 고점 대비 최대 되돌림 깊이
    "pullback_last",      # 직전 10분 고점 대비 현재 되돌림
    "straight_run",       # 돌파 직전 연속 상승 분봉 수
    "vwap_dev",           # 돌파가 / 당일 VWAP − 1
    "range_pos",          # 당일 고저 범위 내 위치 (1이면 고가)
    "high_breaks",        # 당일 신고가 갱신 횟수
    "up_min_ratio",       # 돌파까지 상승 마감 분봉 비율
    "body_ratio",         # 최근 10분 몸통 합 / 범위 합 (추세 강도)
]


@dataclass(frozen=True)
class SurgeEvent:
    """돌파 사건 하나. 피처와 라벨을 잇는 식별자."""

    symbol: str
    day: str
    trig_idx: int          # 당일 분봉 배열에서 돌파 봉의 위치
    trig_ts: str           # YYYYMMDDHHMM
    trig_price: float      # 돌파 판정 가격 (전일종가 × (1+trigger))
    entry_price: float     # 실제 진입가 = 다음 봉 시가
    prev_close: float


def find_trigger(minute: pd.DataFrame, prev_close: float,
                 trigger: float = 0.03,
                 prev_volume: float = 0.0,
                 vol_mult: float = 1.5) -> Optional[int]:
    """가격과 거래량을 **동시에** 만족하는 첫 분봉의 인덱스.

    ```
    ① 고가 ≥ 전일종가 × (1 + trigger)
    ② 그 1분봉 거래량 ≥ vol_mult × (전일 거래량 ÷ 391)
    ```

    ## 왜 고가로 판정하는가

    그 분봉 안에서 가격이 돌파선을 지났다는 뜻이다. 종가로 판정하면 분봉 안에서
    찍고 되돌아온 돌파를 놓치는데, 실거래에서는 그 순간에 주문이 나간다.

    ## 왜 거래량 조건이 필요한가

    가격만 보면 **거래 없이 호가 몇 번에 뛴 것까지 발동한다.** 트리거를 +10%에서
    +3%로 낮추면 그런 사건이 급격히 늘어난다(+3% 도달일 88,275일 vs +10% 13,099일).
    거래량 조건이 그중 "실제로 거래가 붙은" 것만 남긴다.

    ## 왜 돌파 봉 거래량만 보는가 (사용자 결정)

    누적 거래량으로 판정하려면 시점 보정이 필요하다. 실측 장중 프로파일이 U자형이라
    10분 경과 시점에 이미 하루 거래량의 **9.7%**(균등 가정 2.6%의 3.8배)가 돌아간다.
    보정 없이 누적으로 판정하면 아침 돌파가 거의 전부 탈락하고 오후 돌파만 남는데,
    측정에서는 **아침 돌파(62분)가 오후 돌파(106분)보다 좋았다** — 신호를 거꾸로
    거르게 된다. 돌파 봉 하나만 보면 시점 보정이 필요 없고 "지금 거래가 붙었는가"를
    직접 본다. 하루 누적 유입은 `turnover_vs_prev` 피처가 따로 잰다.

    기준선을 전일 분당 평균(전일 거래량 ÷ 391)으로 잡는다. 전일 거래량을 모르면
    (`prev_volume <= 0`) 거래량 조건을 건너뛴다 — 상장 직후 등에서 가격 조건만으로
    판정하는 것이 전부 탈락시키는 것보다 낫다.
    """
    if minute is None or minute.empty or not np.isfinite(prev_close) or prev_close <= 0:
        return None
    level = prev_close * (1.0 + trigger)
    ok = minute["high"].to_numpy(float) >= level
    if prev_volume and prev_volume > 0 and vol_mult > 0:
        floor = vol_mult * (prev_volume / SESSION_MINUTES)
        ok &= minute["volume"].to_numpy(float) >= floor
    hits = np.flatnonzero(ok)
    if not len(hits):
        return None
    idx = int(hits[0])
    # 다음 봉 시가에 진입하므로 마지막 봉에서의 돌파는 쓸 수 없다.
    return idx if idx < len(minute) - 1 else None


def _linreg_slope(y: np.ndarray) -> float:
    """정규화 기울기. 스케일을 없애 종목 간 비교가 되게 한다."""
    n = len(y)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    denom = float(((x - x.mean()) ** 2).sum())
    if denom <= _EPS:
        return 0.0
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / denom)
    scale = float(np.mean(np.abs(y))) + _EPS
    return float(np.clip(slope / scale, -5.0, 5.0))


def build_event_features(o: np.ndarray, h: np.ndarray, l: np.ndarray,
                         c: np.ndarray, v: np.ndarray,
                         prev_close: float, prev_turnover: float,
                         trigger: float = 0.10) -> Dict[str, float]:
    """돌파 봉까지의 분봉 배열로 피처를 만든다.

    **배열은 이미 돌파 봉까지 잘려 있어야 한다**(`arr[:trig_idx + 1]`). 전체 하루를
    받아 내부에서 자르면 실수 한 줄로 미래가 샌다. 호출부가 자르고 넘긴다.
    """
    n = len(c)
    out: Dict[str, float] = {k: np.nan for k in MINUTE_FEATURES}
    # 돌파가 09:00 첫 봉에 나는 경우가 실제로 많다(갭 상승 시작). 그때는 분봉
    # 이력이 1개뿐이라 대부분의 피처가 정의되지 않는다. **버리지 않는다** —
    # 실거래에서도 그 사건은 후보로 올라오므로, 계산 가능한 것만 채우고
    # `gap_trigger`로 "이 집단은 다르다"를 모델에 알린다.
    if n < 1 or prev_close <= 0:
        return out

    trig_price = prev_close * (1.0 + trigger)
    sess_open = float(o[0])
    turnover = c * v
    run_high = np.maximum.accumulate(h)

    # --- 돌파의 성격 ---
    out["trig_minute"] = float(n - 1)                       # 09:00이 0
    # 돌파선(= 전일종가 × 1.10)은 상수라 피처가 될 수 없다. 돌파 **봉의 종가**가
    # 돌파선 위로 얼마나 더 갔는지를 쓴다 — 그 분봉 안에서의 기세다.
    out["trig_close_level"] = float(c[-1] / prev_close - 1.0)
    out["gap_trigger"] = 1.0 if (n - 1) < 5 else 0.0
    out["gap_open"] = float(sess_open / prev_close - 1.0)
    out["run_from_open"] = float(c[-1] / max(sess_open, _EPS) - 1.0)

    # +5%를 처음 넘긴 시점부터 돌파까지의 분 수. 작으면 빠른 급등이다.
    half = np.flatnonzero(h >= prev_close * 1.05)
    out["speed_5pct"] = float((n - 1) - half[0]) if len(half) else float(n - 1)

    # --- 거래량 ---
    med_v = float(np.median(v[v > 0])) if np.any(v > 0) else 0.0
    out["vol_burst"] = float(np.clip(v[-1] / (med_v + _EPS), 0.0, 100.0))
    # 거래량 가속. 20분 이력이 없으면(갭 돌파) 가진 만큼으로 계산하고, 비교할
    # 앞구간이 아예 없으면 1.0(가속 없음)으로 둔다. 결측으로 남기면 트리가
    # 갭 돌파 집단 전체를 한 쪽으로 보내버린다.
    if n >= 8:
        k = min(5, n // 2)
        recent = float(v[-k:].mean())
        prior = float(v[:-k].mean())
        out["vol_accel"] = float(np.clip(recent / (prior + _EPS), 0.0, 50.0))
    else:
        out["vol_accel"] = 1.0
    out["turnover_vs_prev"] = float(np.clip(
        turnover.sum() / (prev_turnover + _EPS), 0.0, 50.0)) if prev_turnover > 0 else np.nan
    out["vol_trend"] = _linreg_slope(v[-30:].astype(float)) if n >= 5 else 0.0

    # --- 가격 구조: 되돌림이 핵심이다 ---
    # 돌파 전까지 고점에서 가장 깊게 밀린 폭. 계단식으로 오른 급등과 한 번
    # 크게 밀렸다 다시 올린 급등은 이후 전개가 다르다.
    dd = l / np.maximum(run_high, _EPS) - 1.0
    out["pullback_max"] = float(dd.min())
    tail = max(1, min(10, n))
    out["pullback_last"] = float(c[-1] / max(h[-tail:].max(), _EPS) - 1.0)

    up = c > o
    streak = 0
    for k in range(n - 1, -1, -1):
        if up[k]:
            streak += 1
        else:
            break
    out["straight_run"] = float(min(streak, 60))

    vwap = float(turnover.sum() / max(v.sum(), _EPS)) if v.sum() > 0 else float(c[-1])
    out["vwap_dev"] = float(c[-1] / max(vwap, _EPS) - 1.0)

    lo, hi = float(l.min()), float(h.max())
    out["range_pos"] = float((c[-1] - lo) / max(hi - lo, _EPS)) if hi > lo else 1.0
    # 신고가 갱신 횟수 — 계단식 상승의 강도
    out["high_breaks"] = float((h > np.concatenate([[-np.inf], run_high[:-1]])).sum())
    out["up_min_ratio"] = float(up.mean())

    body = np.abs(c[-tail:] - o[-tail:]).sum()
    rng = (h[-tail:] - l[-tail:]).sum()
    out["body_ratio"] = float(body / max(rng, _EPS))
    return out


def label_event(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                entry_price: float, target: float = 0.10,
                stop: float = 0.0) -> Dict[str, float]:
    """돌파 **이후** 구간의 결과. 배열은 진입 봉부터 장 마감까지여야 한다.

    여러 라벨을 한 번에 만든다. 어느 것으로 학습할지는 측정으로 정한다.

    - `y_target`: 당일 안에 `target` 이상 더 오르는가 (주 라벨)
    - `max_gain` / `min_gain`: 진입 후 당일 최대·최저 (청산 규칙 설계용)
    - `close_gain`: 종가까지 들고 있었을 때
    - `hit_first`: 목표와 손절 중 어느 쪽이 먼저인가 (−1 손절 / +1 목표 / 0 둘 다 없음)

    `hit_first`가 이 모듈의 존재 이유다. **일봉으로는 목표와 손절의 선후를 알 수
    없다** — 당일 저가가 돌파 전인지 후인지 구분이 안 된다. 분봉은 순서를 안다.
    """
    out = {"y_target": np.nan, "max_gain": np.nan, "min_gain": np.nan,
           "close_gain": np.nan, "hit_first": np.nan, "bars_to_target": np.nan}
    if len(c) == 0 or not np.isfinite(entry_price) or entry_price <= 0:
        return out

    up = h / entry_price - 1.0
    dn = l / entry_price - 1.0
    out["max_gain"] = float(up.max())
    out["min_gain"] = float(dn.min())
    out["close_gain"] = float(c[-1] / entry_price - 1.0)
    out["y_target"] = 1.0 if out["max_gain"] >= target else 0.0

    t_hit = np.flatnonzero(up >= target)
    out["bars_to_target"] = float(t_hit[0]) if len(t_hit) else np.nan
    if stop > 0:
        s_hit = np.flatnonzero(dn <= -stop)
        ti = t_hit[0] if len(t_hit) else np.inf
        si = s_hit[0] if len(s_hit) else np.inf
        # 같은 봉에서 양쪽을 건드리면 보수적으로 손절을 먼저 본다. 1분봉
        # 안에서도 순서는 알 수 없으므로, 유리한 쪽을 가정하면 결과가 부풀려진다.
        out["hit_first"] = 0.0 if (ti == np.inf and si == np.inf) else (
            -1.0 if si <= ti else 1.0)
    return out


def events_for_day(symbol: str, day: str, minute: pd.DataFrame,
                   prev_close: float, prev_turnover: float,
                   trigger: float = 0.03, target: float = 0.10,
                   stop: float = 0.0, prev_volume: float = 0.0,
                   vol_mult: float = 1.5) -> Optional[Dict[str, float]]:
    """하루치 분봉 → 피처 + 라벨 한 행. 돌파가 없으면 `None`.

    하루에 돌파를 **한 번만** 센다. 같은 날 여러 번 돌파선을 넘나드는 종목에서
    사건을 여러 개 만들면 서로 거의 같은 정보를 담아 표본이 중복되고, 검증
    점수가 부풀려진다(일봉 라벨 중첩에 가중치를 주는 것과 같은 이유다).
    """
    idx = find_trigger(minute, prev_close, trigger, prev_volume, vol_mult)
    if idx is None:
        return None
    o = minute["open"].to_numpy(float)
    h = minute["high"].to_numpy(float)
    l = minute["low"].to_numpy(float)
    c = minute["close"].to_numpy(float)
    v = minute["volume"].to_numpy(float)

    feats = build_event_features(o[:idx + 1], h[:idx + 1], l[:idx + 1],
                                 c[:idx + 1], v[:idx + 1],
                                 prev_close, prev_turnover, trigger)
    entry = float(o[idx + 1])
    labels = label_event(h[idx + 1:], l[idx + 1:], c[idx + 1:],
                         entry, target=target, stop=stop)

    row: Dict[str, float] = {"symbol": symbol, "day": day,
                             "trig_ts": str(minute["ts"].iloc[idx]),
                             "trig_idx": float(idx),
                             "entry_price": entry,
                             "prev_close": float(prev_close)}
    row.update(feats)
    row.update(labels)
    return row


def build_event_table(daily: Dict[str, pd.DataFrame],
                      minute_dir: str,
                      trigger: float = 0.03,
                      target: float = 0.10,
                      stop: float = 0.0,
                      min_bars: int = 200,
                      vol_mult: float = 1.5) -> pd.DataFrame:
    """분봉 디렉토리 전체 → 학습용 사건 테이블.

    `daily`에서 전일종가와 전일 거래대금을 가져온다. 분봉만으로는 "전일 대비"를
    알 수 없고, 그 비교가 폭증 판정의 기준이다.

    `min_bars` 미만인 날은 버린다 — 거래정지나 수집 실패로 반쪽만 있는 날의
    피처는 다른 날과 의미가 달라진다.
    """
    from pathlib import Path

    rows: List[Dict[str, float]] = []
    mdir = Path(minute_dir)
    for sym, ddf in daily.items():
        path = mdir / f"{sym}.csv"
        if not path.exists() or ddf is None or len(ddf) < 2:
            continue
        d = ddf.reset_index(drop=True)
        prev_c = dict(zip(d["ts"].astype(str), d["close"].astype(float).shift(1)))
        prev_t = dict(zip(d["ts"].astype(str),
                          (d["close"].astype(float) * d["volume"].astype(float)).shift(1)))
        prev_v = dict(zip(d["ts"].astype(str), d["volume"].astype(float).shift(1)))
        try:
            mdf = pd.read_csv(path, dtype={"ts": str})
        except Exception:
            continue
        if mdf.empty:
            continue
        for day, grp in mdf.groupby(mdf["ts"].str[:8], sort=True):
            if len(grp) < min_bars:
                continue
            pc, pt = prev_c.get(day), prev_t.get(day)
            if pc is None or not np.isfinite(pc) or pc <= 0:
                continue
            row = events_for_day(sym, str(day), grp.reset_index(drop=True),
                                 float(pc), float(pt or 0.0), trigger, target, stop,
                                 float(prev_v.get(day) or 0.0), vol_mult)
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["day", "symbol"]).reset_index(drop=True)
    return out
