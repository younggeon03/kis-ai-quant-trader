"""소재(촉매) 흔적 피처 — 급등의 원인이 아니라 급등의 발자국을 잡는다.

## 왜 이 모듈이 필요한가

30거래일에 +100%를 만드는 종목은 거의 소형주이고, 차트의 추세·평균회귀가 아니라
**특정 소재**(수주·임상·정책·테마)로 오른다. 그런데 기존 피처 60종은 추세와
평균회귀를 보도록 설계돼 있다. 중요도 상위도 `atr_pct`·`ma_dev_60`·`obv_slope`로
전부 그쪽이다. **폭증을 탐지하는 피처가 사실상 없다**(`turnover_z`는 z-score라
5배와 20배를 구분하지 못한다).

실측이 이 격차를 보여준다. 30봉 창마다 '그 창의 최고 종목'을 100% 들었다면:

| 유니버스 | 최고 종목 중앙 | p90 | +100% 종목이 존재한 창 |
|---|---|---|---|
| 기존 40종목(대형주) | +39.3% | +105.7% | 6/49 (12%) |
| 확장 284종목 | **+124.4%** | +270.7% | **34/49 (69%)** |

재료는 확장 유니버스에 충분하다. 병목은 **284개 중 어느 것인지 고르는 일**이다.

## 무엇을 할 수 있고 무엇을 할 수 없는가

**소재가 "무엇"인지는 이 모듈로 알 수 없다.** 원전 수주인지 임상 2상인지는 뉴스나
공시 없이는 모른다(DART OpenAPI로 가능하지만 284종목 × 4.5년 수집 파이프라인이
필요하고, 대회까지 남은 시간에는 다른 작업을 모두 밀어낸다).

**소재가 "있다"는 것은 시장 흔적으로 잡힌다.** 소재가 붙은 종목은 예외 없이
거래대금 폭증·갭·변동폭 확대·동반 상승을 남긴다. 30일 모멘텀을 타는 전략에는
탐지만으로 충분하다 — 무엇인지 몰라도 지금 무언가가 돌고 있다는 것만 알면 된다.

## 두 종류의 피처

**종목 단위**는 폭증의 크기와 신선도를 본다. 비율을 **중앙값 대비**로 잡는 것이
핵심이다 — 평균은 폭증 당일이 평균을 끌어올려 자기 자신을 희석한다.

**테마 단위**는 동반 상승을 본다. 소재는 한 종목이 아니라 **떼로** 움직이므로,
같은 테마의 종목 몇 개가 함께 급등하면 그것이 개별 노이즈가 아니라는 증거다.
테마는 `SECTOR_MAP`의 13개 섹터로는 너무 거칠어서(FINANCE 8 · GAME 6 · IT 5 …)
**수익률 상관으로 데이터에서 유도한다**(`theme_clusters()`).

## 룩어헤드 차단

`utils/indicators.py`의 규칙을 따른다 — 모든 값은 시점 t까지의 정보만 쓴다.
테마 군집도 마찬가지다. 전구간 상관으로 한 번 군집하면 미래 정보가 섞이므로,
`step`봉마다 **직전 `lookback`봉만 보고** 다시 군집하고 그 배정을 다음 블록에
적용한다. `scripts/selftest.py`의 인과성 검증이 이 성질을 확인한다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 폭증 피처의 클립 상한. `strategy/ai_model.py`의 catalyst 진입이 점수를 이 값으로
# 정규화하므로 **두 곳이 갈리면 순위가 조용히 틀어진다.** 상수로 공유한다.
BURST_CLIP_MAX = 6.0

# 종목 단위 소재 흔적 피처
CATALYST_FEATURES: List[str] = [
    "turnover_burst_20",    # 거래대금 / 20봉 중앙값 (log)
    "turnover_burst_60",    # 거래대금 / 60봉 중앙값 (log)
    "burst_peak_5",         # 최근 5봉 중 최대 폭증 — 소재는 며칠 전에 터졌을 수 있다
    "burst_days_20",        # 최근 20봉 중 폭증(3배 이상) 일수
    "surge_days_20",        # 최근 20봉 중 급등(+10% 이상) 일수
    "bars_since_surge",     # 마지막 급등 이후 경과 봉 수 (신선도, 60에서 절단)
    "up_streak",            # 연속 상승 봉 수
    "range_expansion",      # 당일 변동폭 / 20봉 중앙 변동폭
    "gap_burst",            # 갭상승률 × 폭증 (소재는 장 시작부터 뛴다)
    "high_60_break",        # 60봉 신고가 돌파 여부
]

# 테마 단위 동반 상승 피처
THEME_FEATURES: List[str] = [
    "theme_breadth",        # 같은 테마에서 최근 5봉 +10% 이상인 종목 비율
    "theme_ret_5",          # 같은 테마 종목(자신 제외) 5봉 수익률 평균
    "theme_burst",          # 같은 테마 종목(자신 제외) 폭증 평균
    "theme_size",           # 테마 구성 종목 수 (작은 테마는 노이즈)
]

ALL_CATALYST_FEATURES: List[str] = CATALYST_FEATURES + THEME_FEATURES

_BURST_MULT = 3.0        # 폭증 판정 — 중앙값의 3배
_SURGE_RET = 0.10        # 급등 판정 — 일간 +10%
_STALE_CAP = 60          # 신선도 절단


def _rolling_median(x: pd.Series, window: int) -> pd.Series:
    """중앙값을 쓴다. 평균은 폭증 당일이 기준선을 끌어올려 자신을 희석한다."""
    return x.rolling(window, min_periods=max(3, window // 4)).median()


def build_catalyst_features(df: pd.DataFrame) -> pd.DataFrame:
    """종목 하나의 소재 흔적 피처. `df`는 `ts, open, high, low, close, volume` 스키마.

    반환은 입력과 같은 길이이며 `CATALYST_FEATURES` 컬럼을 담는다. 모든 값이
    인과적이다 — 시점 t의 값은 t까지의 봉만 쓴다.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    high, low = df["high"].astype(float), df["low"].astype(float)
    open_ = df["open"].astype(float)

    turnover = close * volume
    ret_1 = close.pct_change()

    # --- 폭증: 중앙값 대비 배수를 log로 눌러 꼬리를 쓸 수 있게 만든다 ---
    # 20배와 5배의 차이가 선형에서는 과하게 벌어져 트리가 극단값만 보게 된다.
    med20 = _rolling_median(turnover, 20)
    med60 = _rolling_median(turnover, 60)
    burst20 = np.log1p(turnover / med20.replace(0.0, np.nan))
    burst60 = np.log1p(turnover / med60.replace(0.0, np.nan))
    out["turnover_burst_20"] = np.clip(burst20, 0.0, BURST_CLIP_MAX)
    out["turnover_burst_60"] = np.clip(burst60, 0.0, BURST_CLIP_MAX)

    # 소재는 오늘 터졌을 수도, 사흘 전에 터졌을 수도 있다. 진입 시점에 이미
    # 폭증이 지나간 경우를 놓치지 않으려면 최근 구간의 최대값이 필요하다.
    out["burst_peak_5"] = out["turnover_burst_20"].rolling(5, min_periods=1).max()

    ratio20 = turnover / med20.replace(0.0, np.nan)
    out["burst_days_20"] = (ratio20 >= _BURST_MULT).rolling(20, min_periods=1).sum()

    # --- 급등 빈도와 신선도 ---
    surge = (ret_1 >= _SURGE_RET)
    out["surge_days_20"] = surge.rolling(20, min_periods=1).sum()

    # 마지막 급등 이후 경과 봉. 급등이 없었던 구간은 상한값으로 둔다.
    idx = np.arange(len(df), dtype=float)
    last = pd.Series(np.where(surge.to_numpy(), idx, np.nan),
                     index=df.index).ffill()
    out["bars_since_surge"] = np.clip(idx - last.to_numpy(), 0, _STALE_CAP)
    out["bars_since_surge"] = out["bars_since_surge"].fillna(_STALE_CAP)

    # --- 연속 상승 ---
    up = (ret_1 > 0).astype(int)
    grp = (up == 0).cumsum()
    out["up_streak"] = up.groupby(grp).cumsum().clip(upper=20)

    # --- 변동폭 확대 ---
    rng = (high - low) / close.replace(0.0, np.nan)
    out["range_expansion"] = np.clip(
        rng / _rolling_median(rng, 20).replace(0.0, np.nan), 0.0, 10.0)

    # --- 갭 × 폭증 ---
    # 소재가 붙으면 장 시작부터 뛴다. 갭만으로는 배당락 같은 것과 섞이므로
    # 폭증과 곱해 "거래가 몰린 갭"만 남긴다.
    gap = (open_ - close.shift(1)) / close.shift(1).replace(0.0, np.nan)
    out["gap_burst"] = np.clip(gap, -0.3, 0.3) * out["turnover_burst_20"]

    # --- 신고가 돌파 ---
    roll_max = close.rolling(60, min_periods=20).max()
    out["high_60_break"] = (close >= roll_max).astype(float)

    return out.replace([np.inf, -np.inf], np.nan)


def theme_clusters(panel: Dict[str, pd.DataFrame],
                   lookback: int = 250,
                   step: int = 60,
                   horizon: int = 5,
                   per_cluster: int = 10,
                   min_names: int = 10) -> Tuple[List[int], List[Dict[str, int]]]:
    """수익률 상관으로 테마를 유도한다. **직전 `lookback`봉만 보고 군집한다.**

    `SECTOR_MAP`의 13개 섹터는 테마보다 거칠다. 같은 IT 안에서도 반도체 소재와
    2차전지 장비는 다른 소재로 움직인다. 그래서 분류를 미리 정하지 않고 **종목들이
    실제로 함께 움직였는지**로 묶는다.

    반환은 `(블록 시작 봉 인덱스 목록, 블록별 {종목: 군집ID})`다. 봉 t의 테마는
    `t`보다 작거나 같은 마지막 블록의 배정을 쓴다 — 그 배정은 블록 시작 이전
    데이터만 보고 만들었으므로 미래를 보지 않는다.

    ## 왜 블록으로 끊는가

    매 봉 284×284 상관을 다시 계산하면 비용이 과하고, 테마는 하루 단위로 바뀌지
    않는다. `step`봉마다 갱신하면 비용은 1/step이고 테마 수명(수 주~수 개월)보다
    촘촘하다.

    ## 군집 방법 (한 번 고친 자리)

    처음에는 **일간** 수익률 상관에 거리 문턱 `1 - 0.5`를 썼다. 실측에서 거의 전부
    단독 군집이 되어 `theme_size`가 전원 2종목, 결측률 76.7%였다. 한국 주식의 일간
    수익률 상관은 같은 테마 안에서도 대체로 0.2~0.4이고, 일간 수익률에는 테마와
    무관한 잡음이 크다.

    그래서 두 가지를 바꿨다.

    - **`horizon`봉 누적 수익률로 상관을 잡는다.** 테마는 하루가 아니라 며칠 단위로
      함께 움직인다. 잡음이 평균되어 같은 테마의 상관이 드러난다.
    - **거리 문턱 대신 군집 수를 정한다**(`maxclust`, 종목수 ÷ `per_cluster`).
      문턱 방식은 상관 수준이 시기마다 달라져(장세에 따라 전체 상관이 오른다)
      어떤 구간에서는 전부 한 덩어리, 어떤 구간에서는 전부 단독이 됐다. 군집 수를
      고정하면 테마 개수는 못 맞히지만 **군집 크기가 항상 쓸 만한 범위**에 있다.

    그렇게 고치자 반대 방향으로 무너졌다 — 57종목 중 **48개가 한 군집**이 됐다.
    원인은 **시장 요인**이다. 모든 종목이 지수와 함께 움직이므로 원시 수익률 상관은
    대부분 "같이 시장을 탄다"를 재고, 평균연결 군집은 그 공통 성분 때문에 하나의
    거대 군집과 단독들로 갈라진다.

    - **횡단면 평균을 빼서 시장 요인을 제거한다.** 봉마다 전 종목 평균 수익률을
      차감하면 남는 것이 상대 움직임이고, 그 잔차의 상관이 테마다.
    - **ward 연결을 쓴다.** 평균연결보다 군집 크기가 고르게 나온다.
    """
    symbols = sorted(panel)
    wide = pd.DataFrame({s: panel[s].set_index("ts")["close"].astype(float)
                         for s in symbols}).sort_index()
    rets = wide.pct_change()
    # 시장 요인 제거 — 봉별 횡단면 평균을 차감한다. 이 한 줄이 "거대 군집 1개 +
    # 단독 다수"를 테마 군집으로 바꾼다(위 주석 참고).
    rets = rets.sub(rets.mean(axis=1), axis=0)
    n = len(rets)
    if n == 0:
        return [], []

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        has_scipy = True
    except ImportError:  # pragma: no cover - scipy는 설치돼 있다
        has_scipy = False

    starts: List[int] = []
    assigns: List[Dict[str, int]] = []
    for t in range(lookback, n, step):
        window = rets.iloc[t - lookback:t]          # t는 포함하지 않는다
        valid = window.columns[window.notna().sum() >= lookback // 2]
        if len(valid) < min_names:
            continue
        # `horizon`봉 누적 수익률. 겹치는 창을 쓰므로 상관이 다소 부풀려지지만,
        # 여기서는 절대 수준이 아니라 **상대 순위**만 쓰므로 군집에는 무해하다.
        agg = window[valid].rolling(horizon, min_periods=horizon).sum().dropna(how="all")
        corr = agg.corr().to_numpy(float)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        k = max(2, min(len(valid) // 2, round(len(valid) / per_cluster)))

        if not has_scipy:
            # 폴백: 상관이 문턱을 넘는 종목을 탐욕적으로 묶는다.
            labels = _greedy_clusters(corr, 0.35)
        else:
            dist = np.clip(1.0 - corr, 0.0, 2.0)
            np.fill_diagonal(dist, 0.0)
            dist = (dist + dist.T) / 2.0
            z = linkage(squareform(dist, checks=False), method="ward")
            labels = fcluster(z, t=k, criterion="maxclust")

        starts.append(t)
        assigns.append({s: int(c) for s, c in zip(valid, labels)})
    return starts, assigns


def _greedy_clusters(corr: np.ndarray, cut: float) -> np.ndarray:
    """scipy가 없을 때의 폴백. 상관이 `cut`을 넘는 종목을 순서대로 묶는다."""
    n = corr.shape[0]
    labels = np.zeros(n, dtype=int)
    nxt = 1
    for i in range(n):
        if labels[i]:
            continue
        labels[i] = nxt
        for j in range(i + 1, n):
            if not labels[j] and corr[i, j] >= cut:
                labels[j] = nxt
        nxt += 1
    return labels


def add_theme_features(featured: Dict[str, pd.DataFrame],
                       panel: Optional[Dict[str, pd.DataFrame]] = None,
                       lookback: int = 250,
                       step: int = 60) -> Dict[str, pd.DataFrame]:
    """`featured`에 테마 동반 상승 피처를 채운다.

    `featured[s]`는 `ts`와 `ret_5`, `turnover_burst_20`을 이미 담고 있어야 한다
    (`build_features` + `build_catalyst_features` 이후). 테마 배정이 없는 구간은
    NaN으로 남긴다 — 0으로 채우면 "동반 상승이 없다"는 거짓 정보가 된다.
    """
    src = panel or featured
    if len(src) < 2:
        for df in featured.values():
            for col in THEME_FEATURES:
                df[col] = np.nan
        return featured

    starts, assigns = theme_clusters(src, lookback=lookback, step=step)
    if not starts:
        for df in featured.values():
            for col in THEME_FEATURES:
                df[col] = np.nan
        return featured

    # ts 기준 와이드 행렬. 테마 집계는 같은 시점의 다른 종목을 봐야 한다.
    symbols = sorted(featured)
    ret5 = pd.DataFrame({s: featured[s].set_index("ts")["ret_5"] for s in symbols})
    burst = pd.DataFrame({s: featured[s].set_index("ts")["turnover_burst_20"]
                          for s in symbols})
    ts_index = ret5.index
    # 블록 시작 봉 인덱스를 ts로 바꾼다. 군집은 `src`의 ts축에서 계산됐다.
    src_ts = sorted({t for df in src.values() for t in df["ts"]})
    block_ts = [src_ts[i] for i in starts if i < len(src_ts)]

    cols = {c: pd.DataFrame(np.nan, index=ts_index, columns=symbols)
            for c in THEME_FEATURES}
    for bi, bts in enumerate(block_ts):
        end = block_ts[bi + 1] if bi + 1 < len(block_ts) else None
        mask = (ts_index >= bts) if end is None else ((ts_index >= bts) & (ts_index < end))
        if not mask.any():
            continue
        assign = assigns[bi]
        groups: Dict[int, List[str]] = {}
        for s, c in assign.items():
            if s in featured:
                groups.setdefault(c, []).append(s)
        for members in groups.values():
            if len(members) < 2:
                continue
            r = ret5.loc[mask, members]
            b = burst.loc[mask, members]
            cnt = len(members)
            # 자신을 제외한 평균 = (합 − 자신) / (n − 1)
            r_sum, b_sum = r.sum(axis=1), b.sum(axis=1)
            for s in members:
                cols["theme_ret_5"].loc[mask, s] = (r_sum - r[s].fillna(0.0)) / (cnt - 1)
                cols["theme_burst"].loc[mask, s] = (b_sum - b[s].fillna(0.0)) / (cnt - 1)
                cols["theme_size"].loc[mask, s] = float(cnt)
            # 동반 상승 폭: 테마 안에서 최근 5봉 +10% 이상인 종목 비율
            breadth = (r >= _SURGE_RET).sum(axis=1) / cnt
            for s in members:
                cols["theme_breadth"].loc[mask, s] = breadth

    for s in symbols:
        df = featured[s]
        for c in THEME_FEATURES:
            df[c] = df["ts"].map(cols[c][s]).astype(float)
        featured[s] = df
    return featured
