"""피처 엔지니어링 파이프라인.

백테스트와 실시간 추론이 **완전히 동일한 함수**를 호출하도록 무상태로 설계했다.
모든 피처는 t시점까지의 데이터만 사용하며(룩어헤드 없음), 가격 수준에 의존하지 않도록
비율/표준화 형태로 만든다(종목 간 비교 가능성 확보).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.indicators import (
    adx,
    atr,
    bollinger,
    donchian_position,
    linreg_slope,
    macd,
    money_flow_index,
    obv,
    realized_volatility,
    rolling_max,
    rolling_min,
    rolling_vwap,
    rsi,
    safe_div,
    sma,
    stochastic,
    zscore,
)

# 종목 단위 피처
BASE_FEATURES: List[str] = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_5", "vol_20", "vol_60", "vol_ratio",
    "rsi_14", "rsi_slope", "macd_hist_n", "macd_cross",
    "bb_pctb", "bb_bw", "atr_pct",
    "obv_slope", "vwap_dev", "turnover_z", "volume_z",
    "poc_dist", "donchian_20", "donchian_60", "adx_14",
    "stoch_k", "stoch_d", "mfi_14",
    "gap", "range_pct", "close_pos",
    "ma_dev_5", "ma_dev_20", "ma_dev_60", "ma_cross",
    "mom_60_5", "trend_20", "dist_high_60", "dist_low_60",
]
# 횡단면(종목 간 비교) 피처
CROSS_FEATURES: List[str] = ["cs_mom", "cs_rev", "cs_turnover", "cs_vol", "cs_rsi"]

# 수급 피처 — 토스 수급 데이터가 있을 때만 채워지고, 없으면 NaN으로 남는다.
# 가격·거래량만으로는 잡히지 않는 국내 시장 고유의 정보다(strategy/flow_features.py).
from strategy.flow_features import FLOW_FEATURES  # noqa: E402

# 소재(촉매) 흔적 피처 — 거래대금 폭증·급등 신선도·테마 동반 상승.
# 기존 피처는 추세와 평균회귀만 보므로 "지금 무언가가 돌고 있다"를 잡지 못한다.
# 30봉 +100%는 거의 소형주의 소재 장세이고, 그것이 이 묶음의 존재 이유다
# (strategy/catalyst_features.py).
from strategy.catalyst_features import (  # noqa: E402
    CATALYST_FEATURES,
    THEME_FEATURES,
)

FEATURE_COLUMNS: List[str] = (BASE_FEATURES + CROSS_FEATURES + FLOW_FEATURES
                              + CATALYST_FEATURES + THEME_FEATURES)

MIN_HISTORY = 80  # 피처가 안정적으로 생성되는 최소 봉 수


def _log_return(close: np.ndarray, periods: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) > periods:
        out[periods:] = np.log(np.maximum(close[periods:], 1e-9) / np.maximum(close[:-periods], 1e-9))
    return out


def build_features(df: pd.DataFrame, flow: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """단일 종목 OHLCV → 피처 DataFrame (원본 컬럼 유지).

    `flow`가 주어지면 수급 피처를 `ts` 기준으로 붙인다. 없으면 해당 컬럼을 NaN으로 채워
    스키마만 맞춘다. 수급이 있는 종목과 없는 종목을 같은 학습 행렬에 넣기 위해서다.
    """
    if df is None or len(df) < 5:
        return pd.DataFrame(columns=list(df.columns) + BASE_FEATURES if df is not None else BASE_FEATURES)

    out = df.copy().reset_index(drop=True)
    close = out["close"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    open_ = out["open"].to_numpy(dtype=float)
    volume = out["volume"].to_numpy(dtype=float)
    n = len(close)

    # --- 수익률 / 변동성 ---
    for p in (1, 3, 5, 10, 20, 60):
        out[f"ret_{p}"] = _log_return(close, p)
    out["vol_5"] = realized_volatility(close, 5)
    out["vol_20"] = realized_volatility(close, 20)
    out["vol_60"] = realized_volatility(close, 60)
    out["vol_ratio"] = safe_div(out["vol_5"].to_numpy(), out["vol_20"].to_numpy(), fill=1.0)

    # --- 모멘텀 / 오실레이터 ---
    rsi14 = rsi(close, 14)
    out["rsi_14"] = (rsi14 - 50.0) / 50.0
    rsi_prev = np.roll(rsi14, 3)
    rsi_prev[:3] = np.nan
    out["rsi_slope"] = (rsi14 - rsi_prev) / 50.0

    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    out["macd_hist_n"] = safe_div(hist, close, fill=0.0) * 100.0
    out["macd_cross"] = safe_div(macd_line - signal_line, np.abs(signal_line) + 1e-6, fill=0.0)
    out.loc[np.isnan(hist), "macd_hist_n"] = np.nan
    out.loc[np.isnan(signal_line), "macd_cross"] = np.nan

    # --- 변동성 밴드 ---
    _, _, _, pctb, bw = bollinger(close, 20, 2.0)
    out["bb_pctb"] = pctb - 0.5
    out["bb_bw"] = bw
    atr14 = atr(high, low, close, 14)
    out["atr_pct"] = safe_div(atr14, close, fill=np.nan)
    out["atr_abs"] = atr14  # 포지션 사이징용(피처로는 사용하지 않음)

    # --- 거래량 / 수급 ---
    obv_series = obv(close, volume)
    out["obv_slope"] = linreg_slope(obv_series, 20)
    vwap20 = rolling_vwap(close, volume, 20)
    out["vwap_dev"] = safe_div(close - vwap20, vwap20, fill=0.0)
    out.loc[np.isnan(vwap20), "vwap_dev"] = np.nan
    turnover = close * volume
    out["turnover_z"] = np.clip(zscore(turnover, 20), -5, 5)
    out["volume_z"] = np.clip(zscore(volume, 20), -5, 5)
    out["turnover"] = turnover

    # --- 가격구조 ---
    from utils.indicators import volume_profile_poc

    poc = volume_profile_poc(close, volume, window=60, bins=20)
    out["poc_dist"] = safe_div(close - poc, close, fill=0.0)
    out.loc[np.isnan(poc), "poc_dist"] = np.nan
    out["donchian_20"] = donchian_position(close, high, low, 20) - 0.5
    out["donchian_60"] = donchian_position(close, high, low, 60) - 0.5
    out["adx_14"] = adx(high, low, close, 14) / 100.0
    k, d = stochastic(high, low, close, 14, 3)
    out["stoch_k"] = (k - 50.0) / 50.0
    out["stoch_d"] = (d - 50.0) / 50.0
    out["mfi_14"] = (money_flow_index(high, low, close, volume, 14) - 50.0) / 50.0

    # --- 봉 내부 구조 ---
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    out["gap"] = safe_div(open_ - prev_close, prev_close, fill=0.0)
    out["range_pct"] = safe_div(high - low, close, fill=0.0)
    out["close_pos"] = safe_div(close - low, high - low, fill=0.5) - 0.5

    # --- 이동평균 편차 ---
    ma5, ma20, ma60 = sma(close, 5), sma(close, 20), sma(close, 60)
    out["ma_dev_5"] = safe_div(close - ma5, ma5, fill=np.nan)
    out["ma_dev_20"] = safe_div(close - ma20, ma20, fill=np.nan)
    out["ma_dev_60"] = safe_div(close - ma60, ma60, fill=np.nan)
    out["ma_cross"] = safe_div(ma5 - ma20, ma20, fill=np.nan)
    for col, base in (("ma_dev_5", ma5), ("ma_dev_20", ma20), ("ma_dev_60", ma60), ("ma_cross", ma20)):
        out.loc[np.isnan(base), col] = np.nan

    # --- 추세 ---
    out["mom_60_5"] = out["ret_60"] - out["ret_5"]
    out["trend_20"] = linreg_slope(close, 20) * 100.0
    hh60 = rolling_max(high, 60)
    ll60 = rolling_min(low, 60)
    out["dist_high_60"] = safe_div(close - hh60, hh60, fill=np.nan)
    out["dist_low_60"] = safe_div(close - ll60, ll60, fill=np.nan)

    # --- 소재 흔적 (가격·거래량만으로 계산, 외부 데이터 불필요) ---
    from strategy.catalyst_features import build_catalyst_features

    cat = build_catalyst_features(out)
    for col in cat.columns:
        out[col] = cat[col].to_numpy()

    # --- 수급 (토스 데이터가 있을 때만) ---
    from strategy.flow_features import merge_flow

    out = merge_flow(out, flow)

    out["symbol_len"] = n  # 디버깅 편의
    return out


def _rank_pct(series: pd.Series) -> pd.Series:
    """횡단면 백분위 랭크를 -0.5~0.5로 정규화."""
    valid = series.notna().sum()
    if valid <= 1:
        return pd.Series(0.0, index=series.index)
    return series.rank(pct=True, na_option="keep").fillna(0.5) - 0.5


def build_panel_features(panel: Dict[str, pd.DataFrame],
                         flow_panel: Optional[Dict[str, pd.DataFrame]] = None
                         ) -> Dict[str, pd.DataFrame]:
    """여러 종목의 피처를 만들고 횡단면 랭크 피처를 추가한다.

    횡단면 피처는 "같은 시점의 다른 종목 대비 상대 위치"이므로,
    시장 전체가 오르내릴 때 생기는 공통 변동을 제거해 순수 알파를 남긴다.

    `flow_panel`({종목: 수급 DataFrame})을 주면 수급 피처도 함께 만든다.
    """
    flow_panel = flow_panel or {}
    featured = {s: build_features(df, flow_panel.get(s))
                for s, df in panel.items() if df is not None and len(df) >= MIN_HISTORY}
    if not featured:
        return {}

    # ts 기준 정렬된 롱포맷을 만들어 시점별 랭크 계산
    frames = []
    for symbol, df in featured.items():
        tmp = df[["ts", "ret_20", "ret_1", "turnover", "vol_20", "rsi_14"]].copy()
        tmp["symbol"] = symbol
        frames.append(tmp)
    long = pd.concat(frames, ignore_index=True)

    grouped = long.groupby("ts", sort=False)
    long["cs_mom"] = grouped["ret_20"].transform(_rank_pct)
    long["cs_rev"] = -grouped["ret_1"].transform(_rank_pct)
    long["cs_turnover"] = grouped["turnover"].transform(_rank_pct)
    long["cs_vol"] = -grouped["vol_20"].transform(_rank_pct)
    long["cs_rsi"] = grouped["rsi_14"].transform(_rank_pct)

    for symbol, df in featured.items():
        sub = long[long["symbol"] == symbol].set_index("ts")
        for col in CROSS_FEATURES:
            df[col] = df["ts"].map(sub[col]).astype(float)
        featured[symbol] = df

    # --- 테마 동반 상승 ---
    # 소재는 한 종목이 아니라 떼로 움직인다. 섹터 13개는 테마보다 거칠어서
    # 수익률 상관으로 군집을 유도한다. 군집은 블록마다 직전 구간만 보고 다시
    # 계산하므로 인과적이다(strategy/catalyst_features.py::theme_clusters).
    from strategy.catalyst_features import add_theme_features

    featured = add_theme_features(featured)
    return featured


def neutralize_cross_section(
    featured: Dict[str, pd.DataFrame],
    columns: Optional[List[str]] = None,
    sector_of=None,
    size_col: str = "turnover",
) -> Dict[str, pd.DataFrame]:
    """피처에서 섹터·규모 노출을 제거하고 잔차로 치환한다.

    `cs_*` 랭크 피처는 "같은 시점 다른 종목 대비 순위"를 보지만, 순위만으로는 공통요인
    노출이 남는다. 예를 들어 반도체가 전부 오른 날 반도체 종목의 모멘텀 피처는 다 같이
    높아지는데, 이건 종목 고유의 정보가 아니라 섹터 베팅이다.

    각 시점에서 피처를 `[상수항, 섹터 더미, log(거래대금)]`에 회귀시키고 **잔차로 바꾼다.**
    남는 값은 "같은 섹터·비슷한 규모의 종목들 대비 이 종목만의 특이값"이다.

    중립화가 불가능한 시점(종목 수가 설명변수보다 적거나 행렬이 특이)은 횡단면 평균만
    빼고 넘어간다. 원본을 훼손하지 않도록 새 딕셔너리를 돌려준다.
    """
    columns = columns or [c for c in BASE_FEATURES if not c.startswith("cs_")]
    frames = []
    for symbol, df in featured.items():
        if df is None or df.empty:
            continue
        cols = [c for c in columns if c in df.columns]
        if not cols:
            continue
        sub = df[["ts"] + cols].copy()
        sub["symbol"] = symbol
        sub["_size"] = np.log1p(df[size_col].to_numpy(dtype=float)) if size_col in df.columns else 0.0
        sub["_sector"] = sector_of(symbol) if sector_of else "ETC"
        frames.append(sub)

    if not frames:
        return featured

    long = pd.concat(frames, ignore_index=True)
    use_cols = [c for c in columns if c in long.columns]
    out = {s: df.copy() for s, df in featured.items()}
    residuals: List[pd.DataFrame] = []

    for ts, group in long.groupby("ts", sort=False):
        if len(group) < 4:
            continue
        sectors = pd.get_dummies(group["_sector"], drop_first=True, dtype=float)
        size = group["_size"].to_numpy(dtype=float).reshape(-1, 1)
        size = np.nan_to_num(size, nan=0.0, posinf=0.0, neginf=0.0)
        design = np.hstack([np.ones((len(group), 1)), sectors.to_numpy(dtype=float), size])
        if design.shape[0] <= design.shape[1]:
            design = np.hstack([np.ones((len(group), 1)), size])   # 설명변수를 줄인다

        values = group[use_cols].to_numpy(dtype=float)
        mask = np.isfinite(values)
        filled = np.where(mask, values, 0.0)
        try:
            coef, *_ = np.linalg.lstsq(design, filled, rcond=None)
            resid = filled - design @ coef
        except np.linalg.LinAlgError:
            resid = filled - filled.mean(axis=0, keepdims=True)
        resid = np.where(mask, resid, np.nan)
        block = pd.DataFrame(resid, columns=use_cols)
        block["ts"] = ts
        block["symbol"] = group["symbol"].to_numpy()
        residuals.append(block)

    if not residuals:
        return out

    # 시점별 루프 안에서 .loc로 한 행씩 쓰면 30,000회 인덱싱이 된다.
    # 잔차를 모아 종목별로 한 번에 매핑한다.
    all_resid = pd.concat(residuals, ignore_index=True)
    for symbol, block in all_resid.groupby("symbol", sort=False):
        target = out.get(symbol)
        if target is None:
            continue
        lookup = block.set_index("ts")[use_cols]
        aligned = lookup.reindex(target["ts"].to_numpy())
        for col in use_cols:
            target[col] = aligned[col].to_numpy()

    return out


def latest_feature_row(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Optional[np.ndarray]:
    """가장 최근 봉의 피처 벡터. 결측이 과도하면 None."""
    columns = columns or FEATURE_COLUMNS
    if df is None or df.empty:
        return None
    available = [c for c in columns if c in df.columns]
    if len(available) < len(columns) * 0.7:
        return None
    row = df.iloc[-1][available].to_numpy(dtype=float)
    if np.isnan(row).mean() > 0.3:
        return None
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def stack_training_matrix(
    featured: Dict[str, pd.DataFrame],
    label_col: str = "label",
    columns: Optional[List[str]] = None,
    weight_col: str = "sample_w",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """학습용 (X, y, 시점인덱스, 컬럼명, 표본가중치) 생성. 시간순으로 정렬한다.

    가중치 컬럼이 없으면 전부 1.0으로 채운다(가중치를 쓰지 않는 경로와 호환).
    """
    columns = columns or FEATURE_COLUMNS
    frames = []
    for symbol, df in featured.items():
        if label_col not in df.columns:
            continue
        cols = [c for c in columns if c in df.columns]
        keep = ["ts"] + cols + [label_col]
        if weight_col in df.columns:
            keep.append(weight_col)
        sub = df[keep].copy()
        if weight_col not in sub.columns:
            sub[weight_col] = 1.0
        sub["symbol"] = symbol
        frames.append(sub)
    if not frames:
        return (np.empty((0, len(columns))), np.empty(0), np.empty(0), columns, np.empty(0))

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=[label_col])
    used_cols = [c for c in columns if c in data.columns]
    data[used_cols] = data[used_cols].replace([np.inf, -np.inf], np.nan)

    # 통째로 비어 있는 피처군(예: 수급 데이터가 없는 경우)은 결측 임계 계산에서 제외한다.
    # 그러지 않으면 "60개 중 80% 이상 채워질 것"이라는 조건에 걸려 전 행이 버려진다.
    populated = [c for c in used_cols if data[c].notna().any()]
    if populated:
        data = data.dropna(thresh=int(len(populated) * 0.8), subset=populated)
    data[used_cols] = data[used_cols].fillna(0.0)
    data = data.sort_values("ts").reset_index(drop=True)

    X = data[used_cols].to_numpy(dtype=np.float32)
    y = data[label_col].to_numpy(dtype=np.int8)
    t = pd.factorize(data["ts"], sort=True)[0].astype(np.int64)
    w = data[weight_col].to_numpy(dtype=np.float64)
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    if w.sum() > 0:
        w = w * (len(w) / w.sum())          # 평균 1로 재정규화
    else:
        w = np.ones(len(y), dtype=np.float64)
    return X, y, t, used_cols, w
