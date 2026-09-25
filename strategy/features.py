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

FEATURE_COLUMNS: List[str] = BASE_FEATURES + CROSS_FEATURES

MIN_HISTORY = 80  # 피처가 안정적으로 생성되는 최소 봉 수


def _log_return(close: np.ndarray, periods: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) > periods:
        out[periods:] = np.log(np.maximum(close[periods:], 1e-9) / np.maximum(close[:-periods], 1e-9))
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """단일 종목 OHLCV → 피처 DataFrame (원본 컬럼 유지)."""
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

    out["symbol_len"] = n  # 디버깅 편의
    return out


def _rank_pct(series: pd.Series) -> pd.Series:
    """횡단면 백분위 랭크를 -0.5~0.5로 정규화."""
    valid = series.notna().sum()
    if valid <= 1:
        return pd.Series(0.0, index=series.index)
    return series.rank(pct=True, na_option="keep").fillna(0.5) - 0.5


def build_panel_features(panel: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """여러 종목의 피처를 만들고 횡단면 랭크 피처를 추가한다.

    횡단면 피처는 "같은 시점의 다른 종목 대비 상대 위치"이므로,
    시장 전체가 오르내릴 때 생기는 공통 변동을 제거해 순수 알파를 남긴다.
    """
    featured = {s: build_features(df) for s, df in panel.items() if df is not None and len(df) >= MIN_HISTORY}
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
    return featured


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """학습용 (X, y, 시점인덱스, 컬럼명) 생성. 시간순으로 정렬한다."""
    columns = columns or FEATURE_COLUMNS
    frames = []
    for symbol, df in featured.items():
        if label_col not in df.columns:
            continue
        cols = [c for c in columns if c in df.columns]
        sub = df[["ts"] + cols + [label_col]].copy()
        sub["symbol"] = symbol
        frames.append(sub)
    if not frames:
        return np.empty((0, len(columns))), np.empty(0), np.empty(0), columns

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=[label_col])
    used_cols = [c for c in columns if c in data.columns]
    data[used_cols] = data[used_cols].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(thresh=int(len(used_cols) * 0.8), subset=used_cols)
    data[used_cols] = data[used_cols].fillna(0.0)
    data = data.sort_values("ts").reset_index(drop=True)

    X = data[used_cols].to_numpy(dtype=np.float32)
    y = data[label_col].to_numpy(dtype=np.int8)
    t = pd.factorize(data["ts"], sort=True)[0].astype(np.int64)
    return X, y, t, used_cols
