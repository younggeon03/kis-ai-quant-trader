"""순수 numpy 기술적 지표 라이브러리.

TA-Lib(C 확장) / pandas-ta 의존성을 제거하기 위해 직접 구현했다. 모든 함수는
입력과 동일한 길이의 배열을 반환하며 워밍업 구간은 NaN으로 채운다.
**미래 정보를 절대 참조하지 않으므로** 백테스트와 실시간 추론에 같은 코드를 쓴다.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "sma", "ema", "wilder_ema", "rolling_std", "rolling_max", "rolling_min", "rolling_sum",
    "zscore", "rsi", "macd", "bollinger", "true_range", "atr", "obv", "rolling_vwap",
    "stochastic", "adx", "linreg_slope", "volume_profile_poc", "donchian_position",
    "realized_volatility", "money_flow_index", "safe_div",
]

_EPS = 1e-12


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def safe_div(a, b, fill: float = 0.0) -> np.ndarray:
    """0 분모를 안전하게 처리하는 나눗셈."""
    a = _arr(a)
    b = _arr(b)
    out = np.full(a.shape, fill, dtype=np.float64)
    mask = np.abs(b) > _EPS
    np.divide(a, b, out=out, where=mask)
    out[~mask] = fill
    return out


def _rolling(x: np.ndarray, window: int) -> np.ndarray:
    """(n-window+1, window) 형태의 롤링 뷰."""
    return sliding_window_view(x, window)


def _pad_front(values: np.ndarray, n: int, window: int) -> np.ndarray:
    out = np.full(n, np.nan, dtype=np.float64)
    if window - 1 < n:
        out[window - 1:] = values
    return out


def sma(x, window: int) -> np.ndarray:
    x = _arr(x)
    if window <= 0 or len(x) < window:
        return np.full(len(x), np.nan)
    return _pad_front(_rolling(x, window).mean(axis=1), len(x), window)


def rolling_sum(x, window: int) -> np.ndarray:
    x = _arr(x)
    if window <= 0 or len(x) < window:
        return np.full(len(x), np.nan)
    return _pad_front(_rolling(x, window).sum(axis=1), len(x), window)


def rolling_std(x, window: int, ddof: int = 1) -> np.ndarray:
    x = _arr(x)
    if window <= 1 or len(x) < window:
        return np.full(len(x), np.nan)
    return _pad_front(_rolling(x, window).std(axis=1, ddof=ddof), len(x), window)


def rolling_max(x, window: int) -> np.ndarray:
    x = _arr(x)
    if window <= 0 or len(x) < window:
        return np.full(len(x), np.nan)
    return _pad_front(_rolling(x, window).max(axis=1), len(x), window)


def rolling_min(x, window: int) -> np.ndarray:
    x = _arr(x)
    if window <= 0 or len(x) < window:
        return np.full(len(x), np.nan)
    return _pad_front(_rolling(x, window).min(axis=1), len(x), window)


def ema(x, span: int) -> np.ndarray:
    """표준 EMA(alpha = 2/(span+1)). 첫 span개 평균으로 시드."""
    x = _arr(x)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < span or span <= 0:
        return out
    alpha = 2.0 / (span + 1.0)
    acc = float(np.mean(x[:span]))
    out[span - 1] = acc
    for i in range(span, n):
        acc = alpha * x[i] + (1.0 - alpha) * acc
        out[i] = acc
    return out


def wilder_ema(x, period: int) -> np.ndarray:
    """Wilder 평활(alpha = 1/period). RSI/ATR/ADX의 표준 평활 방식."""
    x = _arr(x)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period <= 0:
        return out
    acc = float(np.mean(x[:period]))
    out[period - 1] = acc
    for i in range(period, n):
        acc = (acc * (period - 1) + x[i]) / period
        out[i] = acc
    return out


def zscore(x, window: int) -> np.ndarray:
    x = _arr(x)
    mu = sma(x, window)
    sd = rolling_std(x, window)
    return safe_div(x - mu, sd, fill=0.0)


def rsi(close, period: int = 14) -> np.ndarray:
    """Wilder RSI. 0~100."""
    close = _arr(close)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    delta[0] = 0.0
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_ema(gain[1:], period)
    avg_loss = wilder_ema(loss[1:], period)
    rs = safe_div(avg_gain, avg_loss, fill=np.inf)
    rsi_vals = np.where(np.isinf(rs), 100.0, 100.0 - 100.0 / (1.0 + rs))
    rsi_vals = np.where(np.isnan(avg_gain), np.nan, rsi_vals)
    out[1:] = rsi_vals
    return out


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    """(macd_line, signal_line, histogram) 반환."""
    close = _arr(close)
    macd_line = ema(close, fast) - ema(close, slow)
    valid = ~np.isnan(macd_line)
    signal_line = np.full(len(close), np.nan)
    if valid.sum() >= signal:
        sig_valid = ema(macd_line[valid], signal)
        signal_line[valid] = sig_valid
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close, window: int = 20, num_std: float = 2.0):
    """(middle, upper, lower, percent_b, bandwidth) 반환."""
    close = _arr(close)
    mid = sma(close, window)
    sd = rolling_std(close, window)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    percent_b = safe_div(close - lower, upper - lower, fill=0.5)
    percent_b = np.where(np.isnan(mid), np.nan, percent_b)
    bandwidth = safe_div(upper - lower, mid, fill=0.0)
    bandwidth = np.where(np.isnan(mid), np.nan, bandwidth)
    return mid, upper, lower, percent_b, bandwidth


def true_range(high, low, close) -> np.ndarray:
    high, low, close = _arr(high), _arr(low), _arr(close)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return tr


def atr(high, low, close, period: int = 14) -> np.ndarray:
    return wilder_ema(true_range(high, low, close), period)


def obv(close, volume) -> np.ndarray:
    """On-Balance Volume (누적 거래량 흐름)."""
    close, volume = _arr(close), _arr(volume)
    direction = np.sign(np.diff(close, prepend=close[0]))
    direction[0] = 0.0
    return np.cumsum(direction * volume)


def rolling_vwap(close, volume, window: int) -> np.ndarray:
    """롤링 VWAP. 분봉 기준 장중 평균 체결단가 근사."""
    close, volume = _arr(close), _arr(volume)
    pv = rolling_sum(close * volume, window)
    v = rolling_sum(volume, window)
    out = safe_div(pv, v, fill=np.nan)
    return np.where(np.isnan(v), np.nan, out)


def stochastic(high, low, close, k_period: int = 14, d_period: int = 3):
    """(%K, %D) 반환."""
    hh = rolling_max(_arr(high), k_period)
    ll = rolling_min(_arr(low), k_period)
    k = 100.0 * safe_div(_arr(close) - ll, hh - ll, fill=50.0)
    k = np.where(np.isnan(hh), np.nan, k)
    d = sma(k, d_period) if np.isfinite(k).sum() >= d_period else np.full(len(k), np.nan)
    return k, d


def adx(high, low, close, period: int = 14) -> np.ndarray:
    """추세 강도(ADX). 레짐 필터용."""
    high, low, close = _arr(high), _arr(low), _arr(close)
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close)
    atr_v = wilder_ema(tr, period)
    plus_di = 100.0 * safe_div(wilder_ema(plus_dm, period), atr_v, fill=0.0)
    minus_di = 100.0 * safe_div(wilder_ema(minus_dm, period), atr_v, fill=0.0)
    dx = 100.0 * safe_div(np.abs(plus_di - minus_di), plus_di + minus_di, fill=0.0)
    dx = np.where(np.isnan(atr_v), np.nan, dx)
    valid = ~np.isnan(dx)
    out = np.full(n, np.nan)
    if valid.sum() >= period:
        out[valid] = wilder_ema(dx[valid], period)
    return out


def linreg_slope(x, window: int) -> np.ndarray:
    """롤링 선형회귀 기울기(정규화). 추세 방향/강도 피처."""
    x = _arr(x)
    n = len(x)
    if window < 2 or n < window:
        return np.full(n, np.nan)
    t = np.arange(window, dtype=np.float64)
    t_centered = t - t.mean()
    denom = float((t_centered ** 2).sum())
    win = _rolling(x, window)
    slopes = (win - win.mean(axis=1, keepdims=True)) @ t_centered / denom
    scale = np.where(np.abs(win.mean(axis=1)) > _EPS, np.abs(win.mean(axis=1)), 1.0)
    return _pad_front(slopes / scale, n, window)


def volume_profile_poc(close, volume, window: int = 60, bins: int = 20) -> np.ndarray:
    """롤링 Volume Profile의 POC(Point of Control) 가격.

    최근 `window` 구간의 가격을 `bins`개로 나눠 거래량이 가장 많이 쌓인 가격대를 반환.
    지지/저항 근접도 피처로 사용한다.
    """
    close, volume = _arr(close), _arr(volume)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window or window < bins:
        return out
    for i in range(window - 1, n):
        seg_p = close[i - window + 1: i + 1]
        seg_v = volume[i - window + 1: i + 1]
        lo, hi = float(seg_p.min()), float(seg_p.max())
        if hi - lo < _EPS:
            out[i] = lo
            continue
        edges = np.linspace(lo, hi, bins + 1)
        idx = np.clip(np.digitize(seg_p, edges) - 1, 0, bins - 1)
        weights = np.bincount(idx, weights=seg_v, minlength=bins)
        peak = int(np.argmax(weights))
        out[i] = 0.5 * (edges[peak] + edges[peak + 1])
    return out


def donchian_position(close, high, low, window: int = 20) -> np.ndarray:
    """돈치안 채널 내 위치(0=하단, 1=상단). 브레이크아웃 근접도."""
    hh = rolling_max(_arr(high), window)
    ll = rolling_min(_arr(low), window)
    pos = safe_div(_arr(close) - ll, hh - ll, fill=0.5)
    return np.where(np.isnan(hh), np.nan, pos)


def realized_volatility(close, window: int, annualize: int = 252) -> np.ndarray:
    """로그수익률 기반 실현변동성(연율화)."""
    close = _arr(close)
    logret = np.zeros_like(close)
    logret[1:] = np.log(np.maximum(close[1:], _EPS) / np.maximum(close[:-1], _EPS))
    return rolling_std(logret, window) * np.sqrt(annualize)


def money_flow_index(high, low, close, volume, period: int = 14) -> np.ndarray:
    """MFI(거래량 가중 RSI). 0~100."""
    high, low, close, volume = _arr(high), _arr(low), _arr(close), _arr(volume)
    tp = (high + low + close) / 3.0
    raw_flow = tp * volume
    delta = np.diff(tp, prepend=tp[0])
    pos_flow = np.where(delta > 0, raw_flow, 0.0)
    neg_flow = np.where(delta < 0, raw_flow, 0.0)
    pos_sum = rolling_sum(pos_flow, period)
    neg_sum = rolling_sum(neg_flow, period)
    ratio = safe_div(pos_sum, neg_sum, fill=np.inf)
    mfi = np.where(np.isinf(ratio), 100.0, 100.0 - 100.0 / (1.0 + ratio))
    return np.where(np.isnan(pos_sum), np.nan, mfi)
