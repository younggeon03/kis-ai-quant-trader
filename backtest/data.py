"""시세 데이터 소스.

계좌 발급 전에는 KIS 시세 API를 호출할 수 없으므로, **레짐 전환 · 변동성 군집 ·
시장/섹터 공통요인**을 반영한 합성 시세 생성기로 전체 파이프라인을 검증한다.
생성 결과는 KIS 차트 응답을 변환한 DataFrame과 완전히 동일한 스키마
(`ts, open, high, low, close, volume`)를 따르므로, 계좌 발급 후에는
`load_panel_from_dir()` 또는 브로커의 `get_bars()`로 소스만 바꾸면 된다.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils.market_rules import round_to_tick

OHLCV_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

# 레짐: (일간 드리프트, 변동성 배수)
_REGIMES = [
    (0.00075, 0.85),   # 강세
    (0.00000, 1.00),   # 횡보
    (-0.00110, 1.70),  # 약세(변동성 확대)
]
# 레짐 전이 확률(지속성이 높도록 대각 우세)
_TRANSITION = np.array([
    [0.970, 0.025, 0.005],
    [0.030, 0.950, 0.020],
    [0.020, 0.060, 0.920],
])


def _garch_path(n: int, rng: np.random.Generator, base_vol: float,
                omega_ratio: float = 0.05, alpha: float = 0.09, beta: float = 0.88) -> np.ndarray:
    """GARCH(1,1) 변동성 경로. 변동성 군집(clustering)을 재현한다."""
    var_long = base_vol ** 2
    omega = var_long * (1 - alpha - beta) if (1 - alpha - beta) > 0 else var_long * omega_ratio
    sigma2 = np.empty(n)
    sigma2[0] = var_long
    eps = np.empty(n)
    shocks = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)  # 팻테일 정규화
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(max(sigma2[t], 1e-12)) * shocks[t]
    return eps


def _regime_path(n: int, rng: np.random.Generator) -> np.ndarray:
    states = np.empty(n, dtype=int)
    state = rng.integers(0, 3)
    for t in range(n):
        state = rng.choice(3, p=_TRANSITION[state])
        states[t] = state
    return states


def _business_days(n: int, end: Optional[dt.date] = None) -> List[str]:
    end = end or dt.date.today()
    days: List[str] = []
    cur = end
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y%m%d"))
        cur -= dt.timedelta(days=1)
    return list(reversed(days))


def generate_synthetic_panel(
    symbols: Iterable[str],
    n_bars: int = 1000,
    seed: int = 20261012,
    sector_map: Optional[Dict[str, str]] = None,
    annual_vol: float = 0.30,
    end_date: Optional[dt.date] = None,
) -> Dict[str, pd.DataFrame]:
    """합성 OHLCV 패널 생성.

    구조: ``r_i = beta_i * r_market + gamma_i * r_sector + eps_i``
    - r_market: 레짐 전환 드리프트 + GARCH 변동성
    - r_sector: 섹터 공통요인(AR(1))
    - eps_i   : 종목 고유요인(GARCH, 팻테일)
    실제 주식처럼 **예측 가능한 성분(모멘텀/평균회귀)** 을 소량 섞어
    AI 모델이 학습할 신호가 존재하도록 만든다.
    """
    rng = np.random.default_rng(seed)
    symbols = list(symbols)
    sector_map = sector_map or {}
    sectors = sorted({sector_map.get(s, "ETC") for s in symbols})

    daily_vol = annual_vol / np.sqrt(252)

    # --- 시장 요인 ---
    regimes = _regime_path(n_bars, rng)
    mkt_drift = np.array([_REGIMES[s][0] for s in regimes])
    mkt_volmul = np.array([_REGIMES[s][1] for s in regimes])
    mkt_shock = _garch_path(n_bars, rng, base_vol=daily_vol * 0.55) * mkt_volmul
    market_ret = mkt_drift + mkt_shock

    # --- 섹터 요인 (AR(1)) ---
    sector_ret: Dict[str, np.ndarray] = {}
    for sec in sectors:
        noise = _garch_path(n_bars, rng, base_vol=daily_vol * 0.35)
        series = np.zeros(n_bars)
        phi = rng.uniform(0.05, 0.25)
        for t in range(1, n_bars):
            series[t] = phi * series[t - 1] + noise[t]
        sector_ret[sec] = series

    dates = _business_days(n_bars, end_date)
    panel: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        sec = sector_map.get(symbol, "ETC")
        beta = rng.uniform(0.65, 1.45)
        gamma = rng.uniform(0.25, 0.85)
        idio = _garch_path(n_bars, rng, base_vol=daily_vol * rng.uniform(0.55, 1.05))

        ret = beta * market_ret + gamma * sector_ret[sec] + idio

        # 예측 가능한 미세 구조: 단기 평균회귀 + 중기 모멘텀 (알파의 원천)
        mom_coef = rng.uniform(0.010, 0.035)
        rev_coef = rng.uniform(0.020, 0.055)
        adj = np.zeros(n_bars)
        for t in range(21, n_bars):
            mom = ret[t - 21:t - 1].sum()
            rev = ret[t - 1]
            adj[t] = mom_coef * mom - rev_coef * rev
        ret = ret + adj
        ret = np.clip(ret, -0.28, 0.28)  # 가격제한폭

        start_price = float(rng.choice([3_500, 7_800, 15_400, 32_000, 68_000, 125_000, 280_000]))
        close = start_price * np.exp(np.cumsum(ret))

        # --- OHLC 합성 ---
        prev_close = np.concatenate([[start_price], close[:-1]])
        bar_vol = np.abs(ret) + daily_vol * 0.35
        gap = rng.normal(0, 0.30, n_bars) * bar_vol
        open_ = prev_close * (1 + gap)
        hi_wick = np.abs(rng.normal(0, 0.55, n_bars)) * bar_vol
        lo_wick = np.abs(rng.normal(0, 0.55, n_bars)) * bar_vol
        high = np.maximum(open_, close) * (1 + hi_wick)
        low = np.minimum(open_, close) * (1 - lo_wick)

        # --- 거래량: |수익률| 및 변동성과 양의 상관 ---
        base_shares = float(rng.lognormal(mean=12.5, sigma=0.9))
        vol_factor = np.exp(1.6 * np.abs(ret) / max(daily_vol, 1e-6) * 0.25)
        noise = np.exp(rng.normal(0, 0.35, n_bars))
        spikes = np.where(rng.random(n_bars) < 0.02, rng.uniform(2.0, 5.0, n_bars), 1.0)
        volume = np.maximum(1_000.0, base_shares * vol_factor * noise * spikes)

        df = pd.DataFrame(
            {
                "ts": dates,
                "open": [round_to_tick(v) for v in open_],
                "high": [round_to_tick(v) for v in high],
                "low": [round_to_tick(v) for v in low],
                "close": [round_to_tick(v) for v in close],
                "volume": np.round(volume).astype(np.int64),
            }
        )
        # 호가단위 반올림 후 OHLC 정합성 보정
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        df = df[df["close"] > 0].reset_index(drop=True)
        panel[symbol] = df

    return panel


class MarketSimulator:
    """생성된 패널 위를 커서가 전진하는 시세 시뮬레이터.

    Mock 브로커가 "현재 시점"을 갖도록 해 준다. 커서 이후 데이터는 절대 노출하지
    않으므로 look-ahead bias가 구조적으로 불가능하다.
    """

    def __init__(
        self,
        panel: Dict[str, pd.DataFrame],
        warmup: int = 400,
        intrabar_steps: int = 4,
        seed: int = 7,
    ):
        self.panel = {s: df.reset_index(drop=True) for s, df in panel.items() if len(df) > warmup}
        if not self.panel:
            raise ValueError("MarketSimulator: 워밍업 길이를 만족하는 종목이 없습니다")
        self.warmup = warmup
        self.cursor = warmup
        self.max_cursor = min(len(df) for df in self.panel.values()) - 1
        self.intrabar_steps = max(1, intrabar_steps)
        self._intrabar = 0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ 조회
    @property
    def symbols(self) -> List[str]:
        return list(self.panel.keys())

    @property
    def exhausted(self) -> bool:
        return self.cursor >= self.max_cursor

    def history(self, symbol: str, count: int = 300) -> pd.DataFrame:
        """현재 커서까지의 확정 봉(과거→최신)."""
        df = self.panel.get(symbol)
        if df is None:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        end = min(self.cursor + 1, len(df))
        return df.iloc[max(0, end - count):end].reset_index(drop=True)

    def current_bar(self, symbol: str) -> Optional[pd.Series]:
        df = self.panel.get(symbol)
        if df is None or self.cursor >= len(df):
            return None
        return df.iloc[self.cursor]

    def current_price(self, symbol: str) -> float:
        """장중 틱을 흉내낸 현재가(봉 내부를 intrabar_steps로 보간)."""
        bar = self.current_bar(symbol)
        if bar is None:
            return 0.0
        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        frac = (self._intrabar + 1) / self.intrabar_steps
        base = o + (c - o) * frac
        wiggle = self._rng.uniform(-0.35, 0.35) * (h - l) * (1 - frac)
        return float(round_to_tick(min(max(base + wiggle, l), h)))

    def spread(self, symbol: str) -> float:
        bar = self.current_bar(symbol)
        if bar is None:
            return 0.0
        return max(1.0, (float(bar["high"]) - float(bar["low"])) * 0.06)

    def bid_ask(self, symbol: str) -> tuple[float, float]:
        price = self.current_price(symbol)
        half = self.spread(symbol) / 2.0
        return round_to_tick(price - half, mode="down"), round_to_tick(price + half, mode="up")

    # ------------------------------------------------------------------ 전진
    def step(self) -> bool:
        """장중 한 틱 전진. 봉이 바뀌면 True."""
        self._intrabar += 1
        if self._intrabar >= self.intrabar_steps:
            self._intrabar = 0
            if self.cursor < self.max_cursor:
                self.cursor += 1
                return True
        return False

    def timestamp(self) -> str:
        bar = self.current_bar(self.symbols[0])
        return str(bar["ts"]) if bar is not None else ""


# ---------------------------------------------------------------------------- 로더
def load_panel_from_dir(path: str | Path, symbols: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    """`<dir>/<symbol>.csv|parquet` 형태의 로컬 시세를 로드한다."""
    root = Path(path)
    panel: Dict[str, pd.DataFrame] = {}
    if not root.exists():
        return panel
    wanted = set(symbols) if symbols else None
    for file in sorted(list(root.glob("*.csv")) + list(root.glob("*.parquet"))):
        symbol = file.stem.zfill(6)
        if wanted and symbol not in wanted:
            continue
        df = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file)
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            continue
        df["ts"] = df["ts"].astype(str)
        panel[symbol] = df[OHLCV_COLUMNS].reset_index(drop=True)
    return panel


def save_panel(panel: Dict[str, pd.DataFrame], path: str | Path, fmt: str = "csv") -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    for symbol, df in panel.items():
        target = root / f"{symbol}.{fmt}"
        if fmt == "parquet":
            df.to_parquet(target, index=False)
        else:
            df.to_csv(target, index=False)


def get_panel(
    symbols: Iterable[str],
    n_bars: int,
    seed: int,
    sector_map: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str | Path] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """캐시가 있으면 로드하고, 없으면 합성 생성 후 저장한다."""
    symbols = list(symbols)
    if cache_dir and use_cache:
        cached = load_panel_from_dir(cache_dir, symbols)
        if cached and all(len(df) >= n_bars for df in cached.values()) and len(cached) >= len(symbols):
            return {s: cached[s] for s in symbols if s in cached}
    panel = generate_synthetic_panel(symbols, n_bars=n_bars, seed=seed, sector_map=sector_map)
    if cache_dir:
        save_panel(panel, cache_dir)
    return panel
