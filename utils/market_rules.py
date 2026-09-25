"""KRX 매매 제도 규칙 (호가단위, 가격제한폭, 거래비용).

주문 가격이 호가단위를 벗어나면 KIS는 주문을 거부한다(40230000 등).
Mock 엔진과 실주문 경로가 **동일한 규칙**을 쓰도록 여기에 모았다.
"""

from __future__ import annotations

import math
from typing import Literal

Market = Literal["KOSPI", "KOSDAQ"]

# (상한 미만 가격, 호가단위) - 2023.01 KRX 호가가격단위 개편 기준
_KOSPI_TICKS = (
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
    (float("inf"), 1_000),
)
_KOSDAQ_TICKS = (
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (float("inf"), 100),
)

PRICE_LIMIT_PCT = 0.30  # 가격제한폭 ±30%


def tick_size(price: float, market: Market = "KOSPI") -> int:
    table = _KOSPI_TICKS if market == "KOSPI" else _KOSDAQ_TICKS
    for upper, tick in table:
        if price < upper:
            return tick
    return 1_000


def round_to_tick(price: float, market: Market = "KOSPI", mode: str = "nearest") -> int:
    """호가단위에 맞춰 가격을 정렬한다.

    mode: "nearest" | "down"(매수 유리) | "up"(매도 유리)
    """
    if price <= 0:
        return 0
    tick = tick_size(price, market)
    if mode == "down":
        return int(math.floor(price / tick) * tick)
    if mode == "up":
        return int(math.ceil(price / tick) * tick)
    return int(round(price / tick) * tick)


def price_limits(base_price: float, market: Market = "KOSPI") -> tuple[int, int]:
    """(하한가, 상한가). 전일 종가(또는 기준가) 기준 ±30%."""
    upper = round_to_tick(base_price * (1 + PRICE_LIMIT_PCT), market, mode="down")
    lower = round_to_tick(base_price * (1 - PRICE_LIMIT_PCT), market, mode="up")
    return lower, upper


def clamp_to_limits(price: float, base_price: float, market: Market = "KOSPI") -> int:
    lower, upper = price_limits(base_price, market)
    return int(min(max(round_to_tick(price, market), lower), upper))


def buy_cost(qty: int, price: float, fee_bps: float) -> float:
    """매수 수수료(원). 증권거래세는 매도 시에만 부과된다."""
    return qty * price * fee_bps / 10_000.0


def sell_cost(qty: int, price: float, fee_bps: float, tax_bps: float) -> tuple[float, float]:
    """(수수료, 거래세) 반환."""
    gross = qty * price
    return gross * fee_bps / 10_000.0, gross * tax_bps / 10_000.0


def affordable_qty(cash: float, price: float, fee_bps: float) -> int:
    """주어진 현금으로 살 수 있는 최대 수량(수수료 포함)."""
    if price <= 0:
        return 0
    unit_cost = price * (1 + fee_bps / 10_000.0)
    return max(0, int(cash // unit_cost))
