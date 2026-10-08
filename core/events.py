"""이벤트 정의 및 비동기 이벤트 버스.

매매 파이프라인은 다음 이벤트 흐름으로 동작한다.
    Quote/Bar → Signal → RiskDecision → OrderRequest → Fill → Portfolio 갱신
dataclass(slots)로 정의해 핫패스 오버헤드를 최소화했다.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from utils.krx_calendar import KST


def _now() -> dt.datetime:
    return dt.datetime.now(tz=KST)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderType(str, Enum):
    """KIS 주문구분 코드."""

    LIMIT = "00"          # 지정가
    MARKET = "01"         # 시장가
    BEST_LIMIT = "03"     # 최유리지정가
    PRIORITY_LIMIT = "04"  # 최우선지정가


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    SIGNAL = "SIGNAL"
    CRASH_GUARD = "CRASH_GUARD"
    VOL_DECAY = "VOL_DECAY"        # 거래대금이 보유 중 고점 대비 줄어 상승 동력 소멸
    KILL_SWITCH = "KILL_SWITCH"
    DERISK = "DERISK"
    EOD = "EOD"


@dataclass(slots=True)
class Quote:
    """현재가 스냅샷."""

    symbol: str
    price: float
    ts: dt.datetime = field(default_factory=_now)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    change_pct: float = 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.price


@dataclass(slots=True)
class Signal:
    """전략이 생성한 알파 신호."""

    symbol: str
    score: float                     # -1.0 ~ 1.0 (앙상블 최종 점수)
    price: float
    ts: dt.datetime = field(default_factory=_now)
    model_proba: float = 0.5         # AI 모델의 상승확률
    tech_score: float = 0.0          # 기술적 팩터 점수
    cs_score: float = 0.0            # 횡단면 랭크 점수
    regime: float = 1.0              # 레짐 스케일 0~1
    atr: float = 0.0                 # 변동성(원 단위) - 포지션 사이징 입력
    volatility: float = 0.0          # 연율 변동성
    bet: float = -1.0                # 베팅 사이즈 0~1. -1은 '모델 없음'(미설정)을 뜻한다
    reason: str = ""

    @property
    def is_buy(self) -> bool:
        return self.score > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol, "score": round(self.score, 4),
            "proba": round(self.model_proba, 4), "tech": round(self.tech_score, 4),
            "cs": round(self.cs_score, 4), "regime": round(self.regime, 3),
            "bet": round(self.bet, 4),
            "price": self.price, "atr": round(self.atr, 2), "reason": self.reason,
        }


@dataclass(slots=True)
class RiskDecision:
    """리스크 매니저의 최종 판정. approved=False면 주문은 절대 나가지 않는다."""

    approved: bool
    qty: int = 0
    reason: str = ""
    weight: float = 0.0
    stop_price: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: Side
    qty: int
    price: float = 0.0
    order_type: OrderType = OrderType.LIMIT
    reason: str = ""
    ts: dt.datetime = field(default_factory=_now)
    client_id: str = ""


@dataclass(slots=True)
class OrderAck:
    """주문 접수 응답."""

    success: bool
    order_no: str = ""
    org_no: str = ""
    symbol: str = ""
    side: Optional[Side] = None
    qty: int = 0
    price: float = 0.0
    msg_cd: str = ""
    msg: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Fill:
    """체결 통보."""

    symbol: str
    side: Side
    qty: int
    price: float
    fee: float = 0.0
    tax: float = 0.0
    order_no: str = ""
    ts: dt.datetime = field(default_factory=_now)
    reason: str = ""

    @property
    def gross(self) -> float:
        return self.qty * self.price

    @property
    def net_cash(self) -> float:
        """현금 변동량(매수는 음수, 매도는 양수)."""
        if self.side is Side.BUY:
            return -(self.gross + self.fee)
        return self.gross - self.fee - self.tax

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol, "side": self.side.value, "qty": self.qty,
            "price": self.price, "fee": round(self.fee, 1), "tax": round(self.tax, 1),
            "order_no": self.order_no, "reason": self.reason,
        }


@dataclass(slots=True)
class Position:
    """보유 포지션. 트레일링 스톱을 위해 고점을 추적한다."""

    symbol: str
    qty: int = 0
    avg_price: float = 0.0
    last_price: float = 0.0
    peak_price: float = 0.0
    entry_ts: dt.datetime = field(default_factory=_now)
    bars_held: int = 0
    realized_pnl: float = 0.0
    stop_price: float = 0.0
    trailing_active: bool = False
    sector: str = "ETC"

    @property
    def market_value(self) -> float:
        return self.qty * (self.last_price or self.avg_price)

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        return self.qty * ((self.last_price or self.avg_price) - self.avg_price)

    @property
    def pnl_pct(self) -> float:
        if self.avg_price <= 0:
            return 0.0
        return ((self.last_price or self.avg_price) - self.avg_price) / self.avg_price

    def mark(self, price: float) -> None:
        if price > 0:
            self.last_price = price
            self.peak_price = max(self.peak_price, price)

    def apply_fill(self, fill: Fill) -> float:
        """체결 반영. 매도 시 실현손익을 반환한다."""
        realized = 0.0
        if fill.side is Side.BUY:
            total_cost = self.avg_price * self.qty + fill.price * fill.qty
            self.qty += fill.qty
            self.avg_price = total_cost / self.qty if self.qty else 0.0
            self.peak_price = max(self.peak_price, fill.price)
            self.last_price = fill.price
        else:
            sell_qty = min(fill.qty, self.qty)
            realized = sell_qty * (fill.price - self.avg_price) - fill.fee - fill.tax
            self.qty -= sell_qty
            self.realized_pnl += realized
            self.last_price = fill.price
            if self.qty <= 0:
                self.qty = 0
                self.avg_price = 0.0
                self.peak_price = 0.0
                self.trailing_active = False
                self.stop_price = 0.0
        return realized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol, "qty": self.qty, "avg_price": round(self.avg_price, 1),
            "last_price": round(self.last_price, 1), "pnl_pct": round(self.pnl_pct * 100, 2),
            "unrealized": round(self.unrealized_pnl, 0), "stop": round(self.stop_price, 1),
            "bars_held": self.bars_held, "sector": self.sector,
            "trailing": self.trailing_active,
        }


@dataclass
class Portfolio:
    """포트폴리오 상태. 백테스트와 실매매가 동일한 객체를 사용한다."""

    cash: float
    initial_capital: float
    positions: Dict[str, Position] = field(default_factory=dict)
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    trade_returns: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.peak_equity <= 0:
            self.peak_equity = self.initial_capital
        if self.day_start_equity <= 0:
            self.day_start_equity = self.initial_capital

    # ---------------------------------------------------------------- 평가
    @property
    def market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    @property
    def gross_exposure(self) -> float:
        eq = self.equity
        return self.market_value / eq if eq > 0 else 0.0

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.equity - self.day_start_equity) / self.day_start_equity

    @property
    def total_return(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self.equity - self.initial_capital) / self.initial_capital

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total else 0.0

    def open_positions(self) -> Dict[str, Position]:
        return {s: p for s, p in self.positions.items() if p.qty > 0}

    def sector_weights(self) -> Dict[str, float]:
        eq = self.equity
        if eq <= 0:
            return {}
        weights: Dict[str, float] = defaultdict(float)
        for p in self.open_positions().values():
            weights[p.sector] += p.market_value / eq
        return dict(weights)

    # ---------------------------------------------------------------- 갱신
    def mark_prices(self, prices: Dict[str, float]) -> None:
        for symbol, price in prices.items():
            pos = self.positions.get(symbol)
            if pos and pos.qty > 0:
                pos.mark(price)
        self.peak_equity = max(self.peak_equity, self.equity)

    def apply_fill(self, fill: Fill, sector: str = "ETC") -> float:
        pos = self.positions.get(fill.symbol)
        if pos is None:
            pos = Position(symbol=fill.symbol, sector=sector, entry_ts=fill.ts)
            self.positions[fill.symbol] = pos
        entry_price = pos.avg_price
        realized = pos.apply_fill(fill)
        self.cash += fill.net_cash
        self.total_fees += fill.fee + fill.tax
        if fill.side is Side.SELL:
            self.realized_pnl += realized
            self.trade_count += 1
            if entry_price > 0:
                ret = (fill.price - entry_price) / entry_price
                self.trade_returns.append(ret)
            if realized > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
            if pos.qty <= 0:
                self.positions.pop(fill.symbol, None)
        else:
            pos.entry_ts = fill.ts if entry_price == 0 else pos.entry_ts
        self.peak_equity = max(self.peak_equity, self.equity)
        return realized

    def start_new_day(self) -> None:
        self.day_start_equity = self.equity
        for pos in self.positions.values():
            pos.bars_held += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "equity": round(self.equity, 0),
            "cash": round(self.cash, 0),
            "market_value": round(self.market_value, 0),
            "total_return_pct": round(self.total_return * 100, 3),
            "daily_pnl_pct": round(self.daily_pnl_pct * 100, 3),
            "drawdown_pct": round(self.drawdown * 100, 3),
            "gross_exposure": round(self.gross_exposure, 3),
            "positions": [p.to_dict() for p in self.open_positions().values()],
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate * 100, 2),
            "total_fees": round(self.total_fees, 0),
        }


Handler = Callable[[Any], Awaitable[None]]


class EventBus:
    """토픽 기반 비동기 이벤트 버스.

    구독자 예외가 전체 루프를 멈추지 않도록 각 핸들러를 개별 보호한다.
    """

    def __init__(self, maxsize: int = 10_000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._handlers: Dict[str, List[Handler]] = defaultdict(list)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.dropped = 0

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)

    async def publish(self, topic: str, payload: Any) -> None:
        try:
            self._queue.put_nowait((topic, payload))
        except asyncio.QueueFull:
            # 시세 폭주 시 최신성이 중요하므로 가장 오래된 이벤트를 버린다.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.dropped += 1
                self._queue.put_nowait((topic, payload))
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                self.dropped += 1

    async def _consume(self) -> None:
        from utils.logger import logger

        while self._running:
            try:
                topic, payload = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                for handler in self._handlers.get(topic, ()):
                    try:
                        await handler(payload)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.exception(f"[EventBus] 핸들러 오류 topic={topic}: {exc}")
            finally:
                self._queue.task_done()

    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self._consume(), name="event-bus")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    @property
    def pending(self) -> int:
        return self._queue.qsize()


# 토픽 상수
TOPIC_QUOTE = "quote"
TOPIC_SIGNAL = "signal"
TOPIC_ORDER = "order"
TOPIC_FILL = "fill"
TOPIC_RISK = "risk"
TOPIC_ERROR = "error"
