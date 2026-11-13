"""Mock KIS 클라이언트 — 계좌 발급 전 전체 로직 검증용.

실제 서버를 전혀 호출하지 않으면서도 **KIS와 완전히 동일한 JSON 스키마**를 반환한다.
내부에는 가상 호가/체결 엔진이 있어 다음을 모두 재현한다.
  · 지정가/시장가 주문, 미체결 잔량, 부분체결, 주문 취소
  · 호가단위·가격제한폭 위반 거부, 매수가능금액 초과 거부, 보유수량 초과 매도 거부
  · 매수 수수료 / 매도 수수료+증권거래세, 스프레드 기반 슬리피지
  · 임의 주문거부(예외 경로 검증)
따라서 `USE_MOCK_MODE=false`로 바꾸는 순간 상위 코드는 단 한 줄도 수정할 필요가 없다.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from api.base import AbstractBroker, make_response
from config.settings import Settings
from core.events import Fill, OrderType, Side
from core.rate_limiter import PRIORITY_ORDER, PRIORITY_QUOTE, PriorityRateLimiter
from utils.krx_calendar import KST
from utils.logger import logger
from utils.market_rules import (
    affordable_qty,
    buy_cost,
    price_limits,
    round_to_tick,
    sell_cost,
    tick_size,
)


@dataclass
class MockOrder:
    order_no: str
    org_no: str
    symbol: str
    side: Side
    qty: int
    price: float
    order_type: OrderType
    ts: dt.datetime
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    reserved_cash: float = 0.0
    status: str = "OPEN"  # OPEN | FILLED | CANCELLED

    @property
    def remaining(self) -> int:
        return max(0, self.qty - self.filled_qty)

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN" and self.remaining > 0


@dataclass
class MockHolding:
    qty: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def buy(self, qty: int, price: float) -> None:
        total = self.avg_price * self.qty + price * qty
        self.qty += qty
        self.avg_price = total / self.qty if self.qty else 0.0

    def sell(self, qty: int, price: float, cost: float) -> float:
        qty = min(qty, self.qty)
        pnl = qty * (price - self.avg_price) - cost
        self.qty -= qty
        self.realized_pnl += pnl
        if self.qty <= 0:
            self.qty = 0
            self.avg_price = 0.0
        return pnl


class MockKISClient(AbstractBroker):
    """가상 체결 엔진이 내장된 Mock 브로커."""

    is_mock = True

    def __init__(
        self,
        settings: Settings,
        simulator=None,
        rate_limiter: Optional[PriorityRateLimiter] = None,
    ):
        from backtest.data import MarketSimulator, get_panel

        self.settings = settings
        self.limiter = rate_limiter or PriorityRateLimiter(
            rate=settings.effective_rate, burst=settings.REST_BURST
        )
        if simulator is None:
            panel = get_panel(
                settings.UNIVERSE,
                n_bars=settings.MOCK_HISTORY_BARS + settings.MOCK_FUTURE_BARS,
                seed=settings.MOCK_SEED,
                sector_map={s: settings.sector_of(s) for s in settings.UNIVERSE},
                cache_dir=None,
            )
            simulator = MarketSimulator(panel, warmup=settings.MOCK_HISTORY_BARS, seed=settings.MOCK_SEED)
        self.sim: "MarketSimulator" = simulator

        self.cash: float = float(settings.INITIAL_CAPITAL)
        self.reserved: float = 0.0
        self.holdings: Dict[str, MockHolding] = {}
        self.orders: Dict[str, MockOrder] = {}
        self._pending_fills: List[Fill] = []
        self._order_seq = itertools.count(1)
        self._rng = np.random.default_rng(settings.MOCK_SEED + 1)
        self._connected = False
        self.ledger_fills: List[Fill] = []

    # ------------------------------------------------------------------ 수명주기
    async def connect(self) -> None:
        self._connected = True
        logger.info(
            f"[MockBroker] 가상 체결 엔진 기동 | 종목 {len(self.sim.symbols)}개 | "
            f"시작자산 {self.cash:,.0f}원 | 봉 {self.sim.warmup}→{self.sim.max_cursor}"
        )

    async def close(self) -> None:
        self._connected = False
        await self.limiter.close()

    async def health_check(self) -> Dict[str, Any]:
        return {
            "ok": self._connected,
            "mode": "MOCK",
            "symbols": len(self.sim.symbols),
            "cash": self.cash,
            "cursor": self.sim.cursor,
            "max_cursor": self.sim.max_cursor,
        }

    # ------------------------------------------------------------------ 내부 유틸
    async def _throttle(self, priority: int = PRIORITY_QUOTE) -> None:
        """Mock에서도 동일한 유량 제어를 태워 실서버 전환 시 타이밍 차이를 없앤다."""
        await self.limiter.acquire(priority)

    def _base_price(self, symbol: str) -> float:
        hist = self.sim.history(symbol, count=2)
        if len(hist) >= 2:
            return float(hist.iloc[-2]["close"])
        return self.sim.current_price(symbol)

    def _now(self) -> dt.datetime:
        return dt.datetime.now(tz=KST)

    # ------------------------------------------------------------------ 시세 API
    async def inquire_price(self, symbol: str) -> Dict[str, Any]:
        await self._throttle(PRIORITY_QUOTE)
        bar = self.sim.current_bar(symbol)
        if bar is None:
            return make_response(rt_cd="1", msg_cd="MCA00001", msg1="조회할 수 없는 종목입니다.")
        price = self.sim.current_price(symbol)
        prev_close = self._base_price(symbol)
        change = price - prev_close
        output = {
            "stck_prpr": str(int(price)),
            "stck_oprc": str(int(bar["open"])),
            "stck_hgpr": str(int(bar["high"])),
            "stck_lwpr": str(int(bar["low"])),
            "stck_sdpr": str(int(prev_close)),
            "prdy_vrss": str(int(change)),
            "prdy_ctrt": f"{(change / prev_close * 100 if prev_close else 0):.2f}",
            "acml_vol": str(int(bar["volume"])),
            "acml_tr_pbmn": str(int(bar["volume"] * price)),
            "hts_avls": "0",
            "per": "0",
            "pbr": "0",
        }
        return make_response(output)

    async def inquire_asking_price(self, symbol: str) -> Dict[str, Any]:
        await self._throttle(PRIORITY_QUOTE)
        if symbol not in self.sim.panel:
            return make_response(rt_cd="1", msg_cd="MCA00001", msg1="조회할 수 없는 종목입니다.")
        bid, ask = self.sim.bid_ask(symbol)
        tick = tick_size(ask)
        output1 = {"askp1": str(int(ask)), "bidp1": str(int(bid)),
                   "askp_rsqn1": str(int(self._rng.integers(100, 5000))),
                   "bidp_rsqn1": str(int(self._rng.integers(100, 5000)))}
        for i in range(2, 11):
            output1[f"askp{i}"] = str(int(ask + tick * (i - 1)))
            output1[f"bidp{i}"] = str(int(max(tick, bid - tick * (i - 1))))
            output1[f"askp_rsqn{i}"] = str(int(self._rng.integers(100, 8000)))
            output1[f"bidp_rsqn{i}"] = str(int(self._rng.integers(100, 8000)))
        return make_response(None, output1=output1,
                             output2={"antc_cnpr": str(int(self.sim.current_price(symbol)))})

    async def inquire_daily_chart(
        self, symbol: str, start: str = "", end: str = "", period: str = "D", adjust: bool = True
    ) -> Dict[str, Any]:
        await self._throttle(PRIORITY_QUOTE)
        hist = self.sim.history(symbol, count=500)
        if hist.empty:
            return make_response(rt_cd="1", msg_cd="MCA00001", msg1="조회 데이터가 없습니다.")
        rows = [
            {
                "stck_bsop_date": str(r["ts"]),
                "stck_oprc": str(int(r["open"])),
                "stck_hgpr": str(int(r["high"])),
                "stck_lwpr": str(int(r["low"])),
                "stck_clpr": str(int(r["close"])),
                "acml_vol": str(int(r["volume"])),
                "acml_tr_pbmn": str(int(r["volume"] * r["close"])),
            }
            for _, r in hist.iloc[::-1].iterrows()  # KIS는 최신 → 과거 순서
        ]
        last = hist.iloc[-1]
        output1 = {"hts_kor_isnm": f"MOCK{symbol}", "stck_prpr": str(int(last["close"]))}
        return make_response(None, output1=output1, output2=rows)

    async def inquire_multi_price(self, symbols: List[str]) -> Dict[str, Any]:
        await self._throttle(PRIORITY_QUOTE)
        rows = []
        for code in list(symbols)[:30]:
            bar = self.sim.current_bar(code)
            if bar is None:
                continue
            price = self.sim.current_price(code)
            hist = self.sim.history(code, count=2)
            prev_close = float(hist["close"].iloc[-2]) if len(hist) >= 2 else price
            rows.append({
                "inter_shrn_iscd": code,
                "inter2_prpr": str(int(price)),
                "acml_vol": str(int(bar["volume"])),
                "acml_tr_pbmn": str(int(bar["volume"] * price)),
                "prdy_ctrt": f"{((price / prev_close - 1.0) * 100) if prev_close else 0:.2f}",
            })
        return make_response(rows)          # KIS 복수시세는 output 하나에 행 목록이다

    async def inquire_minute_chart(self, symbol: str, hour: str = "", past: bool = True) -> Dict[str, Any]:
        await self._throttle(PRIORITY_QUOTE)
        hist = self.sim.history(symbol, count=120)
        if hist.empty:
            return make_response(rt_cd="1", msg_cd="MCA00001", msg1="조회 데이터가 없습니다.")
        rows = [
            {
                "stck_bsop_date": str(r["ts"]),
                "stck_cntg_hour": f"{9 + (i // 60):02d}{i % 60:02d}00",
                "stck_oprc": str(int(r["open"])),
                "stck_hgpr": str(int(r["high"])),
                "stck_lwpr": str(int(r["low"])),
                "stck_prpr": str(int(r["close"])),
                "cntg_vol": str(int(r["volume"])),
            }
            for i, (_, r) in enumerate(hist.iloc[::-1].iterrows())
        ]
        return make_response(None, output1={"stck_prpr": rows[0]["stck_prpr"]}, output2=rows)

    # ------------------------------------------------------------------ 계좌 API
    async def inquire_balance(self) -> Dict[str, Any]:
        await self._throttle(1)
        output1 = []
        total_eval = 0.0
        total_cost = 0.0
        for symbol, h in self.holdings.items():
            if h.qty <= 0:
                continue
            price = self.sim.current_price(symbol)
            eval_amt = price * h.qty
            cost = h.avg_price * h.qty
            total_eval += eval_amt
            total_cost += cost
            output1.append(
                {
                    "pdno": symbol,
                    "prdt_name": f"MOCK{symbol}",
                    "hldg_qty": str(h.qty),
                    "ord_psbl_qty": str(h.qty - self._locked_sell_qty(symbol)),
                    "pchs_avg_pric": f"{h.avg_price:.2f}",
                    "pchs_amt": str(int(cost)),
                    "prpr": str(int(price)),
                    "evlu_amt": str(int(eval_amt)),
                    "evlu_pfls_amt": str(int(eval_amt - cost)),
                    "evlu_pfls_rt": f"{((eval_amt - cost) / cost * 100) if cost else 0:.2f}",
                }
            )
        available = self.cash - self.reserved
        output2 = [
            {
                "dnca_tot_amt": str(int(available)),
                "prvs_rcdl_excc_amt": str(int(available)),
                "tot_evlu_amt": str(int(self.cash + total_eval)),
                "nass_amt": str(int(self.cash + total_eval)),
                "scts_evlu_amt": str(int(total_eval)),
                "evlu_pfls_smtl_amt": str(int(total_eval - total_cost)),
                "asst_icdc_amt": str(int(self.cash + total_eval - self.settings.INITIAL_CAPITAL)),
            }
        ]
        return make_response(None, output1=output1, output2=output2)

    async def inquire_psbl_order(self, symbol: str, price: float = 0.0) -> Dict[str, Any]:
        await self._throttle(1)
        available = max(0.0, self.cash - self.reserved)
        ref_price = price or self.sim.current_price(symbol)
        qty = affordable_qty(available, ref_price, self.settings.FEE_BPS)
        output = {
            "ord_psbl_cash": str(int(available)),
            "nrcvb_buy_amt": str(int(available)),
            "max_buy_qty": str(qty),
            "nrcvb_buy_qty": str(qty),
        }
        return make_response(output)

    async def inquire_daily_ccld(self, start: str = "", end: str = "") -> Dict[str, Any]:
        await self._throttle(1)
        rows = [
            {
                "odno": f.order_no,
                "pdno": f.symbol,
                "sll_buy_dvsn_cd": "02" if f.side is Side.BUY else "01",
                "ord_qty": str(f.qty),
                "tot_ccld_qty": str(f.qty),
                "avg_prvs": f"{f.price:.1f}",
                "ord_tmd": f.ts.strftime("%H%M%S"),
                "rmn_qty": "0",
            }
            for f in self.ledger_fills[-200:]
        ]
        return make_response(None, output1=rows)

    async def inquire_open_orders(self) -> Dict[str, Any]:
        await self._throttle(1)
        rows = [
            {
                "odno": o.order_no,
                "orgn_odno": o.org_no,
                "pdno": o.symbol,
                "sll_buy_dvsn_cd": "02" if o.side is Side.BUY else "01",
                "ord_qty": str(o.qty),
                "tot_ccld_qty": str(o.filled_qty),
                "rmn_qty": str(o.remaining),
                "ord_unpr": str(int(o.price)),
            }
            for o in self.orders.values()
            if o.is_open
        ]
        return make_response(None, output1=rows)

    def _locked_sell_qty(self, symbol: str) -> int:
        return sum(o.remaining for o in self.orders.values()
                   if o.is_open and o.side is Side.SELL and o.symbol == symbol)

    # ------------------------------------------------------------------ 주문 API
    async def order_cash(
        self,
        symbol: str,
        side: Side,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Dict[str, Any]:
        await self._throttle(PRIORITY_ORDER)

        if qty <= 0:
            return make_response(rt_cd="1", msg_cd="40240000", msg1="주문수량이 올바르지 않습니다.")
        if symbol not in self.sim.panel:
            return make_response(rt_cd="1", msg_cd="40250000", msg1="종목코드 오류입니다.")

        market_price = self.sim.current_price(symbol)
        base = self._base_price(symbol)

        if order_type is OrderType.MARKET:
            price = 0.0
            ref_price = market_price
        else:
            if price <= 0:
                return make_response(rt_cd="1", msg_cd="40230000", msg1="주문단가를 입력하세요.")
            if int(price) % tick_size(price) != 0:
                return make_response(rt_cd="1", msg_cd="40230000", msg1="호가단위 오류입니다.")
            low, high = price_limits(base)
            if not (low <= price <= high):
                return make_response(rt_cd="1", msg_cd="40260000", msg1="가격제한폭을 벗어났습니다.")
            ref_price = price

        if side is Side.BUY:
            need = qty * ref_price * (1 + self.settings.FEE_BPS / 10_000.0)
            if need > (self.cash - self.reserved):
                return make_response(rt_cd="1", msg_cd="40310000", msg1="주문가능금액이 부족합니다.")
        else:
            holding = self.holdings.get(symbol)
            sellable = (holding.qty if holding else 0) - self._locked_sell_qty(symbol)
            if qty > sellable:
                return make_response(rt_cd="1", msg_cd="40320000", msg1="주문가능수량을 초과했습니다.")
            need = 0.0

        # 예외 경로 검증용 임의 거부
        if self._rng.random() < self.settings.MOCK_REJECT_PROB:
            return make_response(rt_cd="1", msg_cd="40910000", msg1="장운영 상태로 주문이 거부되었습니다.")

        seq = next(self._order_seq)
        order = MockOrder(
            order_no=f"{seq:010d}",
            org_no="91252",
            symbol=symbol,
            side=side,
            qty=qty,
            price=float(price),
            order_type=order_type,
            ts=self._now(),
            reserved_cash=need,
        )
        self.orders[order.order_no] = order
        if side is Side.BUY:
            self.reserved += need

        # 시장가는 접수 즉시 체결 시도
        if order_type is OrderType.MARKET:
            self._try_match(order, force=True)

        output = {
            "KRX_FWDG_ORD_ORGNO": order.org_no,
            "ODNO": order.order_no,
            "ORD_TMD": order.ts.strftime("%H%M%S"),
        }
        return make_response(output, msg_cd="APBK0013", msg1="주문 전송 완료 되었습니다.")

    async def cancel_order(
        self, org_no: str, order_no: str, qty: int = 0, all_qty: bool = True
    ) -> Dict[str, Any]:
        await self._throttle(PRIORITY_ORDER)
        order = self.orders.get(order_no)
        if order is None or not order.is_open:
            return make_response(rt_cd="1", msg_cd="40570000", msg1="취소가능한 주문이 없습니다.")
        cancel_qty = order.remaining if all_qty or qty <= 0 else min(qty, order.remaining)
        if order.side is Side.BUY and order.reserved_cash > 0 and order.qty > 0:
            release = order.reserved_cash * (cancel_qty / order.qty)
            self.reserved = max(0.0, self.reserved - release)
            order.reserved_cash = max(0.0, order.reserved_cash - release)
        order.qty -= cancel_qty
        if order.remaining <= 0:
            order.status = "CANCELLED" if order.filled_qty == 0 else "FILLED"
        output = {"KRX_FWDG_ORD_ORGNO": order.org_no, "ODNO": order.order_no,
                  "ORD_TMD": self._now().strftime("%H%M%S")}
        return make_response(output, msg_cd="APBK0014", msg1="주문 취소 완료 되었습니다.")

    # ------------------------------------------------------------------ 체결 엔진
    def _slippage(self, price: float, side: Side) -> float:
        bps = self.settings.SLIPPAGE_BPS / 10_000.0
        return price * (1 + bps) if side is Side.BUY else price * (1 - bps)

    def _try_match(self, order: MockOrder, force: bool = False) -> None:
        """단일 주문에 대해 체결을 시도한다."""
        if not order.is_open:
            return
        bid, ask = self.sim.bid_ask(order.symbol)
        exec_price: Optional[float] = None

        if order.order_type is OrderType.MARKET:
            exec_price = self._slippage(ask if order.side is Side.BUY else bid, order.side)
        elif order.side is Side.BUY and ask <= order.price:
            exec_price = min(order.price, ask)
        elif order.side is Side.SELL and bid >= order.price:
            exec_price = max(order.price, bid)

        if exec_price is None or exec_price <= 0:
            return

        exec_price = float(round_to_tick(exec_price))
        remaining = order.remaining
        if not force and self._rng.random() < self.settings.MOCK_PARTIAL_FILL_PROB and remaining > 1:
            fill_qty = max(1, int(remaining * self._rng.uniform(0.3, 0.7)))
        else:
            fill_qty = remaining

        if order.side is Side.BUY:
            fee = buy_cost(fill_qty, exec_price, self.settings.FEE_BPS)
            cash_need = fill_qty * exec_price + fee
            if cash_need > self.cash:
                return  # 현금 부족 시 체결 보류
            self.cash -= cash_need
            release = order.reserved_cash * (fill_qty / order.qty) if order.qty else 0.0
            self.reserved = max(0.0, self.reserved - release)
            order.reserved_cash = max(0.0, order.reserved_cash - release)
            holding = self.holdings.setdefault(order.symbol, MockHolding())
            holding.buy(fill_qty, exec_price)
            tax = 0.0
        else:
            fee, tax = sell_cost(fill_qty, exec_price, self.settings.FEE_BPS, self.settings.TAX_BPS)
            holding = self.holdings.setdefault(order.symbol, MockHolding())
            holding.sell(fill_qty, exec_price, fee + tax)
            self.cash += fill_qty * exec_price - fee - tax

        prev_qty = order.filled_qty
        order.filled_qty += fill_qty
        order.avg_fill_price = (
            (order.avg_fill_price * prev_qty + exec_price * fill_qty) / order.filled_qty
        )
        if order.remaining <= 0:
            order.status = "FILLED"

        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            qty=fill_qty,
            price=exec_price,
            fee=fee,
            tax=tax,
            order_no=order.order_no,
            ts=self._now(),
        )
        self._pending_fills.append(fill)
        self.ledger_fills.append(fill)

    def match_all(self) -> None:
        """모든 미체결 주문에 대해 체결 시도."""
        for order in list(self.orders.values()):
            if order.is_open:
                self._try_match(order)

    def expire_stale_orders(self, max_age_seconds: float = 180.0) -> int:
        """오래된 미체결 주문 자동 취소(호가 이탈 방지)."""
        now = self._now()
        cancelled = 0
        for order in list(self.orders.values()):
            if order.is_open and (now - order.ts).total_seconds() > max_age_seconds:
                if order.side is Side.BUY and order.reserved_cash > 0:
                    self.reserved = max(0.0, self.reserved - order.reserved_cash)
                    order.reserved_cash = 0.0
                order.status = "CANCELLED" if order.filled_qty == 0 else "FILLED"
                cancelled += 1
        return cancelled

    async def poll_fills(self) -> List[Fill]:
        fills = self._pending_fills
        self._pending_fills = []
        return fills

    # ------------------------------------------------------------------ 시간 전진
    def advance(self) -> bool:
        """가상 시계를 한 틱 전진시키고 체결 엔진을 돌린다."""
        bar_changed = self.sim.step()
        self.match_all()
        return bar_changed

    async def run_clock(self, interval: float = 0.2, stop_event: Optional[asyncio.Event] = None) -> None:
        """백그라운드에서 가상 시계를 굴린다(실시간 루프 테스트용)."""
        while stop_event is None or not stop_event.is_set():
            self.advance()
            if self.sim.exhausted:
                logger.warning("[MockBroker] 시뮬레이션 데이터 소진 - 시계를 정지합니다.")
                break
            await asyncio.sleep(interval)

    def account_snapshot(self) -> Dict[str, Any]:
        equity = self.cash + sum(
            self.sim.current_price(s) * h.qty for s, h in self.holdings.items() if h.qty > 0
        )
        return {
            "cash": round(self.cash, 0),
            "reserved": round(self.reserved, 0),
            "equity": round(equity, 0),
            "holdings": {s: h.qty for s, h in self.holdings.items() if h.qty > 0},
            "open_orders": sum(1 for o in self.orders.values() if o.is_open),
            "cursor": self.sim.cursor,
        }
