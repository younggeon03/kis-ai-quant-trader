"""KIS AI Quant Trader — 시스템 엔트리포인트.

부팅 시퀀스
    설정 로드 → 로거 초기화 → 브로커 생성 → 헬스체크 → 유니버스 확정 →
    모델 로드 → 과거 시세 적재 → (실서버) WebSocket 연결 → 대시보드 →
    메인 이벤트 루프

메인 루프 (LOOP_INTERVAL 주기)
    시세 갱신 → AI 신호 생성 → **리스크 검증(거부권)** → 주문 제출 →
    체결 확인 → 포트폴리오 반영 → 상태 저장

강건성
    · 모든 단계는 개별 예외 처리되며, 실패해도 루프는 멈추지 않는다.
    · WebSocket이 끊기면 REST 폴링으로 자동 폴백한다.
    · Ctrl+C / SIGTERM 시 미체결 주문을 취소하고 상태를 저장한 뒤 종료한다.
    · 재시작하면 브로커 잔고를 기준으로 포지션을 복구한다.

사용법
    python main.py                # Mock 또는 실서버(.env의 USE_MOCK_MODE에 따름)
    python main.py --health-only  # 연결/인증/계좌 점검만 수행하고 종료
    python main.py --duration 60  # 60초만 구동(스모크 테스트)
    python main.py --no-dashboard
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import signal
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from api.base import AbstractBroker, is_ok, to_int
from api.factory import create_broker
from config.settings import Settings, get_settings
from core.events import (
    TOPIC_FILL,
    TOPIC_ORDER,
    TOPIC_SIGNAL,
    EventBus,
    ExitReason,
    Fill,
    OrderRequest,
    OrderType,
    Portfolio,
    Quote,
    Side,
    Signal,
)
from monitor.app import DashboardServer
from strategy.ai_model import EnsembleAlphaStrategy
from strategy.risk_manager import ExitOrder, RiskManager, RiskState
from strategy.selection import select_entries, select_rotation_exits
from strategy.phase import current_phase, phase_config, reset_recommendation
from utils.krx_calendar import KRXCalendar, MarketSession
from utils.logger import JsonLedger, logger, setup_logging
from utils.market_rules import round_to_tick

BAR_REFRESH_LOOPS = 40      # 전체 일봉 재적재 주기(루프 수)
QUOTE_CHUNK = 4             # 루프당 현재가 갱신 종목 수(유량 보호)
RECONCILE_LOOPS = 30        # 브로커 잔고 대조 주기
FILL_POLL_LOOPS = 5         # 체결 폴링 주기(실서버)


class TradingEngine:
    def __init__(self, settings: Optional[Settings] = None, enable_dashboard: bool = True):
        self.s = settings or get_settings()
        self.broker: Optional[AbstractBroker] = None
        self.strategy = EnsembleAlphaStrategy(self.s)
        self.risk = RiskManager(self.s)
        self.portfolio = Portfolio(cash=self.s.INITIAL_CAPITAL, initial_capital=self.s.INITIAL_CAPITAL)
        self.calendar = KRXCalendar()
        self.bus = EventBus()
        self.ledger = JsonLedger(self.s.LEDGER_PATH)
        self.ws = None
        self.dashboard = DashboardServer(self.state_snapshot, self.s.DASHBOARD_HOST,
                                         self.s.DASHBOARD_PORT, settings=self.s) \
            if (enable_dashboard and self.s.DASHBOARD_ENABLED) else None

        # 신호 유니버스: 일봉으로 점수를 계산할 전체 후보(하루 1회 적재)
        self.universe: List[str] = []
        # 시세 관심목록: 실시간 현재가를 받을 종목. KIS WebSocket 구독 한도가 41건
        # 이므로 유니버스 전체를 구독할 수 없다. 보유 종목 + 점수 상위로 채운다.
        self.watchlist: List[str] = []
        self.panel: Dict[str, pd.DataFrame] = {}
        self.latest_prices: Dict[str, float] = {}
        self.atr_map: Dict[str, float] = {}
        # 대회 국면 — 날짜로 결정되며 기간 밖에서는 OFF(평소 동작)다.
        self.phase = phase_config(settings=self.s)
        self._last_phase = self.phase.phase
        self.reset_advice = None
        self.compliance = None
        self.score_map: Dict[str, float] = {}
        self.recent_fills: List[Dict[str, Any]] = []
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        # 대시보드 자산곡선/유량 그래프용 링버퍼 (메모리 상한 고정)
        self.equity_history: deque = deque(maxlen=720)
        self.rps_history: deque = deque(maxlen=90)

        self._running = False
        self._stop = asyncio.Event()
        self._loops = 0
        self._quote_cursor = 0
        self._errors = 0
        self._mock_epoch = dt.datetime(2025, 1, 2, 9, 0)
        self._last_bar_ts: str = ""
        # 신호 캐시. 일봉 전략이므로 봉이 바뀔 때만 재계산한다(아래 _current_signals 참고).
        self._signals: List[Signal] = []
        self._signals_bar_ts: str = ""
        self._started_at: Optional[dt.datetime] = None

    # ================================================================== 시간
    def _now(self) -> dt.datetime:
        """Mock은 가상 시계(봉 1개 = 1거래일), 실서버는 KST 실제 시각."""
        if self.s.USE_MOCK_MODE and self.broker is not None and getattr(self.broker, "is_mock", False):
            cursor = self.broker.sim.cursor  # type: ignore[attr-defined]
            return self._mock_epoch + dt.timedelta(days=cursor)
        return self.calendar.now()

    def _session(self) -> MarketSession:
        if self.s.USE_MOCK_MODE:
            return MarketSession.REGULAR  # Mock은 항상 정규장으로 간주
        return self.calendar.session()

    # ================================================================== 부팅
    async def startup(self) -> bool:
        setup_logging(self.s.LOG_DIR, self.s.LOG_LEVEL)
        self.s.ensure_dirs()
        self._started_at = self.calendar.now()
        logger.info("=" * 78)
        logger.info(" KIS AI Quant Trader — 제16회 뱅키스 대학생 모의투자대회 (AI 부문)")
        logger.info(f" {self.s.describe()}")
        logger.info("=" * 78)

        self.broker = create_broker(self.s)
        await self.broker.connect()

        health = await self.broker.health_check()
        logger.info(f"[Health] {json.dumps(health, ensure_ascii=False, default=str)}")
        if not health.get("ok"):
            logger.error("[Health] 연결 점검 실패 — .env의 키/계좌번호와 네트워크를 확인하세요.")
            if not self.s.USE_MOCK_MODE:
                return False

        self.strategy.load_model()

        if not await self._build_universe():
            logger.error("[Startup] 유니버스 구성 실패")
            return False

        await self._sync_portfolio(initial=True)
        self._restore_state()

        if not self.s.USE_MOCK_MODE:
            await self._start_websocket()

        self.bus.subscribe(TOPIC_FILL, self._on_fill_event)
        self.bus.start()

        if self.dashboard:
            await self.dashboard.start()

        self._running = True
        logger.success(
            f"[Startup] 준비 완료 | 유니버스 {len(self.universe)}종목 | "
            f"자산 {self.portfolio.equity:,.0f}원"
        )
        return True

    async def _build_universe(self) -> bool:
        """일봉 적재 후 유동성 기준으로 매매 대상을 추린다.

        모의투자 도메인은 순위/조건검색 API가 제한적이므로, 사전 정의된 풀에서
        20일 평균 거래대금으로 랭킹해 상위 종목을 선택한다.
        """
        candidates = list(self.s.UNIVERSE)
        logger.info(f"[Universe] 후보 {len(candidates)}종목 일봉 적재 중...")
        loaded: Dict[str, pd.DataFrame] = {}
        for symbol in candidates:
            try:
                df = await self.broker.get_bars(symbol, count=300, period="D")
                if df is None or len(df) < self.strategy.warmup_bars():
                    continue
                loaded[symbol] = df
            except Exception as exc:
                logger.warning(f"[Universe] {symbol} 적재 실패: {exc}")

        if not loaded:
            return False

        scored = []
        for symbol, df in loaded.items():
            value = float((df["close"] * df["volume"]).tail(20).mean())
            if value < self.s.MIN_TRADING_VALUE:
                continue
            scored.append((symbol, value))
        if not scored:  # 유동성 기준이 너무 빡빡하면 전체 사용
            scored = [(s, float((df["close"] * df["volume"]).tail(20).mean()))
                      for s, df in loaded.items()]
        scored.sort(key=lambda kv: -kv[1])

        self.universe = [s for s, _ in scored[: self.s.UNIVERSE_SIZE]]
        self.panel = {s: loaded[s] for s in self.universe}
        # 초기 관심목록은 유동성 상위로 채운다. 첫 신호가 나오면 점수 기준으로 바뀐다.
        self.watchlist = self.universe[: self.s.WS_MAX_SUBSCRIPTIONS]
        for s, df in self.panel.items():
            self.latest_prices[s] = float(df["close"].iloc[-1])
        self._last_bar_ts = str(self.panel[self.universe[0]]["ts"].iloc[-1])
        logger.success(f"[Universe] 확정 {len(self.universe)}종목: {', '.join(self.universe[:10])}"
                       + (" ..." if len(self.universe) > 10 else ""))
        return True

    async def _start_websocket(self) -> None:
        try:
            from core.websocket_client import KISWebSocketClient

            token_manager = getattr(self.broker, "auth", None)
            self.ws = KISWebSocketClient(
                self.s,
                token_manager=token_manager,
                on_trade=self._on_ws_quote,
                on_notice=self._on_ws_notice,
            )
            await self.ws.subscribe_symbols(self.watchlist, orderbook=False)
            await self.ws.subscribe_notice()
            self.ws.start()
        except Exception as exc:
            logger.warning(f"[WS] 초기화 실패 — REST 폴링으로 동작합니다: {exc}")
            self.ws = None

    # ================================================================== 이벤트 핸들러
    async def _on_ws_quote(self, quote: Quote) -> None:
        if quote.price > 0:
            self.latest_prices[quote.symbol] = quote.price

    async def _on_ws_notice(self, notice: Dict[str, Any]) -> None:
        """실시간 체결통보 → 즉시 포트폴리오 반영(폴링보다 빠름)."""
        try:
            if str(notice.get("cntg_yn")) != "2":  # 2 = 체결
                return
            qty = int(notice.get("cntg_qty") or 0)
            price = float(notice.get("cntg_unpr") or 0)
            if qty <= 0 or price <= 0:
                return
            side = Side.SELL if str(notice.get("seln_byov_cls")) == "01" else Side.BUY
            fee = qty * price * self.s.FEE_BPS / 10_000.0
            tax = qty * price * self.s.TAX_BPS / 10_000.0 if side is Side.SELL else 0.0
            fill = Fill(
                symbol=str(notice.get("stck_shrn_iscd", "")).zfill(6),
                side=side, qty=qty, price=price, fee=fee, tax=tax,
                order_no=str(notice.get("oder_no", "")), reason="ws_notice",
            )
            await self.bus.publish(TOPIC_FILL, fill)
        except Exception as exc:
            logger.warning(f"[WS] 체결통보 처리 실패: {exc}")

    async def _on_fill_event(self, fill: Fill) -> None:
        pos = self.portfolio.positions.get(fill.symbol)
        entry = pos.avg_price if pos else 0.0
        realized = self.portfolio.apply_fill(fill, sector=self.s.sector_of(fill.symbol))
        if fill.side is Side.SELL and entry > 0:
            self.risk.on_fill(fill, (fill.price - entry) / entry)

        # 손절선은 **체결 후 평단가 기준**으로 확정한다. 주문 접수 시점에는 포지션이
        # 아직 없으므로 여기서 설정하지 않으면 손절선이 비어버린다(부분체결 포함).
        if fill.side is Side.BUY:
            pos = self.portfolio.positions.get(fill.symbol)
            if pos is not None and pos.qty > 0:
                atr = self.atr_map.get(fill.symbol) or pos.avg_price * 0.02
                new_stop = self.risk.initial_stop(pos.avg_price, atr)
                pos.stop_price = max(pos.stop_price, new_stop) if pos.stop_price > 0 else new_stop
        record = fill.to_dict()
        record["ts"] = fill.ts.isoformat()
        record["realized"] = round(realized, 0)
        record["equity"] = round(self.portfolio.equity, 0)
        self.recent_fills.append(record)
        self.recent_fills = self.recent_fills[-50:]
        self.ledger.write("fill", record)
        logger.success(
            f"[체결] {fill.side.value} {fill.symbol} {fill.qty}주 @{fill.price:,.0f} "
            f"실현손익={realized:,.0f} 자산={self.portfolio.equity:,.0f}"
        )

    # ================================================================== 데이터
    async def _refresh_quotes(self) -> None:
        """현재가 갱신. 유량 보호를 위해 루프당 일부 종목만 회전 조회한다."""
        watch = self.watchlist or self.universe
        if not watch:
            return
        # WebSocket이 건강하면 REST 조회를 생략(유량 절약)
        if self.ws is not None and self.ws.is_healthy():
            return
        n = len(watch)
        chunk = min(QUOTE_CHUNK, n)
        for _ in range(chunk):
            symbol = watch[self._quote_cursor % n]
            self._quote_cursor += 1
            try:
                quote = await self.broker.get_quote(symbol)
                if quote and quote.price > 0:
                    self.latest_prices[symbol] = quote.price
            except Exception as exc:
                logger.debug(f"[Quote] {symbol} 조회 실패: {exc}")

    def _apply_latest_prices_to_panel(self) -> None:
        """마지막 봉의 종가/고가/저가를 실시간 가격으로 갱신(당일 봉 형성)."""
        for symbol, price in self.latest_prices.items():
            df = self.panel.get(symbol)
            if df is None or df.empty or price <= 0:
                continue
            idx = df.index[-1]
            df.at[idx, "close"] = price
            df.at[idx, "high"] = max(float(df.at[idx, "high"]), price)
            df.at[idx, "low"] = min(float(df.at[idx, "low"]), price)

    async def _refresh_bars(self) -> None:
        """일봉 재적재(신규 봉 확정 반영)."""
        for symbol in self.universe:
            try:
                df = await self.broker.get_bars(symbol, count=300, period="D")
                if df is not None and len(df) >= self.strategy.warmup_bars():
                    self.panel[symbol] = df
            except Exception as exc:
                logger.debug(f"[Bars] {symbol} 갱신 실패: {exc}")
        if self.universe and self.universe[0] in self.panel:
            new_ts = str(self.panel[self.universe[0]]["ts"].iloc[-1])
            if new_ts != self._last_bar_ts:
                self._last_bar_ts = new_ts
                logger.info(f"[Bars] 신규 봉 확정: {new_ts}")

    def _refresh_bars_from_sim(self) -> None:
        """Mock 전용 빠른 경로: 시뮬레이터에서 직접 확정 봉을 가져온다.

        REST 경로는 기동 시 `_build_universe()`에서 이미 검증되므로, 루프 중에는
        유량을 주문/시세에 양보한다.
        """
        sim = getattr(self.broker, "sim", None)
        if sim is None:
            return
        for symbol in self.universe:
            df = sim.history(symbol, count=300)
            if df is not None and len(df) >= self.strategy.warmup_bars():
                self.panel[symbol] = df
        if self.universe and self.universe[0] in self.panel:
            new_ts = str(self.panel[self.universe[0]]["ts"].iloc[-1])
            if new_ts != self._last_bar_ts:
                self._last_bar_ts = new_ts

    async def _sync_portfolio(self, initial: bool = False) -> None:
        """브로커 잔고와 내부 포트폴리오를 대조한다(원장 불일치 방지)."""
        try:
            cash, positions = await self.broker.get_portfolio_state()
        except Exception as exc:
            logger.warning(f"[Sync] 잔고 조회 실패: {exc}")
            return

        if initial:
            self.portfolio.cash = cash
            for symbol, pos in positions.items():
                pos.sector = self.s.sector_of(symbol)
                pos.stop_price = self.risk.initial_stop(pos.avg_price, pos.avg_price * 0.02)
                self.portfolio.positions[symbol] = pos
            equity = self.portfolio.equity
            if equity > 0:
                self.portfolio.initial_capital = equity if not positions else self.portfolio.initial_capital
                self.portfolio.peak_equity = max(self.portfolio.peak_equity, equity)
                self.portfolio.day_start_equity = equity
            logger.info(f"[Sync] 초기 동기화: 현금 {cash:,.0f}원 / 보유 {len(positions)}종목")
            return

        # 수량 불일치 시 브로커 값을 신뢰(체결 누락 보정)
        for symbol, remote in positions.items():
            local = self.portfolio.positions.get(symbol)
            if local is None:
                remote.sector = self.s.sector_of(symbol)
                remote.stop_price = self.risk.initial_stop(remote.avg_price, remote.avg_price * 0.02)
                self.portfolio.positions[symbol] = remote
                logger.warning(f"[Sync] 미추적 포지션 발견 → 편입: {symbol} {remote.qty}주")
            elif local.qty != remote.qty:
                logger.warning(f"[Sync] 수량 불일치 {symbol}: 내부 {local.qty} → 브로커 {remote.qty}")
                local.qty = remote.qty
                local.avg_price = remote.avg_price or local.avg_price
        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in positions:
                logger.warning(f"[Sync] 브로커에 없는 포지션 제거: {symbol}")
                self.portfolio.positions.pop(symbol, None)
        self.portfolio.cash = cash

    # ================================================================== 주문
    async def _submit(self, symbol: str, side: Side, qty: int, price: float, reason: str) -> bool:
        if qty <= 0:
            return False
        limit_price = round_to_tick(price * (1.004 if side is Side.BUY else 0.996))
        request = OrderRequest(symbol=symbol, side=side, qty=qty, price=float(limit_price),
                               order_type=OrderType.LIMIT, reason=reason)
        self.ledger.write("order", {
            "symbol": symbol, "side": side.value, "qty": qty, "price": limit_price, "reason": reason,
        })
        try:
            ack = await self.broker.submit_order(request)
        except Exception as exc:
            logger.error(f"[주문] 전송 예외 {symbol}: {exc}")
            return False

        if not ack.success:
            logger.error(f"[주문거부] {side.value} {symbol} {qty}주 :: {ack.msg_cd} {ack.msg}")
            self.ledger.write("order_reject", {"symbol": symbol, "side": side.value,
                                               "qty": qty, "msg": ack.msg, "code": ack.msg_cd})
            return False

        self.open_orders[ack.order_no] = {
            "symbol": symbol, "side": side.value, "qty": qty, "price": limit_price,
            "org_no": ack.org_no, "ts": self._now().isoformat(), "reason": reason,
        }
        await self.bus.publish(TOPIC_ORDER, request)
        logger.info(f"[주문접수] {side.value} {symbol} {qty}주 @{limit_price:,} ({reason})")
        return True

    async def _poll_fills(self) -> None:
        try:
            fills = await self.broker.poll_fills()
        except Exception as exc:
            logger.debug(f"[Fill] 폴링 실패: {exc}")
            return
        for fill in fills:
            # 부분체결이면 잔량이 남아 있으므로 추적을 유지한다.
            # (여기서 바로 제거하면 같은 종목에 중복 진입 주문이 나간다)
            tracked = self.open_orders.get(fill.order_no)
            if tracked is not None:
                # 체결 사유는 주문 시점에만 알 수 있으므로 추적 정보에서 복원한다
                if not fill.reason or fill.reason == "poll":
                    fill.reason = tracked.get("reason", "")
                tracked["qty"] = max(0, int(tracked.get("qty", 0)) - fill.qty)
                if tracked["qty"] <= 0:
                    self.open_orders.pop(fill.order_no, None)
            await self.bus.publish(TOPIC_FILL, fill)

    async def _sync_open_orders(self, max_age_seconds: float = 180.0) -> None:
        """브로커의 미체결 잔량과 내부 추적을 일치시키고, 오래된 주문은 취소한다.

        호가에서 멀어진 지정가를 방치하면 자금이 묶여 다음 기회를 놓치고,
        내부 추적이 남아 있으면 해당 종목에 영영 재진입하지 못한다.
        """
        try:
            payload = await self.broker.inquire_open_orders()
        except Exception as exc:
            logger.debug(f"[Orders] 미체결 조회 실패: {exc}")
            return
        if not is_ok(payload):
            return

        now = self._now()
        live: Dict[str, Dict[str, Any]] = {}
        for row in payload.get("output1") or []:
            order_no = str(row.get("odno", ""))
            remaining = to_int(row.get("rmn_qty"))
            if not order_no or remaining <= 0:
                continue
            prev = self.open_orders.get(order_no, {})
            org_no = prev.get("org_no") or str(row.get("ord_gno_brno") or row.get("orgn_odno") or "")
            placed = prev.get("ts")
            age = 0.0
            if placed:
                try:
                    age = (now - dt.datetime.fromisoformat(placed)).total_seconds()
                except ValueError:
                    age = 0.0
            if age > max_age_seconds:
                try:
                    await self.broker.cancel_order(org_no, order_no, all_qty=True)
                    logger.info(f"[Orders] 장기 미체결 취소: {prev.get('symbol', '')} {order_no} "
                                f"({age:.0f}s 경과)")
                    continue
                except Exception as exc:
                    logger.debug(f"[Orders] 취소 실패 {order_no}: {exc}")
            live[order_no] = {
                **prev,
                "symbol": str(row.get("pdno", "")).zfill(6) or prev.get("symbol", ""),
                "qty": remaining,
                "org_no": org_no,
                "ts": placed or now.isoformat(),
            }
        self.open_orders = live

    # ================================================================== 메인 루프
    async def _tick(self) -> None:
        now = self._now()
        session = self._session()

        # --- Mock 가상 시계 전진 ---
        if self.s.USE_MOCK_MODE and getattr(self.broker, "is_mock", False):
            bar_changed = self.broker.advance()  # type: ignore[attr-defined]
            self.broker.expire_stale_orders(max_age_seconds=120)  # type: ignore[attr-defined]
            if bar_changed:
                self._refresh_bars_from_sim()
            for symbol in self.universe:
                price = self.broker.sim.current_price(symbol)  # type: ignore[attr-defined]
                if price > 0:
                    self.latest_prices[symbol] = price
            if self.broker.sim.exhausted:  # type: ignore[attr-defined]
                logger.warning("[Engine] Mock 데이터 소진 → 루프를 종료합니다.")
                self._stop.set()
                return

        # --- 장 상태 체크 ---
        if session is not MarketSession.REGULAR:
            if self._loops % 20 == 0:
                logger.info(f"[Engine] 장 상태 {session.value} — 대기 중 "
                            f"(다음 개장까지 {self.calendar.seconds_until_next_open() / 60:.0f}분)")
            if session is MarketSession.CLOSED and self.portfolio.open_positions():
                # 장 종료 후에는 평가만 갱신
                self.portfolio.mark_prices(self.latest_prices)
            return

        # --- 시세 갱신 ---
        await self._refresh_quotes()
        if self._loops % BAR_REFRESH_LOOPS == 0 and self._loops > 0 and not self.s.USE_MOCK_MODE:
            await self._refresh_bars()
        self._apply_latest_prices_to_panel()

        # --- 평가 및 리스크 상태 ---
        prices = {s: p for s, p in self.latest_prices.items() if p > 0}
        self.portfolio.mark_prices(prices)
        for symbol, price in prices.items():
            self.risk.track_price(symbol, price)
        self.equity_history.append({
            "t": now.strftime("%H:%M:%S"),
            "equity": round(self.portfolio.equity, 0),
            "dd": round(self.portfolio.drawdown * 100, 3),
            "exposure": round(self.portfolio.gross_exposure * 100, 2),
        })
        limiter = getattr(self.broker, "limiter", None)
        if limiter is not None:
            self.rps_history.append(round(limiter.stats.measured_rps(), 2))

        prev_state = self.risk.state
        # --- 대회 국면 적용 (날짜만 보고 전환한다) ---
        self.phase = phase_config(settings=self.s, now=now)
        self.risk.set_phase(self.phase if self.phase.is_competition else None)
        if self.phase.phase != self._last_phase:
            logger.warning(f"[대회] 국면 전환 {self._last_phase} → {self.phase.phase} :: {self.phase.label}")
            self.ledger.write("phase", {"from": self._last_phase, "to": self.phase.phase,
                                        "top_k": self.phase.top_k,
                                        "use_stop_loss": self.phase.use_stop_loss,
                                        "use_mdd_gate": self.phase.use_mdd_gate})
            self._last_phase = self.phase.phase

        if self.phase.is_competition:
            self._check_reset_window(now)
            self._check_compliance(now)

        self.risk.update(self.portfolio, now)
        if prev_state != self.risk.state:
            self.ledger.write("risk_state", {"from": prev_state, "to": self.risk.state,
                                             "equity": self.portfolio.equity,
                                             "drawdown": self.portfolio.drawdown})

        # --- 신호 (봉이 바뀔 때만 재계산) ---
        signals = self._current_signals(now)
        if signals:
            await self.bus.publish(TOPIC_SIGNAL, signals[0])

        # --- 청산(리스크 우선) ---
        exits = self.risk.evaluate_exits(self.portfolio, prices, self.atr_map, self.score_map, now)
        if self.s.USE_TOP_K:
            exits.extend(self._rotation_exits(signals, prices, exits))
        for order in exits:
            price = prices.get(order.symbol, order.price)
            ok = await self._submit(order.symbol, Side.SELL, order.qty, price,
                                    f"{order.reason.value}|{order.detail}")
            if ok:
                self.ledger.write("exit", {"symbol": order.symbol, "qty": order.qty,
                                           "reason": order.reason.value, "detail": order.detail})

        # --- 진입 ---
        if self.risk.state == RiskState.NORMAL:
            # 백테스트(`backtest/engine.py`)와 반드시 같은 선정 로직을 쓴다.
            # 두 경로가 갈라지면 백테스트는 실거래를 예측하지 못한다.
            if self.s.USE_TOP_K:
                candidates = select_entries(
                    signals, self.portfolio, self.s,
                    top_k=self._top_k(),
                    regime_scale=self.strategy.regime.scale,
                    pending=set(self.open_orders_symbols()))
            else:
                candidates = [x for x in signals
                              if x.score >= self.s.SIGNAL_BUY_THRESHOLD]
            for sig in candidates:
                if sig.symbol in self.open_orders_symbols():
                    continue
                decision = self.risk.evaluate_entry(sig, self.portfolio, now, pending=self._pending_notional())
                self.ledger.write("risk_decision", {
                    "symbol": sig.symbol, "score": round(sig.score, 4),
                    "approved": decision.approved, "qty": decision.qty,
                    "reason": decision.reason, **decision.detail,
                })
                if not decision.approved:
                    continue
                price = prices.get(sig.symbol, sig.price)
                await self._submit(sig.symbol, Side.BUY, decision.qty, price,
                                   f"ENTRY|score={sig.score:.3f}")

        # --- 체결 확인 ---
        if self.s.USE_MOCK_MODE or self._loops % FILL_POLL_LOOPS == 0:
            await self._poll_fills()

        # --- 잔고/미체결 대조 ---
        if self._loops % RECONCILE_LOOPS == 0 and self._loops > 0:
            await self._sync_portfolio()
            await self._sync_open_orders()

        # --- 상태 저장 ---
        if self._loops % 5 == 0:
            self._save_state()

        if self._loops % 20 == 0:
            logger.info(
                f"[Engine] loop={self._loops} equity={self.portfolio.equity:,.0f} "
                f"({self.portfolio.total_return:+.2%}) dd={self.portfolio.drawdown:.2%} "
                f"pos={len(self.portfolio.open_positions())} risk={self.risk.state} "
                f"regime={self.strategy.regime.label}"
            )

    def _rotation_exits(self, signals, prices, already) -> List[ExitOrder]:
        """순위에서 밀려난 보유 종목을 교체 청산한다(백테스트와 동일 규칙).

        손절·급락과 달리 "더 좋은 후보가 나타났다"는 이유의 청산이다. 이미 다른
        사유로 청산 예정인 종목은 중복해서 팔지 않는다.
        """
        taken = {order.symbol for order in already}
        out: List[ExitOrder] = []
        for symbol in select_rotation_exits(
                signals, self.portfolio, self.s,
                top_k=self._top_k(),
                regime_scale=self.strategy.regime.scale):
            if symbol in taken or symbol in self.open_orders_symbols():
                continue
            pos = self.portfolio.positions.get(symbol)
            if pos is None or pos.qty <= 0:
                continue
            out.append(ExitOrder(symbol, pos.qty, ExitReason.SIGNAL,
                                 prices.get(symbol, pos.last_price),
                                 "순위 이탈 — 더 좋은 후보로 교체"))
        return out

    def _top_k(self) -> int:
        """이번 봉의 목표 보유 종목 수. 대회 기간이면 국면 값을 쓴다."""
        phase = getattr(self, "phase", None)
        if phase is not None and phase.is_competition:
            return phase.top_k
        return int(self.s.MAX_POSITIONS)

    def _check_reset_window(self, now) -> None:
        """수익률 초기화 권고를 로그와 대시보드에 띄운다.

        **실행하지 않는다.** 되돌릴 수 없는 1회성 계좌 설정 변경이므로 사용자가
        KIS 화면에서 직접 수행한다.
        """
        advice = reset_recommendation(self.portfolio, now, self.s)
        self.reset_advice = advice
        if not advice.window_open:
            return
        if advice.should_reset:
            logger.error(f"[대회·초기화] {advice.reason} — KIS 화면에서 직접 초기화하십시오")
        else:
            logger.info(f"[대회·초기화] {advice.reason}")
        self.ledger.write("reset_advice", {
            "should_reset": advice.should_reset, "total_return": advice.total_return,
            "deadline": advice.deadline, "reason": advice.reason})

    def _check_compliance(self, now) -> None:
        """대회 최소 요건 충족 현황을 갱신한다.

        요건을 놓치면 수익률과 무관하게 실격이므로, 빠듯해지면 회전율 통제를
        완화해 거래를 늘린다. 거래비용보다 실격이 비싸다.
        """
        from strategy.compliance import holding_bars_override, status_from_ledger

        try:
            self.compliance = status_from_ledger(
                Path(self.s.LOG_DIR) / "ledger.jsonl", self.s, now=now)
        except Exception as exc:
            logger.warning(f"[대회·요건] 집계 실패: {exc}")
            return
        override = holding_bars_override(self.compliance, self.s)
        if override is not None and override != self.s.MIN_HOLDING_BARS:
            logger.warning(
                f"[대회·요건] 미충족 항목이 있어 최소 보유기간을 "
                f"{self.s.MIN_HOLDING_BARS} → {override}봉으로 완화합니다")
            self.s.MIN_HOLDING_BARS = override

    def _current_signals(self, now) -> List[Signal]:
        """이번 루프에서 쓸 신호. **일봉이 바뀔 때만** 다시 계산한다.

        ## 왜 캐시가 필요한가

        이 전략은 일봉으로 판단한다. 그런데 루프는 `LOOP_INTERVAL`(3초)마다 돌고,
        예전에는 매 루프 `generate_signals()`를 호출했다. 일봉은 장중에 바뀌지 않으므로
        같은 계산을 반복한 것이다.

        실측: 20종목 **5.63초**, 116종목 **17.83초**. 루프 간격 3초를 이미 넘겼고
        유니버스를 넓히면 더 심해진다. 장중에 실제로 필요한 것은 손절·트레일링
        판단용 **현재가**이고, 그것은 `latest_prices`가 따로 갱신한다.

        그래서 `_last_bar_ts`가 달라졌을 때만 재계산한다. 봉 재적재는
        `BAR_REFRESH_LOOPS` 주기로 일어나므로 하루 1회 수준이다.
        """
        if self._signals and self._signals_bar_ts == self._last_bar_ts:
            return self._signals
        try:
            signals = self.strategy.generate_signals(
                self.panel, self.portfolio.open_positions(), now)
        except Exception as exc:
            self._errors += 1
            logger.exception(f"[Engine] 신호 생성 실패: {exc}")
            return self._signals          # 직전 신호를 유지한다(빈 목록으로 떨어뜨리지 않는다)
        self._signals = signals
        self._signals_bar_ts = self._last_bar_ts
        self.score_map = {x.symbol: x.score for x in signals}
        self.atr_map = {x.symbol: x.atr for x in signals}
        if signals:
            self._refresh_watchlist(signals)
            logger.info(f"[Signal] {len(signals)}종목 재계산 (봉 {self._last_bar_ts}) · "
                        f"상위 {', '.join(f'{x.symbol}:{x.score:.2f}' for x in signals[:3])}")
        return signals

    def _refresh_watchlist(self, signals) -> None:
        """실시간 시세를 받을 종목을 점수 기준으로 갱신한다.

        ## 왜 유니버스 전체를 구독하지 않는가

        KIS WebSocket 구독 한도가 41건(체결+호가 합산)이다. 유니버스가 284종목이면
        전체 구독이 불가능하다. 반면 **신호 계산은 일봉만 쓰고 일봉은 하루 1회
        적재**하므로, 실시간 시세가 필요한 것은 실제로 거래할 종목뿐이다.

        순서는 보유 종목이 먼저다. 손절·트레일링 판단이 현재가에 달려 있어서
        보유 종목의 시세가 밀리면 방어가 늦어진다. 남는 자리를 점수 상위 후보로 채운다.

        구독 대상이 바뀌면 재구독한다. 매 루프 재구독하면 유량을 낭비하므로
        **집합이 실제로 달라졌을 때만** 한다.
        """
        cap = max(1, int(self.s.WS_MAX_SUBSCRIPTIONS))
        held = [c for c in self.portfolio.open_positions()]
        ranked = [x.symbol for x in signals if x.symbol not in held]
        new_watch = (held + ranked)[:cap]
        if set(new_watch) == set(self.watchlist):
            return
        self.watchlist = new_watch
        if self.ws is not None:
            try:
                asyncio.create_task(self.ws.subscribe_symbols(new_watch, orderbook=False))
            except Exception as exc:
                logger.debug(f"[WS] 관심목록 재구독 실패: {exc}")

    def open_orders_symbols(self) -> set:
        return {o["symbol"] for o in self.open_orders.values()}

    def _pending_notional(self) -> Dict[str, float]:
        """미체결 매수 주문의 평가금액. 리스크 한도가 체결 지연 동안 뚫리지 않게 한다."""
        out: Dict[str, float] = {}
        for o in self.open_orders.values():
            if o.get("side") != Side.BUY.value:
                continue
            out[o["symbol"]] = out.get(o["symbol"], 0.0) + float(o.get("qty", 0)) * float(o.get("price", 0))
        return out

    async def run(self, duration: Optional[float] = None) -> None:
        if not await self.startup():
            return
        deadline = None if duration is None else (asyncio.get_running_loop().time() + duration)
        backoff = 1.0

        try:
            while not self._stop.is_set():
                start = asyncio.get_running_loop().time()
                try:
                    await self._tick()
                    self._loops += 1
                    backoff = 1.0
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    self._errors += 1
                    logger.exception(f"[Engine] 루프 오류({self._errors}회): {exc}")
                    self.ledger.write("error", {"error": str(exc), "trace": traceback.format_exc()[-2000:]})
                    await asyncio.sleep(min(30.0, backoff))
                    backoff *= 2

                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    logger.info("[Engine] 지정 구동 시간 종료")
                    break

                elapsed = asyncio.get_running_loop().time() - start
                try:
                    await asyncio.wait_for(self._stop.wait(),
                                           timeout=max(0.05, self.s.LOOP_INTERVAL - elapsed))
                    break
                except asyncio.TimeoutError:
                    continue
        finally:
            await self.shutdown()

    # ================================================================== 종료
    async def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        logger.warning("[Shutdown] 종료 시퀀스를 시작합니다...")

        # 1) 미체결 주문 취소
        try:
            if hasattr(self.broker, "cancel_all_open_orders"):
                await self.broker.cancel_all_open_orders()  # type: ignore[attr-defined]
            else:
                for order_no, info in list(self.open_orders.items()):
                    await self.broker.cancel_order(info.get("org_no", ""), order_no, all_qty=True)
        except Exception as exc:
            logger.warning(f"[Shutdown] 주문 취소 실패: {exc}")

        # 2) 잔여 체결 수집 및 상태 저장
        try:
            await self._poll_fills()
        except Exception:
            pass
        self._save_state()

        # 3) 리소스 정리 (하나가 실패해도 나머지는 계속 정리한다)
        async def _safe(coro_factory, label: str) -> None:
            try:
                await coro_factory()
            except Exception as exc:
                logger.debug(f"[Shutdown] {label} 정리 중 예외(무시): {exc}")

        if self.ws:
            await _safe(self.ws.stop, "websocket")
        if self.dashboard:
            await _safe(self.dashboard.stop, "dashboard")
        await _safe(self.bus.stop, "eventbus")
        if self.broker:
            await _safe(self.broker.close, "broker")

        snap = self.portfolio.snapshot()
        logger.info("=" * 78)
        logger.info(f" 최종 자산   : {snap['equity']:,.0f}원 ({snap['total_return_pct']:+.2f}%)")
        logger.info(f" 최대 낙폭   : {snap['drawdown_pct']:.2f}%")
        logger.info(f" 청산 거래   : {snap['trade_count']}회 (승률 {snap['win_rate']:.1f}%)")
        logger.info(f" 총 비용     : {snap['total_fees']:,.0f}원 | 루프 {self._loops}회 | 오류 {self._errors}회")
        logger.info("=" * 78)

    # ================================================================== 상태
    def _state_path(self) -> Path:
        return Path(self.s.STATE_DIR) / "engine_state.json"

    def _save_state(self) -> None:
        try:
            payload = {
                "saved_at": self.calendar.now().isoformat(),
                "mode": "MOCK" if self.s.USE_MOCK_MODE else self.s.KIS_ENV,
                "portfolio": self.portfolio.snapshot(),
                "risk": self.risk.snapshot(self.portfolio),
                "universe": self.universe,
                "loops": self._loops,
                "peak_equity": self.portfolio.peak_equity,
            }
            self._state_path().write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.debug(f"[State] 저장 실패: {exc}")

    def _restore_state(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            peak = float(data.get("peak_equity") or 0)
            if peak > 0:
                self.portfolio.peak_equity = max(self.portfolio.peak_equity, peak)
            logger.info(f"[State] 이전 상태 복구 (peak_equity={self.portfolio.peak_equity:,.0f})")
        except Exception as exc:
            logger.debug(f"[State] 복구 실패: {exc}")

    def _progress(self) -> Dict[str, Any]:
        """Mock 시뮬레이션 진행률. 실서버에서는 장 마감까지 남은 시간을 쓴다."""
        sim = getattr(self.broker, "sim", None)
        if sim is not None:
            span = max(1, sim.max_cursor - sim.warmup)
            done = max(0, sim.cursor - sim.warmup)
            return {"kind": "mock", "done": done, "total": span,
                    "pct": round(min(100.0, done / span * 100), 1),
                    "label": f"{done}/{span}봉"}
        left = self.calendar.minutes_to_close()
        total = 390.0  # 09:00~15:30
        return {"kind": "session", "done": round(total - left), "total": round(total),
                "pct": round(min(100.0, max(0.0, (total - left) / total * 100)), 1),
                "label": f"마감까지 {left:.0f}분"}

    def _competition_snapshot(self) -> Dict[str, Any]:
        """대회 국면·초기화 권고·최소 요건 현황.

        대회 기간이 아니면 `active=False`만 돌려준다. 대시보드는 그때 이 영역을 숨긴다.

        **초기화 권고가 화면에 떠야 하는 이유**: 수익률 초기화는 10/12~10/16 중
        단 1회이고 코드가 대신 실행하지 않는다(되돌릴 수 없는 계좌 설정 변경이다).
        마지막 날을 놓치면 기회가 사라지므로 사람이 볼 수 있어야 한다.
        """
        from strategy.phase import trading_days_remaining

        phase = getattr(self, "phase", None)
        if phase is None or not phase.is_competition:
            return {"active": False}

        advice = getattr(self, "reset_advice", None)
        comp = getattr(self, "compliance", None)
        return {
            "active": True,
            "phase": phase.phase,
            "label": phase.label,
            "top_k": phase.top_k,
            "use_stop_loss": phase.use_stop_loss,
            "use_mdd_gate": phase.use_mdd_gate,
            "days_remaining": trading_days_remaining(self._now(), self.s),
            "period": f"{self.s.COMP_START} ~ {self.s.COMP_END}",
            "reset": ({
                "window_open": advice.window_open,
                "should_reset": advice.should_reset,
                "deadline": advice.deadline,
                "total_return_pct": round(advice.total_return * 100, 2),
                "reason": advice.reason,
            } if advice is not None else None),
            "requirements": comp.to_dict() if comp is not None else None,
        }

    def state_snapshot(self) -> Dict[str, Any]:
        """대시보드/헬스체크용 통합 스냅샷."""
        limiter = getattr(self.broker, "limiter", None)
        auth = getattr(self.broker, "auth", None)
        return {
            "competition": self._competition_snapshot(),
            "portfolio": self.portfolio.snapshot(),
            "risk": self.risk.snapshot(self.portfolio),
            "strategy": self.strategy.snapshot(),
            "recent_fills": self.recent_fills[-20:],
            "history": list(self.equity_history),
            "rps_history": list(self.rps_history),
            "system": {
                "running": self._running,
                "mode": "MOCK(가상체결)" if self.s.USE_MOCK_MODE
                        else ("실전" if self.s.is_prod else "KIS 모의투자"),
                "session": self._session().value,
                "loops": self._loops,
                "errors": self._errors,
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "universe": len(self.universe),
                "watchlist": len(self.watchlist),
                "open_orders": len(self.open_orders),
                "event_queue": self.bus.pending,
                "elapsed_sec": round((self.calendar.now() - self._started_at).total_seconds())
                               if self._started_at else 0,
                "progress": self._progress(),
                "rate_limiter": limiter.snapshot() if limiter else {},
                "websocket": self.ws.snapshot() if self.ws else {"connected": False},
                "auth": auth.snapshot() if auth else {},
            },
        }


# ====================================================================== CLI
async def run_health_check(settings: Settings) -> int:
    setup_logging(settings.LOG_DIR, settings.LOG_LEVEL)
    logger.info(f"[HealthCheck] {settings.describe()}")
    broker = create_broker(settings)
    try:
        await broker.connect()
        result = await broker.health_check()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if result.get("ok"):
            logger.success("[HealthCheck] 모든 점검을 통과했습니다.")
            return 0
        logger.error("[HealthCheck] 점검 실패 — 위 상세 결과를 확인하세요.")
        return 1
    except Exception as exc:
        logger.error(f"[HealthCheck] 예외: {exc}")
        return 1
    finally:
        await broker.close()


def _install_signal_handlers(engine: "TradingEngine") -> None:
    loop = asyncio.get_running_loop()

    def _handler() -> None:
        logger.warning("[Signal] 종료 신호 수신 — graceful shutdown")
        engine._stop.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _handler)
        except (NotImplementedError, RuntimeError):
            # Windows의 ProactorEventLoop는 add_signal_handler를 지원하지 않는다
            signal.signal(sig, lambda *_: _handler())


async def amain(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.mock is not None:
        settings.USE_MOCK_MODE = args.mock
    if args.health_only:
        return await run_health_check(settings)

    engine = TradingEngine(settings, enable_dashboard=not args.no_dashboard)
    _install_signal_handlers(engine)
    await engine.run(duration=args.duration)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="KIS AI Quant Trader")
    parser.add_argument("--health-only", action="store_true", help="연결/인증/계좌 점검만 수행")
    parser.add_argument("--duration", type=float, default=None, help="구동 시간(초). 미지정 시 무한")
    parser.add_argument("--no-dashboard", action="store_true", help="대시보드 비활성화")
    parser.add_argument("--mock", dest="mock", action="store_true", default=None,
                        help="강제로 Mock 모드 실행")
    parser.add_argument("--live", dest="mock", action="store_false",
                        help="강제로 실서버 모드 실행")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        logger.warning("사용자 중단(KeyboardInterrupt)")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
