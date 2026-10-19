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

import numpy as np
import pandas as pd

from api.base import UNKNOWN_RESULT_CODE, AbstractBroker, is_ok, to_float, to_int
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
from strategy.exit_rules import daily_alive_burst, vwap_fade
from strategy.sliding import (LAST_ENTRY_MIN, SlidingWatcher, minute_turnover, rolling_sum,
                              same_time_base)
from strategy.opening import (minute_index, opening_ratio, rank_triggers, same_window_ratio,
                              window_turnover)
from strategy.eligibility import vts_tradable
from strategy.risk_manager import ExitOrder, RiskManager, RiskState
from strategy.selection import select_entries, select_rotation_exits
from strategy.intraday import IntradayWatcher, TriggerSignal
from strategy.phase import current_phase, phase_config, reset_recommendation
from utils.krx_calendar import KRXCalendar, MarketSession
from utils.logger import JsonLedger, logger, setup_logging
from utils.orderbook_log import OrderbookRecorder
from utils.market_rules import round_to_tick

BAR_REFRESH_LOOPS = 40      # 전체 일봉 재적재 주기(루프 수)
QUOTE_CHUNK = 4             # 루프당 현재가 갱신 종목 수(유량 보호)
RECONCILE_LOOPS = 30        # 브로커 잔고 대조 주기
DISK_WARN_GB = 1.0          # 남은 디스크가 이 아래면 경고(서버 디스크 10GB — 원장·상태 저장 실패 방지)
FLATTEN_ORDER_TTL = 30.0    # 마감 정리 구간의 미체결 매도 취소 기준(초)
OPENING_LATE_MAX_MIN = 2    # 장 시작 판정을 이만큼(분) 넘겨 기동했으면 건너뛴다(누적이 창보다 길어진다)
FILL_POLL_LOOPS = 5         # 체결 폴링 주기(실서버)
# 규칙 기반 진입(burst·opening)의 신호 점수. 진입 결정은 규칙이 이미 내렸으므로 리스크
# 관문의 일봉 모델용 문턱(SIGNAL_BUY_THRESHOLD)에 걸리지 않게 둔다. 고정 사이징에서는
# 점수가 비중에 쓰이지 않는다.
RULE_TRIGGER_SCORE = 1.0


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
        # 당일 **누적** 거래량(`Quote.volume` = KIS `acml_vol`). 거래량 감소
        # 청산(`EXIT_RULES`의 vol_decay)의 입력이다. 패널 마지막 봉의 거래량은
        # `_apply_latest_prices_to_panel()`이 갱신하지 않으므로(종가·고가·저가만
        # 덮는다) 여기에 따로 들고 간다.
        self.latest_volumes: Dict[str, float] = {}
        # 호가·체결 전방 수집. **이번 대회 전략에는 쓰이지 않는다** —
        # 호가는 과거 이력을 받을 수 없어 다음을 위해 지금부터 쌓는 것이다.
        self.obrec = (OrderbookRecorder(Path(self.s.DATA_DIR) / "cache",
                                        interval_sec=self.s.ORDERBOOK_SAMPLE_SEC)
                      if self.s.COLLECT_ORDERBOOK else None)
        self.atr_map: Dict[str, float] = {}
        self.vol_map: Dict[str, float] = {}
        # 대회 국면 — 날짜로 결정되며 기간 밖에서는 OFF(평소 동작)다.
        self.phase = phase_config(settings=self.s)
        self._last_phase = self.phase.phase
        self.reset_advice = None
        self.compliance = None
        self.score_map: Dict[str, float] = {}
        self.recent_fills: List[Dict[str, Any]] = []
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        # --- 장중 분봉 전략 (strategy/intraday.py) ---
        # `USE_INTRADAY`가 꺼져 있어도 생성한다. 조건 분기를 한 곳(`_feed_intraday`)에
        # 모아 두면 켜고 끌 때 다른 경로를 건드리지 않는다.
        self.intraday = IntradayWatcher(
            trigger_pct=self.s.INTRADAY_TRIGGER_PCT,
            vol_mult=self.s.INTRADAY_VOL_MULT,
            entry_mode=self.s.INTRADAY_ENTRY_MODE,
            burst_mult=self.s.INTRADAY_BURST_MULT,
            flatten=self.s.INTRADAY_FLATTEN,
            target_pct=self.s.INTRADAY_TARGET_PCT,
            stop_pct=self.s.INTRADAY_STOP_PCT,
            trail_pct=self.s.INTRADAY_TRAIL_PCT,
            trail_activate_pct=self.s.INTRADAY_TRAIL_ACTIVATE_PCT,
            flatten_hhmm=self.s.INTRADAY_FLATTEN_HHMM,
            candidate_ttl=self.s.INTRADAY_CANDIDATE_TTL,
            swap_margin=self.s.INTRADAY_SWAP_MARGIN,
            max_swaps=self.s.INTRADAY_MAX_SWAPS)
        self._intraday_queue: List[TriggerSignal] = []
        self._scanned_day: str = ""
        # 진입 시점 점수. 교체 판정에 쓰므로 체결 때까지 들고 있어야 한다.
        self._entry_scores: Dict[str, float] = {}
        # 아침 스캔은 158초가 걸려 루프를 막으므로 별도 태스크로 돌린다.
        self._scan_task: Optional[asyncio.Task] = None
        # 전일 종가·거래량. 패널은 실시간 가격으로 덮어써지므로 한 번만 떠서 보관한다.
        self._prev_daily: Dict[str, tuple] = {}
        # --- 다일 보유 청산(`vol_alive`) — 하루 한 번, 확정 일봉으로 판정한다 ---
        # 종목 → 어제 확정 거래대금 / 그 전일까지 20봉 중앙값(`daily_alive_burst`).
        # 장중 누적으로 판정하면 매일 아침 시가에 팔린다(함수 주석 참고).
        self._alive_burst: Dict[str, float] = {}
        # 종목 → 첫 매수 체결일(YYYYMMDD). 진입 당일은 판정하지 않으므로(폭증일이라
        # 당연히 생존) 반드시 알아야 하고, 재시작해도 남도록 상태 파일에 저장한다.
        self._entry_dates: Dict[str, str] = {}
        # 패널 밖 보유 종목의 일봉(유니버스 순위가 바뀌어 빠진 경우). 판정에만 쓴다.
        self._held_daily: Dict[str, pd.DataFrame] = {}
        # 전일 기준값(`_prev_daily`·`_alive_burst`)을 만든 날. 날짜가 바뀌면
        # `_roll_daily_reference()`가 일봉을 다시 받고 둘 다 새로 만든다.
        self._ref_day: str = ""
        # --- opening 모드 (장 시작 N분 거래대금) — 하루 한 번 판정, 한 번 진입 ---
        self._opening_day: str = ""                      # 판정을 시작한 날
        self._opening_task: Optional[asyncio.Task] = None
        self._opening_picks: Optional[List[tuple]] = None  # [(종목, 배수)] 배수 큰 순
        self._opening_record_task: Optional[asyncio.Task] = None  # 기준선 분봉 기록(진입 뒤)
        self._opening_done_day: str = ""                 # 진입을 처리한 날
        # 오늘 '매매불가'로 거부된 종목(종목 → 날짜). 같은 날 다시 주문하지 않는다
        self._reject_today: Dict[str, str] = {}
        # --- 장중 VWAP 청산 — 보유 종목 누적 거래대금 표본(직전 W분 거래대금을 차분으로 만든다) ---
        self._vwap_profile: Dict[str, List[Optional[float]]] = {}
        self._vwap_hist: Dict[str, deque] = {}           # 종목 → deque[(초, 누적 거래대금)]
        self._vwap_day: str = ""
        self._vwap_last_poll: float = 0.0
        # --- 장중 추가 매수(SLIDE_ENTRY) — 유니버스 누적 거래대금 표본 · 오늘 처음 발동한 종목 ---
        self._slide_profile: Dict[str, List[Optional[float]]] = {}
        self._slide_hist: Dict[str, deque] = {}
        self._slide_day: str = ""
        self._slide_last_poll: float = 0.0
        self._slide_task: Optional[asyncio.Task] = None
        self._slide_fired: set = set()                  # 오늘 이미 한 번 발동한 종목(첫 발동만 쓴다)
        self._slide_queue: List[tuple] = []             # [(배수, 종목)] 이번 조회에서 새로 발동
        self._sold_today: Dict[str, str] = {}           # 오늘 판 종목(같은 날 다시 사지 않는다)
        self._locked = False                              # 목표 도달 잠금(LOCK_RETURN) — 켜지면 끝까지 유지
        self._lock_seen: Optional[dt.datetime] = None     # 목표 첫 확인 시각(시세 튐 방지용 확인 대기)
        # 관제 화면용 오늘의 진입 판정 — 장 시작 판정 결과와 장중 발동, 각각 무엇이 됐는지(매매에는 안 쓴다)
        self._board: Dict[str, Any] = {"day": "", "opening": None, "slide": []}
        # same_window 기준선 — 종목 → 지난 거래일들의 09:00~09:0N 거래대금(오래된 것부터, 최대 20)
        self._opening_hist: Dict[str, List[float]] = {}
        # 종목 → 분봉 파일의 마지막 날짜. 같은 날을 두 번 기록하지 않으려고 들고 있다.
        self._opening_file_last: Dict[str, str] = {}
        # --- sliding 모드 (장중 직전 W분 거래대금) ---
        self.sliding = SlidingWatcher(self.s.SLIDING_WINDOW_MIN, self.s.SLIDING_MULT)
        self._sl_task: Optional[asyncio.Task] = None
        self._sl_queue: List[tuple] = []        # (발동 분, 배수, 종목) — _tick 이 처리한다
        self._sl_file_last: Dict[str, str] = {}  # 종목 → 하루 전체 분봉 파일의 마지막 날짜
        # --- 요건 전용 소액 매매 — 종목 → 매수일. 전략 자리 수에서 뺀다 ---
        self._compliance_syms: Dict[str, str] = {}
        self._compliance_day: str = ""
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
        self._signals_failed_ts: str = ""   # 신호 계산이 실패한 봉 — 같은 봉에서 재시도하지 않는다
        self._disk_free_gb: Optional[float] = None
        self._disk_warned_at: Optional[dt.datetime] = None
        self._started_at: Optional[dt.datetime] = None

    # ================================================================== 시간
    def _prime_intraday(self) -> None:
        """일봉 패널에서 전일 종가·거래량을 감시기에 넣는다.

        전일 **거래량**이 분당 하한의 기준이므로 종가만으로는 트리거가 성립하지
        않는다. 일봉이 갱신될 때마다 다시 넣어야 날짜가 밀리지 않는다.
        """
        if not self.s.USE_INTRADAY:
            return
        # --- 전일 기준값은 **한 번만** 떠서 보관한다 ---
        #
        # `_apply_latest_prices_to_panel()`이 패널 **마지막 봉**의 종가·고가·저가를
        # 실시간 가격으로 덮어쓴다(당일 봉 형성). 그래서 패널에서 매번 다시 읽으면
        # 전일 종가가 오염될 수 있다.
        #
        # 처음에는 `ts < 오늘`로 걸러 해결했다고 봤지만 그것으로는 부족하다 —
        # 브로커가 당일 봉을 아직 주지 않으면 **마지막 봉이 어제 것**이고, 그 행이
        # 덮어써진다. 그러면 필터가 오염된 어제 행을 그대로 집어 기준선이 현재가를
        # 따라다니고, 트리거가 영원히 발동하지 않거나 즉시 발동한다.
        #
        # 그래서 `startup()`에서 **갓 적재한(아직 덮어쓰지 않은) 패널**로 한 번
        # 떠 두고 이후에는 그 값을 쓴다. 날짜가 바뀌면 `_roll_daily_reference()`가
        # 일봉을 새로 받은 직후 `_prev_daily`를 비우고 이 함수를 다시 부른다.
        # (예전 주석은 "`startup()`이 다시 돈다"였지만 `run()`은 `startup()`을 한
        # 번만 부른다 — 여러 날 켜 두면 기준값이 첫날 것으로 고정돼 있었다.)
        today = self._now().strftime("%Y%m%d")
        if not self._prev_daily:
            for symbol, df in self.panel.items():
                if df is None or len(df) < 2:
                    continue
                hist = df[df["ts"].astype(str) < today]
                if hist.empty:
                    hist = df.iloc[:-1]
                if hist.empty:
                    continue
                prev = hist.iloc[-1]
                # 20봉 거래대금 중앙값 — `burst` 모드의 분모다.
                # `turnover = 종가 x 거래량` 정의는 `catalyst_features`와 같지만
                # **창이 한 칸 다르다**: 일봉 피처는 `rolling(20)`으로 당일을
                # 포함하는데(인과적이지만 09:00에는 모르는 값), 여기서는
                # `hist`(전일까지)의 마지막 20봉을 쓴다. 폭증일 자신이 중앙값에
                # 들어가지 않으므로 같은 배수에서 약간 더 쉽게 발동한다.
                tail = hist.tail(20)
                med20 = float(
                    (tail["close"].astype(float) * tail["volume"].astype(float)).median()
                ) if len(tail) >= 10 else 0.0
                self._prev_daily[symbol] = (float(prev["close"]), float(prev["volume"]),
                                            med20)
        primed = 0
        no_med = 0
        for symbol, vals in self._prev_daily.items():
            pc, pv = vals[0], vals[1]
            med20 = vals[2] if len(vals) > 2 else 0.0
            self.intraday.prime(symbol, pc, pv, turnover_med20=med20)
            primed += 1
            if med20 <= 0:
                no_med += 1
        if primed:
            if self.s.INTRADAY_ENTRY_MODE == "burst":
                cond = (f"거래대금 x{self.s.INTRADAY_BURST_MULT:g} (장중 누적)")
            else:
                cond = (f"돌파 +{self.s.INTRADAY_TRIGGER_PCT:.1%}, "
                        f"거래량 x{self.s.INTRADAY_VOL_MULT}")
            logger.info(f"[장중] 전일 기준 {primed}종목 적재 ({cond}"
                        f"{'' if self.s.INTRADAY_FLATTEN else ' · 마감정리 OFF → 다일 보유'})")
            # **burst 모드에서 분모가 0이면 그 종목은 영원히 발동하지 않는다.**
            # 조용한 미작동이므로 세어서 경고한다.
            if self.s.INTRADAY_ENTRY_MODE == "burst" and no_med:
                logger.warning(f"[장중] 거래대금 중앙값 없음 {no_med}종목 — "
                               "이 종목들은 burst 모드에서 발동하지 않는다(이력 부족)")

    def _refresh_alive(self, today: str) -> None:
        """`vol_alive` 판정 입력을 **확정 일봉**에서 만든다. 하루 한 번.

        `_prime_intraday`와 같은 이유로 **갓 적재한 패널**에서만 불러야 한다
        (`startup()`·`_roll_daily_reference()`). 장중에 부르면 마지막 봉이 실시간
        가격으로 덮어써진 상태라 분자가 틀어진다.
        """
        frames = {**self._held_daily, **self.panel}
        alive: Dict[str, float] = {}
        ref_ts: Dict[str, str] = {}
        for symbol, df in frames.items():
            if df is None or df.empty:
                continue
            hist = df[df["ts"].astype(str) < today]
            b = daily_alive_burst(hist)
            if b > 0:
                alive[symbol] = b
                ref_ts[symbol] = str(hist["ts"].iloc[-1])
        # 일봉 재적재가 실패했으면(서버 점검·네트워크) 패널이 묵어 있다. 그대로 쓰면
        # 어제 진입한 종목이 **폭증 전날** 거래대금으로 판정돼 잘못 팔린다. 판정을
        # 보류하는 쪽이 안전하다 — 최악이 하루 더 보유하는 것이다.
        stale = self._stale_reference(today)
        if stale is not None:
            logger.warning(f"[기준] 마지막 확정 봉이 {stale or '없음'} — 직전 거래일이 아니다"
                           "(일봉 재적재 실패?). 오늘 vol_alive 판정을 보류한다"
                           "(보유 유지 · 손절·상한만). 재시작하면 다시 받는다")
            alive = {}
        self._alive_burst = alive
        # 보유 종목의 오늘 판정을 미리 찍는다 — 아침에 무엇이 왜 팔릴지 로그로 보인다.
        mult = float(self.s.EXIT_VOL_ALIVE_MULT)
        for symbol in self.portfolio.open_positions():
            entered = self._entry_dates.get(symbol, today)
            if entered >= today:
                logger.info(f"[보유] {symbol} 진입 당일 — vol_alive 판정 없음")
            elif symbol not in alive:
                logger.warning(f"[보유] {symbol} 확정 일봉 거래대금을 계산할 수 없음 — "
                               f"vol_alive 판정 불가(손절·{self.s.MAX_HOLDING_BARS}봉 상한만)")
            else:
                verdict = "유지" if alive[symbol] >= mult else "청산(시가)"
                logger.info(f"[보유] {symbol} {ref_ts[symbol]} 거래대금 20봉중앙 "
                            f"x{alive[symbol]:.2f} (기준 x{mult:g}) → {verdict}")

    def _stale_reference(self, today: str) -> Optional[str]:
        """실서버에서 패널의 마지막 확정 봉이 **직전 거래일보다 오래됐으면** 그 날짜를,
        정상이면 `None`을 돌려준다. Mock은 가상 날짜라 보지 않는다.

        종목 하나는 거래정지로 봉이 빠질 수 있으므로 패널 전체의 최신 봉을 본다.
        """
        if self.s.USE_MOCK_MODE:
            return None
        latest = ""
        for df in self.panel.values():
            if df is None or df.empty:
                continue
            ts = df["ts"].astype(str)
            ts = ts[ts < today]
            if not ts.empty:
                latest = max(latest, str(ts.iloc[-1]))
        try:
            d = dt.datetime.strptime(today, "%Y%m%d").date() - dt.timedelta(days=1)
        except ValueError:
            return None
        for _ in range(15):                      # 연휴를 넘어도 2주면 충분하다
            if self.calendar.is_trading_day(d):
                break
            d -= dt.timedelta(days=1)
        return latest if latest < d.strftime("%Y%m%d") else None

    def _vol_alive_map(self, today: str) -> Dict[str, float]:
        """`evaluate_exits(burst_map=...)`에 넘길 값 — 보유 종목만, 진입 당일 제외.

        **진입 당일은 넘기지 않는다**(배수 0 = 판정 안 함). 폭증일이라 측정에서도
        당연히 생존이고, 넘기면 어제(폭증 전날) 거래대금으로 판정돼 사자마자 판다.
        """
        return {sym: self._alive_burst[sym]
                for sym in self.portfolio.open_positions()
                if sym in self._alive_burst
                and self._entry_dates.get(sym, today) < today}

    async def _load_held_bars(self) -> None:
        """패널에 없는 보유 종목의 일봉을 받는다(판정 전용).

        유니버스는 기동할 때마다 20일 거래대금 순위로 다시 뽑히므로, 보유 중인
        종목이 순위 밖으로 밀리면 패널에서 빠진다. 그러면 `vol_alive`를 판정할
        일봉이 없어 30봉 상한까지 조용히 들고 간다.
        """
        held = [s_ for s_ in self.portfolio.open_positions() if s_ not in self.panel]
        frames: Dict[str, pd.DataFrame] = {}
        for symbol in held:
            try:
                df = await self.broker.get_bars(symbol, count=60, period="D")
                if df is not None and not df.empty:
                    frames[symbol] = df
            except Exception as exc:
                logger.warning(f"[보유] {symbol} 일봉 적재 실패: {exc}")
        self._held_daily = frames

    async def _roll_daily_reference(self, today: str) -> None:
        """날짜가 바뀌면 일봉을 새로 받고 전일 기준값을 전부 다시 만든다.

        `run()`은 `startup()`을 한 번만 부르므로 프로세스를 여러 날 켜 두면
        전일 종가·20봉 중앙값(진입 기준)과 `vol_alive` 입력(청산 기준)이 첫날 것으로
        고정된다. 실서버에서는 자정 직후(장외)에 돈다 — 유니버스 전체 일봉 적재가
        모의투자 유량으로 수 분 걸리므로 장중에 돌면 안 된다.
        """
        self._drop_stale_day_state(today)
        if self.s.USE_MOCK_MODE and getattr(self.broker, "is_mock", False):
            self._refresh_bars_from_sim()
        else:
            logger.info(f"[기준] 날짜 변경 {self._ref_day} → {today} — 일봉 재적재")
            await self._refresh_bars()
        await self._load_held_bars()
        self._prev_daily = {}
        self._prime_intraday()
        self._refresh_alive(today)
        self._load_opening_history(today)
        self._load_vwap_profile()
        self._load_slide_profile()
        self._flush_sliding_bars()
        self._load_sliding_history(today)
        self._ref_day = today
        if not self.s.USE_MOCK_MODE and self.universe and self.universe[0] in self.panel:
            df = self.panel[self.universe[0]]
            last = df[df["ts"].astype(str) < today]
            if not last.empty:
                logger.info(f"[기준] 마지막 확정 봉 {last['ts'].iloc[-1]} (기준 종목 {self.universe[0]})")

    def _drop_stale_day_state(self, today: str) -> None:
        """전날 것으로 남은 주문 추적·요건 매매 표시를 버린다.

        지정가는 당일 주문이라 장 마감에 증권사가 취소한다. 실서버는 미체결 조회
        (`_sync_open_orders`)가 그것을 지워 주지만 **모의투자는 미체결 조회가 안 된다**
        (`90000000`). 그러면 끝내 체결되지 않은 매수가 `open_orders`에 남아 그 자리를
        영영 막고(재시작 전까지), 매도는 `_sellable_qty`가 그 수량을 빼서 다시 팔지 못한다.

        요건 매매 표시(`_compliance_syms`)는 주문 접수 때 남기므로 체결되지 않으면
        보유하지 않은 종목이 남는다. 나중에 전략이 그 종목을 사면 요건 포지션으로 읽혀
        자리 수에서 빠지고 14:30에 팔린다(2026-10-18 노트북 테스트 잔재로 실제 발견).
        """
        stale = [no for no, o in self.open_orders.items()
                 if str(o.get("ts", ""))[:10].replace("-", "") < today]
        for no in stale:
            o = self.open_orders.pop(no)
            logger.warning(f"[Orders] 전날 미체결 추적 정리: {o.get('symbol', '')} {no} "
                           f"잔량 {o.get('qty', 0)} (장 마감에 취소된 주문)")
        held = set(self.portfolio.open_positions())
        gone = [s_ for s_ in self._compliance_syms if s_ not in held]
        for s_ in gone:
            self._compliance_syms.pop(s_, None)
        if gone:
            logger.warning(f"[대회·요건] 보유하지 않은 요건 표시 정리: {', '.join(gone)}")
        if stale or gone:
            self._save_state()

    def _load_vwap_profile(self) -> None:
        """장중 VWAP 청산 기준선 표를 읽는다(없으면 청산 판정을 하지 않는다)."""
        if not self.s.VWAP_EXIT:
            return
        path = Path(self.s.VWAP_PROFILE_FILE)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"[장중·VWAP] 기준선 표 없음({path}) — VWAP 청산 판정 안 함: {exc}")
            self._vwap_profile = {}
            return
        if int(data.get("window", 0)) != int(self.s.VWAP_EXIT_WINDOW):
            logger.warning(f"[장중·VWAP] 기준선 창 {data.get('window')}분 ≠ 설정 "
                           f"{self.s.VWAP_EXIT_WINDOW}분 — 판정 안 함")
            self._vwap_profile = {}
            return
        self._vwap_profile = data.get("profile") or {}
        logger.info(f"[장중·VWAP] 기준선 {len(self._vwap_profile)}종목 · 최신 {data.get('as_of')} · "
                    f"VWAP {self.s.VWAP_EXIT_BELOW:.0%} 아래 + 직전 {self.s.VWAP_EXIT_WINDOW}분 "
                    f"< 평소 x{self.s.VWAP_EXIT_MULT:g}")

    LOCK_CONFIRM_SECONDS = 60.0   # 목표를 처음 넘고 이만큼 지나도 넘어 있어야 잠근다(시세 한 건 튐 방지)

    async def _lock_check(self, now: dt.datetime, prices: Dict[str, float]) -> None:
        """목표 도달 잠금 — 누적 수익률이 `LOCK_RETURN` 이상이면 전략 포지션을 팔고 매수를 멈춘다.

        심사는 마지막 날(11/20) 수익률만 본다. 꼬리 전략은 출렁임이 커서 중간에 목표를 찍고 내려오는
        경로가 많다 — 그 경로를 목표로 바꾼다. 측정(`combo_study.py --lock-study`, 지문
        `a12a9eb45ea44d2d`, 25거래일): +100% 잠금 P(+100%) 5.12% → 6.59%. 측정은 종가 한 번 판정이고
        여기서는 장중에 보므로 측정보다 불리하지 않다. 요건 매매(`_compliance_trades`)는 계속한다.
        """
        target = float(self.s.LOCK_RETURN)
        if target <= 0 or not self.phase.is_competition:
            return
        if not self._locked:
            ret = float(self.portfolio.total_return)
            if ret < target:
                self._lock_seen = None
                return
            if self._lock_seen is None:
                self._lock_seen = now
                logger.warning(f"[잠금] 누적 {ret:+.2%} ≥ 목표 {target:+.0%} — "
                               f"{self.LOCK_CONFIRM_SECONDS:.0f}초 뒤에도 넘어 있으면 잠근다")
                return
            if (now - self._lock_seen).total_seconds() < self.LOCK_CONFIRM_SECONDS:
                return
            self._locked = True
            self.ledger.write("lock", {"total_return": ret, "target": target,
                                       "equity": self.portfolio.equity})
            logger.warning(f"[잠금] 목표 도달 {ret:+.2%} — 전략 포지션 전량 매도 · 대회 끝까지 신규 매수 없음")
            self._save_state()
        for symbol, pos in list(self.portfolio.open_positions().items()):
            if symbol in self._compliance_syms:
                continue                      # 요건 포지션은 요건 매매가 당일 판다
            qty = self._sellable_qty(symbol, pos.qty)
            if qty <= 0:
                continue
            px = prices.get(symbol, pos.last_price or pos.avg_price)
            if await self._submit(symbol, Side.SELL, qty, px, "LOCK|목표 도달 잠금"):
                self.ledger.write("exit", {"symbol": symbol, "qty": qty, "reason": "lock",
                                           "detail": "목표 도달 잠금"})

    async def _vwap_exits(self, now: dt.datetime) -> None:
        """보유 종목을 장중 VWAP 청산 규칙으로 본다(`vwap_fade`). 복수시세 한 번으로 보유 종목을 다 본다.

        직전 W분 거래대금은 누적 거래대금(`acml_tr_pbmn`)의 차분이다 — W분 전 표본이 쌓여야 판정한다
        (재시작 직후 W분은 판정하지 않는다). 기준선은 '분 m 에서 끝나는 직전 W분'이므로 지금 막 끝난
        분(경과 분 - 1)의 값을 쓴다. 요건 전용 소액 매매 포지션은 보지 않는다.
        """
        if not self.s.VWAP_EXIT or not self._vwap_profile:
            return
        mins = (now.hour - 9) * 60 + now.minute
        if mins < int(self.s.VWAP_EXIT_START_MIN):
            return
        t = now.timestamp()
        if t - self._vwap_last_poll < float(self.s.VWAP_EXIT_POLL_SEC):
            return
        self._vwap_last_poll = t
        day = now.strftime("%Y%m%d")
        if day != self._vwap_day:
            self._vwap_hist.clear()
            self._vwap_day = day
        held = sorted(self._strategy_held())
        if not held:
            return
        try:
            snap = await self.broker.get_turnovers(held)
        except Exception as exc:
            logger.debug(f"[장중·VWAP] 복수시세 실패: {exc}")
            return
        W = int(self.s.VWAP_EXIT_WINDOW) * 60
        for symbol in held:
            row = snap.get(symbol)
            if not row:
                continue
            price = float(row.get("price", 0.0))
            vol = float(row.get("acml_vol", 0.0))
            tov = float(row.get("turnover", 0.0))
            hist = self._vwap_hist.setdefault(symbol, deque())
            hist.append((t, tov))
            while hist and t - hist[0][0] > W + 300:
                hist.popleft()
            old = [h for h in hist if t - h[0] >= W]
            if not old:
                continue                            # W분 전 표본이 아직 없다
            t0, tov0 = old[-1]
            recent = (tov - tov0) * W / max(1.0, t - t0)
            prof = self._vwap_profile.get(symbol) or []
            m = mins - 1
            base = prof[m] if 0 <= m < len(prof) else None
            reason = vwap_fade(price, tov, vol, recent, base,
                               below=float(self.s.VWAP_EXIT_BELOW), mult=float(self.s.VWAP_EXIT_MULT))
            if reason is None:
                continue
            pos = self.portfolio.positions.get(symbol)
            if pos is None or pos.qty <= 0:
                continue
            qty = self._sellable_qty(symbol, pos.qty)
            if qty <= 0:
                continue                            # 이미 낸 매도가 체결 대기 중이다
            logger.info(f"[장중·VWAP] {symbol} 매도 — {reason}")
            ok = await self._submit(symbol, Side.SELL, qty, price or pos.last_price, f"VWAP|{reason}")
            if ok:
                self.ledger.write("exit", {"symbol": symbol, "qty": qty, "reason": "vwap",
                                           "detail": reason})

    def _load_slide_profile(self) -> None:
        """장중 추가 매수 기준선 표(같은 시각 직전 W분 거래대금, 지난 20일 중앙)를 읽는다."""
        if not self.s.SLIDE_ENTRY:
            return
        path = Path(self.s.SLIDE_PROFILE_FILE)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"[장중·추가] 기준선 표 없음({path}) — 장중 추가 매수 안 함: {exc}")
            self._slide_profile = {}
            return
        if int(data.get("window", 0)) != int(self.s.SLIDE_WINDOW):
            logger.warning(f"[장중·추가] 기준선 창 {data.get('window')}분 ≠ 설정 {self.s.SLIDE_WINDOW}분 — 안 함")
            self._slide_profile = {}
            return
        self._slide_profile = data.get("profile") or {}
        logger.info(f"[장중·추가] 기준선 {len(self._slide_profile)}종목 · 최신 {data.get('as_of')} · "
                    f"직전 {self.s.SLIDE_WINDOW}분 ≥ 평소 x{self.s.SLIDE_MULT:g} · "
                    f"{self.s.SLIDE_POLL_SEC:g}초 간격")

    def _slide_ready(self, now: dt.datetime) -> bool:
        """장중 추가 매수를 볼 때인가 — 장 시작 진입을 처리한 뒤, 마지막 진입 분 전."""
        if not self.s.SLIDE_ENTRY or not self._slide_profile:
            return False
        mins = (now.hour - 9) * 60 + now.minute
        if mins > int(self.s.SLIDE_LAST_MIN):
            return False
        if self._opening_mode() and self._opening_done_day != now.strftime("%Y%m%d"):
            return False
        return True

    def _slide_kick(self, now: dt.datetime) -> None:
        """조회 간격이 됐으면 유니버스 누적 거래대금 조회를 태스크로 띄운다(루프를 막지 않는다).

        유니버스 289종목이면 복수시세 10호출, 모의 유량(초당 1건)으로 약 10초다 — 루프 안에서 기다리면
        그동안 VWAP 청산·체결 처리가 멈추므로 태스크로 떼어 내고 결과(`_slide_queue`)만 루프가 읽는다.
        """
        if not self._slide_ready(now):
            return
        if self._slide_task is not None and not self._slide_task.done():
            return
        t = now.timestamp()
        if t - self._slide_last_poll < float(self.s.SLIDE_POLL_SEC):
            return
        self._slide_last_poll = t
        self._slide_task = asyncio.create_task(self._slide_poll(now))

    async def _slide_poll(self, now: dt.datetime) -> None:
        """유니버스 누적 거래대금을 받아 직전 W분 거래대금(차분)이 평소 같은 시각의 k배를 **처음** 넘은 종목을 찾는다.

        측정(`sliding_study.py`)처럼 종목마다 그날 **첫 발동**만 쓴다 — 자리가 없어 못 산 발동은 버린다.
        """
        day = now.strftime("%Y%m%d")
        if day != self._slide_day:
            self._slide_hist.clear()
            self._slide_fired = set()
            self._slide_queue = []
            self._slide_day = day
        syms = [x for x in self.universe if x in self._slide_profile]
        if not syms:
            return
        try:
            snap = await self.broker.get_turnovers(syms)
        except Exception as exc:
            logger.debug(f"[장중·추가] 복수시세 실패: {exc}")
            return
        t = now.timestamp()                      # 조회를 시작한 시각(10초 안팎 차이는 무시한다)
        mins = (now.hour - 9) * 60 + now.minute
        W = int(self.s.SLIDE_WINDOW) * 60
        k = float(self.s.SLIDE_MULT)
        fired = []
        for symbol in syms:
            row = snap.get(symbol)
            if not row:
                continue
            tov = float(row.get("turnover", 0.0))
            hist = self._slide_hist.setdefault(symbol, deque())
            hist.append((t, tov))
            while hist and t - hist[0][0] > W + 300:
                hist.popleft()
            if symbol in self._slide_fired:
                continue
            old = [h for h in hist if t - h[0] >= W]
            if not old:
                continue
            t0, tov0 = old[-1]
            recent = (tov - tov0) * W / max(1.0, t - t0)
            prof = self._slide_profile.get(symbol) or []
            m = mins - 1
            base = prof[m] if 0 <= m < len(prof) else None
            if base and base > 0 and recent >= k * base:
                self._slide_fired.add(symbol)
                fired.append((recent / base, symbol))
        if fired:
            fired.sort(reverse=True)
            self._slide_queue = fired
            rows = self._board_day(day)["slide"]
            rows.extend(self._board_row(sy, r, now, "대기") for r, sy in fired)
            del rows[:-80]
            logger.info(f"[장중·추가] 발동 {len(fired)}건 · 상위 " +
                        ", ".join(f"{sy} x{r:.1f}" for r, sy in fired[:5]))

    async def _slide_entries(self, now: dt.datetime, prices: Dict[str, float]) -> None:
        """이번 조회에서 새로 발동한 종목으로 빈자리를 채운다(배수 큰 순, 오늘 판 종목·보유·매매불가 제외)."""
        if not self._slide_queue:
            return
        queue, self._slide_queue = self._slide_queue, []
        day = now.strftime("%Y%m%d")
        inflight = self.open_orders_symbols() - self._compliance_exempt()
        held = self._strategy_held()
        free = max(0, int(self.s.MAX_POSITIONS) - len(held | set(inflight)))
        board = self._board_day(day)["slide"]
        mark = lambda sym, out: self._board_mark(board, sym, out)
        for ratio, symbol in queue:
            if free <= 0:
                mark(symbol, "자리 없음")
                continue
            if symbol in held or symbol in inflight:
                mark(symbol, "보유 중")
                continue
            if self._sold_today.get(symbol) == day:
                mark(symbol, "오늘 매도")
                continue
            if self._reject_today.get(symbol) == day:
                mark(symbol, "매매불가")
                continue
            status: Dict[str, Any] = {}
            try:
                raw = await self.broker.inquire_price(symbol)
                status = raw.get("output") or {}
                px = to_float(status.get("stck_prpr")) or prices.get(symbol, 0.0)
            except Exception:
                px = prices.get(symbol, 0.0)
            if self._vts_live() and not vts_tradable(status, self.s.VTS_BLOCK_FULL_MARGIN):
                logger.info(f"[장중·추가] {symbol} 건너뜀 — 모의투자 매매불가 "
                            f"(상태 {status.get('iscd_stat_cls_code')} · 증거금 {status.get('marg_rate')}%)")
                self._reject_today[symbol] = day
                mark(symbol, "매매불가")
                continue
            if px <= 0:
                mark(symbol, "시세 없음")
                continue
            min_chase = float(self.s.SLIDE_MIN_CHASE)
            if min_chase > 0:
                prev_close = float((self._prev_daily.get(symbol) or (0.0,))[0])
                if not (prev_close > 0 and px / prev_close - 1.0 >= min_chase):
                    logger.info(f"[장중·추가] {symbol} 건너뜀 — 전일 대비 "
                                f"{(px / prev_close - 1.0) if prev_close > 0 else float('nan'):+.2%} "
                                f"< {min_chase:+.0%}")
                    chase = (px / prev_close - 1.0) if prev_close > 0 else 0.0
                    mark(symbol, f"전일 대비 {chase:+.1%} < {min_chase:+.0%}")
                    continue
            self.latest_prices[symbol] = px
            sig = Signal(symbol=symbol, score=RULE_TRIGGER_SCORE, price=px, ts=now,
                         atr=self.atr_map.get(symbol, 0.0),
                         volatility=self.vol_map.get(symbol, 0.0),
                         bet=-1.0, reason="slide_turnover")
            decision = self.risk.evaluate_entry(sig, self.portfolio, now,
                                                pending=self._pending_notional(),
                                                slot_exempt=self._compliance_exempt())
            self.ledger.write("risk_decision", {
                "symbol": symbol, "score": round(sig.score, 4), "ratio": round(ratio, 3),
                "approved": decision.approved, "qty": decision.qty,
                "reason": decision.reason, "path": "slide", **decision.detail})
            if not decision.approved:
                logger.info(f"[장중·추가] {symbol} 진입 거부 — {decision.reason}")
                mark(symbol, f"거부 · {decision.reason}")
                continue
            ok = await self._submit(symbol, Side.BUY, decision.qty, px,
                                    f"SLIDE|직전 {self.s.SLIDE_WINDOW}분 거래대금 x{ratio:.1f}")
            mark(symbol, "매수 주문" if ok else "주문 실패")
            if ok:
                free -= 1

    def _vts_live(self) -> bool:
        """모의투자 실서버에 붙어 있는가(매매불가 종목 거름이 필요한 곳)."""
        return (not self.s.USE_MOCK_MODE) and str(self.s.KIS_ENV).lower() == "vts"

    def _opening_mode(self) -> bool:
        return bool(self.s.USE_INTRADAY) and self.s.INTRADAY_ENTRY_MODE == "opening"

    def _sliding_mode(self) -> bool:
        return bool(self.s.USE_INTRADAY) and self.s.INTRADAY_ENTRY_MODE == "sliding"

    def _rule_entry_mode(self) -> bool:
        """규칙이 정한 시점에 판정하는 모드(opening·sliding) — 누적 감시·아침 스캔을 쓰지 않는다."""
        return self._opening_mode() or self._sliding_mode()

    def _load_sliding_history(self, today: str) -> None:
        """sliding 기준선을 하루 한 번 만든다(기동·날짜 변경).

        same_time: `SLIDING_HISTORY_DIR`의 하루 전체 분봉에서 오늘 이전 마지막 20거래일의
        직전 W분 합을 시각별로 만들고 중앙값을 낸다(`same_time_base`, 측정과 같은 함수).
        day_avg: 평소 하루 거래대금(20봉 중앙) x W/390.
        """
        if not self._sliding_mode():
            return
        w = int(self.s.SLIDING_WINDOW_MIN)
        self.sliding = SlidingWatcher(w, self.s.SLIDING_MULT)
        self.sliding.reset_day(today)
        same = self.s.SLIDING_BASELINE == "same_time"
        ok = 0
        for symbol in self.universe:
            if not same:
                vals = self._prev_daily.get(symbol)
                med20 = vals[2] if vals and len(vals) > 2 else 0.0
                if med20 > 0:
                    self.sliding.set_base(symbol, med20 * w / 390.0)
                    ok += 1
                continue
            path = Path(self.s.SLIDING_HISTORY_DIR) / f"{symbol}.csv"
            if not path.exists():
                continue
            try:
                m = pd.read_csv(path, dtype={"ts": str})
            except Exception as exc:
                logger.warning(f"[장중·슬라이딩] {symbol} 과거 분봉 읽기 실패: {exc}")
                continue
            if m.empty:
                continue
            self._sl_file_last[symbol] = str(m["ts"].str[:8].max())
            m = m[m["ts"].str[:8] < today]
            days = sorted(m["ts"].str[:8].unique())[-20:]
            if len(days) < 10:
                continue
            m = m[m["ts"].str[:8].isin(days)]
            rows = []
            for _, g in m.groupby(m["ts"].str[:8], sort=True):
                tv = minute_turnover(minute_index(g["ts"]), g["close"].to_numpy(float),
                                     g["volume"].to_numpy(float), LAST_ENTRY_MIN + 1)
                rows.append(rolling_sum(tv, w))
            self.sliding.set_base(symbol, same_time_base(np.vstack(rows)))
            ok += 1
        msg = (f"[장중·슬라이딩] 기준선({self.s.SLIDING_BASELINE}) {ok}/{len(self.universe)}종목 · "
               f"W={w}분 · x{self.s.SLIDING_MULT:g}")
        if self.universe and ok < 0.8 * len(self.universe):
            logger.warning(msg + " — 기준선이 없는 종목은 발동하지 않는다")
        else:
            logger.info(msg)

    def _flush_sliding_bars(self) -> None:
        """감시하며 받은 그날 분봉을 하루 전체 분봉 파일에 이어 붙인다(날짜 변경·종료 때).

        KIS 당일분봉은 그날만 받을 수 있다. 저장하지 않으면 대회 중 same_time 기준선이 토스
        수집분에서 멈춘다. 추가 조회 없이 이미 받은 봉을 쓴다.
        """
        day = self.sliding.day
        if not (self._sliding_mode() and day and self.s.SLIDING_BASELINE == "same_time"):
            return
        n = 0
        for symbol, book in self.sliding.bars.items():
            if not book or self._sl_file_last.get(symbol, "") >= day:
                continue
            rows = [(ts_, *v) for ts_, v in sorted(book.items()) if ts_[:8] == day]
            if not rows:
                continue
            path = Path(self.s.SLIDING_HISTORY_DIR) / f"{symbol}.csv"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]
                             ).to_csv(path, mode="a", header=not path.exists(), index=False)
                self._sl_file_last[symbol] = day
                n += 1
            except Exception as exc:
                logger.warning(f"[장중·슬라이딩] {symbol} 분봉 기록 실패: {exc}")
        if n:
            logger.info(f"[장중·슬라이딩] {day} 분봉 {n}종목 기록 → {self.s.SLIDING_HISTORY_DIR}")

    async def _sliding_poll_once(self, symbol: str) -> None:
        """한 종목의 당일분봉(최근 30분)을 받아 감시기에 넣고, 발동하면 대기열에 쌓는다."""
        now = self._now()
        try:
            bars = await self.broker.get_minute_bars(symbol, now.strftime("%H%M%S"))
        except Exception as exc:
            logger.debug(f"[장중·슬라이딩] {symbol} 분봉 실패: {exc}")
            return
        if bars is None or bars.empty:
            return
        now_min = (now.hour - 9) * 60 + now.minute
        hit = self.sliding.update(
            symbol, bars["ts"].astype(str).tolist(), bars["open"].tolist(),
            bars["high"].tolist(), bars["low"].tolist(), bars["close"].tolist(),
            bars["volume"].tolist(), now_min)
        last_px = float(bars["close"].iloc[-1])
        if last_px > 0:
            self.latest_prices[symbol] = last_px
        if hit is not None and symbol not in self.portfolio.open_positions():
            mm, ratio = hit
            self._sl_queue.append((mm, ratio, symbol))
            logger.warning(f"[장중·슬라이딩] 발동 {symbol} — {9 + (mm + 1) // 60:02d}:"
                           f"{(mm + 1) % 60:02d}까지 {self.s.SLIDING_WINDOW_MIN}분 거래대금 "
                           f"x{ratio:.1f} (감지 {now.strftime('%H:%M:%S')})")
            self.ledger.write("sliding_trigger", {
                "symbol": symbol, "minute": mm, "ratio": round(ratio, 3),
                "detected": now.isoformat()})

    async def _sliding_poll_loop(self) -> None:
        """장중 내내 유니버스 당일분봉을 돌아가며 받는다(한 바퀴 약 2.6분 — 측정의 '지연 4분').

        시세 P2라 주문(P0)·계좌(P1)를 밀지 않는다. 주문은 하지 않는다 — 발동은 대기열에
        쌓고 `_tick`이 리스크 관문을 거쳐 낸다.
        """
        while self._running:
            if self._session() is not MarketSession.REGULAR:
                await asyncio.sleep(5.0)
                continue
            self.sliding.reset_day(self._now().strftime("%Y%m%d"))
            for symbol in list(self.universe):
                if not self._running:
                    return
                await self._sliding_poll_once(symbol)
            await asyncio.sleep(0.1)

    async def _sliding_entries(self, now: dt.datetime, prices: Dict[str, float]) -> None:
        """대기열의 발동으로 빈 슬롯을 채운다 — 발동이 이른 순, 같은 분이면 배수가 큰 순.

        측정과 같은 규칙이다. 슬롯이 없어 못 산 발동은 버린다(청산은 시가에 나서 장중에
        슬롯이 다시 비지 않는다).
        """
        if not self._sl_queue:
            return
        queued = sorted(self._sl_queue, key=lambda x: (x[0], -x[1]))
        self._sl_queue = []
        inflight = self.open_orders_symbols() - self._compliance_exempt()
        held = self._strategy_held()
        free = max(0, int(self.s.MAX_POSITIONS) - len(held | set(inflight)))
        for mm, ratio, symbol in queued:
            if free <= 0:
                logger.info(f"[장중·슬라이딩] {symbol} 발동 — 슬롯 없음, 버림")
                continue
            if symbol in held or symbol in inflight:
                continue
            px = prices.get(symbol) or self.latest_prices.get(symbol, 0.0)
            if not px or px <= 0:
                continue
            sig = Signal(symbol=symbol, score=RULE_TRIGGER_SCORE, price=px, ts=now,
                         atr=self.atr_map.get(symbol, 0.0),
                         volatility=self.vol_map.get(symbol, 0.0),
                         bet=-1.0, reason="sliding_turnover")
            decision = self.risk.evaluate_entry(sig, self.portfolio, now,
                                                pending=self._pending_notional(),
                                                slot_exempt=self._compliance_exempt())
            self.ledger.write("risk_decision", {
                "symbol": symbol, "score": round(sig.score, 4), "ratio": round(ratio, 3),
                "approved": decision.approved, "qty": decision.qty,
                "reason": decision.reason, "path": "sliding", **decision.detail})
            if not decision.approved:
                logger.info(f"[장중·슬라이딩] {symbol} 진입 거부 — {decision.reason}")
                continue
            ok = await self._submit(symbol, Side.BUY, decision.qty, px,
                                    f"SLIDING|{self.s.SLIDING_WINDOW_MIN}분 거래대금 x{ratio:.1f}")
            if ok:
                free -= 1
                inflight = self.open_orders_symbols()

    def _compliance_exempt(self) -> set:
        """전략 자리 수에서 빼는 종목 — 요건 전용 소액 매매 포지션."""
        return set(self._compliance_syms)

    def _strategy_held(self) -> set:
        return set(self.portfolio.open_positions()) - self._compliance_exempt()

    async def _compliance_trades(self, now: dt.datetime, prices: Dict[str, float]) -> None:
        """지수 종목 요건 — 모자라면 소액으로 사서 당일 판다(2026-10-10, 사용자 결정).

        측정(scripts/compliance_study.py)에서 지금 진입 규칙은 30거래일에 지수 종목을 중앙
        2개만 샀다(5종목 요건 충족 10.7%). 요건을 놓치면 수익률과 무관하게 실격이다.
        """
        # 매도는 `COMPLIANCE_TRADES`를 꺼도 돈다 — 요건이 찬 뒤 끄면(10/19 사용자) 이미 산 요건 포지션이
        # 당일 청산되지 않고 남는다
        if not (self._compliance_syms or self.s.COMPLIANCE_TRADES) or not self.phase.is_competition:
            return
        day = now.strftime("%Y%m%d")
        hhmm = now.strftime("%H%M")
        # --- 매도: 당일 매도 시각 이후, 또는 전날 남은 것은 바로 ---
        for symbol, bought in list(self._compliance_syms.items()):
            pos = self.portfolio.positions.get(symbol)
            if pos is None or pos.qty <= 0:
                continue
            if bought < day or hhmm >= self.s.COMPLIANCE_SELL_HHMM:
                qty = self._sellable_qty(symbol, pos.qty)
                if qty > 0:
                    px = prices.get(symbol, pos.last_price or pos.avg_price)
                    await self._submit(symbol, Side.SELL, qty, px, "COMPLIANCE|요건 매매 당일 청산")
        # --- 매수: 하루 한 번, 시작일 이후 ---
        if not self.s.COMPLIANCE_TRADES:
            return
        try:
            start = dt.date.fromisoformat(self.s.COMPLIANCE_START)
        except ValueError:
            return
        if (self._compliance_day == day or now.date() < start
                or hhmm < self.s.COMPLIANCE_BUY_HHMM or hhmm >= self.s.COMPLIANCE_SELL_HHMM):
            return
        self._compliance_day = day
        from strategy.compliance import compliance_picks, index_symbols_traded, read_fills
        members = {x for lst in self.s.index_members().values() for x in lst}
        if not members:
            logger.warning("[대회·요건] 지수 멤버십 정보가 없어 요건 매매를 하지 않는다")
            return
        fills = read_fills(Path(self.s.LOG_DIR) / "ledger.jsonl", since=start)
        traded = index_symbols_traded(fills, members)
        # 거래대금이 큰 지수 종목부터 — 체결이 확실하고 호가가 촘촘하다
        # 1주 가격이 예산을 넘는 종목은 뺀다(매수 지정가 폭 포함). 남겨 두면 수량 0으로 매번 거부되는데
        # 거래대금 상위라 날마다 다시 뽑혀 요건이 영영 안 찬다(10/19: SK하이닉스 122만원·삼성전기 111만원)
        cap_px = float(self.s.COMPLIANCE_TRADE_KRW) / (1.0 + float(self.s.BUY_LIMIT_BAND)) * 0.97
        affordable = {s_ for s_ in self.universe
                      if 0 < (self._prev_daily.get(s_, (0.0,))[0] or 0.0) <= cap_px}
        # 증거금 100% 종목은 뺀다 — 모의투자 API가 거부할 수 있고(10/14 40070000) 요건 매매는 체결이 확실해야 한다
        risky = set(self.s.full_margin_symbols()) | {s_ for s_, d in self._reject_today.items() if d == day}
        ranked = sorted((s_ for s_ in self.universe
                         if s_ in members and s_ in affordable and s_ not in risky),
                        key=lambda s_: -(self._prev_daily.get(s_, (0, 0, 0.0))[2]
                                         if len(self._prev_daily.get(s_, ())) > 2 else 0.0))
        exclude = set(self.portfolio.open_positions()) | set(self.open_orders_symbols())
        picks = compliance_picks(traded, members, ranked, int(self.s.MIN_TRADE_SYMBOLS),
                                 exclude=exclude)
        if not picks:
            return
        logger.warning(f"[대회·요건] 지수 종목 {len(traded)}/{self.s.MIN_TRADE_SYMBOLS} — "
                       f"소액 매매 {len(picks)}건: {', '.join(picks)}")
        for symbol in picks:
            try:
                q = await self.broker.get_quote(symbol)
                px = float(q.price) if q and q.price > 0 else prices.get(symbol, 0.0)
            except Exception:
                px = prices.get(symbol, 0.0)
            if px <= 0:
                continue
            sig = Signal(symbol=symbol, score=RULE_TRIGGER_SCORE, price=px, ts=now,
                         bet=-1.0, reason="compliance")
            decision = self.risk.evaluate_entry(sig, self.portfolio, now,
                                                pending=self._pending_notional(),
                                                max_notional=float(self.s.COMPLIANCE_TRADE_KRW))
            self.ledger.write("risk_decision", {
                "symbol": symbol, "approved": decision.approved, "qty": decision.qty,
                "reason": decision.reason, "path": "compliance", **decision.detail})
            if not decision.approved:
                logger.info(f"[대회·요건] {symbol} 거부 — {decision.reason}")
                continue
            if await self._submit(symbol, Side.BUY, decision.qty, px, "COMPLIANCE|지수 종목 요건"):
                self._compliance_syms[symbol] = day
                self._save_state()

    def _opening_dir(self) -> Path:
        return Path(self.s.OPENING_HISTORY_DIR)

    def _load_opening_history(self, today: str) -> None:
        """same_window 기준선 — 종목별로 **오늘 이전** 거래일들의 09:00~09:0N 거래대금을 적재한다.

        측정(`scripts/opening_study.py`)과 같은 함수(`window_turnover`)로 계산하고, 판정은
        `same_window_base`가 마지막 20일(최소 10일)의 중앙값으로 한다. 파일은 토스 수집분
        (`collect_opening.py`)에 대회 중 09:0N 판정 때 받은 KIS 분봉이 이어 붙는다.
        """
        if not (self._opening_mode() and self.s.OPENING_BASELINE == "same_window"):
            return
        n = int(self.s.OPENING_WINDOW_MIN)
        hist: Dict[str, List[float]] = {}
        last: Dict[str, str] = {}
        for symbol in self.universe:
            path = self._opening_dir() / f"{symbol}.csv"
            if not path.exists():
                continue
            try:
                m = pd.read_csv(path, dtype={"ts": str})
            except Exception as exc:
                logger.warning(f"[장중·개장] {symbol} 과거 분봉 읽기 실패: {exc}")
                continue
            if m.empty:
                continue
            last[symbol] = str(m["ts"].str[:8].max())
            m = m[m["ts"].str[:8] < today]
            vals: List[float] = []
            for _, g in m.groupby(m["ts"].str[:8], sort=True):
                vals.append(window_turnover(minute_index(g["ts"]), g["close"].to_numpy(float),
                                            g["volume"].to_numpy(float), n))
            hist[symbol] = vals[-20:]
        self._opening_hist = hist
        self._opening_file_last = last
        ok = sum(1 for v in hist.values() if len(v) >= 10)
        newest = max(last.values()) if last else "없음"
        msg = (f"[장중·개장] 같은 시간대 기준선 적재 {ok}/{len(self.universe)}종목(10일 이상) · "
               f"최신 {newest}")
        if self.universe and ok < 0.8 * len(self.universe):
            logger.warning(msg + " — 기준선이 없는 종목은 발동하지 않는다")
        else:
            logger.info(msg)

    def _record_opening_bars(self, symbol: str, day: str, bars: pd.DataFrame,
                             tv: float) -> None:
        """오늘 09:00~09:0(N-1) 분봉을 과거 분봉 파일에 붙이고 기준선에 넣는다.

        KIS 당일분봉은 그날만 받을 수 있어서, 판정 때 받은 것을 저장하지 않으면 대회 중
        기준선이 토스 수집분(10/8)에서 멈춘다. 추가 조회 없이 이미 받은 분봉을 쓴다.
        """
        if self._opening_file_last.get(symbol, "") >= day:
            return                                      # 오늘을 이미 기록했다(재시작 등)
        cols = ["ts", "open", "high", "low", "close", "volume"]
        path = self._opening_dir() / f"{symbol}.csv"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            bars[cols].to_csv(path, mode="a", header=not path.exists(), index=False)
            self._opening_file_last[symbol] = day
        except Exception as exc:
            logger.warning(f"[장중·개장] {symbol} 분봉 기록 실패: {exc}")
        h = self._opening_hist.setdefault(symbol, [])
        h.append(float(tv))
        del h[:-20]

    async def _opening_scan(self, now: dt.datetime) -> None:
        """09:0N 직후 유니버스 전 종목의 **누적 거래대금을 복수시세로 한꺼번에** 받아 판정한다.

        ## 왜 복수시세인가 (2026-10-13)

        예전에는 종목마다 당일분봉을 받았다. KIS 모의투자의 **실효 유량은 초당 약 1건**이라
        (1.0건 거절 0 · 1.2건 30중 5 · 1.5건 30중 10 · 1.8건 30중 13 — 실측) 284종목에 약 5분이
        걸려 진입이 09:15로 밀렸다. 복수시세(`FHKST11300006`)는 30종목/호출이라 10번, 약 10초다.

        `acml_tr_pbmn`(장 시작부터의 실제 체결대금)을 09:0N 직후에 읽으므로 측정의 "09:00~09:0(N-1)
        분봉 종가 x 거래량 합"과 정의가 거의 같다(차이는 조회까지의 몇 초와 분봉 근사).
        판정은 `strategy/opening.py` 순수 함수가 한다(측정과 같은 코드).

        **내일 기준선용 분봉 기록은 진입 뒤로 미룬다**(`_record_opening_history`). 기록은 종목마다
        분봉이 필요해 5분이 걸리므로, 판정 앞에 두면 진입이 다시 늦어진다.
        """
        n = int(self.s.OPENING_WINDOW_MIN)
        day = now.strftime("%Y%m%d")
        hour = f"{9 + n // 60:02d}{n % 60:02d}00"
        held = set(self.portfolio.open_positions())
        same = self.s.OPENING_BASELINE == "same_window"
        ratios: Dict[str, float] = {}
        quotes = await self.broker.get_turnovers(list(self.universe))
        for symbol, q in quotes.items():
            if q.get("price", 0) > 0:
                self.latest_prices[symbol] = float(q["price"])
            if symbol in held:
                continue
            tv = float(q.get("turnover", 0.0))
            vals = self._prev_daily.get(symbol)
            med20 = vals[2] if vals and len(vals) > 2 else 0.0
            r = (same_window_ratio(tv, self._opening_hist.get(symbol, [])) if same
                 else opening_ratio(tv, med20, n))
            if r > 0:
                ratios[symbol] = r
        picks = rank_triggers(ratios, float(self.s.OPENING_MULT))
        self._opening_picks = picks
        self._board_day(day)["opening"] = {
            "time": self._now().strftime("%H:%M:%S"), "window": n,
            "mult": float(self.s.OPENING_MULT), "checked": len(quotes), "skipped": None,
            "rows": [self._board_row(s_, r, self._now(), "진입 대기") for s_, r in picks[:30]]}
        top = ", ".join(f"{s_} x{r:.1f}" for s_, r in picks[:5]) or "없음"
        logger.info(f"[장중·개장] 09:00~{hour[:2]}:{hour[2:4]} 판정 {len(quotes)}종목 "
                    f"({self.s.OPENING_BASELINE} · 복수시세 {self._now().strftime('%H:%M:%S')}) → "
                    f"발동 {len(picks)}건 (x{self.s.OPENING_MULT:g} 이상 · 상위: {top})")
        self.ledger.write("opening_scan", {
            "window_min": n, "mult": float(self.s.OPENING_MULT), "checked": len(quotes),
            "baseline": self.s.OPENING_BASELINE,
            "picks": [{"symbol": s_, "ratio": round(r, 3)} for s_, r in picks[:20]]})
        if same:
            self._opening_record_task = asyncio.create_task(self._record_opening_history(day, hour))

    async def _record_opening_history(self, day: str, hour: str) -> None:
        """오늘 09:00~09:0(N-1) 분봉을 받아 기준선 파일에 붙인다(진입 뒤 백그라운드).

        기준선은 측정과 같은 정의(분봉 종가 x 거래량 합)로 이어져야 한다 — 그래서 판정에 쓴
        복수시세 값이 아니라 분봉으로 기록한다. 시세 P2라 주문을 밀지 않는다.
        """
        n = int(self.s.OPENING_WINDOW_MIN)
        done = 0
        for symbol in list(self.universe):
            try:
                bars = await self.broker.get_minute_bars(symbol, hour)
            except Exception as exc:
                logger.debug(f"[장중·개장] {symbol} 분봉 실패: {exc}")
                continue
            bars = bars[bars["ts"].astype(str).str[:8] == day] if not bars.empty else bars
            if bars.empty:
                continue
            mi = minute_index(bars["ts"])
            tv = window_turnover(mi, bars["close"].to_numpy(float),
                                 bars["volume"].to_numpy(float), n)
            self._record_opening_bars(symbol, day, bars[(mi >= 0) & (mi < n)], tv)
            done += 1
            if done == 1:   # 시각 표기(시작/끝) 확인용 — 실서버 첫날 확인
                logger.info(f"[장중·개장] 분봉 시각 확인 {symbol}: "
                            f"{', '.join(bars['ts'].astype(str).str[8:12].tolist()[:8])}")
        logger.info(f"[장중·개장] 기준선 분봉 기록 {done}/{len(self.universe)}종목 ({day})")

    async def _opening_entries(self, now: dt.datetime, prices: Dict[str, float]) -> None:
        """판정 결과로 **하루 한 번** 빈 슬롯을 채운다.

        측정과 같은 규칙이다 — 배수가 큰 순으로 채우고, 슬롯이 없어 못 산 발동은
        버린다(다음 날로 넘기지 않는다). 측정은 "구간 끝 + 지연분 시가"에 들어가므로
        그 전에는 사지 않는다. 판정이 그보다 늦게 끝나면 끝난 뒤 사고 지연을 기록한다.
        """
        n, lag = int(self.s.OPENING_WINDOW_MIN), int(self.s.OPENING_LAG_MIN)
        day = now.strftime("%Y%m%d")
        if (self._opening_done_day == day or self._opening_day != day
                or self._opening_picks is None):
            return
        start = n + lag
        if now.strftime("%H%M") < f"{9 + start // 60:02d}{start % 60:02d}":
            return
        self._opening_done_day = day
        late = (now.hour - 9) * 60 + now.minute - start
        if late > 0 and self._opening_picks:
            logger.warning(f"[장중·개장] 진입이 측정보다 {late}분 늦다(분봉 판정 지연)")
        inflight = self.open_orders_symbols() - self._compliance_exempt()
        held = self._strategy_held()
        free = max(0, int(self.s.MAX_POSITIONS) - len(held | set(inflight)))
        board = (self._board_day(day).get("opening") or {}).get("rows") or []
        mark = lambda sym, out: self._board_mark(board, sym, out)
        for symbol, ratio in self._opening_picks:
            if free <= 0:
                mark(symbol, "자리 없음")
                continue
            if symbol in held or symbol in inflight:
                mark(symbol, "보유 중")
                continue
            if self._reject_today.get(symbol) == day:
                mark(symbol, "매매불가")
                continue
            # 현재가와 종목 상태를 한 호출로 받는다. 관심목록 밖 종목은 현재가가 묵어 있을 수 있고,
            # 모의투자는 증거금 100%·경보 종목 주문을 거부한다(`vts_tradable`) — 거부 주문 한 건이
            # 유량 대기까지 약 3초라, 미리 걸러 다음 순위로 바로 내려간다(2026-10-14 5건 연속 거부).
            status: Dict[str, Any] = {}
            try:
                raw = await self.broker.inquire_price(symbol)
                status = raw.get("output") or {}
                px = to_float(status.get("stck_prpr")) or prices.get(symbol, 0.0)
            except Exception:
                px = prices.get(symbol, 0.0)
            if self._vts_live() and not vts_tradable(status, self.s.VTS_BLOCK_FULL_MARGIN):
                logger.info(f"[장중·개장] {symbol} 건너뜀 — 모의투자 매매불가 "
                            f"(상태 {status.get('iscd_stat_cls_code')} · 증거금 {status.get('marg_rate')}%)")
                self._reject_today[symbol] = day
                mark(symbol, "매매불가")
                continue
            if px <= 0:
                mark(symbol, "시세 없음")
                continue
            self.latest_prices[symbol] = px
            sig = Signal(symbol=symbol, score=RULE_TRIGGER_SCORE, price=px, ts=now,
                         atr=self.atr_map.get(symbol, 0.0),
                         volatility=self.vol_map.get(symbol, 0.0),
                         bet=-1.0, reason="opening_turnover")
            decision = self.risk.evaluate_entry(sig, self.portfolio, now,
                                                pending=self._pending_notional(),
                                                slot_exempt=self._compliance_exempt())
            self.ledger.write("risk_decision", {
                "symbol": symbol, "score": round(sig.score, 4), "ratio": round(ratio, 3),
                "approved": decision.approved, "qty": decision.qty,
                "reason": decision.reason, "path": "opening", **decision.detail})
            if not decision.approved:
                logger.info(f"[장중·개장] {symbol} 진입 거부 — {decision.reason}")
                mark(symbol, f"거부 · {decision.reason}")
                continue
            ok = await self._submit(symbol, Side.BUY, decision.qty, px,
                                    f"OPENING|09:00~{n}분 거래대금 x{ratio:.1f}")
            mark(symbol, "매수 주문" if ok else "주문 실패")
            if ok:
                free -= 1
        self._opening_picks = None

    async def _scan_candidates(self, now: dt.datetime) -> None:
        """아침에 전 후보의 상승률·거래량을 한 번 훑어 **오늘 움직이는** 종목을 구독한다.

        ## 왜 필요한가

        `config/universe.json`은 생성 시점(2026-09-29)의 거래대금 상위 목록이다.
        거래대금 하한은 측정에서 "그날 뭔가 돌고 있던 종목을 남기는 필터"로
        작동했다 — 하한을 낮추면 p95 변동성이 12.80%에서 5.32%로 **떨어진다**.
        즉 그 목록은 생성 시점의 테마이고, 대회 30일 사이엔 다른 종목이 돈다.
        스냅샷에 묶이면 대회 기간의 신선한 테마를 전부 놓친다.

        ## 왜 KIS로 조회하는가

        토스에 순위 API(`/api/v1/rankings`)가 있지만 우리 자격증명에 권한이 없고
        (검증 400 뒤 인가 401), 무엇보다 **이 세션에서 토스 IP가 여섯 번 끊겼다.**
        주문이 KIS로 나가니 조회도 KIS로 해서 실거래 경로의 의존성을 하나로 둔다.

        유량은 걱정하지 않아도 된다. `PriorityRateLimiter`가 시세를 P2, 주문을 P0로
        두므로 스캔이 몰려도 손절 주문이 밀리지 않는다.
        """
        if not self.s.USE_INTRADAY:
            return
        day = now.strftime("%Y%m%d")
        if self._scanned_day == day or now.strftime("%H%M") < self.s.INTRADAY_SCAN_HHMM:
            return
        self._scanned_day = day          # 실패해도 하루 한 번만 시도한다
        pool = [s_ for s_ in self.s.UNIVERSE if s_ not in self.portfolio.open_positions()]
        logger.info(f"[장중·스캔] 후보 {len(pool)}종목 상승률 조회 시작")
        rows: List[tuple] = []
        for symbol in pool:
            try:
                q = await self.broker.get_quote(symbol)
            except Exception as exc:
                logger.debug(f"[장중·스캔] {symbol} 실패: {exc}")
                continue
            if q is None or q.price <= 0:
                continue
            self.latest_prices[symbol] = q.price
            st = self.intraday.states.get(symbol)
            prev_close = st.prev_close if st else 0.0
            if prev_close <= 0:
                continue
            chg = q.price / prev_close - 1.0
            # 전일 거래량 대비 오늘 누적 — 아직 덜 돈 종목을 선호한다(신선도).
            ratio = (q.volume / st.prev_volume) if (st and st.prev_volume > 0) else 0.0
            rows.append((symbol, chg, ratio))
        if not rows:
            logger.warning("[장중·스캔] 유효한 시세가 없어 관심목록을 바꾸지 않습니다")
            return

        # 이미 돌파선을 넘긴 종목은 지금 들어가도 늦다(신선도 측정). 아직 아래에
        # 있으면서 많이 오른 종목이 오늘 돌파할 후보다.
        below = [r for r in rows if r[1] < self.s.INTRADAY_TRIGGER_PCT]
        pick = sorted(below or rows, key=lambda r: -r[1])
        cap = max(1, min(int(self.s.INTRADAY_SCAN_TOP), int(self.s.WS_MAX_SUBSCRIPTIONS)))
        held = list(self.portfolio.open_positions())
        chosen = held + [r[0] for r in pick][: max(0, cap - len(held))]
        self.watchlist = chosen
        if self.ws is not None:
            try:
                await self.ws.subscribe_symbols(chosen, orderbook=False)
            except Exception as exc:
                logger.warning(f"[장중·스캔] 재구독 실패: {exc}")
        top = ", ".join(f"{s_}{c:+.1%}" for s_, c, _ in pick[:5])
        logger.info(f"[장중·스캔] {len(rows)}종목 조회 → {len(chosen)}종목 구독 (상위: {top})")
        self.ledger.write("intraday_scan", {
            "scanned": len(rows), "subscribed": len(chosen),
            "top": [{"symbol": s_, "chg": round(c, 5), "vol_ratio": round(v, 3)}
                    for s_, c, v in pick[:10]]})

    def _intraday_score(self, sig: TriggerSignal) -> float:
        """돌파 사건의 선정 점수.

        지금은 **신선도 고정 공식**이다. 분봉 모델을 쓰려면 이 함수만 바꾸면
        되도록 분리해 뒀다 — 점수 산정이 감시기 밖에 있는 이유도 같다.

        모델 전환은 측정이 끝난 뒤에 한다. 부분 데이터에서 모델 선택 평균이
        무작위보다 낮았고(+0.567% vs +0.575%), 개선된 것은 꼬리뿐이었다
        (P(+100%) 1.76배). 공정한 비교 전에 바꾸면 그 상태를 실거래로 들고 간다.
        """
        return float(sig.freshness)

    def _feed_intraday(self, symbol: str, price: float, cum_volume: float,
                       now: dt.datetime) -> None:
        """시세 한 건을 감시기에 흘린다. 발동하면 대기열에 쌓고 `_tick()`이 처리한다.

        여기서 바로 주문하지 않는다 — WebSocket 콜백에서 주문을 내면 네트워크
        대기가 시세 수신을 막는다. 루프 주기가 3초이므로 지연은 그 안이고,
        백테스트의 '다음 분봉 시가 체결'(최대 60초)보다 빠르다.
        """
        if not self.s.USE_INTRADAY or price <= 0:
            return
        sig = self.intraday.on_quote(symbol, price, cum_volume, now)
        if sig is not None:
            self._intraday_queue.append(sig)
            logger.warning(f"[장중] 돌파 {symbol} {sig.describe()}")
            self.ledger.write("intraday_trigger", {
                "symbol": sig.symbol, "price": sig.price,
                "prev_close": sig.prev_close, "level_pct": round(sig.level_pct, 5),
                "minute_volume": sig.minute_volume, "vol_floor": sig.vol_floor})

    def _now(self) -> dt.datetime:
        """Mock은 가상 시계(봉 1개 = 1거래일), 실서버는 KST 실제 시각."""
        if self.s.USE_MOCK_MODE and self.broker is not None and getattr(self.broker, "is_mock", False):
            cursor = self.broker.sim.cursor  # type: ignore[attr-defined]
            step = int(self.s.MOCK_MINUTES_PER_LOOP)
            if step > 0:
                # 테스트 전용: 날짜를 고정하고 루프마다 분을 전진시킨다. 이렇게
                # 해야 아침 스캔(09:05)·마감 정리(15:15)가 실제로 발동한다
                # (기본 동작은 일 단위라 가상 시각이 항상 09:00이다).
                return self._mock_epoch + dt.timedelta(minutes=self._loops * step)
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
        # 상태 파일이 없거나 모드가 달라도 반드시 돈다(기록 없는 보유 = 오늘 진입).
        self._reconcile_entry_dates()
        # 장중 감시기에 전일 종가·거래량을 넣는다. 일봉 적재 뒤여야 한다.
        self._prime_intraday()
        # 다일 보유 청산 입력도 **갓 적재한 패널**에서 같이 만든다(`_refresh_alive`).
        today = self._now().strftime("%Y%m%d")
        await self._load_held_bars()
        self._refresh_alive(today)
        self._load_opening_history(today)
        self._load_vwap_profile()
        self._load_slide_profile()
        self._load_sliding_history(today)
        self._ref_day = today
        # 장중 전략은 일봉 신호를 **하루에 한 번도 다시 계산하지 않으므로**
        # (`_current_signals` 참고) 첫 계산을 장 전 기동 시점에 끝낸다. 그대로 두면
        # 09:00 첫 루프에서 돌아 실데이터 259종목 기준 **18.7초** 루프가 멈춘다 —
        # 바로 `vol_alive` 시가 매도가 나가야 하는 때다. WebSocket 구독 전에 해야
        # 관심목록 씨앗(`_refresh_watchlist`)이 첫 구독에 반영된다.
        if self.s.USE_INTRADAY:
            self._current_signals(self._now())

        if not self.s.USE_MOCK_MODE:
            await self._start_websocket()

        self.bus.subscribe(TOPIC_FILL, self._on_fill_event)
        self.bus.start()

        if self.dashboard:
            await self.dashboard.start()

        self._running = True
        if self._sliding_mode():
            self._sl_task = asyncio.create_task(self._sliding_poll_loop())
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
        # 규칙 진입 모드는 일봉 신호를 쓰지 않는다 — 모델 워밍업(80봉)과 20일 평균 거래대금 하한(30억)은
        # 측정(combo_study: 유니버스 파일 전 종목)에 없는 필터라, 신규 상장주 15개·거래대금 30억 미만 28개가
        # 실거래에서만 빠져 측정 485종목 대 실거래 443종목이 됐다(10/19). 여기서는 `vol_alive`의
        # 20봉 중앙값만 있으면 되고 유동성은 유니버스 파일(10억 하한)이 이미 걸렀다.
        rule_mode = self._rule_entry_mode()
        min_bars = 21 if rule_mode else self.strategy.warmup_bars()
        loaded: Dict[str, pd.DataFrame] = {}
        for symbol in candidates:
            try:
                df = await self.broker.get_bars(symbol, count=300, period="D")
                if df is None or len(df) < min_bars:
                    continue
                loaded[symbol] = df
            except Exception as exc:
                logger.warning(f"[Universe] {symbol} 적재 실패: {exc}")

        if not loaded:
            return False

        scored = []
        for symbol, df in loaded.items():
            value = float((df["close"] * df["volume"]).tail(20).mean())
            if not rule_mode and value < self.s.MIN_TRADING_VALUE:
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
        n_short = len(candidates) - len(loaded)
        n_cap = max(0, len(scored) - len(self.universe))
        if n_short:
            logger.info(f"[Universe] 후보 중 {n_short}종목 제외(적재 실패·봉 {min_bars}개 미만)")
        if len(loaded) - len(scored):
            logger.info(f"[Universe] 거래대금 하한으로 {len(loaded) - len(scored)}종목 제외")
        if n_cap:
            logger.warning(f"[Universe] UNIVERSE_SIZE={self.s.UNIVERSE_SIZE} 상한으로 거래대금 하위 "
                           f"{n_cap}종목을 잘랐다 — 유니버스 파일 전부를 쓰려면 상한을 올린다")
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
                on_orderbook=self._on_ws_orderbook if self.obrec else None,
            )
            # 호가 구독은 체결과 41건 한도를 나눠 쓴다. 수집을 켤 때만 요청한다.
            await self.ws.subscribe_symbols(self.watchlist,
                                            orderbook=bool(self.obrec))
            await self.ws.subscribe_notice()
            self.ws.start()
        except Exception as exc:
            logger.warning(f"[WS] 초기화 실패 — REST 폴링으로 동작합니다: {exc}")
            self.ws = None

    # ================================================================== 이벤트 핸들러
    async def _on_ws_quote(self, quote: Quote) -> None:
        if quote.price > 0:
            self.latest_prices[quote.symbol] = quote.price
            self.latest_volumes[quote.symbol] = float(quote.volume or 0.0)
            if self.obrec is not None:
                # 체결강도 분류 규칙(가격 >= 매도1호가 / 가격 > 중간가 ...)은
                # 하나가 아니다. 지금 박으면 다른 규칙으로 재계산할 수 없으므로
                # **원재료만** 남기고 분류는 분석 시점에 고른다.
                try:
                    self.obrec.observe_tape(quote.symbol, quote.price,
                                            float(quote.volume or 0.0),
                                            bid=float(quote.bid or 0.0),
                                            ask=float(quote.ask or 0.0),
                                            ts=self._now())
                except Exception as exc:
                    logger.debug(f"[호가수집] 테이프 실패: {exc}")
            # `Quote.volume`은 KIS `acml_vol`(누적거래량)이다. 감시기가 분이 바뀔 때
            # 기준값을 잡고 차분으로 이번 분의 거래량을 만든다.
            self._feed_intraday(quote.symbol, quote.price, quote.volume, self._now())

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
            self._settle_open_order(fill)
            await self.bus.publish(TOPIC_FILL, fill)
        except Exception as exc:
            logger.warning(f"[WS] 체결통보 처리 실패: {exc}")

    async def _on_ws_orderbook(self, ob: Dict[str, Any]) -> None:
        """호가 10단 수신 → 전방 수집.

        **전략은 이 값을 쓰지 않는다.** 호가는 과거 이력을 받을 수 없어 학습·검증이
        불가능하고(REST에 날짜 파라미터가 없고 WS는 구독 시점부터만), 검증 없이
        쓰는 것은 측정이 아니라 도박이다. 다음 대회를 위해 쌓기만 한다.

        예외를 삼킨다 — 여기서 던지면 WebSocket 수신 루프가 깨져 시세가 끊긴다.
        """
        if self.obrec is None:
            return
        try:
            self.obrec.observe_orderbook(ob, self._now())
        except Exception as exc:
            logger.debug(f"[호가수집] 처리 실패: {exc}")

    async def _on_fill_event(self, fill: Fill) -> None:
        pos = self.portfolio.positions.get(fill.symbol)
        entry = pos.avg_price if pos else 0.0
        was_flat = pos is None or pos.qty <= 0
        realized = self.portfolio.apply_fill(fill, sector=self.s.sector_of(fill.symbol))
        if fill.side is Side.SELL and entry > 0:
            self.risk.on_fill(fill, (fill.price - entry) / entry)

        # 손절선은 **체결 후 평단가 기준**으로 확정한다. 주문 접수 시점에는 포지션이
        # 아직 없으므로 여기서 설정하지 않으면 손절선이 비어버린다(부분체결 포함).
        if fill.side is Side.BUY:
            pos = self.portfolio.positions.get(fill.symbol)
            # 진입일은 **무포지션에서 처음 체결된 날**이다. 부분체결·추가 체결이 날짜를
            # 밀면 안 되고, 청산 후 재진입이면 새 날짜로 덮어야 한다(잔고 대조로
            # 포지션이 사라진 경로에서는 묵은 날짜가 남아 있을 수 있다).
            # 바로 저장한다 — 5루프 주기 저장 사이에 재시작하면 오늘 산 종목의 날짜가
            # 없어진다.
            if pos is not None and pos.qty > 0 and (was_flat or fill.symbol not in self._entry_dates):
                self._entry_dates[fill.symbol] = self._now().strftime("%Y%m%d")
                self._save_state()
            if pos is not None and pos.qty > 0:
                atr = self.atr_map.get(fill.symbol) or pos.avg_price * 0.02
                new_stop = self.risk.initial_stop(pos.avg_price, atr)
                pos.stop_price = max(pos.stop_price, new_stop) if pos.stop_price > 0 else new_stop
                # 장중 청산 감시도 **체결 시점**에 평단가로 시작한다. 접수 시점에는
                # 평단가가 없어 목표·손절 기준이 어긋난다(위 손절선과 같은 이유).
                # **마감 정리를 끄면 장중 감시를 열지 않는다.** 그러면 청산이
                # 전부 일봉 경로(`risk_manager`: 손절 + 감쇠 3종 + 시간손절)로
                # 가서 다일 보유가 된다. 감시를 열어 두면 장중 손절·트레일이
                # 일봉 규칙과 이중으로 걸려 어느 쪽이 청산했는지 알 수 없다.
                if self.s.USE_INTRADAY and self.s.INTRADAY_FLATTEN:
                    # 진입 점수를 함께 넣는다 — 교체 판정의 기준이 된다.
                    self.intraday.open_watch(
                        fill.symbol, pos.avg_price,
                        self._entry_scores.get(fill.symbol, 0.0))
        if fill.side is Side.SELL:
            pos_after = self.portfolio.positions.get(fill.symbol)
            if pos_after is None or pos_after.qty <= 0:
                self.intraday.close_watch(fill.symbol)
                if self._entry_dates.pop(fill.symbol, None) is not None:
                    self._save_state()
                if self._compliance_syms.pop(fill.symbol, None) is not None:
                    self._save_state()
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
        """현재가 갱신. 유량 보호를 위해 루프당 일부 종목만 회전 조회한다.

        ## 장중 전략에서는 WebSocket이 건강해도 돈다

        폭증 감시 대상은 유니버스 전체(284종목)인데 WebSocket 구독 한도는 40이다.
        WS만 보면 나머지 244종목의 폭증을 놓치고, 측정 모집단(257종목)과 실거래가
        갈라진다. 그래서 **WS가 받지 않는 종목을 REST로 돌아가며** 조회한다.

        한 바퀴가 루프당 4종목 x 약 3.5초로 3~4분이다. 측정된 감지 시각 중앙이
        개장 후 119분이고 판정 입력이 **당일 누적** 거래대금이라 드문드문 봐도
        값이 정확하다(분당 차분을 쓰는 breakout 모드와 다르다). 유량은 시세 P2라
        주문(P0)·계좌(P1)를 밀지 않는다.

        **WS가 받지 않는 보유 종목은 매 루프 맨 앞에 넣는다.** WS가 끊긴 경우와,
        REST 순회로 감지해 산 종목(관심목록 밖이라 구독돼 있지 않다)이 해당한다.
        284종목 순회에 섞이면 보유 종목 시세가 3분 넘게 멈춰 손절 판단이 늦어진다.
        """
        ws_ok = self.ws is not None and self.ws.is_healthy()
        held: List[str] = []
        if self._rule_entry_mode():
            # opening·sliding 모드는 진입 판정이 09:0N 분봉 한 번이라 유니버스를 하루 종일 돌
            # 이유가 없다. 오히려 그 유량이 판정 조회를 늦춘다. **WS 밖 보유 종목만** 본다.
            covered = set(self.watchlist) if ws_ok else set()
            held = [h for h in self.portfolio.open_positions() if h not in covered]
            watch = []
        elif self.s.USE_INTRADAY:
            covered = set(self.watchlist) if ws_ok else set()
            held = [h for h in self.portfolio.open_positions() if h not in covered]
            watch = [s_ for s_ in self.universe if s_ not in covered and s_ not in held]
        else:
            # WebSocket이 건강하면 REST 조회를 생략(유량 절약)
            if ws_ok:
                return
            watch = self.watchlist or self.universe
        targets: List[str] = list(held)
        n = len(watch)
        for _ in range(min(max(1, QUOTE_CHUNK - len(held)), n)):
            targets.append(watch[self._quote_cursor % n])
            self._quote_cursor += 1
        for symbol in targets:
            try:
                quote = await self.broker.get_quote(symbol)
                if quote and quote.price > 0:
                    self.latest_prices[symbol] = quote.price
                    self.latest_volumes[symbol] = float(quote.volume or 0.0)
                    self._feed_intraday(symbol, quote.price, quote.volume, self._now())
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
                # 진입일이 없으면 `vol_alive`가 영원히 판정하지 않는다. 오늘로 둔다
                # (`_reconcile_entry_dates`와 같은 규칙 — 최악이 하루 더 보유).
                self._entry_dates.setdefault(symbol, self._now().strftime("%Y%m%d"))
                logger.warning(f"[Sync] 미추적 포지션 발견 → 편입: {symbol} {remote.qty}주")
            elif local.qty != remote.qty:
                logger.warning(f"[Sync] 수량 불일치 {symbol}: 내부 {local.qty} → 브로커 {remote.qty}")
                local.qty = remote.qty
                local.avg_price = remote.avg_price or local.avg_price
        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in positions:
                logger.warning(f"[Sync] 브로커에 없는 포지션 제거: {symbol}")
                self.portfolio.positions.pop(symbol, None)
                self._entry_dates.pop(symbol, None)
        self.portfolio.cash = cash

    # ================================================================== 주문
    async def _submit(self, symbol: str, side: Side, qty: int, price: float, reason: str) -> bool:
        if qty <= 0:
            return False
        # 지정가 폭은 **매수와 매도가 비대칭**이다 (실측 33,482 돌파 사건 · 13.4M 분봉)
        #
        # 매수: 가격이 올라가는 중에 그 아래로 걸어 추격해야 한다. +0.4%로는
        # 체결률이 **61.5%**뿐이고, 못 사는 38.5%가 하필 "빠르게 뛴 급등"이다 —
        # 측정에서 가장 좋았던 집단(속도 26.5분 vs 61.8분)을 골라서 놓친다.
        #
        # | 폭 | 체결률 | 실제 체결가(중앙) |
        # |---|---|---|
        # | 0.4% | 61.5% | +0.121% |
        # | **2.0%** | **85.7%** | **+0.313%** |
        # | 5.0% | 95.5% | +0.409% |
        #
        # 지정가는 **상한**이고 대부분 그보다 싸게 체결되므로, 폭을 0.4%→2.0%로
        # 열면 체결률이 24%p 오르는 대신 체결가는 0.19%p만 불리해진다. 5%까지
        # 열지 않는 이유는 돌파선 대비 5% 오른 종목까지 사는 것이 되어 신선도
        # 신호와 반대 방향이기 때문이다.
        #
        # 매도: 가격이 내려가는 중에도 **아래로** 걸어 양보하는 쪽이라
        # −0.4%에서 체결률 **99.71%**(마감구간 99.84%)다. 넓힐 이유가 없다.
        band = self.s.BUY_LIMIT_BAND if side is Side.BUY else self.s.SELL_LIMIT_BAND
        limit_price = round_to_tick(price * (1.0 + band if side is Side.BUY
                                             else 1.0 - band))
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
            if ack.msg_cd == UNKNOWN_RESULT_CODE:
                # 응답을 못 받았다 — 접수됐을 수도 있다. 재전송하지 않고 **즉시 대조**한다.
                # 접수됐다면 미체결 목록에 잡혀 `open_orders`로 들어오고(이후 같은 종목 중복
                # 주문을 막는다), 체결됐다면 잔고에 잡힌다.
                logger.error(f"[주문불명] {side.value} {symbol} {qty}주 — 미체결·잔고 즉시 대조")
                self.ledger.write("order_unknown", {"symbol": symbol, "side": side.value,
                                                    "qty": qty, "price": limit_price})
                await self._sync_open_orders()
                await self._sync_portfolio()
                return False
            logger.error(f"[주문거부] {side.value} {symbol} {qty}주 :: {ack.msg_cd} {ack.msg}")
            if ack.msg_cd == "40070000":      # 모의투자 매매불가 종목 — 오늘은 다시 주문하지 않는다
                self._reject_today[symbol] = self._now().strftime("%Y%m%d")
            self.ledger.write("order_reject", {"symbol": symbol, "side": side.value,
                                               "qty": qty, "msg": ack.msg, "code": ack.msg_cd})
            return False

        if side is Side.SELL:
            self._sold_today[symbol] = self._now().strftime("%Y%m%d")
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
            if tracked is not None and (not fill.reason or fill.reason == "poll"):
                # 체결 사유는 주문 시점에만 알 수 있으므로 추적 정보에서 복원한다
                fill.reason = tracked.get("reason", "")
            self._settle_open_order(fill)
            await self.bus.publish(TOPIC_FILL, fill)

    def _settle_open_order(self, fill: Fill) -> None:
        """체결 수량만큼 미체결 추적 잔량을 줄이고, 0이 되면 지운다.

        **체결통보(WS)와 폴링 양쪽에서 부른다.** 모의투자는 미체결 조회
        (`inquire-psbl-rvsecncl`)가 `90000000 해당업무가 제공되지 않습니다`이고 당일
        체결조회도 비어 와서, 대조·폴링으로는 추적이 영영 안 지워진다. 그러면 판
        종목이 재시작 전까지 자리 하나를 차지해 장중 추가 매수가 멈춘다(10/16 실측).
        """
        tracked = self.open_orders.get(fill.order_no)
        if tracked is None:
            return
        tracked["qty"] = max(0, int(tracked.get("qty", 0)) - fill.qty)
        if tracked["qty"] <= 0:
            self.open_orders.pop(fill.order_no, None)

    def _check_disk(self, now: dt.datetime) -> None:
        """남은 디스크를 재고 모자라면 경고한다(한 시간에 한 번). 디스크가 차면 원장·상태를 못 쓴다."""
        self._disk_free_gb = disk_free_gb(str(Path(self.s.LOG_DIR)))
        if self._disk_free_gb is None or self._disk_free_gb >= DISK_WARN_GB:
            return
        if self._disk_warned_at and (now - self._disk_warned_at).total_seconds() < 3600:
            return
        self._disk_warned_at = now
        logger.warning(f"[디스크] 남은 용량 {self._disk_free_gb:.2f}GB < {DISK_WARN_GB}GB — "
                       f"로그·백업 정리 필요(원장 저장 실패 위험)")

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

        # --- 날짜 롤오버: 전일 기준값 재생성 ---
        # **장 상태 체크보다 앞에 둔다.** 실서버에서는 자정 직후(장외)에 날짜가
        # 바뀌므로, 뒤에 두면 아래 `return`에 막혀 09:00 첫 루프에서야 돌고 그동안
        # (일봉 재적재 수 분) 시가 청산과 진입 감시가 멈춘다.
        today = now.strftime("%Y%m%d")
        if self._ref_day and today != self._ref_day:
            await self._roll_daily_reference(today)

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
        # 일봉 재적재는 **장중 전략에서는 하지 않는다.**
        #
        # `_refresh_bars()`는 유니버스 전체(284종목) 일봉을 받으며 블로킹한다.
        # 모의투자 유량 1.8건/초로 **158초**가 걸리고, `BAR_REFRESH_LOOPS=40`
        # (약 2분)마다 돌면 장중의 절반 이상이 멈춘다.
        #
        # 그리고 분봉 전략에 **필요하지 않다.** 트리거가 쓰는 일봉 입력은 전일
        # 종가·거래량뿐이고 그것은 하루 내내 고정이다. 장 마감 후 다음 날 시작
        # 시점에 `startup()`이 새로 적재한다.
        #
        # 장중 호가는 `_refresh_quotes()`가 `QUOTE_CHUNK=4`로 루프당 4종목씩만
        # 돌려 블로킹하지 않는다(그쪽은 안전하다).
        if (self._loops % BAR_REFRESH_LOOPS == 0 and self._loops > 0
                and not self.s.USE_MOCK_MODE and not self.s.USE_INTRADAY):
            await self._refresh_bars()
        self._apply_latest_prices_to_panel()
        if self.s.USE_INTRADAY and self._loops % BAR_REFRESH_LOOPS == 0:
            self._prime_intraday()
        # 아침 후보 스캔(하루 1회). 전일 기준 적재 뒤여야 상승률을 계산할 수 있다.
        #
        # **await 하지 않고 별도 태스크로 띄운다.** 유니버스 284종목을 모의투자
        # 유량(2건/초, 안전마진 후 1.8건/초)으로 조회하면 **158초**가 걸린다.
        # `_tick()` 안에서 await 하면 그 2.6분 동안 돌파 감지도 청산도 처리되지
        # 않는다 — 측정에서 이른 돌파가 좋고(상위군 36.8분 vs 하위군 74.9분)
        # 진입 슬리피지가 크다고 나왔으므로 09:05~09:07을 날리는 것은 치명적이다.
        #
        # 재진입은 `_scanned_day`가 막는다(함수 진입 직후에 설정한다).
        #
        # **opening 모드에서는 돌지 않는다.** 그 스캔은 누적 폭증을 실시간으로 보려고
        # WS 40종목을 고르는 것인데 opening은 09:0N 분봉 판정 한 번이 전부다. 그리고
        # 같은 유량(284건)을 판정 조회와 나눠 쓰면 판정이 진입 시각(09:0N+지연)을 넘긴다.
        if (self.s.USE_INTRADAY and not self._rule_entry_mode()
                and self._scanned_day != now.strftime("%Y%m%d")):
            if self._scan_task is None or self._scan_task.done():
                self._scan_task = asyncio.create_task(self._scan_candidates(now))
        # opening 판정 — 09:0N이 지나면 하루 한 번 띄운다. 결과를 먼저 비워서 전날
        # 판정이 오늘 진입으로 새지 않게 한다.
        n_open = int(self.s.OPENING_WINDOW_MIN)
        if (self._opening_mode() and self._opening_day != now.strftime("%Y%m%d")
                and now.strftime("%H%M") >= f"{9 + n_open // 60:02d}{n_open % 60:02d}"):
            self._opening_day = now.strftime("%Y%m%d")
            self._opening_picks = None
            late = (now.hour - 9) * 60 + now.minute - n_open
            if late > OPENING_LATE_MAX_MIN:
                # **늦게 기동했으면 판정하지 않는다.** 복수시세의 누적 거래대금은 09:00부터
                # 지금까지라, 11시에 재면 5분 기준선 대비 거의 전 종목이 발동한다(측정에 없는 진입).
                # 빈 결과로 두면 `_opening_entries`가 오늘 처리를 끝내고 장중 추가 매수가 열린다.
                self._opening_picks = []
                self._board_day(self._opening_day)["opening"] = {
                    "time": now.strftime("%H:%M:%S"), "window": n_open,
                    "mult": float(self.s.OPENING_MULT), "checked": 0, "rows": [],
                    "skipped": f"{now:%H:%M} 기동 — 판정 시각이 {late}분 지나 건너뜀"}
                logger.warning(f"[장중·개장] {now:%H:%M} 기동 — 09:0{n_open} 판정이 {late}분 지나 건너뜀 "
                               f"(장중 추가 매수만 본다)")
            else:
                self._opening_task = asyncio.create_task(self._opening_scan(now))

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
        # 거래량 감소 청산(`EXIT_RULES`의 vol_decay) 입력.
        # 거래대금 = 현재가 x 당일 누적 거래량으로, 백테스트
        # (`backtest/engine.py`: 종가 x 거래량)와 **같은 정의**를 쓴다.
        # 한쪽만 넘기면 두 경로가 갈려 측정이 실거래를 반영하지 못한다.
        turnover_map = {sym: float(px) * float(self.latest_volumes.get(sym, 0.0))
                        for sym, px in prices.items()}
        # (`vol_decay`가 쓰는 이 장중 누적값은 아래 `vol_alive`와 같은 문제 — 아침에
        #  누적이 0에 가깝다 — 를 갖는다. `vol_decay`는 측정에서 기각돼 꺼져 있다.
        #  다시 켜려면 먼저 확정 일봉 입력으로 바꿔야 한다.)
        #
        # `vol_alive` 입력 — **어제 확정된** 거래대금 / 그 전일까지 20봉 중앙값.
        #
        # 예전에는 여기서 장중 누적 거래대금(위 `turnover_map`)을 20봉 중앙값으로
        # 나눴다. 측정은 "봉 t 종가에 하루치로 판정 → t+1 시가 매도"인데, 장중 누적은
        # 다음 날 09:00에 거의 0이라 **살아 있는 종목도 매일 아침 팔렸다**(재현:
        # 하루 x25인 날이 09:00에 x0.25로 읽혔다). 그래서 하루 한 번 확정 일봉으로
        # 만든 값(`_refresh_alive`)을 쓰고, 첫 루프(09:00)에 판정이 나면 시가에 판다.
        #
        burst_map = self._vol_alive_map(today)
        exits = self.risk.evaluate_exits(self.portfolio, prices, self.atr_map,
                                         self.score_map, now,
                                         turnover_map=turnover_map,
                                         burst_map=burst_map)
        if self.s.USE_TOP_K:
            exits.extend(self._rotation_exits(signals, prices, exits))
        for order in exits:
            price = prices.get(order.symbol, order.price)
            qty = self._sellable_qty(order.symbol, order.qty)
            if qty <= 0:
                continue  # 이미 낸 매도가 체결 대기 중이다
            ok = await self._submit(order.symbol, Side.SELL, qty, price,
                                    f"{order.reason.value}|{order.detail}")
            if ok:
                self.ledger.write("exit", {"symbol": order.symbol, "qty": qty,
                                           "reason": order.reason.value, "detail": order.detail})

        # --- 목표 도달 잠금 (설정 LOCK_RETURN) ---
        await self._lock_check(now, prices)

        # --- 장중 VWAP 청산 (설정 VWAP_EXIT) ---
        await self._vwap_exits(now)

        # --- 장중 청산 (분봉 전략) ---
        # 리스크 청산 뒤에 본다. 킬스위치·급락은 장중 규칙보다 우선해야 한다.
        if self.s.USE_INTRADAY:
            for symbol in list(self.intraday.watches):
                pos = self.portfolio.positions.get(symbol)
                if pos is None or pos.qty <= 0:
                    self.intraday.close_watch(symbol)
                    continue
                price = prices.get(symbol, 0.0)
                if price <= 0:
                    continue
                reason = self.intraday.exit_reason(symbol, price, now)
                if reason is None:
                    continue
                qty = self._sellable_qty(symbol, pos.qty)
                if qty <= 0:
                    continue  # 이미 낸 매도가 체결 대기 중이다
                ok = await self._submit(symbol, Side.SELL, qty, price, f"INTRADAY|{reason}")
                if ok:
                    self.ledger.write("exit", {"symbol": symbol, "qty": qty,
                                               "reason": "intraday", "detail": reason})

        # --- 장중 진입: opening 모드 (09:0N 판정 → 09:(N+지연) 한 번) ---
        if self._opening_mode() and self.risk.state == RiskState.NORMAL and not self._locked:
            await self._opening_entries(now, prices)

        # --- 장중 추가 매수 (SLIDE_ENTRY) — 장 시작 진입이 끝난 뒤 빈자리를 장중 발동으로 ---
        if self.s.SLIDE_ENTRY and self.risk.state == RiskState.NORMAL and not self._locked:
            self._slide_kick(now)
            await self._slide_entries(now, prices)

        # --- 장중 진입: sliding 모드 (감시 루프가 쌓은 발동) ---
        if self._sliding_mode() and self.risk.state == RiskState.NORMAL and not self._locked:
            await self._sliding_entries(now, prices)

        # --- 장중 진입 (분봉 전략) ---
        if self.s.USE_INTRADAY and not self._rule_entry_mode():
            # 발동한 사건을 **명부에 올린다.** 매 루프 비우면 같은 분에 동시 발동한
            # 것들 사이에서만 고르게 되고, 실측에서 슬롯 2개가 09:00~09:01에 차는
            # 날이 76.9%라 나머지 12건이 선택 기회 없이 버려졌다
            # (`IntradayWatcher.pick_entries()` 주석 참고).
            queued, self._intraday_queue = self._intraday_queue, []
            for sig in queued:
                self.intraday.add_candidate(sig, self._intraday_score(sig))

            inflight = self.open_orders_symbols()
            if self.risk.state != RiskState.NORMAL:
                if queued:
                    logger.info(f"[장중] 리스크 상태 {self.risk.state} — 돌파 {len(queued)}건 보류")
            elif self.intraday.should_flatten(now):
                if queued:
                    logger.info(f"[장중] 마감 정리 시각 — 돌파 {len(queued)}건 진입하지 않음")
            else:
                held = self._strategy_held()          # 요건 매매 포지션은 자리 수에서 뺀다
                free = max(0, int(self.s.MAX_POSITIONS)
                           - len(held | (set(inflight) - self._compliance_exempt())))
                picks = self.intraday.pick_entries(now, free, exclude=set(inflight))
                # 슬롯이 꽉 찼으면 교체를 검토한다. 매도만 내고, 매수는 슬롯이
                # 비는 다음 루프에서 `pick_entries()`가 집는다.
                if free <= 0:
                    tgt = self.intraday.swap_target(now, exclude=set(inflight))
                    if tgt is not None:
                        drop, incoming = tgt
                        pos = self.portfolio.positions.get(drop)
                        if pos is not None and pos.qty > 0:
                            px = prices.get(drop, pos.avg_price)
                            ok = await self._submit(
                                drop, Side.SELL, pos.qty, px,
                                f"INTRADAY_SWAP|{incoming.symbol}로 교체")
                            if ok:
                                self.intraday.note_swap(incoming)
                                logger.warning(
                                    f"[장중·교체] {drop} 매도 → {incoming.symbol} "
                                    f"({self.intraday.swaps_today}/{self.s.INTRADAY_MAX_SWAPS}회)")
                                self.ledger.write("intraday_swap", {
                                    "out": drop, "in": incoming.symbol,
                                    "in_freshness": round(incoming.freshness, 4)})
                for sig in picks:
                    if sig.symbol in inflight:
                        continue
                    # 리스크 매니저가 모든 주문의 최종 관문이다. 장중 경로도 우회하지 않는다.
                    #
                    # 점수는 `RULE_TRIGGER_SCORE`(1.0)다. 예전에는 "가격 상승률 x 10"을
                    # 넣었는데, 리스크 관문의 `SIGNAL_BUY_THRESHOLD`(0.22, 일봉 모델용)에
                    # 걸려 **전일 대비 +2.2% 미만에서 터진 폭증이 전부 거부됐다.** burst
                    # 판정은 일부러 가격 조건을 걸지 않고(위 감시기 주석) 측정에도 그 필터가
                    # 없다 — 측정에 없는 규칙이 실거래에만 있었다(2026-10-09 수정).
                    # 고정 사이징에서는 점수가 비중에 쓰이지 않으므로 수량은 그대로다.
                    entry_sig = Signal(
                        symbol=sig.symbol, score=RULE_TRIGGER_SCORE,
                        price=sig.price, ts=sig.ts,
                        atr=self.atr_map.get(sig.symbol, 0.0),
                        volatility=self.vol_map.get(sig.symbol, 0.0),
                        bet=-1.0, reason="intraday_breakout")
                    decision = self.risk.evaluate_entry(entry_sig, self.portfolio, now,
                                                        pending=self._pending_notional(),
                                                slot_exempt=self._compliance_exempt())
                    self.ledger.write("risk_decision", {
                        "symbol": sig.symbol, "score": round(entry_sig.score, 4),
                        "approved": decision.approved, "qty": decision.qty,
                        "reason": decision.reason, "path": "intraday", **decision.detail})
                    if not decision.approved:
                        logger.info(f"[장중] {sig.symbol} 진입 거부 — {decision.reason}")
                        continue
                    # 체결 시점에 `open_watch`가 교체 기준 점수로 쓴다. 접수와
                    # 체결 사이에 신호 객체가 사라지므로 여기서 남겨 둔다.
                    self._entry_scores[sig.symbol] = self._intraday_score(sig)
                    await self._submit(sig.symbol, Side.BUY, decision.qty, sig.price,
                                       f"INTRADAY|{sig.describe()}")

        # --- 진입 (일봉 신호) ---
        # 장중 전략을 쓰면 진입은 돌파가 결정한다. 두 경로가 같이 돌면 익스포저가
        # 의도의 두 배가 되고, 측정한 분포(하루 2종목 x 45%)와 달라진다.
        if self.risk.state == RiskState.NORMAL and not self.s.USE_INTRADAY:
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
                decision = self.risk.evaluate_entry(sig, self.portfolio, now, pending=self._pending_notional(),
                                                slot_exempt=self._compliance_exempt())
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

        # --- 요건 전용 소액 매매(지수 종목 5개) ---
        if self.risk.state == RiskState.NORMAL or self._compliance_syms:
            await self._compliance_trades(now, prices)

        # --- 체결 확인 ---
        if self.s.USE_MOCK_MODE or self._loops % FILL_POLL_LOOPS == 0:
            await self._poll_fills()

        # --- 잔고/미체결 대조 ---
        if self._loops % RECONCILE_LOOPS == 0 and self._loops > 0:
            await self._sync_portfolio()
            await self._sync_open_orders()
            self._check_disk(now)
        elif (self.s.USE_INTRADAY and self.intraday.should_flatten(now)
              and any(o.get("side") == Side.SELL.value for o in self.open_orders.values())):
            # **마감 정리 구간만 촘촘하게 본다.** 미체결 매도는 취소되고 나서야
            # 다시 접수되는데(`_sellable_qty` 가드), 평시 주기는 90초이고 TTL은
            # 180초라 15:15~15:30의 재시도가 4회뿐이다. 이 구간의 미체결은
            # **오버나이트로 직결**되므로 TTL을 짧게 잡고 매 루프 확인한다.
            # 비용은 미체결 조회(P1) 한 건이고, 매도 주문이 남아 있을 때만 돈다.
            await self._sync_open_orders(max_age_seconds=FLATTEN_ORDER_TTL)

        # --- 상태 저장 ---
        if self._loops % 5 == 0:
            self._save_state()
            if self.obrec is not None:
                self.obrec.flush()

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
        # **규칙 진입 모드(opening·sliding)에서는 일봉 신호를 계산하지 않는다** (2026-10-13).
        #
        # 그 모드에서 신호는 진입에 쓰이지 않는다 — 손절선은 ATR 이 있든 없든 25%로 같고(넓은 쪽을
        # 쓴다), 사이징은 고정 비중, 신호 반전 청산은 꺼져 있다. 그런데 계산은 실서버에서 **실패**한다:
        # KIS 일봉 조회가 한 번에 100봉만 주는데 피처가 120봉 이상을 요구한다(`index -120 is out of
        # bounds ... size 100`). 실패하면 매 루프 다시 시도해 장중 루프가 약 20초씩 멈췄을 것이다.
        if self._rule_entry_mode():
            return []
        # 같은 봉에서 이미 실패했으면 다시 시도하지 않는다(매 루프 수십 초씩 막힌다)
        if self._signals_failed_ts and self._signals_failed_ts == self._last_bar_ts:
            return self._signals
        if self._signals and self._signals_bar_ts == self._last_bar_ts:
            return self._signals
        # 장중 전략에서는 **한 번 계산하면 그날은 다시 하지 않는다.**
        #
        # 일봉 신호가 쓰이는 곳은 관심목록 씨앗(`_refresh_watchlist`)과 사이징 보조
        # 값(`vol_map`·`atr_map`)뿐이다. 진입은 돌파가 결정하고 신호 반전 청산도
        # 해제했으므로, 장중에 다시 계산할 이유가 없다.
        #
        # 그런데 비용이 크다 — 20종목에 **9~13초**(테마 군집이 42%)이고 유니버스가
        # 넓어지면 더 늘어난다. 루프 간격은 3초다. Mock 스모크에서 이 계산 하나로
        # 70초 동안 루프가 **1회**만 돌았다. 장중에 그렇게 멈추면 돌파를 놓친다.
        if self.s.USE_INTRADAY and self._signals:
            return self._signals
        try:
            signals = self.strategy.generate_signals(
                self.panel, self.portfolio.open_positions(), now)
        except Exception as exc:
            self._errors += 1
            logger.exception(f"[Engine] 신호 생성 실패: {exc}")
            self._signals_failed_ts = self._last_bar_ts
            return self._signals          # 직전 신호를 유지한다(빈 목록으로 떨어뜨리지 않는다)
        self._signals = signals
        self._signals_bar_ts = self._last_bar_ts
        self.score_map = {x.symbol: x.score for x in signals}
        self.atr_map = {x.symbol: x.atr for x in signals}
        # 장중 경로가 사이징에 쓸 변동성. 일봉 신호에서 같이 받아 둔다.
        self.vol_map = {x.symbol: x.volatility for x in signals}
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
        if self.s.USE_INTRADAY:
            # 장중 전략에서는 '점수 상위'가 아니라 **오늘 돌파할 가능성**이 기준이다.
            # 돌파선에 가까운 종목을 우선 구독한다 — 멀리 있는 종목의 시세는
            # 지금 쓸 일이 없고, 구독 한도가 41건이라 자리가 귀하다.
            # 아직 시세를 못 본 종목은 거리가 무한이라 뒤로 밀린다. 그래서
            # 일봉 점수 상위를 소수 섞어 초기 탐색이 끊기지 않게 한다.
            seed = [x.symbol for x in signals if x.symbol not in held][: max(5, cap // 4)]
            cand = [x.symbol for x in signals if x.symbol not in held and x.symbol not in seed]
            cand.sort(key=self.intraday.near_trigger)
            ranked = seed + cand
        else:
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

    def _pending_sell_qty(self, symbol: str) -> int:
        """미체결 **매도** 주문의 잔량 합.

        청산 판정은 포지션 수량을 보는데, 주문을 낸 직후에도 체결되기 전까지
        수량이 그대로 남아 있다. 가드가 없으면 매 루프 같은 수량을 다시 접수해
        `40320000 주문가능수량을 초과했습니다`가 쌓인다 — 실측에서 매도 4건에
        거부 15건이 났다. 하필 마감 정리(15:15~15:30)에 집중되는데, 그 구간은
        주문 레인(P0)을 낭비하면 안 되는 구간이다.
        """
        return sum(int(o.get("qty", 0)) for o in self.open_orders.values()
                   if o.get("symbol") == symbol and o.get("side") == Side.SELL.value)

    def _sellable_qty(self, symbol: str, pos_qty: int) -> int:
        """지금 추가로 낼 수 있는 매도 수량. 보류 잔량을 뺀다."""
        return max(0, int(pos_qty) - self._pending_sell_qty(symbol))

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
        # 아침 스캔 태스크가 남아 있으면 취소한다. 158초짜리 작업이라 종료가
        # 그만큼 늦어지고, 종료 중 주문을 내면 안 된다.
        if self._scan_task is not None and not self._scan_task.done():
            self._scan_task.cancel()
        if self._opening_task is not None and not self._opening_task.done():
            self._opening_task.cancel()        # 같은 이유 — 284종목 분봉 조회
        if self._opening_record_task is not None and not self._opening_record_task.done():
            self._opening_record_task.cancel()
        if self._slide_task is not None and not self._slide_task.done():
            self._slide_task.cancel()          # 장중 추가 매수 조회 — 종료 중에는 주문을 내지 않는다
        if self._sl_task is not None and not self._sl_task.done():
            self._sl_task.cancel()
        self._flush_sliding_bars()             # 오늘 받은 분봉을 남긴다(내일 기준선)
        if not self._running:
            return
        self._running = False
        logger.warning("[Shutdown] 종료 시퀀스를 시작합니다...")

        # 0) 호가·체결 버퍼를 먼저 비운다. 뒤쪽 단계가 실패하면 수집분이 사라진다.
        if self.obrec is not None:
            n = self.obrec.flush()
            snap = self.obrec.snapshot()
            logger.info(f"[호가수집] 종료 flush {n:,}행 · "
                        f"호가 {snap['symbols_ob']}종목 · 테이프 {snap['symbols_tape']}종목 · "
                        f"오류 {snap['errors']}")

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
                "entry_dates": self._entry_dates,
                "compliance_syms": self._compliance_syms,
                "sold_today": self._sold_today,
                "locked": self._locked,
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
            # **다른 모드의 상태는 복구하지 않는다.** 같은 파일을 Mock과 실서버가
            # 같이 쓰므로, Mock 실행의 자산 고점·진입일(가상 날짜)이 실계좌로 넘어오면
            # 첫날부터 낙폭이 잡히고 보유 봉 수가 엉뚱하게 커진다.
            mode = "MOCK" if self.s.USE_MOCK_MODE else self.s.KIS_ENV
            if data.get("mode") != mode:
                logger.info(f"[State] 저장 모드 {data.get('mode')} ≠ 현재 {mode} — 복구하지 않음")
                return
            peak = float(data.get("peak_equity") or 0)
            if peak > 0:
                self.portfolio.peak_equity = max(self.portfolio.peak_equity, peak)
            logger.info(f"[State] 이전 상태 복구 (peak_equity={self.portfolio.peak_equity:,.0f})")
            saved = data.get("entry_dates") or {}
            self._entry_dates = {str(k): str(v) for k, v in saved.items()}
            comp = data.get("compliance_syms") or {}
            self._compliance_syms = {str(k): str(v) for k, v in comp.items()}
            # 오늘 판 종목 — 장중 재시작 뒤 같은 날 다시 사지 않게(날짜를 같이 저장하므로 어제 것은 무해)
            sold = data.get("sold_today") or {}
            self._sold_today = {str(k): str(v) for k, v in sold.items()}
            self._locked = bool(data.get("locked", False))
            if self._locked:
                logger.warning("[잠금] 목표 도달 잠금 상태 복구 — 신규 매수 없음")
        except Exception as exc:
            logger.debug(f"[State] 복구 실패: {exc}")

    def _reconcile_entry_dates(self) -> None:
        """재시작 후 보유 종목의 진입일과 보유 봉 수를 맞춘다.

        - 기록이 없는 보유 종목은 **오늘 진입으로 간주**한다. 어제로 간주하면 오늘
          산 종목이 폭증 전날의 낮은 거래대금으로 판정돼 곧바로 팔린다. 오늘로
          두면 최악이 하루 더 보유하는 것이다.
        - `bars_held`는 잔고 대조로 0부터 시작하므로 진입일로부터 다시 센다.
          그래야 `MAX_HOLDING_BARS`(안전 상한)가 재시작으로 무력화되지 않는다.
          연속 실행에서 날짜가 바뀔 때 1씩 오르는 것과 같은 값이다(진입 당일 0).
        """
        today = self._now().date()
        today_s = today.strftime("%Y%m%d")
        held = self.portfolio.open_positions()
        self._entry_dates = {k: v for k, v in self._entry_dates.items() if k in held}
        for symbol, pos in held.items():
            entered = self._entry_dates.get(symbol)
            if not entered:
                self._entry_dates[symbol] = today_s
                logger.warning(f"[State] {symbol} 진입일 기록 없음 — 오늘 진입으로 간주"
                               "(vol_alive 판정은 내일부터)")
                continue
            if self.s.USE_MOCK_MODE:
                continue    # Mock 가상 날짜는 거래일 달력과 무관하다
            try:
                start = dt.datetime.strptime(entered, "%Y%m%d").date()
            except ValueError:
                continue
            days = 0
            cur = start + dt.timedelta(days=1)
            while cur <= today:
                if self.calendar.is_trading_day(cur):
                    days += 1
                cur += dt.timedelta(days=1)
            pos.bars_held = max(pos.bars_held, days)
            logger.info(f"[State] {symbol} 진입 {entered} · 보유 {pos.bars_held}봉")

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

    def _board_day(self, day: str) -> Dict[str, Any]:
        """오늘 판정판. 날짜가 바뀌면 비운다."""
        if self._board["day"] != day:
            self._board = {"day": day, "opening": None, "slide": []}
        return self._board

    def _board_row(self, symbol: str, ratio: float, now: dt.datetime, outcome: str) -> Dict[str, Any]:
        return {"symbol": symbol, "name": self.s.name_of(symbol), "ratio": round(float(ratio), 2),
                "time": now.strftime("%H:%M:%S"), "outcome": outcome}

    @staticmethod
    def _board_mark(rows: List[Dict[str, Any]], symbol: str, outcome: str) -> None:
        """그 종목의 가장 최근 행에 결과를 적는다."""
        for row in reversed(rows):
            if row["symbol"] == symbol:
                row["outcome"] = outcome
                return

    def state_snapshot(self) -> Dict[str, Any]:
        """대시보드/헬스체크용 통합 스냅샷."""
        limiter = getattr(self.broker, "limiter", None)
        auth = getattr(self.broker, "auth", None)
        return {
            "competition": self._competition_snapshot(),
            "portfolio": self.portfolio.snapshot(),
            "risk": self.risk.snapshot(self.portfolio),
            "strategy": self.strategy.snapshot(),
            "triggers": (self._board if self._board["day"] == self._now().strftime("%Y%m%d")
                         else {"day": "", "opening": None, "slide": []}),
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
                # 외부 API(TR)별 왕복 지연 p50/p95/p99 · 오류 — `core/latency.py`
                "http_latency": (self.broker.http_stats.snapshot()
                                 if getattr(self.broker, "http_stats", None) else {}),
                "http_latency_prom": (self.broker.http_stats.prometheus()
                                      if getattr(self.broker, "http_stats", None) else []),
                "event_dropped": self.bus.dropped,
                "clock_skew_s": getattr(self.broker, "clock_skew_s", None),
                "disk_free_gb": self._disk_free_gb,
            },
        }


# ====================================================================== 디스크
def disk_free_gb(path: str = ".") -> Optional[float]:
    """원장·로그가 쓰이는 디스크의 남은 용량(GB). 확인 실패면 None."""
    import shutil
    try:
        return round(shutil.disk_usage(path).free / 1024 ** 3, 2)
    except OSError:
        return None


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
