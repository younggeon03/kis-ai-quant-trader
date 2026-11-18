"""전 모듈 스모크 테스트 — 계좌 발급 전 로직 검증용.

    python scripts/selftest.py

검증 항목
  1. 설정/TR_ID 환경 분기          9. 백테스트 엔진 + 성과지표
  2. 호가단위·가격제한폭 규칙     10. 이벤트 버스
  3. 기술적 지표 정확성/인과성    11. 실서버 클라이언트(오프라인)
  4. 유량 제어기(실측 RPS)        12. 과적합 검증(CPCV·DSR·PBO)
  5. 우선순위 큐(주문 우선)       13. 표본 가중치·베팅 사이징
  6. Mock 체결 엔진               14. 대회 규정 필터·요건 추적
  7. 리스크 매니저                15. 종목 선정·대회 국면
  8. 피처·라벨 파이프라인         16. 패널 정렬 이력 하한

모든 항목이 PASS여야 실전 연동 준비가 된 것이다.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import enable_utf8_console  # noqa: E402

enable_utf8_console()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import Settings, get_settings  # noqa: E402
from core.events import (  # noqa: E402
    EventBus,
    ExitReason,
    Fill,
    OrderType,
    Portfolio,
    Position,
    Side,
    Signal,
)
from core.rate_limiter import (  # noqa: E402
    PRIORITY_ORDER,
    PRIORITY_QUOTE,
    PriorityRateLimiter,
)
from utils.indicators import atr, bollinger, ema, macd, rsi, sma  # noqa: E402
from utils.logger import setup_logging  # noqa: E402
from utils.market_rules import price_limits, round_to_tick, tick_size  # noqa: E402

RESULTS: List[Tuple[str, bool, str, float]] = []


def check(name: str):
    """테스트 데코레이터(동기/비동기 모두 지원)."""

    def wrapper(fn: Callable):
        async def runner():
            start = time.perf_counter()
            try:
                out = fn()
                if asyncio.iscoroutine(out):
                    out = await out
                RESULTS.append((name, True, str(out or "OK"), time.perf_counter() - start))
            except AssertionError as exc:
                RESULTS.append((name, False, f"검증 실패: {exc}", time.perf_counter() - start))
            except Exception as exc:
                detail = traceback.format_exc().strip().splitlines()[-1]
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc} | {detail}",
                                time.perf_counter() - start))
        runner.__name__ = fn.__name__
        return runner

    return wrapper


import tempfile as _tempfile

_SELFTEST_TMP = Path(_tempfile.mkdtemp(prefix="bankis_selftest_"))


def make_settings() -> Settings:
    """테스트 전용 경량 설정(종목 수/봉 수 축소)."""
    s = get_settings()
    s.UNIVERSE = s.UNIVERSE[:6]
    s.UNIVERSE_SIZE = 6
    s.MOCK_HISTORY_BARS = 260
    s.MOCK_FUTURE_BARS = 120
    s.REST_RATE_LIMIT = 20.0
    s.REST_BURST = 10
    s.MIN_TRADING_VALUE = 0.0
    # --- 환경 의존을 끊는다 ---
    #
    # 테스트가 `.env`를 읽으면 **환경 설정을 검증하는 것이 되어** 코드가 맞는지
    # 알 수 없다. 실제로 두 번 깨졌다 — `.env`를 `SIZING_MODE=fixed`로 바꾸자
    # 변동성 타게팅 검증이, 분봉 전략을 위해 리스크 청산을 해제하자 킬스위치·국면
    # 검증이 깨졌다. 둘 다 코드 버그가 아니라 설정 변경이었다.
    #
    # 그래서 테스트가 단정하는 값은 **코드 기본값으로 고정**한다. 각 모드·규칙의
    # on/off 동작은 해당 테스트 안에서 따로 검증한다.
    fields = Settings.model_fields
    for key in ("SIZING_MODE", "MDD_SOFT_LIMIT", "MDD_HARD_LIMIT", "DAILY_LOSS_LIMIT",
                "TAKE_PROFIT_PCT", "TRAILING_ACTIVATE_PCT", "CRASH_DROP_PCT",
                "STOP_LOSS_PCT", "MAX_POSITIONS", "MAX_POSITION_WEIGHT",
                "MAX_SECTOR_WEIGHT", "USE_INTRADAY",
                "COMP_START", "COMP_END", "PHASE1_END",
                "PHASE1_TOP_K", "PHASE2_TOP_K"):
        if key in fields:
            setattr(s, key, fields[key].default)
    # --- 실거래 파일과 격리한다 ---
    #
    # 셀프테스트가 `TradingEngine`을 만들면 원장(`logs/ledger.jsonl`)·상태 파일·로그에 기록한다.
    # 실거래 중에 셀프테스트를 돌렸더니 **대회 근거자료인 원장에 테스트 기록 65건이 섞였다**
    # (2026-10-13 — 체결은 0건이라 요건 집계는 무사). 그래서 항상 임시 폴더로 돌린다.
    s.LEDGER_PATH = str(_SELFTEST_TMP / "ledger.jsonl")
    s.LOG_DIR = str(_SELFTEST_TMP / "logs")
    s.STATE_DIR = str(_SELFTEST_TMP / "state")
    return s


# ============================================================ 1. 설정 / TR_ID
@check("1. 설정·TR_ID 환경 분기")
def test_settings():
    s = make_settings()
    s.KIS_ENV = "vts"
    assert s.tr("order_buy") == "VTTC0802U", s.tr("order_buy")
    assert s.tr("order_sell") == "VTTC0801U"
    assert s.tr("inquire_balance") == "VTTC8434R"
    assert "openapivts" in s.rest_base_url
    s.KIS_ENV = "prod"
    assert s.tr("order_buy") == "TTTC0802U"
    assert s.tr("inquire_balance") == "TTTC8434R"
    assert s.rest_base_url == "https://openapi.koreainvestment.com:9443"
    s.KIS_ENV = "vts"
    assert s.tr("inquire_price") == "FHKST01010100"  # 시세는 공통
    assert s.cano.isdigit() and len(s.cano) == 8
    cost = s.round_trip_cost
    assert 0.001 < cost < 0.01, cost
    return f"VTS/PROD TR 분기 정상, 왕복비용 {cost * 100:.3f}%"


# ============================================================ 2. 시장 규칙
@check("2. 호가단위·가격제한폭")
def test_market_rules():
    assert tick_size(1_500) == 1
    assert tick_size(3_000) == 5
    assert tick_size(12_000) == 10
    assert tick_size(35_000) == 50
    assert tick_size(75_000) == 100
    assert tick_size(300_000) == 500
    assert tick_size(700_000) == 1_000
    assert round_to_tick(75_049) == 75_000
    assert round_to_tick(75_051) == 75_100
    assert round_to_tick(75_090, mode="down") == 75_000
    assert round_to_tick(75_010, mode="up") == 75_100
    low, high = price_limits(100_000)
    assert 69_000 <= low <= 70_100, low
    assert 129_900 <= high <= 130_100, high
    assert low % tick_size(low) == 0 and high % tick_size(high) == 0
    return f"제한폭 {low:,}~{high:,} (기준가 100,000)"


# ============================================================ 3. 지표
@check("3. 기술적 지표 정확성·인과성")
def test_indicators():
    rng = np.random.default_rng(3)
    n = 300
    close = 50_000 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))

    # SMA는 pandas rolling과 일치해야 한다
    ours = sma(close, 20)
    ref = pd.Series(close).rolling(20).mean().to_numpy()
    assert np.allclose(ours[19:], ref[19:], rtol=1e-10), "SMA 불일치"

    # RSI 범위
    r = rsi(close, 14)
    valid = r[~np.isnan(r)]
    assert valid.min() >= 0 and valid.max() <= 100, f"RSI 범위 이탈 {valid.min()}~{valid.max()}"

    # ATR 양수
    a = atr(high, low, close, 14)
    assert np.nanmin(a) > 0, "ATR 음수"

    # MACD 정의 검증
    macd_line, sig, hist = macd(close)
    idx = ~np.isnan(hist)
    assert np.allclose(hist[idx], (macd_line - sig)[idx]), "MACD 히스토그램 불일치"

    # 인과성: 앞부분만 잘라 계산해도 동일 시점 값이 같아야 한다(룩어헤드 없음)
    k = 200
    for fn in (lambda x: rsi(x, 14), lambda x: ema(x, 12), lambda x: sma(x, 20)):
        full = fn(close)
        prefix = fn(close[:k])
        assert np.allclose(full[:k][~np.isnan(prefix)], prefix[~np.isnan(prefix)], rtol=1e-9), \
            "지표가 미래 정보를 참조합니다"

    _, upper, lower, pctb, _ = bollinger(close, 20, 2.0)
    ok = ~np.isnan(upper)
    assert np.all(upper[ok] >= lower[ok]), "볼린저 상/하단 역전"
    return f"SMA/RSI/ATR/MACD/BB 검증 완료 (인과성 OK, n={n})"


# ============================================================ 4. 유량 제어
@check("4. 유량 제어기 실측 RPS")
async def test_rate_limiter():
    rate, burst, total = 5.0, 5, 25
    limiter = PriorityRateLimiter(rate=rate, burst=burst)
    start = time.monotonic()
    await asyncio.gather(*[limiter.acquire(PRIORITY_QUOTE) for _ in range(total)])
    elapsed = time.monotonic() - start

    expected_min = (total - burst) / rate * 0.85
    assert elapsed >= expected_min, f"너무 빨리 통과({elapsed:.2f}s < {expected_min:.2f}s) — 제한 미작동"
    peak = limiter.stats.peak_rps(window=1.0)
    assert peak <= rate + burst + 1, f"1초 피크 {peak} > 허용 {rate + burst}"
    assert limiter.stats.granted == total
    await limiter.close()
    return f"{total}건 / {elapsed:.2f}s (설정 {rate}/s, 1초 피크 {peak:.0f})"


# ============================================================ 5. 우선순위
@check("5. 우선순위 큐(주문 최우선)")
async def test_priority():
    limiter = PriorityRateLimiter(rate=10.0, burst=1)
    order_completed: List[int] = []
    counter = {"n": 0}

    async def acquire(priority: int, tag: str):
        await limiter.acquire(priority)
        counter["n"] += 1
        if tag == "ORDER":
            order_completed.append(counter["n"])

    tasks = [asyncio.create_task(acquire(PRIORITY_QUOTE, "QUOTE")) for _ in range(12)]
    await asyncio.sleep(0.05)
    tasks.append(asyncio.create_task(acquire(PRIORITY_ORDER, "ORDER")))
    await asyncio.gather(*tasks)
    await limiter.close()

    assert order_completed, "주문 태스크가 완료되지 않음"
    position = order_completed[0]
    assert position <= 4, f"주문이 {position}번째로 밀림(시세 폴링에 선점당함)"
    return f"시세 12건 대기 중 주문이 {position}번째로 처리됨"


# ============================================================ 6. Mock 체결 엔진
@check("6. Mock 체결 엔진(체결·거부·취소)")
async def test_mock_broker():
    from api.base import is_ok, to_float, to_int
    from api.mock_client import MockKISClient

    s = make_settings()
    broker = MockKISClient(s)
    await broker.connect()
    symbol = broker.sim.symbols[0]

    # 현재가
    price_resp = await broker.inquire_price(symbol)
    assert is_ok(price_resp), price_resp
    price = to_float(price_resp["output"]["stck_prpr"])
    assert price > 0

    # 차트 → DataFrame 변환(과거→최신 정렬)
    bars = await broker.get_bars(symbol, count=100)
    assert len(bars) >= 100, f"봉 부족 {len(bars)}"
    assert list(bars.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert bars["ts"].iloc[0] < bars["ts"].iloc[-1], "시간 정렬 오류"

    # 호가단위 위반 → 거부
    bad = await broker.order_cash(symbol, Side.BUY, 1, price + 1 if tick_size(price) > 1 else price + 0.5)
    if tick_size(price) > 1:
        assert not is_ok(bad) and bad["msg_cd"] == "40230000", bad

    # 매수 가능 금액 초과 → 거부
    huge = await broker.order_cash(symbol, Side.BUY, 10_000_000, price)
    assert not is_ok(huge) and huge["msg_cd"] == "40310000", huge

    # 보유수량 초과 매도 → 거부
    oversell = await broker.order_cash(symbol, Side.SELL, 10, price)
    assert not is_ok(oversell) and oversell["msg_cd"] == "40320000", oversell

    # 정상 시장가 매수 → 즉시 체결
    buy = await broker.order_cash(symbol, Side.BUY, 10, 0, OrderType.MARKET)
    assert is_ok(buy), buy
    assert buy["output"]["ODNO"], "주문번호 없음"
    fills = await broker.poll_fills()
    assert fills and fills[0].side is Side.BUY, "체결 미발생"
    filled_qty = sum(f.qty for f in fills)
    assert fills[0].fee > 0, "매수 수수료 미반영"

    # 잔고 반영 확인
    balance = await broker.inquire_balance()
    assert is_ok(balance)
    holding = [r for r in balance["output1"] if r["pdno"] == symbol]
    assert holding and to_int(holding[0]["hldg_qty"]) == filled_qty, balance["output1"]

    # 체결되지 않는 지정가 → 미체결로 남고 취소 가능
    # 하한가(전일 종가 기준 -30%)로 낸다 — 현재가 x0.7은 현재가가 전일보다 낮은 종목에서 하한 아래라 거부된다
    far = price_limits(to_float(price_resp["output"]["stck_sdpr"]) or price)[0]
    pending = await broker.order_cash(symbol, Side.BUY, 1, far)
    assert is_ok(pending), pending
    broker.match_all()
    open_orders = await broker.inquire_open_orders()
    assert any(o["odno"] == pending["output"]["ODNO"] for o in open_orders["output1"]), "미체결 주문 누락"
    cancel = await broker.cancel_order(pending["output"]["KRX_FWDG_ORD_ORGNO"],
                                       pending["output"]["ODNO"], all_qty=True)
    assert is_ok(cancel), cancel

    # 매도 → 거래세 반영
    sell = await broker.order_cash(symbol, Side.SELL, filled_qty, 0, OrderType.MARKET)
    assert is_ok(sell), sell
    sell_fills = await broker.poll_fills()
    assert sell_fills and sell_fills[0].tax > 0, "매도 거래세 미반영"

    await broker.close()
    return f"체결 {filled_qty}주, 거부 3종 정상, 취소 정상"


# ============================================================ 7. 리스크 매니저
@check("7. 리스크 매니저(사이징·손절·킬스위치)")
def test_risk_manager():
    from strategy.risk_manager import RiskManager, RiskState

    s = make_settings()
    s.MAX_POSITIONS = 3
    risk = RiskManager(s)
    capital = 100_000_000.0
    pf = Portfolio(cash=capital, initial_capital=capital)

    sig = Signal(symbol="005930", score=0.6, price=70_000, atr=1_400, volatility=0.25, regime=1.0)
    decision = risk.evaluate_entry(sig, pf)
    assert decision.approved, decision.reason
    assert decision.qty > 0
    weight = decision.qty * sig.price / capital
    assert weight <= s.MAX_POSITION_WEIGHT + 1e-6, f"종목 비중 초과 {weight:.2%}"
    assert 0 < decision.stop_price < sig.price, decision.stop_price

    # 신호 미달 → 거부
    weak = Signal(symbol="000660", score=0.05, price=100_000, atr=2_000, volatility=0.3)
    assert not risk.evaluate_entry(weak, pf).approved

    # 변동성이 크면 수량이 줄어야 한다.
    # 가짜 종목코드("A"/"B")를 쓰면 대회 규정 관문(`strategy/eligibility.py`)이
    # "유니버스 제외 종목"으로 거부해 수량이 0이 된다. 실제 유니버스 종목을 쓴다.
    sym_calm, sym_wild = s.UNIVERSE[0], s.UNIVERSE[1]
    calm = Signal(symbol=sym_calm, score=0.6, price=70_000, atr=700, volatility=0.15)
    wild = Signal(symbol=sym_wild, score=0.6, price=70_000, atr=4_000, volatility=0.75)
    q_calm = risk.evaluate_entry(calm, pf).qty
    q_wild = risk.evaluate_entry(wild, pf).qty
    assert q_calm > q_wild, f"변동성 타게팅 미작동 ({q_calm} vs {q_wild})"

    # --- 고정 비중 모드 ---
    # 변동성·켈리·레짐을 곱하지 않으므로 변동성이 달라도 비중이 같아야 하고,
    # 실제로 `FIXED_POSITION_WEIGHT`에 닿아야 한다. 이 모드가 없으면 돌파 진입이
    # 종목당 6~14%로 들어가 측정한 구조(2종목 x 42%)와 달라진다.
    s_fixed = s.model_copy(deep=True)
    s_fixed.SIZING_MODE = "fixed"
    s_fixed.FIXED_POSITION_WEIGHT = 0.42
    s_fixed.MAX_POSITION_WEIGHT = 0.42
    s_fixed.MAX_SECTOR_WEIGHT = 0.35          # fixed 모드는 섹터 상한을 면제한다
    risk_fixed = RiskManager(s_fixed)
    pf2 = Portfolio(cash=100_000_000.0, initial_capital=100_000_000.0)
    d_calm = risk_fixed.evaluate_entry(calm, pf2)
    d_wild = risk_fixed.evaluate_entry(wild, pf2)
    assert d_calm.approved and d_wild.approved, "고정 모드에서 진입 거부"
    assert d_calm.qty == d_wild.qty, (
        f"고정 모드인데 변동성이 수량을 바꿨다 ({d_calm.qty} vs {d_wild.qty})")
    w_fixed = d_calm.qty * calm.price / pf2.equity
    assert 0.40 <= w_fixed <= 0.42, f"고정 비중이 목표에서 벗어남 {w_fixed:.3f}"

    # 대회 규정 관문: 1,000원 미만 매수 불가
    cheap = Signal(symbol=s.UNIVERSE[0], score=0.6, price=900, atr=30, volatility=0.3)
    cheap_dec = risk.evaluate_entry(cheap, pf)
    assert not cheap_dec.approved and "1,000원" in cheap_dec.reason, cheap_dec.reason

    # 대회 규정 관문: 유니버스 외 종목 차단 (유니버스 파일이 있을 때만 의미 있다)
    from strategy.eligibility import is_buyable, position_weight_cap
    if s._universe_blob():
        assert not is_buyable("999999", 50_000, s)[0], "유니버스 외 종목이 통과했다"
    # 단일 종목 한도는 규정 50%를 넘지 못한다
    s_over = s.model_copy(deep=True)
    s_over.MAX_POSITION_WEIGHT = 0.80
    assert position_weight_cap(s_over) == 0.50, position_weight_cap(s_over)

    # 최대 보유 종목 수 제한.
    # 진입 후보는 규정 관문을 먼저 통과해야 하므로 실제 유니버스 종목을 쓴다
    # (가짜 코드는 "유니버스 제외 종목"으로 먼저 거부되어 이 검사에 닿지 못한다).
    held = s.UNIVERSE[2:5]
    for i, code in enumerate(held):
        pf.positions[code] = Position(symbol=code, qty=100, avg_price=10_000,
                                      last_price=10_000, sector=f"S{i}")
    pf.cash = capital - 3_000_000
    fresh = next(c for c in s.UNIVERSE if c not in pf.positions)
    blocked = risk.evaluate_entry(Signal(symbol=fresh, score=0.9, price=10_000,
                                         atr=200, volatility=0.2), pf)
    assert not blocked.approved and "최대 보유" in blocked.reason, blocked.reason

    # 섹터 집중 제한
    risk2 = RiskManager(s)
    pf2 = Portfolio(cash=capital * 0.5, initial_capital=capital)
    big = Position(symbol="005930", qty=800, avg_price=70_000, last_price=70_000, sector="IT")
    pf2.positions["005930"] = big
    sector_blocked = risk2.evaluate_entry(
        Signal(symbol="000660", score=0.8, price=150_000, atr=3_000, volatility=0.25), pf2)
    assert not sector_blocked.approved, "섹터 비중 제한 미작동"

    # 손절 트리거
    risk3 = RiskManager(s)
    pf3 = Portfolio(cash=0.0, initial_capital=capital)
    p = Position(symbol="005930", qty=100, avg_price=70_000, last_price=70_000, sector="IT")
    p.stop_price = 67_550
    pf3.positions["005930"] = p
    exits = risk3.evaluate_exits(pf3, {"005930": 67_000}, {"005930": 1_400})
    assert exits and exits[0].reason is ExitReason.STOP_LOSS, exits

    # 익절 — 0이면 비활성화, 양수면 작동해야 한다.
    # 기본값은 0(익절 없음)이다. 실데이터 21구간 절제 실험에서 9% 익절은 상위 20%
    # 종목의 10봉 상승폭 +16.1% 중 +8.9%만 실현해 45%를 버렸고, 제거하니 Sharpe가
    # -0.348 → 0.523으로 바뀌었다. 그래서 켠 경로와 끈 경로를 모두 검증한다.
    s_tp = s.model_copy(deep=True)
    s_tp.TAKE_PROFIT_PCT = 0.09
    s_tp.TRAILING_ACTIVATE_PCT = 0.0      # 트레일링을 끄고 익절만 본다
    exits = RiskManager(s_tp).evaluate_exits(pf3, {"005930": 78_000}, {"005930": 1_400})
    assert exits and exits[0].reason is ExitReason.TAKE_PROFIT, exits

    s_off = s.model_copy(deep=True)
    s_off.TAKE_PROFIT_PCT = 0.0
    s_off.TRAILING_ACTIVATE_PCT = 0.0
    exits_off = RiskManager(s_off).evaluate_exits(pf3, {"005930": 78_000}, {"005930": 1_400})
    assert not exits_off, f"익절 0인데 청산이 나왔다: {exits_off}"

    # 하드 MDD → 킬스위치 + 전량 청산
    risk4 = RiskManager(s)
    pf4 = Portfolio(cash=0.0, initial_capital=capital)
    pf4.positions["005930"] = Position(symbol="005930", qty=1000, avg_price=100_000,
                                       last_price=80_000, sector="IT")
    pf4.peak_equity = 100_000_000
    state = risk4.update(pf4)
    assert state == RiskState.HALT, f"킬스위치 미작동 (dd={pf4.drawdown:.2%}, state={state})"
    kill_exits = risk4.evaluate_exits(pf4, {"005930": 80_000})
    assert kill_exits and kill_exits[0].reason is ExitReason.KILL_SWITCH
    assert kill_exits[0].qty == 1000, "전량 청산 아님"
    assert not risk4.evaluate_entry(sig, pf4).approved, "HALT 상태에서 신규 진입이 허용됨"

    # 소프트 MDD → 디리스킹
    risk5 = RiskManager(s)
    pf5 = Portfolio(cash=10_000_000.0, initial_capital=capital)
    # 포지션 자체는 손절 대상이 아니지만(소폭 이익) 포트폴리오 낙폭이 소프트 리밋을 넘은 상황
    pf5.positions["005930"] = Position(symbol="005930", qty=1000, avg_price=80_000,
                                       last_price=82_000, sector="IT")
    pf5.peak_equity = 100_000_000
    # 낙폭은 며칠에 걸쳐 쌓인 상황으로 가정(당일 손실 한도가 먼저 걸리지 않도록)
    pf5.day_start_equity = pf5.equity
    assert risk5.update(pf5) == RiskState.DERISK, risk5.state
    derisk = risk5.evaluate_exits(pf5, {"005930": 82_000})
    assert any(e.reason is ExitReason.DERISK for e in derisk), derisk
    return (f"사이징({q_calm}주 vs {q_wild}주) · 손절 · 익절(on/off) · "
            f"킬스위치 · 디리스킹 · 대회 규정 관문 모두 작동")


# ============================================================ 8. 피처/라벨
@check("8. 피처·라벨 파이프라인")
def test_features():
    from backtest.data import generate_synthetic_panel
    from strategy.ai_model import make_triple_barrier_labels
    from strategy.features import FEATURE_COLUMNS, build_panel_features, stack_training_matrix

    s = make_settings()
    panel = generate_synthetic_panel(s.UNIVERSE[:4], n_bars=400, seed=11,
                                     sector_map={c: s.sector_of(c) for c in s.UNIVERSE[:4]})
    featured = build_panel_features(panel)
    assert featured, "피처 생성 실패"

    df = next(iter(featured.values()))
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    assert not missing, f"누락 피처 {missing}"

    # 수급 피처는 토스 데이터가 있을 때만 채워진다. 합성 데이터에는 없으므로
    # 가격 기반 피처(종목 + 횡단면)만 결측률을 본다.
    from strategy.features import BASE_FEATURES, CROSS_FEATURES
    from strategy.flow_features import FLOW_FEATURES

    price_cols = BASE_FEATURES + CROSS_FEATURES
    tail = df[price_cols].iloc[-50:]
    nan_ratio = float(tail.isna().to_numpy().mean())
    assert nan_ratio < 0.05, f"최근 구간 결측률 과다 {nan_ratio:.1%}"
    # 수급이 없으면 해당 컬럼은 전부 NaN이어야 한다(스키마만 맞춘 상태)
    assert df[FLOW_FEATURES].isna().all().all(), "수급 데이터가 없는데 값이 채워졌습니다"

    labels = make_triple_barrier_labels(df, horizon=5, pt_atr=1.5, sl_atr=1.0, cost=0.003)
    valid = labels.dropna()
    assert len(valid) > 200, f"라벨 부족 {len(valid)}"
    assert set(valid.unique()) <= {0.0, 1.0}, set(valid.unique())
    pos_rate = float(valid.mean())
    assert 0.05 < pos_rate < 0.95, f"라벨 편향 {pos_rate:.2%}"

    for sym, f in featured.items():
        f["label"] = make_triple_barrier_labels(f, 5, 1.5, 1.0, 0.003)
    X, y, t, cols, w = stack_training_matrix(featured)
    assert len(X) > 500 and X.shape[1] == len(cols), X.shape
    assert np.isfinite(X).all(), "학습 행렬에 NaN/Inf 존재"
    assert (np.diff(t) >= 0).all(), "시점 인덱스가 정렬되지 않음"
    # 가중치 컬럼이 없으면 1.0으로 채워져야 한다(하위 호환)
    assert len(w) == len(y) and abs(w.mean() - 1.0) < 1e-9, w.mean()
    return (f"피처 {len(FEATURE_COLUMNS)}개(가격 {len(price_cols)} + 수급 {len(FLOW_FEATURES)}), "
            f"학습샘플 {len(X):,}, 양성비율 {pos_rate:.1%}")


# ============================================================ 9. 백테스트
@check("9. 백테스트 엔진·성과지표")
def test_backtest():
    from backtest.data import generate_synthetic_panel
    from backtest.engine import BacktestEngine
    from backtest.metrics import compute_metrics
    from strategy.ai_model import EnsembleAlphaStrategy
    from strategy.risk_manager import RiskManager

    # 지표 계산 자체 검증(알려진 입력)
    eq = [100.0, 110.0, 99.0, 121.0]
    m = compute_metrics(eq)
    assert abs(m.total_return - 0.21) < 1e-9, m.total_return
    assert abs(m.max_drawdown - 0.1) < 1e-9, m.max_drawdown

    s = make_settings()
    panel = generate_synthetic_panel(s.UNIVERSE[:5], n_bars=500, seed=5,
                                     sector_map={c: s.sector_of(c) for c in s.UNIVERSE[:5]})
    engine = BacktestEngine(panel, strategy=EnsembleAlphaStrategy(s),
                            risk_manager=RiskManager(s), settings=s,
                            initial_capital=50_000_000)
    result = engine.run(verbose=False)

    met = result.metrics
    assert met.periods > 300, f"자산곡선 길이 {met.periods}"
    assert math.isfinite(met.sharpe) and math.isfinite(met.max_drawdown)
    assert 0 <= met.max_drawdown <= 1, met.max_drawdown
    assert met.final_equity > 0
    assert len(result.curve) == met.periods
    assert (result.curve["drawdown"] <= 1e-9).all(), "낙폭이 양수로 계산됨"
    # 비용이 반영되어야 한다
    if met.trades > 0:
        assert met.total_fees > 0, "거래비용 미반영"
    return (f"{met.periods}봉, 거래 {met.trades}회, Sharpe {met.sharpe:.2f}, "
            f"MDD {met.max_drawdown * 100:.1f}%, 수익률 {met.total_return * 100:.1f}%")


# ============================================================ 10. 이벤트 버스
@check("10. 이벤트 버스")
async def test_event_bus():
    bus = EventBus(maxsize=100)
    received: List[Fill] = []
    errors: List[str] = []

    async def good(fill: Fill):
        received.append(fill)

    async def bad(fill: Fill):
        errors.append("boom")
        raise RuntimeError("의도적 예외 — 버스가 죽으면 안 됨")

    bus.subscribe("fill", bad)     # 예외를 던지는 핸들러가 먼저 등록되어도
    bus.subscribe("fill", good)    # 뒤 핸들러는 정상 수신되어야 한다
    bus.start()
    for i in range(5):
        await bus.publish("fill", Fill(symbol="005930", side=Side.BUY, qty=i + 1, price=70_000))
    await asyncio.sleep(0.3)
    await bus.stop()

    assert len(received) == 5, f"수신 {len(received)}/5"
    assert len(errors) == 5, "예외 핸들러가 호출되지 않음"
    assert sum(f.qty for f in received) == 15
    return "핸들러 예외 격리 확인, 5/5 수신"


# ============================================================ 11. 실서버 경로
@check("11. 실서버 클라이언트(오프라인 검증)")
async def test_live_client_offline():
    """네트워크 없이 실서버 경로의 헤더·바디·응답 파싱을 검증한다.

    계좌 발급 후 처음 붙일 때 TR_ID/파라미터 오타로 시간을 날리지 않도록,
    전송 직전까지의 모든 구성을 미리 확인한다.
    """
    import datetime as dtime

    from api.base import is_ok
    from api.client import KISClient
    from core.auth import Token

    s = make_settings()
    s.USE_MOCK_MODE = False
    s.KIS_ENV = "vts"
    s.DRY_RUN = True
    s.KIS_APP_KEY = "TEST-APP-KEY"
    s.KIS_APP_SECRET = "TEST-APP-SECRET"
    s.KIS_ACCOUNT_NO = "50123456-01"

    client = KISClient(s)
    now = dtime.datetime.now(dtime.timezone.utc)
    client.auth._token = Token(access_token="TESTTOKEN", issued_at=now,
                               expires_at=now + dtime.timedelta(hours=24))

    headers = await client._headers(s.tr("order_buy"))
    assert headers["tr_id"] == "VTTC0802U", headers["tr_id"]
    assert headers["authorization"] == "Bearer TESTTOKEN"
    assert headers["appkey"] == "TEST-APP-KEY" and headers["custtype"] == "P"

    # DRY_RUN: 실제 전송 없이 접수 응답 형태를 반환해야 한다
    resp = await client.order_cash("005930", Side.BUY, 10, 70_000, OrderType.LIMIT)
    assert is_ok(resp) and resp["output"]["ODNO"], resp
    assert resp["msg_cd"] == "DRYRUN"

    # 계좌번호 분해
    assert s.cano == "50123456" and s.acnt_prdt_cd == "01"

    # 실제 KIS 응답 형태(최신→과거) 파싱 검증
    sample = {
        "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "정상처리 되었습니다.",
        "output2": [
            {"stck_bsop_date": "20261012", "stck_oprc": "70000", "stck_hgpr": "71000",
             "stck_lwpr": "69500", "stck_clpr": "70500", "acml_vol": "12000000"},
            {"stck_bsop_date": "20261011", "stck_oprc": "69000", "stck_hgpr": "70200",
             "stck_lwpr": "68800", "stck_clpr": "69800", "acml_vol": "9000000"},
        ],
    }
    frame = KISClient._chart_rows_to_frame(sample["output2"])
    assert list(frame["ts"]) == ["20261011", "20261012"], "차트가 과거→최신으로 정렬되지 않음"
    assert float(frame["close"].iloc[-1]) == 70_500

    # 잔고 응답 파싱
    balance = {
        "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "정상",
        "output1": [{"pdno": "005930", "hldg_qty": "100", "pchs_avg_pric": "69500.00",
                     "prpr": "70500"}],
        "output2": [{"dnca_tot_amt": "45000000", "tot_evlu_amt": "52050000"}],
    }
    original = client.inquire_balance

    async def fake_balance():
        return balance

    client.inquire_balance = fake_balance  # type: ignore[assignment]
    cash, positions = await client.get_portfolio_state()
    client.inquire_balance = original  # type: ignore[assignment]
    assert cash == 45_000_000, cash
    assert positions["005930"].qty == 100 and positions["005930"].avg_price == 69_500.0

    await client.close()
    return "헤더/TR_ID/DRY_RUN 주문/차트·잔고 파싱 검증 완료"


# ============================================================ 12. 과적합 검증 도구
@check("12. 과적합 검증(CPCV·DSR·PBO)")
def test_validation():
    from backtest.validation import (CombinatorialPurgedCV, deflated_sharpe_ratio,
                                     pbo_cscv, sharpe_ratio)

    cv = CombinatorialPurgedCV(n_groups=6, n_test=2, horizon=5, embargo=5)
    assert cv.n_splits == 15 and cv.n_paths == 5, (cv.n_splits, cv.n_paths)

    splits = list(cv.split(1200))
    assert len(splits) == 15, len(splits)
    sp = splits[0]
    assert len(np.intersect1d(sp.train_idx, sp.test_idx)) == 0, "학습/테스트 인덱스 중복"
    # purge: 테스트 경계 직전 10봉(horizon+embargo)이 학습에서 빠져야 한다
    lo, _ = sp.group_slices[sp.test_groups[0]]
    train = set(sp.train_idx.tolist())
    assert not any(i in train for i in range(max(0, lo - 10), lo)), "purge 미적용"

    rng = np.random.default_rng(7)
    gr = {(x.split_id, g): rng.normal(0.0005, 0.01, 40) for x in splits for g in x.test_groups}
    paths = cv.assemble_paths(gr)
    assert len(paths) >= cv.n_paths - 1, len(paths)

    # 무작위 수익률은 유의하지 않다고 판정해야 한다
    trials = [sharpe_ratio(rng.normal(0, 0.012, 400)) for _ in range(20)]
    noise = deflated_sharpe_ratio(rng.normal(0, 0.012, 400), trials)
    assert noise["dsr"] < 0.95, f"무작위인데 유의 판정 {noise['dsr']}"

    # 진짜 알파는 PSR이 높아야 한다
    real = deflated_sharpe_ratio(rng.normal(0.0012, 0.010, 400), trials)
    assert real["psr"] > noise["psr"], (real["psr"], noise["psr"])

    # PBO: 전부 무작위면 0.5 부근, 하나만 우월하면 낮아야 한다
    flat = pbo_cscv(rng.normal(0, 0.01, (400, 10)), n_splits=8)
    edge = rng.normal(0, 0.01, (400, 10)); edge[:, 3] += 0.0015
    better = pbo_cscv(edge, n_splits=8)
    assert 0.15 < flat["pbo"] < 0.85, flat["pbo"]
    assert better["pbo"] <= flat["pbo"] + 0.1, (better["pbo"], flat["pbo"])

    return (f"CPCV {cv.n_splits}분할/{cv.n_paths}경로 · 무작위 DSR {noise['dsr']:.3f} · "
            f"PBO {flat['pbo']:.2f}→{better['pbo']:.2f}")


# ============================================================ 13. 표본 가중치
@check("13. 표본 가중치·베팅 사이징")
def test_weights_and_bets():
    from strategy.ai_model import bet_size
    from strategy.sample_weights import average_uniqueness, compute_sample_weights

    # 겹치지 않으면 고유도 1, 완전히 겹치면 1/n
    assert np.allclose(average_uniqueness(np.array([0, 5]), np.array([4, 9]), 10), 1.0)
    assert np.allclose(average_uniqueness(np.zeros(3, int), np.full(3, 4), 5), 1 / 3)

    # 삼중배리어처럼 H봉 중첩되면 고유도가 1/(H+1) 부근
    n, H = 200, 5
    t0 = np.arange(n); t1 = np.minimum(t0 + H, n + H - 1)
    w, info = compute_sample_weights(t0, t1, np.random.default_rng(0).normal(0, 0.02, n + H),
                                     n_bars=n + H)
    assert abs(w.mean() - 1.0) < 1e-9, w.mean()
    assert info["effective_n"] < n, "중첩이 있는데 유효 표본이 줄지 않음"

    # 베팅 사이즈: 중립점에서 0, 확신이 커질수록 단조 증가
    assert abs(bet_size(0.5, 0.5)) < 1e-9
    sizes = [bet_size(p, 0.5) for p in (0.55, 0.65, 0.75, 0.9)]
    assert all(b < a for b, a in zip(sizes, sizes[1:])), sizes
    assert 0 < sizes[-1] <= 1.0

    return (f"고유도 {info['mean_uniqueness']:.3f} · 유효표본 {info['effective_n']:.0f}/{n} · "
            f"베팅 0.9→{sizes[-1]:.3f}")


# ============================================================ 14. 대회 규정
@check("14. 대회 규정 필터·요건 추적")
def test_competition_rules():
    """매매 가능 판정과 최소 자격 요건 집계.

    규정 위반 주문은 거부되거나 실격 사유가 되므로, 전략 성과와 무관하게 반드시
    작동해야 한다.
    """
    from strategy.compliance import evaluate
    from strategy.eligibility import (MAX_SINGLE_WEIGHT, MIN_BUY_PRICE, MIN_SHARES,
                                      index_membership_count, is_buyable, is_sellable,
                                      position_weight_cap)

    s = make_settings()

    # 1,000원 미만은 매수 불가, 매도는 가능해야 한다(손절을 막으면 더 위험하다)
    sym = s.UNIVERSE[0]
    assert not is_buyable(sym, MIN_BUY_PRICE - 1, s)[0]
    assert is_buyable(sym, MIN_BUY_PRICE, s)[0], is_buyable(sym, MIN_BUY_PRICE, s)
    assert is_sellable(sym, s)[0], "매도가 막히면 손절이 불가능하다"

    # 단일 종목 한도는 규정 50%를 넘지 못한다
    over = s.model_copy(deep=True)
    over.MAX_POSITION_WEIGHT = 0.90
    assert position_weight_cap(over) == MAX_SINGLE_WEIGHT, position_weight_cap(over)

    # 유니버스 파일이 있으면 목록 외 종목을 차단한다
    blob = s._universe_blob()
    if blob.get("tradable"):
        assert not is_buyable("999999", 50_000, s)[0]
        # 규정 필터가 실제로 적용됐는지 산출물로 확인
        for code in blob["tradable"][:50]:
            assert s.shares_of(code) >= MIN_SHARES or s.shares_of(code) == 0, code

    # 최소 자격 요건: "코스피200·코스닥150 중 5종목"은 지수 멤버십으로 센다
    members = {x for lst in s.index_members().values() for x in lst}
    s_req = s.model_copy(deep=True)
    s_req.MIN_TRADE_COUNT = 5
    s_req.MIN_TRADE_AMOUNT = 500_000_000
    s_req.MIN_TRADE_DAYS = 5
    s_req.MIN_TRADE_SYMBOLS = 5

    def make_fills(codes):
        out = []
        for day in (12, 13, 14, 15, 16):
            for code in codes:
                for side in ("BUY", "SELL"):
                    out.append({"event": "fill", "symbol": code, "side": side,
                                "qty": 100, "price": 120_000,
                                "ts": f"2026-10-{day:02d}T09:05:00+09:00"})
        return out

    if members:
        inside = [c for c in s.UNIVERSE if c in members][:5]
        outside = [c for c in s.UNIVERSE if c not in members][:5]
        assert index_membership_count(inside, s) == len(inside)
        assert index_membership_count(outside, s) == 0
        if len(inside) == 5:
            assert evaluate(make_fills(inside), s_req, 20).all_met
        if len(outside) == 5:
            # 지수 외 종목만 거래하면 종목수 요건 미달로 잡혀야 한다
            st = evaluate(make_fills(outside), s_req, 20)
            assert not st.all_met, st.describe()

    # 거래가 전혀 없으면 4개 요건 모두 미충족
    empty = evaluate([], s_req, 20)
    assert len(empty.unmet) == 4, empty.describe()

    # 모의투자 매매불가 — 2026-10-14 실측 상태값(거부 5종목 · 체결 2종목)
    from strategy.eligibility import vts_tradable
    assert not vts_tradable({"iscd_stat_cls_code": "57", "marg_rate": "100.00"}), "증거금 100% 통과"
    assert not vts_tradable({"iscd_stat_cls_code": "53", "marg_rate": "100.00"}), "투자경고 통과"
    assert not vts_tradable({"iscd_stat_cls_code": "53", "marg_rate": "40.00"}), "경고인데 증거금만 봄"
    assert vts_tradable({"iscd_stat_cls_code": "55", "marg_rate": "40.00"}), "신용 가능 종목을 막음"
    assert vts_tradable({}), "상태를 모르면 막지 않는다(거부되면 당일 차단 목록이 막는다)"
    full = {"iscd_stat_cls_code": "57", "marg_rate": "100.00"}
    assert vts_tradable(full, block_full_margin=False), "증거금 100% 허용인데 막음"
    assert not vts_tradable({"iscd_stat_cls_code": "53", "marg_rate": "100.00"}, block_full_margin=False),         "증거금 허용이 투자경고까지 풀었다"
    assert not vts_tradable(full), "증거금 100%를 막지 않음"
    assert Settings.model_fields["VTS_BLOCK_FULL_MARGIN"].default is True, "기본값은 증거금 100%를 뺀다(11/04)"
    return (f"매수/매도 판정 · 한도 {MAX_SINGLE_WEIGHT:.0%} 클립 · "
            f"지수 멤버십 {len(members)}종목 · 요건 4항목 집계 · 모의투자 매매불가 판정")


# ============================================================ 15. 선정·국면
@check("15. 종목 선정·대회 국면")
def test_selection_and_phase():
    """Top-K 선정과 날짜 기반 국면 전환.

    레짐을 종목 수에 곱하지 않는 것이 핵심이다. 예전에 리스크 매니저가 비중에
    곱하는 것과 겹쳐 익스포저가 37% → 12%로 붕괴한 적이 있다.
    """
    import datetime as dt

    from strategy.phase import PHASE1, PHASE2, PHASE_OFF, current_phase, phase_config, \
        reset_recommendation, trading_days_remaining
    from strategy.risk_manager import RiskManager, RiskState
    from strategy.selection import (_REGIME_CUT, desired_holdings, locked_symbols,
                                    select_entries, select_rotation_exits, target_count)

    s = make_settings()
    s.MAX_POSITIONS = 5
    s.MIN_HOLDING_BARS = 3
    s.SIGNAL_SELL_THRESHOLD = -0.15

    # --- 레짐은 종목 수에 곱하지 않는다(문턱값으로만 축소) ---
    assert target_count(s, 5, 1.00) == 5
    assert target_count(s, 5, 0.80) == 5, "레짐 중앙값에서 종목 수가 줄었다(이중 적용 버그)"
    assert target_count(s, 5, _REGIME_CUT) == 5
    assert target_count(s, 5, _REGIME_CUT - 0.01) == 2, "극단 레짐에서 절반으로 줄어야 한다"
    assert target_count(s, 8, 1.0) == 5, "MAX_POSITIONS 상한을 넘었다"

    # make_settings()가 UNIVERSE를 6종목으로 자르므로 원본 목록에서 8개를 가져온다.
    from config.settings import DEFAULT_UNIVERSE
    pool = s.UNIVERSE if len(s.UNIVERSE) >= 8 else DEFAULT_UNIVERSE
    codes = list(pool[:8])
    signals = [Signal(symbol=c, score=0.9 - 0.1 * i, price=10_000)
               for i, c in enumerate(codes)]
    pf = Portfolio(cash=100_000_000.0, initial_capital=100_000_000.0)
    assert [x.symbol for x in select_entries(signals, pf, s, top_k=4)] == codes[:4]

    # 매도 신호 종목은 담지 않는다
    bad = Signal(symbol=codes[0], score=-0.4, price=10_000)
    assert codes[0] not in desired_holdings([bad] + signals[1:], pf, s, top_k=8)

    # 최소 보유 기간 잠금 — 순위 밖이어도 유지하고, 잠금 해제분만 교체한다
    pf2 = Portfolio(cash=100_000_000.0, initial_capital=100_000_000.0)
    pf2.positions[codes[7]] = Position(symbol=codes[7], qty=10, avg_price=10_000,
                                       last_price=10_000, bars_held=1)
    pf2.positions[codes[6]] = Position(symbol=codes[6], qty=10, avg_price=10_000,
                                       last_price=10_000, bars_held=9)
    want = desired_holdings(signals, pf2, s, top_k=3)
    assert codes[7] in want and codes[6] not in want, want
    assert locked_symbols(pf2, s) == {codes[7]}
    assert select_rotation_exits(signals, pf2, s, top_k=3) == [codes[6]]

    # --- 국면은 날짜만 보고 전환한다 ---
    s.COMP_START, s.COMP_END, s.PHASE1_END = "2026-10-12", "2026-11-20", "2026-10-16"
    s.PHASE1_TOP_K, s.PHASE2_TOP_K = 2, 4
    cases = {"2026-10-09": PHASE_OFF, "2026-10-12": PHASE1, "2026-10-16": PHASE1,
             "2026-10-19": PHASE2, "2026-11-20": PHASE2, "2026-11-23": PHASE_OFF}
    for day, want_phase in cases.items():
        got = current_phase(dt.date.fromisoformat(day), s)
        assert got == want_phase, f"{day}: {got} != {want_phase}"
    assert trading_days_remaining(dt.date(2026, 10, 12), s) == 30

    # 1단계는 손절·MDD 게이트를 끄고, 2단계는 켠다
    c1 = phase_config(settings=s, now=dt.date(2026, 10, 13))
    c2 = phase_config(settings=s, now=dt.date(2026, 10, 20))
    assert c1.top_k == 2 and not c1.use_stop_loss and not c1.use_mdd_gate
    assert c2.top_k == 4 and c2.use_stop_loss and c2.use_mdd_gate

    # 1단계에서 MDD 30%여도 HALT되지 않고, 2단계에서는 된다
    deep = Portfolio(cash=70_000_000.0, initial_capital=100_000_000.0)
    deep.peak_equity = 100_000_000.0
    deep.day_start_equity = 100_000_000.0
    r1 = RiskManager(s)
    r1.set_phase(c1)
    assert r1.update(deep, dt.datetime(2026, 10, 13, 15, 30)) == RiskState.NORMAL
    r2 = RiskManager(s)
    r2.set_phase(c2)
    assert r2.update(deep, dt.datetime(2026, 10, 20, 15, 30)) == RiskState.HALT

    # --- 1단계가 끄는 것은 손절뿐이다. 감쇠 청산(vol_alive)은 1단계에도 돈다 ---
    #
    # 예전에는 1단계 분기가 감쇠까지 건너뛰어 1주차가 "사서 30봉 상한까지 보유"가
    # 됐다. 백테스트에는 대회 날짜가 없어 어떤 측정에도 나타나지 않았다.
    saved ={k: getattr(s, k) for k in ("EXIT_RULES", "EXIT_VOL_ALIVE_MULT",
                                        "EXIT_VOL_ALIVE_MIN_BARS", "STOP_LOSS_PCT",
                                        "MAX_HOLDING_BARS", "PHASE1_DISABLE_RISK",
                                        "TAKE_PROFIT_PCT")}
    try:
        s.EXIT_RULES, s.EXIT_VOL_ALIVE_MULT, s.EXIT_VOL_ALIVE_MIN_BARS = "vol_alive", 2.0, 1
        s.STOP_LOSS_PCT, s.MAX_HOLDING_BARS, s.PHASE1_DISABLE_RISK = 0.25, 30, True
        s.TAKE_PROFIT_PCT = 0.0
        c1 = phase_config(settings=s, now=dt.date(2026, 10, 13))

        def exits_for(cfg, price: float, burst: float):
            rm = RiskManager(s)
            rm.set_phase(cfg)
            pf_ = Portfolio(cash=55_000_000.0, initial_capital=100_000_000.0)
            pf_.positions["AAA"] = Position(symbol="AAA", qty=4_500, avg_price=10_000,
                                            last_price=10_000, bars_held=2)
            return rm.evaluate_exits(pf_, {"AAA": price}, burst_map={"AAA": burst},
                                     now=dt.datetime(2026, 10, 13, 9, 0))

        for cfg, label in ((c1, "1단계"), (c2, "2단계")):
            ex = exits_for(cfg, 10_000.0, 1.0)          # 거래 소멸(x1 < x2)
            assert ex and ex[0].reason == ExitReason.VOL_DECAY, \
                f"{label}에서 vol_alive 청산이 안 났다 — 1주차가 매수 후 보유가 된다: {ex}"
            assert not exits_for(cfg, 10_000.0, 5.0), f"{label}: 살아 있는데(x5) 청산했다"
        # 손절은 1단계에서만 꺼진다(−30%, 거래는 살아 있음)
        assert not exits_for(c1, 7_000.0, 5.0), "1단계인데 손절이 걸렸다(초기화가 하방을 덮는다)"
        ex = exits_for(c2, 7_000.0, 5.0)
        assert ex and ex[0].reason == ExitReason.STOP_LOSS, f"2단계 손절 미작동: {ex}"
        # 익절은 전략 청산이라 1단계에도 돈다(꺼져 있으면 안 판다)
        assert not exits_for(c1, 12_500.0, 5.0), "익절 0인데 +25%에서 팔았다"
        s.TAKE_PROFIT_PCT = 0.2
        for cfg, label in ((c1, "1단계"), (c2, "2단계")):
            ex = exits_for(cfg, 12_500.0, 5.0)
            assert ex and ex[0].reason == ExitReason.TAKE_PROFIT, f"{label} 익절 미작동: {ex}"
            assert not exits_for(cfg, 11_500.0, 5.0), f"{label}: +15%인데 익절 20%가 걸렸다"
    finally:
        for k, v in saved.items():
            setattr(s, k, v)

    # 초기화는 마지막 날에만 권고하고, 이익이면 권고하지 않는다
    loss = Portfolio(cash=90_000_000.0, initial_capital=100_000_000.0)
    gain = Portfolio(cash=108_000_000.0, initial_capital=100_000_000.0)
    assert not reset_recommendation(loss, dt.date(2026, 10, 13), s).should_reset
    assert reset_recommendation(loss, dt.date(2026, 10, 16), s).should_reset
    assert not reset_recommendation(gain, dt.date(2026, 10, 16), s).should_reset
    assert not reset_recommendation(loss, dt.date(2026, 10, 19), s).should_reset
    return ("레짐 단일 적용 · 잠금/교체 · 국면 6일자 · 1단계 손절만 해제(vol_alive 유지) · "
            "초기화 권고 모두 작동")


# ============================================================ 16. 패널 정렬
@check("17. 장중 돌파 감시·청산")
def test_intraday():
    """분봉 전략의 집행 레이어. 네트워크 없이 순수 로직만 검증한다.

    가격·거래량 **동시** 조건, 누적거래량 차분으로 만드는 분당 거래량, 하루 1회
    발동, 청산 우선순위(손절 > 트레일링 > 익절 > 마감정리), 날짜 롤오버를 본다.
    롤오버가 깨지면 **어제 발동한 종목이 오늘 영원히 발동하지 않는다.**
    """
    import datetime as dt
    from strategy.intraday import IntradayWatcher

    w = IntradayWatcher(trigger_pct=0.03, vol_mult=1.5, stop_pct=0.05,
                        trail_pct=0.03, trail_activate_pct=0.05, flatten_hhmm="1515")
    w.prime("A", prev_close=10_000.0, prev_volume=391_000.0)   # 분당 1,000주 → 하한 1,500
    base = dt.datetime(2026, 10, 12, 9, 0)

    def q(mi, price, cum):
        return w.on_quote("A", price, cum, base + dt.timedelta(minutes=mi))

    assert abs(w.vol_floor("A") - 1500.0) < 1e-6, f"하한 {w.vol_floor('A')}"
    assert q(0, 10_100, 500) is None, "가격 미달인데 발동"
    assert q(1, 10_400, 1_000) is None, "거래량 미달인데 발동"
    sig = q(1, 10_400, 2_600)          # 같은 분 누적 2,600 → 이번 분 2,100주
    assert sig is not None, "가격·거래량 모두 충족인데 미발동"
    assert abs(sig.level_pct - 0.04) < 1e-6, f"돌파율 {sig.level_pct}"
    assert q(2, 10_800, 9_000) is None, "하루 두 번 발동"

    # --- 청산 우선순위 ---
    w.open_watch("A", 10_400.0)
    t = base + dt.timedelta(minutes=30)
    assert w.exit_reason("A", 10_700.0, t) is None, "+2.9%에서 청산"
    assert w.exit_reason("A", 11_000.0, t) is None, "고점 갱신 중 청산"
    assert w.exit_reason("A", 10_900.0, t) is None, "고점 대비 -0.9%에서 청산"
    assert "트레일링" in (w.exit_reason("A", 10_650.0, t) or ""), "트레일링 미작동"
    w.open_watch("A", 10_400.0)
    assert "손절" in (w.exit_reason("A", 9_800.0, t) or ""), "손절 미작동"
    w.open_watch("A", 10_400.0)
    assert "마감" in (w.exit_reason("A", 10_400.0,
                                  dt.datetime(2026, 10, 12, 15, 16)) or ""), "마감 정리 미작동"

    # --- 날짜 롤오버 ---
    w.reset_day("20261013")
    nxt = w.on_quote("A", 10_400.0, 99_999.0, dt.datetime(2026, 10, 13, 9, 0))
    assert nxt is not None, "날짜가 바뀌었는데 발동 이력이 남아 있다"

    # --- burst 모드: 당일 누적 거래대금만 본다 ---
    #
    # 일봉 catalyst 조건(`turnover = 종가 x 거래량` >= 20봉 중앙값 x 10)을 **장중
    # 누적으로** 보는 모드다. 일봉은 장 마감 후에야 확정되므로 T+1 시가에 들어가고
    # 폭증일(시가→종가 평균 +5.29%, 승률 71.8%)을 통째로 버린다.
    #
    # **가격 조건을 걸지 않는 것이 핵심이다.** "거래대금은 10배인데 가격은 안 오른
    # 날"이 분별 대상의 일부이고, 가격 조건을 같이 걸면 그 집단이 사라진다.
    b = IntradayWatcher(entry_mode="burst", burst_mult=10.0, trigger_pct=0.03)
    b.prime("A", prev_close=10_000.0, prev_volume=391_000.0, turnover_med20=1_000e6)
    t0 = dt.datetime(2026, 10, 12, 9, 0)
    # 가격이 **내려가도** 거래대금만 차면 발동해야 한다(가격 조건 없음).
    assert b.on_quote("A", 9_500.0, 500_000.0,
                      t0) is None, "거래대금 미달인데 발동 (9,500x50만 = 47.5억 < 100억)"
    sig = b.on_quote("A", 9_500.0, 1_100_000.0, t0 + dt.timedelta(minutes=1))
    assert sig is not None, "거래대금 10.45배인데 미발동 — 가격이 내려가도 발동해야 한다"

    # 분모가 0이면 영원히 발동하지 않는다(조용한 미작동 — 호출부가 채워야 한다)
    b2 = IntradayWatcher(entry_mode="burst", burst_mult=10.0)
    b2.prime("B", prev_close=10_000.0, prev_volume=391_000.0)   # turnover_med20 미지정
    assert b2.on_quote("B", 10_000.0, 1e9, t0) is None, "분모 0인데 발동"

    # --- 마감 정리 OFF: 다일 보유로 넘긴다 ---
    nf = IntradayWatcher(flatten=False, flatten_hhmm="1515")
    assert nf.should_flatten(dt.datetime(2026, 10, 12, 15, 30)) is False,         "flatten=False 인데 마감 정리가 True"
    nf.open_watch("A", 10_000.0)
    assert nf.exit_reason("A", 10_000.0, dt.datetime(2026, 10, 12, 15, 30)) is None,         "flatten=False 인데 마감 정리로 청산했다"
    assert w.should_flatten(dt.datetime(2026, 10, 12, 15, 30)) is True,         "flatten=True 인데 마감 정리가 False"
    return ("가격·거래량 동시 조건 · 분당 차분 · 하루 1회 · "
            "청산 우선순위(손절>트레일>익절>마감) · 롤오버 · "
            "burst 모드(가격 무관·분모 0 미작동) · 마감정리 OFF 모두 작동")


@check("18. 장중 후보 명부·교체")
def test_intraday_book():
    """돌파 사건 선정의 핵심 경로. 틀리면 **조용히 잘못된 종목을 고른다.**

    하루 돌파가 중앙 14건인데 보유 한도는 2종목이다. 27건을 버리는 그 선택이
    전략의 거의 전부이고, `tail_optimizer` 기준 P(+100%)를 1.04%(무작위)에서
    6.17%(스킬 20%)까지 바꾼다 — 구조보다 큰 레버다.

    명부를 넣은 이유도 측정에서 나왔다. 발동 신호를 매 루프 비우면 **같은 분에
    동시 발동한 것들 사이에서만** 고르게 되고, 2번째 돌파가 5분 안에 나는 날이
    76.9%라 슬롯이 09:00~09:01에 차서 나머지가 선택 기회조차 없이 버려졌다.

    여기서 보는 것: 점수 순 선택 · 슬롯 한도 · 보유 종목 제외 · 선택분 명부 제거 ·
    TTL 만료 · 교체 여유(margin) · 하루 교체 한도 · 날짜 롤오버.
    """
    import datetime as dt
    from strategy.intraday import IntradayWatcher, TriggerSignal

    def sig(sym, ts, fresh):
        return TriggerSignal(symbol=sym, price=1000.0, prev_close=970.0,
                             minute_volume=5000.0, vol_floor=1000.0, ts=ts,
                             freshness=fresh)

    t0 = dt.datetime(2026, 10, 12, 9, 0)
    w = IntradayWatcher(candidate_ttl=5, swap_margin=0.10, max_swaps=1)
    w.reset_day("20261012")

    # --- 점수 순 선택 + 슬롯 한도 + 명부 제거 ---
    w.add_candidate(sig("AAA", t0, 0.50), 0.50)
    w.add_candidate(sig("BBB", t0, 0.80), 0.80)
    w.add_candidate(sig("CCC", t0, 0.65), 0.65)
    got = w.pick_entries(t0, 2)
    assert [s.symbol for s in got] == ["BBB", "CCC"], [s.symbol for s in got]
    assert set(w.book) == {"AAA"}, f"선택분이 명부에 남았다: {sorted(w.book)}"

    # --- 보유 종목은 제외한다 (중복 진입 방지) ---
    w.open_watch("BBB", 1000.0, 0.80)
    w.add_candidate(sig("BBB", t0, 0.99), 0.99)
    assert [s.symbol for s in w.pick_entries(t0, 2)] == ["AAA"], "보유 종목을 또 골랐다"

    # --- TTL 만료 ---
    w2 = IntradayWatcher(candidate_ttl=5, max_swaps=0)
    w2.reset_day("20261012")
    w2.add_candidate(sig("OLD", t0, 0.9), 0.9)
    w2.prune_book(t0 + dt.timedelta(minutes=4))
    assert "OLD" in w2.book, "TTL 안인데 버려졌다"
    w2.prune_book(t0 + dt.timedelta(minutes=6))
    assert "OLD" not in w2.book, "TTL이 지났는데 남아 있다"

    # --- 교체: 여유 미달이면 바꾸지 않는다 ---
    w3 = IntradayWatcher(candidate_ttl=30, swap_margin=0.10, max_swaps=1)
    w3.reset_day("20261012")
    w3.open_watch("HELD", 1000.0, 0.60)
    w3.add_candidate(sig("NEW", t0, 0.65), 0.65)      # 차이 0.05 < 0.10
    assert w3.swap_target(t0) is None, "여유가 모자란데 교체했다"
    w3.add_candidate(sig("BIG", t0, 0.80), 0.80)      # 차이 0.20 > 0.10
    tgt = w3.swap_target(t0)
    assert tgt is not None and tgt[0] == "HELD" and tgt[1].symbol == "BIG", tgt

    # --- 하루 교체 한도 ---
    w3.note_swap(tgt[1])
    assert "BIG" in w3.book, "note_swap이 후보를 명부에서 빼버렸다(다음 루프에 사야 한다)"
    assert w3.swap_target(t0) is None, "하루 한도를 넘겨 교체했다"

    # --- max_swaps=0이면 교체하지 않는다 (현재 기본값) ---
    w4 = IntradayWatcher(candidate_ttl=30, swap_margin=0.0, max_swaps=0)
    w4.reset_day("20261012")
    w4.open_watch("HELD", 1000.0, 0.10)
    w4.add_candidate(sig("NEW", t0, 0.99), 0.99)
    assert w4.swap_target(t0) is None, "max_swaps=0인데 교체했다"

    # --- 날짜 롤오버가 명부·교체 카운터를 비운다 ---
    w3.reset_day("20261013")
    assert not w3.book and w3.swaps_today == 0, "롤오버가 명부/카운터를 안 비웠다"
    return ("점수 순 선택 · 슬롯 한도 · 보유 제외 · 명부 제거 · TTL 만료 · "
            "교체 여유/한도 · max_swaps=0 · 롤오버 모두 작동")


@check("16. 패널 정렬 이력 하한")
def test_panel_min_bars():
    """상장이 늦은 종목이 전체 백테스트 구간을 잘라내지 않는지.

    `align_panel`은 공통 날짜의 교집합을 잡으므로, 이력이 짧은 종목 하나가 패널
    전체를 그만큼으로 줄인다. 실제로 신규 상장이 섞인 유니버스에서 수십 봉으로
    붕괴할 수 있어 `MIN_HISTORY_BARS`로 막는다.
    """
    from backtest.data import generate_synthetic_panel
    from backtest.engine import align_panel

    s = make_settings()
    panel = generate_synthetic_panel(s.UNIVERSE[:4], n_bars=400, seed=3,
                                     sector_map={c: s.sector_of(c) for c in s.UNIVERSE[:4]})
    # 한 종목만 최근 30봉으로 잘라 '신규 상장'을 흉내낸다
    late = s.UNIVERSE[3]
    panel[late] = panel[late].iloc[-30:].reset_index(drop=True)

    without = align_panel(panel, min_bars=0)
    assert len(next(iter(without.values()))) == 30, \
        f"하한이 없으면 30봉으로 잘려야 한다: {len(next(iter(without.values())))}"

    # 하한은 중앙값의 90%로 자동 완화되므로(짧은 패널 보호) 400봉 기준 360봉이 적용된다.
    with_floor = align_panel(panel, min_bars=400)
    assert late not in with_floor, "이력 부족 종목이 제외되지 않았다"
    kept = len(next(iter(with_floor.values())))
    assert kept == 400, f"나머지 종목은 온전해야 한다: {kept}"
    return (f"하한 없음 {len(without)}종목x30봉 → 하한 적용 {len(with_floor)}종목x{kept}봉 (상대 하한으로 짧은 패널 보호)")


@check("19. 감쇠 청산(거래량·상승·급락)")
def test_decay_exits():
    """감쇠 청산 모듈 — 거래량 감소 · 상승 감쇠 · 급격한 하락.

    `strategy/exit_rules.py`는 주문도 포트폴리오도 모르는 순수 로직이라 네트워크
    없이 검증된다. 보는 것은 (1) 세 규칙이 각각 발동하는가, (2) `EXIT_RULES`의
    **순서**가 결과를 바꾸는가, (3) 임계값이 0이면 꺼지는가, (4) 같은 종목에
    재진입할 때 **묵은 거래대금 고점이 지워지는가**다.

    (4)가 중요하다. 상태를 남겨 두면 과거 폭증일의 거래대금 고점이 살아 있어
    재진입 첫 봉에서 `vol_decay`가 바로 걸린다(지금은 평상 거래량이므로 비율이
    낮다). 실제로 그렇게 될 수 있는 구조였고 `evaluate_exits`가 매 호출마다
    미보유 종목의 상태를 버리도록 고쳤다.
    """
    from strategy.exit_rules import (RULE_CRASH, RULE_TRAILING, RULE_VOL_DECAY,
                                     DecayExitTracker)

    s = make_settings()
    s.CRASH_DROP_PCT = 0.10
    s.CRASH_LOOKBACK = 10
    s.TRAILING_ACTIVATE_PCT = 0.05
    s.TRAILING_ATR_MULT = 2.0
    s.EXIT_VOL_DECAY_RATIO = 0.30
    s.EXIT_VOL_DECAY_MIN_BARS = 2

    # --- 1) 거래량 감소 ---
    s.EXIT_RULES = RULE_VOL_DECAY
    t = DecayExitTracker(s)
    t.open("A", 10_000.0)
    assert t.check("A", 10_000.0, turnover=1_000e6) is None, "첫 봉에 걸려선 안 된다"
    assert t.check("A", 10_200.0, turnover=900e6) is None, "고점의 90%는 기준 위다"
    hit = t.check("A", 10_100.0, turnover=200e6)      # 고점의 20% < 30%
    assert hit is not None and hit.rule == RULE_VOL_DECAY, f"거래량 감소 미작동: {hit}"
    # 최소 보유 봉 수 — 1봉만 지난 상태에서는 걸리지 않아야 한다
    t2 = DecayExitTracker(s)
    t2.open("B", 10_000.0)
    assert t2.check("B", 10_000.0, turnover=10e6) is None, "최소 봉 수 미적용"

    # --- 2) 상승 감쇠(트레일링) ---
    s.EXIT_RULES = RULE_TRAILING
    t = DecayExitTracker(s)
    t.open("A", 10_000.0)
    assert t.check("A", 10_200.0, atr=100.0) is None, "+2%는 무장 전이다"
    assert t.check("A", 11_000.0, atr=100.0) is None, "고점 갱신 중에는 청산이 아니다"
    assert t.trail_stop("A", atr=100.0) == 11_000.0 - 200.0, "트레일링 손절선 계산 오류"
    hit = t.check("A", 10_700.0, atr=100.0)           # 고점 11,000 − 2x100 = 10,800
    assert hit is not None and hit.rule == RULE_TRAILING, f"상승 감쇠 미작동: {hit}"
    assert hit.stop_price == 10_800.0, f"손절선 전달 오류: {hit.stop_price}"

    # --- 3) 급격한 하락 ---
    s.EXIT_RULES = RULE_CRASH
    t = DecayExitTracker(s)
    t.open("A", 10_000.0)
    for px in (10_000.0, 10_500.0, 11_000.0):
        assert t.check("A", px) is None, "상승 중에 급락이 걸려선 안 된다"
    hit = t.check("A", 9_800.0)                        # 고점 11,000 대비 −10.9%
    assert hit is not None and hit.rule == RULE_CRASH, f"급락 미작동: {hit}"

    # --- 4) 순서가 결과를 바꾼다 ---
    #
    # 같은 봉에서 급락과 거래량 감소가 동시에 충족되게 만들고, `EXIT_RULES`의
    # 앞쪽에 둔 규칙으로 기록되는지 본다. 사유별 통계가 순서에 의존한다.
    def first_rule(order: str) -> str:
        s.EXIT_RULES = order
        tr = DecayExitTracker(s)
        tr.open("A", 10_000.0)
        tr.check("A", 10_000.0, turnover=1_000e6)
        tr.check("A", 11_000.0, turnover=1_000e6)
        h = tr.check("A", 9_700.0, turnover=100e6)     # 급락 −11.8% + 거래 10%
        assert h is not None, f"어느 규칙도 걸리지 않았다({order})"
        return h.rule

    assert first_rule(f"{RULE_CRASH},{RULE_VOL_DECAY}") == RULE_CRASH, "순서 무시됨"
    assert first_rule(f"{RULE_VOL_DECAY},{RULE_CRASH}") == RULE_VOL_DECAY, "순서 무시됨"

    # --- 5) 임계값 0이면 꺼진다 / 빈 목록이면 전부 꺼진다 ---
    s.EXIT_RULES = f"{RULE_CRASH},{RULE_TRAILING},{RULE_VOL_DECAY}"
    s.EXIT_VOL_DECAY_RATIO = 0.0
    s.TRAILING_ACTIVATE_PCT = 0.0
    s.CRASH_DROP_PCT = 1.0
    t = DecayExitTracker(s)
    t.open("A", 10_000.0)
    for px, tv in ((10_000.0, 1_000e6), (12_000.0, 1_000e6), (8_000.0, 1e6)):
        assert t.check("A", px, turnover=tv, atr=100.0) is None,             f"임계값 0/1인데 발동했다 (price={px})"
    s.EXIT_RULES = ""
    s.CRASH_DROP_PCT = 0.10
    t = DecayExitTracker(s)
    t.open("A", 10_000.0)
    for px in (10_000.0, 11_000.0, 8_000.0):
        assert t.check("A", px) is None, "빈 목록인데 발동했다"

    # --- 5-1) vol_alive — 분모가 20봉 중앙값이다(보유 중 고점이 아니다) ---
    #
    # `vol_decay`와 **분모만 다르고 그것이 전부다.** `vol_decay`는 보유 중 고점을
    # 분모로 써서 폭증일 자신이 고점이 되어 발동률 80.9%가 됐다(P(+100%) 2.09%
    # → 1.01%). `vol_alive`는 진입 조건과 같은 분모를 써서 "아직 평소보다 거래가
    # 많다"를 판정하고, 이것이 `MAX_HOLDING_BARS`를 대체한다
    # (측정: 같은 라운드에서 단일 2배 0.64% → 0.86%, 포트 2.21% → 4.06%).
    from strategy.exit_rules import RULE_VOL_ALIVE

    s.EXIT_RULES = RULE_VOL_ALIVE
    s.EXIT_VOL_ALIVE_MULT = 2.0
    s.EXIT_VOL_ALIVE_MIN_BARS = 1
    t = DecayExitTracker(s)
    t.open("A", 10_000.0)
    for b in (10.0, 5.0, 2.5):
        assert t.check("A", 10_000.0, turnover=1e9, burst=b) is None,             f"배수 x{b}는 기준 x2 위인데 청산했다"
    hit = t.check("A", 10_000.0, turnover=1e9, burst=1.9)
    assert hit is not None and hit.rule == RULE_VOL_ALIVE, f"vol_alive 미작동: {hit}"
    # **배수를 못 받으면(0) 판정하지 않는다.** 배관이 끊겼을 때 전 종목이 첫 봉에
    # 청산되는 것을 막는 장치다 — 0을 "거래 소멸"로 읽으면 그렇게 된다.
    t2 = DecayExitTracker(s)
    t2.open("B", 10_000.0)
    assert t2.check("B", 10_000.0, turnover=1e9, burst=0.0) is None,         "배수 0(모름)을 거래 소멸로 읽었다 — 배관이 끊기면 전 종목이 털린다"
    s.EXIT_VOL_ALIVE_MULT = 0.0
    t3 = DecayExitTracker(s)
    t3.open("C", 10_000.0)
    assert t3.check("C", 10_000.0, turnover=1e9, burst=0.1) is None, "배수 0인데 발동"

    # --- 5-2) 실거래 입력이 백테스트와 같은 창이다 ---
    #
    # 실거래는 `daily_alive_burst(확정 봉)`, 백테스트는
    # `rolling(20, min_periods=10).median().shift(1)`이다. 창이 한 칸만 어긋나도
    # 측정한 규칙과 다른 것을 실거래로 들고 간다(수집기에서 `.shift(1)`을 빼먹어
    # 커버리지가 83.1%였던 전례가 있다). 모든 길이에서 대조한다.
    from strategy.exit_rules import daily_alive_burst
    rng = np.random.default_rng(16)
    n = 60
    frame = pd.DataFrame({
        "ts": [f"2026{(i // 28) + 1:02d}{(i % 28) + 1:02d}" for i in range(n)],
        "close": rng.uniform(900, 1_100, n),
        "volume": rng.lognormal(10, 1.0, n),
    })
    tov = (frame["close"] * frame["volume"]).to_numpy()
    med = pd.Series(tov).rolling(20, min_periods=10).median().shift(1).to_numpy()
    for k in range(1, n + 1):
        want = tov[k - 1] / med[k - 1] if np.isfinite(med[k - 1]) else 0.0
        got = daily_alive_burst(frame.iloc[:k])
        assert abs(got - want) < 1e-9, f"창 불일치 (확정 봉 {k}개): 실거래 {got} vs 백테스트 {want}"

    # --- 6) 재진입 시 묵은 고점이 지워진다 ---
    s.EXIT_RULES = RULE_VOL_DECAY
    s.EXIT_VOL_DECAY_RATIO = 0.30
    s.EXIT_VOL_DECAY_MIN_BARS = 2
    t = DecayExitTracker(s)
    t.open("A", 10_000.0)
    t.check("A", 10_000.0, turnover=1_000e6)      # 폭증일 고점 1,000억
    t.close("A")                                   # 청산
    t.open("A", 9_000.0)                           # 같은 종목 재진입(평상 거래량)
    assert t.check("A", 9_000.0, turnover=50e6) is None, "재진입 첫 봉에 걸렸다"
    assert t.check("A", 9_100.0, turnover=50e6) is None,         "묵은 거래대금 고점이 남아 재진입 직후 청산됐다"
    return ("거래량 감소 · 상승 감쇠(손절선 전달) · 급락 · vol_alive(20봉중앙 분모 · "
            "배수 0 보호 · 실거래 창 = 백테스트 창) · EXIT_RULES 순서 반영 · "
            "임계값 0 비활성 · 재진입 상태 초기화 모두 작동")


@check("20. 호가·체결 전방 수집")
def test_orderbook_recorder():
    """`utils/orderbook_log.py` — 샘플링 · 컬럼 계약 · 실패 격리.

    이 수집은 **이번 대회 전략에 쓰이지 않는다**(호가는 과거 이력을 받을 수 없어
    학습·검증이 불가능하다). 다음을 위해 쌓는 것이고, 그래서 **매매를 방해하지
    않는 것**이 가장 중요한 성질이다 — 기록 실패가 예외로 바깥에 나가면 호가
    콜백이 WebSocket 수신 루프를 깨고 시세가 끊긴다.

    보는 것: (1) 간격 안의 중복을 버리는가, (2) 컬럼 계약이 헤더와 맞는가,
    (3) 10단 합계가 상위 5단만 저장하는 손실을 보전하는가, (4) 쓸 수 없는 경로에서
    **예외를 던지지 않는가**.
    """
    import datetime as dt
    import tempfile
    from pathlib import Path

    from utils.orderbook_log import OB_COLUMNS, TAPE_COLUMNS, OrderbookRecorder

    tmp = Path(tempfile.mkdtemp())
    r = OrderbookRecorder(tmp, interval_sec=5.0, flush_rows=1000)
    t = dt.datetime(2026, 10, 12, 9, 0, 0)
    ob = {"symbol": "005930",
          "ask": [100.0 + i for i in range(10)],
          "bid": [99.0 - i for i in range(10)],
          "ask_qty": [10.0 * (i + 1) for i in range(10)],
          "bid_qty": [20.0 * (i + 1) for i in range(10)]}

    assert r.observe_orderbook(ob, t) is True, "첫 호가가 기록되지 않았다"
    assert r.observe_orderbook(ob, t + dt.timedelta(seconds=3)) is False,         "샘플링 간격 안인데 기록했다"
    assert r.observe_orderbook(ob, t + dt.timedelta(seconds=6)) is True,         "간격을 넘겼는데 기록하지 않았다"
    assert r.observe_tape("005930", 100.0, 5_000.0, bid=99.0, ask=100.0,
                          ts=t) is True, "테이프가 기록되지 않았다"
    assert r.observe_tape("005930", 0.0, 1.0, ts=t) is False, "가격 0을 기록했다"
    assert r.flush() == 3, "flush 행수가 다르다"

    ob_lines = (tmp / "orderbook" / "005930.csv").read_text(
        encoding="utf-8").strip().splitlines()
    tape_lines = (tmp / "tape" / "005930.csv").read_text(
        encoding="utf-8").strip().splitlines()
    assert ob_lines[0] == ",".join(OB_COLUMNS), "호가 헤더가 계약과 다르다"
    assert tape_lines[0] == ",".join(TAPE_COLUMNS), "테이프 헤더가 계약과 다르다"
    assert len(ob_lines) == 3 and len(tape_lines) == 2,         f"행 수 불일치: 호가 {len(ob_lines) - 1} · 테이프 {len(tape_lines) - 1}"
    first = ob_lines[1].split(",")
    assert len(first) == len(OB_COLUMNS), "호가 열 수가 헤더와 다르다"
    # 상위 5단만 저장하므로 10단 합계가 따로 있어야 한다(합 = 10+20+...+100 = 550 / 1100)
    assert float(first[OB_COLUMNS.index("ask_qty_sum")]) == 550.0, "10단 매도 잔량 합 오류"
    assert float(first[OB_COLUMNS.index("bid_qty_sum")]) == 1100.0, "10단 매수 잔량 합 오류"

    # --- 쓸 수 없는 경로에서도 예외를 던지지 않는다 ---
    bad = OrderbookRecorder(tmp / "orderbook" / "005930.csv" / "nope",
                            interval_sec=0.0)
    bad.observe_orderbook(ob, t)
    wrote = bad.flush()          # 파일 아래 디렉토리를 만들 수 없다
    assert wrote == 0, "쓸 수 없는 경로인데 썼다고 보고했다"
    assert bad.snapshot()["errors"] > 0, "실패를 세지 않았다"
    return ("샘플링 간격 · 컬럼 계약 · 10단 합계 보전 · 가격 0 거부 · "
            "쓰기 실패 격리(예외 미전파) 모두 작동")


@check("21. 장중 burst · 보유 청산 배관 (main.py)")
async def test_burst_wiring():
    """`_prime_intraday` → `turnover_med20` → `_feed_intraday` → 대기열.

    **Mock으로는 이 경로를 검증할 수 없다.** Mock 합성 데이터의 거래대금 10배
    폭증은 0.0723%(46/63,600)뿐이라 73일 x 40종목 스모크에서 기댓값이 2.1건이고,
    실제로 0건이 나왔다. 실데이터는 약 2.6%로 **36배 더 자주** 터진다. 그래서
    데이터에 의존하지 않고 **배관만** 고정한다.

    17번이 `IntradayWatcher`를 직접 검증하지만, `main.py`가 20봉 거래대금 중앙값을
    계산해 `prime()`에 넘기는 구간은 아무도 보지 않았다. **분모가 0이면 그 종목은
    조용히 영원히 발동하지 않는다.**
    """
    import datetime as dt

    import pandas as pd

    from main import TradingEngine

    s = make_settings()
    s.USE_INTRADAY = True
    s.INTRADAY_ENTRY_MODE = "burst"
    s.INTRADAY_BURST_MULT = 10.0
    s.INTRADAY_FLATTEN = False
    eng = TradingEngine(s, enable_dashboard=False)

    # 20봉 거래대금 중앙값이 1,000만원이 되도록 패널을 만든다
    # (종가 1,000원 x 거래량 10,000주). 마지막 봉은 '오늘'이라 기준에서 빠져야 한다.
    days = [f"2026090{i}" if i < 10 else f"202609{i}" for i in range(1, 26)]
    base = pd.DataFrame({
        "ts": days,
        "open": [1000.0] * len(days), "high": [1010.0] * len(days),
        "low": [990.0] * len(days), "close": [1000.0] * len(days),
        "volume": [10_000.0] * len(days),
    })
    eng.panel = {"AAA": base.copy()}
    eng._prev_daily = {}
    eng._now = lambda: dt.datetime(2026, 9, 26, 9, 5)   # 패널 마지막 봉보다 뒤
    eng._prime_intraday()

    st = eng.intraday.states.get("AAA")
    assert st is not None, "prime 이 종목 상태를 만들지 않았다"
    assert st.turnover_med20 > 0, "거래대금 중앙값이 0 — burst 가 영원히 발동하지 않는다"
    assert abs(st.turnover_med20 - 10_000_000.0) < 1.0,         f"중앙값 계산 오류: {st.turnover_med20:,.0f} (기대 10,000,000)"

    now = dt.datetime(2026, 9, 26, 9, 5)
    # 임계 = 1,000만 x 10 = 1억. 가격 1,000원이면 10만주가 필요하다.
    eng._intraday_queue.clear()
    eng._feed_intraday("AAA", 1000.0, 50_000.0, now)       # 5,000만 — 미달
    assert not eng._intraday_queue, "임계 미달인데 대기열에 들어갔다"
    eng._feed_intraday("AAA", 1000.0, 110_000.0, now + dt.timedelta(minutes=1))
    assert len(eng._intraday_queue) == 1,         f"임계 초과인데 발동하지 않았다 (대기열 {len(eng._intraday_queue)})"

    # USE_INTRADAY 가 꺼져 있으면 아무것도 흐르지 않는다
    eng._intraday_queue.clear()
    eng.intraday.reset_day("20260927")
    s.USE_INTRADAY = False
    eng._feed_intraday("AAA", 1000.0, 999_999.0, now + dt.timedelta(days=1))
    assert not eng._intraday_queue, "USE_INTRADAY=False 인데 흘렀다"

    msg = await _check_live_vol_alive(s)
    return ("20봉 거래대금 중앙값 전달 · 임계 미달/초과 판정 · USE_INTRADAY 분기 · "
            + msg + " 모두 작동 (Mock 데이터는 폭증이 0.07%뿐이라 burst 발생은 검증 불가)")


async def _check_live_vol_alive(s) -> str:
    """실거래 `vol_alive` 배관 — **확정 일봉으로 하루 한 번**, 진입 당일 제외.

    예전에는 장중 누적 거래대금(KIS `acml_vol`)을 20봉 중앙값으로 나눠 매 루프
    판정했다. 다음 날 09:00에 누적이 거의 0이라 하루 전체로 x25인 날도 x0.25로
    읽혀 **매일 아침 시가에 팔렸다.** Mock은 `acml_vol`에 일봉 거래량 전체를 넣어
    스모크로는 드러나지 않았다. 측정은 "봉 t 종가 판정 → t+1 시가 매도"다.
    """
    import datetime as dt
    import json
    import tempfile
    import types

    from main import TradingEngine

    keys = ("USE_INTRADAY", "INTRADAY_ENTRY_MODE", "USE_MOCK_MODE", "STATE_DIR",
            "EXIT_RULES", "EXIT_VOL_ALIVE_MULT", "EXIT_VOL_ALIVE_MIN_BARS",
            "MAX_HOLDING_BARS")
    saved = {k: getattr(s, k) for k in keys}
    try:
        s.USE_INTRADAY, s.INTRADAY_ENTRY_MODE = True, "burst"
        s.EXIT_RULES, s.EXIT_VOL_ALIVE_MULT, s.EXIT_VOL_ALIVE_MIN_BARS = "vol_alive", 2.0, 1
        s.MAX_HOLDING_BARS = 30

        def frame(days, vols, closes=None):
            closes = closes or [1000.0] * len(days)
            return pd.DataFrame({"ts": days, "open": closes, "high": closes,
                                 "low": closes, "close": closes, "volume": vols})

        days = [f"202609{i:02d}" for i in range(1, 26)]          # 9/01 ~ 9/25 확정
        today = "20260926"
        flat = [10_000.0] * 24                                     # 거래대금 1,000만
        # AAA: 어제(9/25) x25 · 오늘 미완성 봉(09:00, 거래 거의 없음)이 붙어 있다
        aaa = frame(days + [today], flat + [250_000.0, 100.0])
        bbb = frame(days, flat + [5_000.0])                        # 어제 x0.5 → 소멸
        ccc = frame(days, flat + [5_000.0])                        # 오늘 진입
        eng = TradingEngine(s, enable_dashboard=False)
        eng.panel = {"AAA": aaa, "BBB": bbb, "CCC": ccc}
        for sym, held in (("AAA", 1), ("BBB", 1), ("CCC", 0)):
            eng.portfolio.positions[sym] = Position(symbol=sym, qty=100, avg_price=1000.0,
                                                    last_price=1000.0, bars_held=held)
        eng._entry_dates = {"AAA": "20260924", "BBB": "20260924", "CCC": today}
        eng._refresh_alive(today)
        m = eng._vol_alive_map(today)
        assert abs(m.get("AAA", 0.0) - 25.0) < 1e-9, \
            f"어제 확정 거래대금을 안 썼다(오늘 미완성 봉이 섞였나): {m.get('AAA')}"
        assert abs(m.get("BBB", 0.0) - 0.5) < 1e-9, f"BBB 배수 오류: {m.get('BBB')}"
        assert "CCC" not in m, "진입 당일 종목을 판정했다 — 폭증 전날 거래대금으로 사자마자 판다"

        # 장중 누적이 아무리 작아도(09:00) 판정 입력은 그대로다
        eng.latest_volumes = {"AAA": 1.0, "BBB": 1.0, "CCC": 1.0}
        assert eng._vol_alive_map(today) == m, "판정 입력에 장중 누적이 섞였다"
        prices = {"AAA": 1000.0, "BBB": 1000.0, "CCC": 1000.0}
        exits = eng.risk.evaluate_exits(eng.portfolio, prices, burst_map=m,
                                        now=dt.datetime(2026, 9, 26, 9, 0))
        assert {e.symbol for e in exits} == {"BBB"}, \
            f"살아 있는 종목까지 팔았거나 죽은 종목을 놓쳤다: {[e.symbol for e in exits]}"

        # --- 날짜 롤오버: 여러 날 켜 두면 기준값이 새로 만들어져야 한다 ---
        # 9/26이 확정되면서 거래가 죽었다(x0.5). 전일 종가도 1,234원으로 바뀐다.
        aaa2 = frame(days + [today], flat + [250_000.0, 5_000.0],
                     closes=[1000.0] * 25 + [1234.0])
        eng.s.USE_MOCK_MODE = True
        eng.broker = types.SimpleNamespace(is_mock=True)
        eng._refresh_bars_from_sim = lambda: eng.panel.update({"AAA": aaa2})
        eng._now = lambda: dt.datetime(2026, 9, 29, 0, 1)        # 월요일 자정 직후
        eng._ref_day = today
        await eng._roll_daily_reference("20260929")
        assert eng._ref_day == "20260929", "롤오버가 기준일을 갱신하지 않았다"
        want = 1234.0 * 5_000.0 / 10_000_000.0                     # 9/26 거래대금 ÷ 20봉 중앙
        assert abs(eng._vol_alive_map("20260929").get("AAA", 0.0) - want) < 1e-9, \
            "롤오버 후에도 어제 판정값이 남았다 — 여러 날 켜 두면 기준이 고정된다"
        assert eng.intraday.states["AAA"].prev_close == 1234.0, \
            "롤오버가 전일 종가를 갱신하지 않았다(진입 기준이 첫날에 고정)"
        assert "CCC" in eng._vol_alive_map("20260929"), "다음 날에도 진입 당일로 취급했다"

        # --- 재시작: 진입일로 보유 봉 수 복원 · 기록 없는 보유는 오늘 진입 ---
        s.USE_MOCK_MODE = False
        eng3 = TradingEngine(s, enable_dashboard=False)
        eng3._now = lambda: dt.datetime(2026, 10, 14, 9, 0)        # 수요일
        for sym in ("AAA", "BBB"):
            eng3.portfolio.positions[sym] = Position(symbol=sym, qty=10, avg_price=1000.0,
                                                     last_price=1000.0)
        eng3._entry_dates = {"AAA": "20261012", "ZZZ": "20261001"}  # 월요일 진입 · 청산된 종목
        eng3._reconcile_entry_dates()
        assert eng3.portfolio.positions["AAA"].bars_held == 2, \
            f"보유 봉 수 복원 오류: {eng3.portfolio.positions['AAA'].bars_held} (기대 2: 화·수)"
        assert eng3._entry_dates.get("BBB") == "20261014", "기록 없는 보유를 오늘 진입으로 두지 않았다"
        assert "ZZZ" not in eng3._entry_dates, "보유하지 않는 종목의 진입일이 남았다"

        # --- 일봉 재적재가 실패해 패널이 묵었으면 그날 판정을 보류한다 ---
        # 묵은 패널로 판정하면 어제 진입한 종목이 폭증 **전날** 거래대금으로 읽혀
        # 잘못 팔린다. 10/13(화)의 직전 거래일은 10/12(월)다.
        old_days = [f"202609{i:02d}" for i in range(1, 26)] + ["20261008"]
        eng3.panel = {"AAA": frame(old_days, flat + [250_000.0, 5_000.0])}
        eng3._entry_dates = {"AAA": "20261008", "BBB": "20261008"}
        eng3._refresh_alive("20261013")
        assert not eng3._vol_alive_map("20261013"), \
            "마지막 확정 봉(10/08)이 직전 거래일(10/12)이 아닌데 판정했다"
        eng3._refresh_alive("20261012")          # 10/09 한글날 → 직전 거래일 10/08
        assert eng3._vol_alive_map("20261012"), \
            "연휴 다음 날 정상 패널을 묵었다고 판정했다(휴장일 달력 확인)"

        # --- 다른 모드(Mock)의 상태 파일은 실서버에 복구하지 않는다 ---
        with tempfile.TemporaryDirectory() as tmp:
            s.STATE_DIR = tmp
            Path(tmp, "engine_state.json").write_text(json.dumps(
                {"mode": "MOCK", "peak_equity": 2e8, "entry_dates": {"AAA": "20250101"}}),
                encoding="utf-8")
            eng4 = TradingEngine(s, enable_dashboard=False)
            eng4._restore_state()
            assert eng4.portfolio.peak_equity < 2e8, "Mock 자산 고점이 실서버로 넘어왔다"
            assert not eng4._entry_dates, "Mock 진입일(가상 날짜)이 실서버로 넘어왔다"

        # --- 폭증 감시는 유니버스 전체다: WS가 받지 않는 종목을 REST로 돈다 ---
        #
        # WS 구독 한도가 40이라 WS만 보면 284 중 244종목의 폭증을 놓친다. 예전에는
        # WS가 건강하면 REST 조회를 아예 건너뛰었다.
        class _Broker:
            def __init__(self):
                self.calls = []

            async def inquire_price(self, symbol):   # 진입 직전 현재가·종목 상태(모의 매매불가 거름)
                return {"output": {"stck_prpr": "10000", "iscd_stat_cls_code": "55", "marg_rate": "40.00"}}

            async def get_quote(self, symbol):
                self.calls.append(symbol)
                return types.SimpleNamespace(price=1000.0, volume=1.0)

        s.USE_MOCK_MODE = False
        eng5 = TradingEngine(s, enable_dashboard=False)
        eng5._now = lambda: dt.datetime(2026, 10, 14, 10, 0)
        eng5.broker = _Broker()
        eng5.universe = [f"U{i:02d}" for i in range(12)]
        eng5.watchlist = ["U00", "U01", "U02", "U03"]                # WS 구독 중
        eng5.ws = types.SimpleNamespace(is_healthy=lambda: True)
        for _ in range(4):
            await eng5._refresh_quotes()
        seen = set(eng5.broker.calls)
        assert seen and not (seen & set(eng5.watchlist)), \
            f"WS가 받는 종목을 REST로 중복 조회했다: {sorted(seen & set(eng5.watchlist))}"
        assert seen == set(eng5.universe) - set(eng5.watchlist), \
            f"WS 밖 종목을 다 돌지 못했다: {sorted(set(eng5.universe) - seen)}"
        # WS가 받지 않는 보유 종목은 매 루프 맨 앞에서 본다(손절 판단이 밀리지 않게).
        # REST 순회로 감지해 산 종목은 관심목록 밖이라 WS가 건강해도 여기에 해당한다.
        eng5.portfolio.positions["U07"] = Position(symbol="U07", qty=10, avg_price=1000.0,
                                                   last_price=1000.0)
        for healthy in (True, False):
            eng5.ws = types.SimpleNamespace(is_healthy=lambda h=healthy: h)
            for _ in range(3):
                eng5.broker.calls = []
                await eng5._refresh_quotes()
                assert eng5.broker.calls[0] == "U07", \
                    f"WS 밖 보유 종목이 맨 앞이 아니다(WS {healthy}): {eng5.broker.calls}"
                assert eng5.broker.calls.count("U07") == 1, "보유 종목을 한 루프에 두 번 조회했다"
        # WS가 받는 보유 종목은 REST로 중복 조회하지 않는다
        eng5.ws = types.SimpleNamespace(is_healthy=lambda: True)
        eng5.portfolio.positions.pop("U07")
        eng5.portfolio.positions["U01"] = Position(symbol="U01", qty=10, avg_price=1000.0,
                                                   last_price=1000.0)
        eng5.broker.calls = []
        await eng5._refresh_quotes()
        assert "U01" not in eng5.broker.calls, "WS가 받는 보유 종목을 REST로 중복 조회했다"
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
    return ("vol_alive 확정 일봉 판정(오늘 미완성 봉·장중 누적 배제 · 진입 당일 제외) · "
            "날짜 롤오버 · 재적재 실패 시 보류 · 재시작 복원 · 모드 격리 · "
            "WS 밖 종목 REST 순회")


@check("22. 장 시작 구간 진입 (opening 모드)")
async def test_opening_entry():
    """09:0N 분봉 판정 → 09:(N+지연)에 한 번 진입. **측정과 같은 순수 함수**를 쓴다.

    측정(`scripts/opening_study.py`)과 실거래(`main.py::_opening_scan`)가 창을 각자
    계산하면 한 칸만 어긋나도 다른 규칙을 실거래로 들고 간다. 그래서 (1) 창 경계,
    (2) 두 경로가 같은 함수 객체를 쓰는지, (3) KIS 분봉 파서, (4) 진입 배선을 본다.
    """
    import datetime as dt
    import types

    import scripts.opening_study as study
    from main import TradingEngine
    from strategy import opening as op
    from strategy.intraday import IntradayWatcher

    # --- 1) 창 경계: 09:00~09:0(N-1)만, 장전·09:0N 이후는 빼고 ---
    hhmm = ["0859", "0900", "0901", "0904", "0905", "0910"]
    mi = op.minute_index(hhmm)
    assert list(mi) == [-1, 0, 1, 4, 5, 10], f"경과 분 계산 오류: {list(mi)}"
    close = np.full(6, 1_000.0)
    vol = np.array([999.0, 1.0, 2.0, 3.0, 999.0, 999.0])
    assert op.window_turnover(mi, close, vol, 5) == 6_000.0, "창 경계가 틀렸다(09:00~09:04만 합산)"
    assert op.day_avg_base(390e6, 5) == 5e6
    assert op.opening_ratio(50e6, 390e6, 5) == 10.0
    assert op.opening_ratio(50e6, 0.0, 5) == 0.0, "기준선 0인데 배수를 냈다(발동 위험)"
    assert op.rank_triggers({"A": 3.0, "B": 12.0, "C": 10.0}, 10.0) == [("B", 12.0), ("C", 10.0)]

    # --- 2) 측정과 실거래가 같은 함수를 쓴다 ---
    for name in ("window_turnover", "minute_index", "day_avg_base", "opening_ratio"):
        assert getattr(study, name) is getattr(op, name), \
            f"측정 스크립트가 {name}을 따로 구현했다 — 측정과 실거래가 갈린다"

    # --- 3) KIS 당일 분봉 파서: 분 단위 ts · 분당 체결량 · 과거→최신 ---
    from api.base import AbstractBroker
    rows = [{"stck_bsop_date": "20261012", "stck_cntg_hour": f"09{m:02d}00",
             "stck_prpr": "1000", "stck_oprc": "1000", "stck_hgpr": "1000",
             "stck_lwpr": "1000", "cntg_vol": str(10 * (m + 1)), "acml_vol": "999999"}
            for m in range(4, -1, -1)]                     # KIS는 최신순
    fake = types.SimpleNamespace()

    async def _chart(symbol, hour="", past=True):
        return {"rt_cd": "0", "msg_cd": "", "msg1": "", "output2": rows}
    fake.inquire_minute_chart = _chart
    bars = await AbstractBroker.get_minute_bars(fake, "AAA", "090500")
    assert bars["ts"].tolist()[0] == "202610120900" and len(bars) == 5, \
        f"분봉 ts가 분 단위가 아니거나 정렬이 틀렸다: {bars['ts'].tolist()}"
    assert bars["volume"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0], \
        "분당 체결량(cntg_vol) 대신 누적(acml_vol)을 읽었다"

    # --- 3-1) 규칙 진입 점수가 리스크 관문의 모델 문턱에 걸리지 않는다 ---
    # 예전 burst 경로는 "가격 상승률 x 10"을 점수로 넣어 +2.2% 미만 폭증이 전부
    # `SIGNAL_BUY_THRESHOLD`(0.22)에 거부됐다. 측정에 없는 가격 필터였다.
    import main as main_mod
    assert main_mod.RULE_TRIGGER_SCORE >= Settings.model_fields["SIGNAL_BUY_THRESHOLD"].default,         "규칙 진입 점수가 모델 문턱보다 낮다 — 측정에 없는 필터가 생긴다"

    # --- 3-2) 복수시세 파서: 30종목씩 나눠 요청 · KIS 필드명 해석 ---
    calls_ = []

    async def _multi(symbols):
        calls_.append(list(symbols))
        return {"rt_cd": "0", "msg_cd": "", "msg1": "", "output": [
            {"inter_shrn_iscd": c, "inter2_prpr": "1000", "acml_vol": "10",
             "acml_tr_pbmn": "12345"} for c in symbols]}
    fake2 = types.SimpleNamespace(inquire_multi_price=_multi)
    syms_ = [f"{i:06d}" for i in range(65)]
    tq = await AbstractBroker.get_turnovers(fake2, syms_)
    assert [len(c) for c in calls_] == [30, 30, 5], f"30종목씩 나눠 요청하지 않았다: {[len(c) for c in calls_]}"
    assert len(tq) == 65 and tq["000007"]["turnover"] == 12345.0 and tq["000007"]["price"] == 1000.0, \
        "복수시세 필드(inter_shrn_iscd·inter2_prpr·acml_tr_pbmn) 해석 오류"

    # --- 4) 엔진 배선 ---
    s = get_settings()
    keys = ("USE_INTRADAY", "INTRADAY_ENTRY_MODE", "OPENING_WINDOW_MIN", "OPENING_MULT",
            "OPENING_LAG_MIN", "MAX_POSITIONS", "SIZING_MODE", "FIXED_POSITION_WEIGHT",
            "MAX_POSITION_WEIGHT", "MAX_GROSS_EXPOSURE", "USE_MOCK_MODE",
            "OPENING_BASELINE", "OPENING_HISTORY_DIR")
    saved = {k: getattr(s, k) for k in keys}
    try:
        s.USE_INTRADAY, s.INTRADAY_ENTRY_MODE = True, "opening"
        s.OPENING_WINDOW_MIN, s.OPENING_MULT, s.OPENING_LAG_MIN = 5, 10.0, 3
        s.MAX_POSITIONS, s.SIZING_MODE, s.FIXED_POSITION_WEIGHT = 2, "fixed", 0.45
        s.MAX_POSITION_WEIGHT, s.MAX_GROSS_EXPOSURE, s.USE_MOCK_MODE = 0.50, 0.95, False
        s.OPENING_BASELINE = "day_avg"            # 아래 5)에서 same_window 를 따로 본다

        # 시세 한 건으로는 발동하지 않는다(breakout 분기로 떨어지면 안 된다)
        w = IntradayWatcher(entry_mode="opening")
        w.prime("AAA", 1000.0, 1e6, turnover_med20=1e6)
        assert w.on_quote("AAA", 2000.0, 1e9, dt.datetime(2026, 10, 12, 9, 1)) is None, \
            "opening 모드인데 시세 한 건으로 발동했다(가격 돌파로 진입한다)"

        codes = list(get_settings().UNIVERSE[:5])
        # 평소 20봉 중앙 3.9억 → 5분 몫 500만. 배수: c0 x20 · c1 x12 · c2 x15 · c3 x5 · c4(보유)
        mult = {codes[0]: 20.0, codes[1]: 12.0, codes[2]: 15.0, codes[3]: 5.0, codes[4]: 50.0}

        class _Broker:
            def __init__(self):
                self.minute_calls = []
                self.turnover_calls = 0

            async def get_turnovers(self, symbols):
                # 복수시세 — 장 시작부터 지금까지의 누적 거래대금(5분치 = 500만 x 배수)
                self.turnover_calls += 1
                return {c: {"price": 10_000.0, "acml_vol": 0.0, "turnover": 5e6 * mult[c]}
                        for c in symbols}

            async def get_minute_bars(self, symbol, hour):
                self.minute_calls.append((symbol, hour))
                per_min = 5e6 * mult[symbol] / 5
                return pd.DataFrame({"ts": [f"20261012090{m}" for m in range(5)],
                                     "open": [10_000.0] * 5, "high": [10_000.0] * 5,
                                     "low": [10_000.0] * 5, "close": [10_000.0] * 5,
                                     "volume": [per_min / 10_000.0] * 5})

            async def inquire_price(self, symbol):   # 진입 직전 현재가·종목 상태(모의 매매불가 거름)
                return {"output": {"stck_prpr": "10000", "iscd_stat_cls_code": "55", "marg_rate": "40.00"}}

            async def get_quote(self, symbol):
                return types.SimpleNamespace(price=10_000.0, volume=0.0)

        eng = TradingEngine(s, enable_dashboard=False)
        eng.broker = _Broker()
        eng.universe = codes
        eng._prev_daily = {c: (10_000.0, 1e5, 390e6) for c in codes}
        eng.portfolio.positions[codes[4]] = Position(symbol=codes[4], qty=10,
                                                     avg_price=10_000.0, last_price=10_000.0)
        submitted = []

        async def _submit(symbol, side, qty, price, reason):
            submitted.append(symbol)
            eng.open_orders[f"o{len(submitted)}"] = {"symbol": symbol, "side": side.value,
                                                     "qty": qty, "price": price}
            return True
        eng._submit = _submit

        await eng._opening_scan(dt.datetime(2026, 10, 12, 9, 5, 1))
        # 판정은 복수시세 한 번(30종목/호출)이다 — 종목마다 분봉을 받으면 5분이 걸린다
        assert eng.broker.turnover_calls == 1 and not eng.broker.minute_calls, \
            "판정에 분봉을 종목마다 받았다(복수시세여야 한다 — day_avg 는 기록도 하지 않는다)"
        picks = [c for c, _ in eng._opening_picks]
        assert picks == [codes[0], codes[2], codes[1]], f"배수 큰 순·문턱 적용 오류: {picks}"

        eng._opening_day = "20261012"
        await eng._opening_entries(dt.datetime(2026, 10, 12, 9, 7), {})
        assert not submitted, "측정은 09:08(구간 끝+3분) 진입인데 그 전에 샀다"
        await eng._opening_entries(dt.datetime(2026, 10, 12, 9, 8), {})
        assert submitted == [codes[0]], \
            f"빈 슬롯 1개(보유 1)를 배수 1위로 채우지 않았다: {submitted}"
        await eng._opening_entries(dt.datetime(2026, 10, 12, 9, 30), {})
        assert submitted == [codes[0]], "하루 한 번인데 다시 진입했다"
        # 다음 날: 오늘 판정이 아직 없으면 어제 결과로 사지 않는다
        eng._opening_picks = [(codes[1], 12.0)]
        await eng._opening_entries(dt.datetime(2026, 10, 13, 9, 30), {})
        assert submitted == [codes[0]], "전날 판정 결과로 오늘 진입했다"

        # opening 모드의 REST 시세는 WS 밖 보유 종목만 본다(유니버스 순회 안 함)
        calls = []

        async def _q(symbol):
            calls.append(symbol)
            return types.SimpleNamespace(price=10_000.0, volume=0.0)
        eng.broker.get_quote = _q
        eng.ws = types.SimpleNamespace(is_healthy=lambda: True)
        eng.watchlist = []
        await eng._refresh_quotes()
        assert calls == [codes[4]], f"opening 모드에서 보유 외 종목을 조회했다: {calls}"

        # --- 5) same_window 기준선 (채택안) ---
        # (a) 측정의 pandas 와 같은 값
        rng = np.random.default_rng(22)
        seq = pd.Series(rng.lognormal(16, 1.0, 40))
        want = seq.rolling(20, min_periods=10).median().shift(1)
        for j in range(len(seq)):
            got = op.same_window_base(seq.iloc[:j].tolist())
            w_ = float(want.iloc[j]) if np.isfinite(want.iloc[j]) else 0.0
            assert abs(got - w_) < 1e-6 * max(1.0, w_),                 f"같은 시간대 기준선이 측정과 다르다(과거 {j}일): {got} vs {w_}"
        # (b) 엔진 — 과거 파일에서 적재, 판정 후 기록, 보유도 기록, 같은 날 중복 기록 없음
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            s.OPENING_BASELINE, s.OPENING_HISTORY_DIR = "same_window", tmp
            s.OPENING_WINDOW_MIN = 5
            past = [f"202609{d:02d}" for d in range(1, 21)]           # 과거 20일
            for c in codes:
                # 과거 매일 09:00~09:04 거래대금 500만(분당 100만 = 1만원 x 100주)
                rows_ = [(f"{dd}090{m}", 10_000.0, 10_000.0, 10_000.0, 10_000.0, 100.0)
                         for dd in past for m in range(5)]
                pd.DataFrame(rows_, columns=["ts", "open", "high", "low", "close", "volume"]
                             ).to_csv(Path(tmp) / f"{c}.csv", index=False)
            eng2 = TradingEngine(s, enable_dashboard=False)
            eng2.broker = _Broker()                                   # 오늘: 500만 x mult
            eng2.universe = codes
            eng2._prev_daily = {}                                     # same_window 는 일봉이 필요 없다
            eng2.portfolio.positions[codes[4]] = Position(symbol=codes[4], qty=10,
                                                          avg_price=10_000.0, last_price=10_000.0)
            eng2._load_opening_history("20261012")
            assert all(len(eng2._opening_hist[c]) == 20 for c in codes), "과거 20일 적재 실패"
            assert abs(eng2._opening_hist[codes[0]][-1] - 5e6) < 1.0, "과거 구간 거래대금 계산 오류"
            await eng2._opening_scan(dt.datetime(2026, 10, 12, 9, 5, 1))
            assert eng2._opening_hist[codes[0]][-1] == 5e6, \
                "판정 직후 기준선에 오늘이 들어갔다(기록은 진입 뒤여야 한다)"
            await eng2._opening_record_task                           # 진입 뒤 기록
            got_picks = dict(eng2._opening_picks)
            assert abs(got_picks.get(codes[0], 0) - 20.0) < 1e-6,                 f"기준선에 오늘이 섞였거나 배수가 틀렸다: {got_picks}"
            assert codes[4] not in got_picks, "보유 종목을 후보로 올렸다"
            for c in codes:                                           # 보유 포함 전부 기록
                d_ = pd.read_csv(Path(tmp) / f"{c}.csv", dtype={"ts": str})
                assert (d_["ts"].str[:8] == "20261012").sum() == 5, f"{c} 오늘 분봉 기록 오류"
            assert len(eng2._opening_hist[codes[0]]) == 20 and                 abs(eng2._opening_hist[codes[0]][-1] - 1e8) < 1.0, "오늘 값이 기준선에 이어지지 않았다"
            # 같은 날 다시 판정(재시작)해도 파일에 두 번 쓰지 않는다
            eng3 = TradingEngine(s, enable_dashboard=False)
            eng3.broker, eng3.universe, eng3._prev_daily = _Broker(), codes, {}
            eng3._load_opening_history("20261012")
            await eng3._opening_scan(dt.datetime(2026, 10, 12, 9, 6))
            await eng3._opening_record_task
            d_ = pd.read_csv(Path(tmp) / f"{codes[0]}.csv", dtype={"ts": str})
            assert (d_["ts"].str[:8] == "20261012").sum() == 5, "같은 날을 두 번 기록했다"
            # 기준선이 10일 미만이면 발동하지 않는다
            eng3._opening_hist = {c: [5e6] * 9 for c in codes}
            await eng3._opening_scan(dt.datetime(2026, 10, 13, 9, 5))
            eng3._opening_record_task.cancel()
            assert not eng3._opening_picks, "기준선 9일인데 발동했다(최소 10일)"
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
    return ("창 경계(09:00~09:0N-1) · 측정과 같은 함수 · KIS 분봉 파서(분 단위·분당 체결량) · "
            "복수시세 1회 판정 · 판정→지연 후 1회 진입(배수 순·빈 슬롯·보유 제외·전날 결과 차단) · "
            "시세 1건 미발동 · 보유만 시세 조회 · same_window(측정과 같은 값·판정 후 기록·"
            "보유 기록·중복 기록 없음·10일 미만 미발동) 모두 작동")


@check("23. 장중 슬라이딩 윈도우 판정")
def test_sliding():
    """최근 W분 거래대금이 기준선의 k배가 되는 첫 분 — 측정과 같은 순수 함수.

    창이 한 칸 어긋나면(분 m에서 m의 거래가 들어가는지) 측정이 미래를 보게 되거나 실거래가
    다른 규칙을 들고 간다. 경계와 기준선을 pandas 계산과 대조한다.
    """
    import scripts.sliding_study as study
    from strategy import sliding as sl

    tv = np.arange(1.0, 11.0)                                     # 분당 1,2,...,10
    rs = sl.rolling_sum(tv, 3)
    assert np.isnan(rs[0]) and np.isnan(rs[1]), "창이 차기 전에 값을 냈다"
    assert rs[2] == 6.0 and rs[9] == 27.0, f"창 경계 오류: {rs}"     # 1+2+3 · 8+9+10
    mt = sl.minute_turnover(np.array([0, 0, 2, 999]), np.array([10.0, 10.0, 5.0, 1.0]),
                            np.array([1.0, 2.0, 4.0, 1.0]), 5)
    assert list(mt) == [30.0, 0.0, 20.0, 0.0, 0.0], f"분 집계 오류(범위 밖 제외): {list(mt)}"

    # 같은 시각 기준선 = 시각마다 pandas rolling(20, min_periods=10) 중앙(오늘 제외)
    rng = np.random.default_rng(23)
    H = rng.lognormal(10, 1.0, (30, 4))
    H[3, 1] = np.nan
    want = pd.DataFrame(H).rolling(20, min_periods=10).median().shift(1).to_numpy()
    for j in range(30):
        got = sl.same_time_base(H[:j]) if j else np.full(4, np.nan)
        for col in range(4):
            a, b = got[col], want[j, col]
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9 * max(1.0, b),                 f"같은 시각 기준선이 pandas 와 다르다(과거 {j}일, 열 {col}): {a} vs {b}"

    base = np.array([10.0, 10.0, 10.0, 10.0])
    assert sl.first_trigger(np.array([np.nan, 20.0, 60.0, 90.0]), base, 5.0) == (2, 6.0)
    assert sl.first_trigger(np.array([1.0, 2.0, 3.0, 4.0]), 1.0, 5.0) is None
    assert sl.first_trigger(np.array([1.0, 2.0, 3.0, 99.0]), 1.0, 5.0, last_min=2) is None,         "진입 마감(15:19) 이후 발동을 냈다"

    for name in ("rolling_sum", "minute_turnover", "same_time_base", "first_trigger"):
        assert getattr(study, name) is getattr(sl, name), f"측정 스크립트가 {name}을 따로 구현했다"
    return "창 경계 · 분 집계 · 같은 시각 기준선(pandas 대조) · 첫 발동·마감 · 측정과 같은 함수 모두 작동"


@check("24. 장중 슬라이딩 배관 (sliding 모드)")
async def test_sliding_wiring():
    """감시 루프 → 대기열 → 리스크 관문 → 주문, 그리고 그날 분봉 기록.

    (1) 진행 중인 분은 판정하지 않는다(거래량이 덜 찼다 — 측정은 끝난 분만 쓴다)
    (2) 같은 분이 다시 오면 완성된 값으로 덮어쓴다 (3) 하루 한 번만 발동한다
    (4) 기준선은 과거 하루 전체 분봉에서 같은 시각으로 만든다(측정과 같은 함수)
    (5) 발동이 이른 순으로 빈 슬롯만 채우고 나머지는 버린다 (6) 그날 분봉을 한 번만 기록한다
    """
    import datetime as dt
    import tempfile
    import types

    from main import TradingEngine
    from strategy.sliding import LAST_ENTRY_MIN, SlidingWatcher

    def bars_of(day, vols, start=0):
        ts = [f"{day}{9 + (start + k) // 60:02d}{(start + k) % 60:02d}" for k in range(len(vols))]
        px = [1_000.0] * len(vols)
        return ts, px, px, px, px, list(map(float, vols))

    # --- 감시기 단독 ---
    w = SlidingWatcher(window=3, mult=5.0)
    w.reset_day("20261012")
    w.set_base("A", np.full(LAST_ENTRY_MIN + 1, 3e6))            # 3분 기준 300만(=1천원 x 3천주)
    # 09:00~09:03 거래량 1천주씩, 09:04는 진행 중(아직 10주) — 지금 09:04
    assert w.update("A", *bars_of("20261012", [1000, 1000, 1000, 1000, 10]), now_minute=4) is None, \
        "평소 수준(x1)인데 발동했다"
    # 09:04가 완성되며 2만주로 바뀌었다 → 09:02~09:04 합 = 1천원 x 2.2만주 = 2,200만 ≥ 5 x 300만
    hit = w.update("A", *bars_of("20261012", [1000, 1000, 1000, 1000, 20000, 5]), now_minute=5)
    assert hit is not None and hit[0] == 4, f"완성된 분 덮어쓰기·발동 분 오류: {hit}"
    assert w.update("A", *bars_of("20261012", [9e9] * 8), now_minute=8) is None, "하루 두 번 발동했다"
    w.reset_day("20261013")
    assert "A" not in w.fired and not w.bars, "날짜가 바뀌었는데 상태가 남았다"
    # 진행 중인 분만 크면 발동하지 않는다
    w2 = SlidingWatcher(window=3, mult=5.0)
    w2.reset_day("20261012")
    w2.set_base("B", 3e6)
    assert w2.update("B", *bars_of("20261012", [1000, 1000, 1000, 9e9]), now_minute=3) is None, \
        "진행 중인 분(거래량이 덜 찬 분)으로 발동했다"

    # --- 엔진 배선 ---
    s = get_settings()
    keys = ("USE_INTRADAY", "INTRADAY_ENTRY_MODE", "SLIDING_WINDOW_MIN", "SLIDING_MULT",
            "SLIDING_BASELINE", "SLIDING_HISTORY_DIR", "MAX_POSITIONS", "SIZING_MODE",
            "FIXED_POSITION_WEIGHT", "MAX_POSITION_WEIGHT", "MAX_GROSS_EXPOSURE", "USE_MOCK_MODE")
    saved = {k: getattr(s, k) for k in keys}
    try:
        s.USE_INTRADAY, s.INTRADAY_ENTRY_MODE = True, "sliding"
        s.SLIDING_WINDOW_MIN, s.SLIDING_MULT, s.SLIDING_BASELINE = 3, 5.0, "same_time"
        s.MAX_POSITIONS, s.SIZING_MODE, s.FIXED_POSITION_WEIGHT = 2, "fixed", 0.45
        s.MAX_POSITION_WEIGHT, s.MAX_GROSS_EXPOSURE, s.USE_MOCK_MODE = 0.50, 0.95, False
        codes = list(get_settings().UNIVERSE[:4])
        with tempfile.TemporaryDirectory() as tmp:
            s.SLIDING_HISTORY_DIR = tmp
            past = [f"202609{d:02d}" for d in range(1, 21)]
            for c in codes:                                       # 과거 매 분 1천원 x 1천주
                rows_ = [(f"{dd}{9 + m // 60:02d}{m % 60:02d}", 1e3, 1e3, 1e3, 1e3, 1000.0)
                         for dd in past for m in range(0, 30)]
                pd.DataFrame(rows_, columns=["ts", "open", "high", "low", "close", "volume"]
                             ).to_csv(Path(tmp) / f"{c}.csv", index=False)
            eng = TradingEngine(s, enable_dashboard=False)
            eng.universe = codes
            eng._load_sliding_history("20261012")
            base = np.asarray(eng.sliding.base[codes[0]])
            assert abs(base[5] - 3e6) < 1.0 and np.isnan(base[0]), \
                f"같은 시각 기준선 오류(3분 합 300만, 첫 2분은 창이 안 참): {base[:6]}"

            # 오늘: c0 은 09:03에, c1 은 09:02에, c2 는 09:04에 터진다. c3 는 보유 중
            spike = {codes[0]: 3, codes[1]: 2, codes[2]: 4, codes[3]: 2}

            class _Broker:
                async def get_minute_bars(self, symbol, hour):
                    vols = [1000.0] * 10
                    vols[spike[symbol]] = 50_000.0
                    ts, o, h, lo, c, v = bars_of("20261012", vols)
                    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": lo,
                                         "close": c, "volume": v})

            eng.broker = _Broker()
            eng._now = lambda: dt.datetime(2026, 10, 12, 9, 9, 30)
            eng.portfolio.positions[codes[3]] = Position(symbol=codes[3], qty=10,
                                                         avg_price=1e3, last_price=1e3)
            for c in codes:
                await eng._sliding_poll_once(c)
            got = sorted(eng._sl_queue)
            assert [x[2] for x in got] == [codes[1], codes[0], codes[2]], \
                f"발동 순서 오류(보유 제외, 이른 순): {got}"
            submitted = []

            async def _submit(symbol, side, qty, price, reason):
                submitted.append(symbol)
                eng.open_orders[f"o{len(submitted)}"] = {"symbol": symbol, "side": side.value,
                                                         "qty": qty, "price": price}
                return True
            eng._submit = _submit
            await eng._sliding_entries(dt.datetime(2026, 10, 12, 9, 10), {})
            assert submitted == [codes[1]], f"빈 슬롯 1개를 가장 이른 발동으로 채우지 않았다: {submitted}"
            assert not eng._sl_queue, "못 산 발동을 다음 루프로 넘겼다(측정은 버린다)"

            # 그날 분봉을 한 번만 기록한다
            eng._flush_sliding_bars()
            eng._flush_sliding_bars()
            d_ = pd.read_csv(Path(tmp) / f"{codes[0]}.csv", dtype={"ts": str})
            assert (d_["ts"].str[:8] == "20261012").sum() == 10, "오늘 분봉 기록 누락·중복"
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
    return ("완성된 분만 판정 · 진행 중 분 덮어쓰기 · 하루 1회 · 날짜 초기화 · 같은 시각 기준선 적재 · "
            "이른 순 빈 슬롯만(보유 제외·나머지 버림) · 그날 분봉 1회 기록 모두 작동")


@check("25. 요건 전용 소액 매매 (지수 종목 5개)")
async def test_compliance_trades():
    """지수 종목 요건 — 모자라면 소액으로 사서 당일 판다. 전략 자리 2개를 먹지 않는다.

    측정에서 지금 진입 규칙은 30거래일에 지수 종목을 중앙 2개만 샀다(충족 10.7%). 요건을
    놓치면 수익률과 무관하게 실격이다. 보는 것: (1) 모자란 수 + 여유만 고른다 (2) 관문이
    금액을 묶고 자리 한도를 면제한다 (3) 요건 포지션이 전략 자리를 막지 않는다 (4) 시작일
    전에는 사지 않고, 매수 시각에 한 번 사고, 매도 시각에 판다.
    """
    import datetime as dt
    import json
    import tempfile
    import types

    from main import TradingEngine
    from strategy.compliance import compliance_picks, index_symbols_traded
    from strategy.phase import phase_config
    from strategy.risk_manager import RiskManager

    members = {"I1", "I2", "I3", "I4", "I5", "I6", "I7"}
    ranked = ["X1", "I1", "I2", "I3", "I4", "I5", "I6", "I7"]
    assert compliance_picks({"I1", "I2"}, members, ranked, 5) == ["I3", "I4", "I5", "I6"], \
        "모자란 3개 + 여유 1개를 거래대금 순으로 골라야 한다(이미 매매한 것·지수 밖 제외)"
    assert compliance_picks({"I1", "I2", "I3", "I4", "I5"}, members, ranked, 5) == [], "충족인데 골랐다"
    assert compliance_picks({"I1"}, members, ranked, 5, exclude={"I2"})[0] == "I3", "보유·미체결 제외 실패"
    assert index_symbols_traded([{"symbol": "I1"}, {"symbol": "X1"}], members) == {"I1"}

    s = get_settings()
    keys = ("SIZING_MODE", "FIXED_POSITION_WEIGHT", "MAX_POSITIONS", "MAX_POSITION_WEIGHT",
            "MAX_GROSS_EXPOSURE", "COMPLIANCE_TRADES", "COMPLIANCE_TRADE_KRW", "COMPLIANCE_START",
            "COMPLIANCE_BUY_HHMM", "COMPLIANCE_SELL_HHMM", "MIN_TRADE_SYMBOLS", "LOG_DIR",
            "LEDGER_PATH", "COMP_START", "COMP_END", "PHASE1_END", "USE_MOCK_MODE", "UNIVERSE")
    saved = {k: getattr(s, k) for k in keys}
    try:
        s.SIZING_MODE, s.FIXED_POSITION_WEIGHT, s.MAX_POSITIONS = "fixed", 0.45, 2
        s.MAX_POSITION_WEIGHT, s.MAX_GROSS_EXPOSURE = 0.50, 0.95
        codes = list(get_settings().UNIVERSE[:8])
        # --- (2) 관문: 금액을 묶고 자리 한도를 면제한다 ---
        rm = RiskManager(s)
        pf = Portfolio(cash=10_000_000.0, initial_capital=100_000_000.0)
        for c in codes[:2]:                                        # 전략 2자리 꽉 참
            pf.positions[c] = Position(symbol=c, qty=4_500, avg_price=10_000.0, last_price=10_000.0)
        now = dt.datetime(2026, 10, 19, 9, 30)
        sig = Signal(symbol=codes[2], score=1.0, price=10_000.0, ts=now, bet=-1.0)
        d1 = rm.evaluate_entry(sig, pf, now, pending={}, max_notional=1_000_000.0)
        assert d1.approved and d1.qty == 100, f"요건 매매가 자리 한도에 막혔거나 금액이 안 묶였다: {d1.reason} {d1.qty}"
        d2 = rm.evaluate_entry(sig, pf, now, pending={})
        assert not d2.approved, "전략 진입이 자리 한도를 넘었다"
        # 요건 포지션은 전략 자리를 먹지 않는다
        pf.positions[codes[1]].qty = 100
        d3 = rm.evaluate_entry(sig, pf, now, pending={}, slot_exempt={codes[1]})
        assert d3.approved, f"요건 포지션이 전략 자리를 막았다: {d3.reason}"

        # --- (4) 엔진: 시작일 · 매수 시각 · 매도 시각 ---
        with tempfile.TemporaryDirectory() as tmp:
            s.LOG_DIR, s.LEDGER_PATH = tmp, str(Path(tmp) / "ledger.jsonl")
            s.COMP_START, s.COMP_END, s.PHASE1_END = "2026-10-12", "2026-11-20", "2026-10-16"
            s.COMPLIANCE_TRADES, s.COMPLIANCE_TRADE_KRW = True, 1_000_000.0
            s.COMPLIANCE_START, s.COMPLIANCE_BUY_HHMM, s.COMPLIANCE_SELL_HHMM = "2026-10-19", "0930", "1430"
            s.MIN_TRADE_SYMBOLS, s.USE_MOCK_MODE = 5, False
            mem = sorted(x for lst in s.index_members().values() for x in lst)
            tradable = set((s._universe_blob() or {}).get("tradable") or mem)
            fm = set(s.full_margin_symbols())   # 증거금 100%(10/14) — 요건 매매는 피한다
            uni = [m_ for m_ in mem if m_ in tradable and m_ not in fm][:8]   # 실거래처럼 유니버스 ∩ 지수
            fm_mem = [m_ for m_ in mem if m_ in fm][:1]
            s.UNIVERSE = list(uni) + fm_mem     # 규정 필터(유니버스 밖 거부)를 통과시킨다
            assert len(uni) >= 7, "지수 멤버십 정보가 부족하다(유니버스 파일 확인)"
            # 이미 지수 종목 2개를 매매했다(10/19 이후 체결)
            with open(s.LEDGER_PATH, "w", encoding="utf-8") as f:
                for c in uni[:2]:
                    f.write(json.dumps({"event": "fill", "ts": "2026-10-19T09:13:00",
                                        "symbol": c, "qty": 1, "price": 10000.0}) + "\n")
            eng = TradingEngine(s, enable_dashboard=False)
            eng.universe = uni + fm_mem
            eng._prev_daily = {c: (1e4, 1e5, 1e9 - k) for k, c in enumerate(uni)}
            for c in fm_mem:                    # 거래대금 1위여도 증거금 100%면 뽑지 않는다
                eng._prev_daily[c] = (1e4, 1e5, 2e9)
            # 거래대금 1위인 미매매 종목이 1주 120만원 — 예산(100만원)으로 못 사므로 건너뛰어야 한다(10/19 실측)
            eng._prev_daily[uni[2]] = (1.2e6, 1e5, 1e9 - 2)

            class _Broker:
                async def inquire_price(self, symbol):   # 진입 직전 현재가·종목 상태(모의 매매불가 거름)
                    return {"output": {"stck_prpr": "10000", "iscd_stat_cls_code": "55", "marg_rate": "40.00"}}

                async def get_quote(self, symbol):
                    return types.SimpleNamespace(price=10_000.0, volume=0.0)
            eng.broker = _Broker()
            bought = []

            async def _submit(symbol, side, qty, price, reason):
                bought.append((symbol, side.value, qty))
                return True
            eng._submit = _submit
            eng.phase = phase_config(settings=s, now=dt.date(2026, 10, 15))
            await eng._compliance_trades(dt.datetime(2026, 10, 15, 9, 31), {})
            assert not bought, "시작일(10/19) 전에 요건 매매를 했다(초기화가 실적을 지울 수 있다)"
            eng.phase = phase_config(settings=s, now=dt.date(2026, 10, 19))
            await eng._compliance_trades(dt.datetime(2026, 10, 19, 9, 29), {})
            assert not bought, "매수 시각 전에 샀다"
            await eng._compliance_trades(dt.datetime(2026, 10, 19, 9, 31), {})
            buys = [b for b in bought if b[1] == "BUY"]
            assert [b[0] for b in buys] == uni[3:7] and all(b[2] == 100 for b in buys), \
                f"모자란 3 + 여유 1을 100만원씩 사야 한다(1주가 예산보다 비싼 종목 제외): {buys}"
            await eng._compliance_trades(dt.datetime(2026, 10, 19, 10, 0), {})
            assert len([b for b in bought if b[1] == "BUY"]) == 4, "하루에 두 번 샀다"
            # 체결됐다고 치고 매도 시각에 판다. 전략 자리는 비어 있어야 한다
            for c in uni[3:7]:
                eng.portfolio.positions[c] = Position(symbol=c, qty=100, avg_price=1e4, last_price=1e4)
            assert not eng._strategy_held(), "요건 포지션이 전략 자리를 차지했다"
            await eng._compliance_trades(dt.datetime(2026, 10, 19, 14, 31), {})
            sells = [b for b in bought if b[1] == "SELL"]
            assert sorted(b[0] for b in sells) == sorted(uni[3:7]), f"당일 청산 실패: {sells}"
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
    return ("모자란 수+여유 선택(1주가 예산보다 비싼 종목 제외) · 관문 금액 제한·자리 면제 · 전략 자리 미점유 · 시작일 전 금지 · "
            "하루 1회 매수 · 당일 청산 모두 작동")


@check("26. 서버 신뢰성 (주문 멱등성·무손실 채널·지연 지표·시계 오차)")
async def test_server_reliability():
    """실무 백엔드 관점의 장치 넷 — 네트워크 없이 httpx MockTransport 로 검증한다.

    (1) 주문은 '결과를 모르는' 실패(읽기 타임아웃·5xx)에 재시도하지 않는다 — 이중 주문 방지.
        접속 실패(요청이 나가지 않음)는 재시도한다. 시세 조회는 멱등이라 재시도한다.
    (2) 시세가 폭주해 큐가 넘쳐도 체결 이벤트는 버려지지 않는다.
    (3) 지연 히스토그램의 분위수와 Prometheus 출력.
    (4) 증권사 `Date` 헤더로 시계 오차를 잰다.
    """
    import datetime as dt

    import httpx

    from api.base import UNKNOWN_RESULT_CODE
    from api.client import KISClient
    from core.events import EventBus
    from core.latency import LatencyStats

    s = get_settings()
    saved = {k: getattr(s, k) for k in ("HTTP_MAX_RETRY", "HTTP_BACKOFF_BASE", "DRY_RUN")}
    try:
        s.HTTP_MAX_RETRY, s.HTTP_BACKOFF_BASE, s.DRY_RUN = 3, 0.0, False

        def client_with(handler):
            c = KISClient(s)
            c._client = httpx.AsyncClient(base_url="https://kis.test",
                                          transport=httpx.MockTransport(handler))

            async def _headers(tr_id, hashkey=None):
                return {"tr_id": tr_id}

            async def _hashkey(body):
                return "h"
            c._headers, c._hashkey = _headers, _hashkey
            return c

        ok_body = {"rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok", "output": {"ODNO": "1"}}

        # (1-a) 주문 + 읽기 타임아웃 → 재시도하지 않고 결과 불명
        calls = {"n": 0}

        def read_timeout(request):
            calls["n"] += 1
            raise httpx.ReadTimeout("응답 대기 초과", request=request)
        c = client_with(read_timeout)
        r = await c._request("POST", "/order", "TTTC0802U", body={"a": 1}, idempotent=False)
        assert calls["n"] == 1 and r.get("msg_cd") == UNKNOWN_RESULT_CODE, \
            f"주문이 읽기 타임아웃 뒤 재시도됐다(이중 주문 위험): 호출 {calls['n']}회 {r.get('msg_cd')}"

        # (1-b) 주문 + 접속 실패(요청이 나가지 않음) → 재시도해서 성공
        calls["n"] = 0

        def connect_then_ok(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("접속 실패", request=request)
            return httpx.Response(200, json=ok_body)
        c = client_with(connect_then_ok)
        r = await c._request("POST", "/order", "TTTC0802U", body={"a": 1}, idempotent=False)
        assert calls["n"] == 2 and r.get("rt_cd") == "0", "접속 실패는 재시도해야 한다(요청이 나가지 않았다)"

        # (1-c) 주문 + 5xx → 결과 불명(재시도 안 함)
        calls["n"] = 0

        def server_error(request):
            calls["n"] += 1
            return httpx.Response(500, json={"rt_cd": "1", "msg_cd": "E500", "msg1": "err"})
        c = client_with(server_error)
        r = await c._request("POST", "/order", "TTTC0802U", body={"a": 1}, idempotent=False)
        assert calls["n"] == 1 and r.get("msg_cd") == UNKNOWN_RESULT_CODE, "주문 5xx 를 재시도했다"

        # (1-d) 시세 조회(멱등) + 읽기 타임아웃 → 재시도
        calls["n"] = 0

        def timeout_then_ok(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("x", request=request)
            return httpx.Response(200, json=ok_body,
                                  headers={"date": "Tue, 13 Oct 2026 08:00:00 GMT"})
        c = client_with(timeout_then_ok)
        r = await c._request("GET", "/quote", "FHKST01010100", params={})
        assert calls["n"] == 2 and r.get("rt_cd") == "0", "멱등한 조회는 재시도해야 한다"
        snap = c.http_stats.snapshot()["FHKST01010100"]
        assert snap["count"] == 2 and snap["errors"] == 1, f"지연 지표 기록 오류: {snap}"
        # (4) 시계 오차 — 서버 08:00:00 UTC 를 기준으로 로컬과의 차이를 낸다
        assert c.clock_skew_s is not None, "Date 헤더로 시계 오차를 재지 않았다"
        for v in (c._client,):
            await v.aclose()

        # (2) 무손실 채널 — 손실 채널을 넘치게 해도 체결은 다 온다
        bus = EventBus(maxsize=5)
        got_fills, got_quotes = [], []

        async def on_fill(x):
            got_fills.append(x)

        async def on_quote(x):
            got_quotes.append(x)
        bus.subscribe("fill", on_fill)
        bus.subscribe("quote", on_quote)
        for i in range(50):
            await bus.publish("quote", i)
            if i % 10 == 0:
                await bus.publish("fill", f"F{i}")
        bus.start()
        for _ in range(50):
            await asyncio.sleep(0)
            if len(got_fills) == 5 and not bus.pending:
                break
        await bus.stop()
        assert got_fills == ["F0", "F10", "F20", "F30", "F40"], f"체결 이벤트가 유실됐다: {got_fills}"
        assert bus.dropped == 45 and got_quotes == list(range(45, 50)), \
            f"손실 채널은 최신 5개만 남겨야 한다: dropped={bus.dropped} quotes={got_quotes}"

        # (3) 히스토그램 분위수·Prometheus
        ls = LatencyStats()
        for ms in [20] * 90 + [400] * 9 + [3000]:
            ls.observe("TR", ms / 1000)
        q = ls.snapshot()["TR"]
        assert q["p50_ms"] == 25.0 and q["p95_ms"] == 500.0 and q["p99_ms"] == 500.0 \
            and q["max_ms"] == 3000.0, f"분위수 오류: {q}"
        prom = "\n".join(ls.prometheus())
        assert 'kis_http_request_seconds_bucket{tr="TR",le="+Inf"} 100' in prom, "Prometheus 누적 버킷 오류"
    finally:
        for k, v in saved.items():
            setattr(s, k, v)

    # --- 장중 VWAP 청산: 순수 판정 + 엔진 배관(누적 거래대금 차분 · W분 전 표본 · 기준선 분 인덱스) ---
    import datetime as dt
    import types
    from collections import deque
    from core.events import Position
    from main import TradingEngine
    from strategy.exit_rules import vwap_fade
    # VWAP = 10억/10만주 = 10,000원
    assert vwap_fade(9_900, 1e9, 1e5, 4e6, 5e6) is not None, "VWAP 아래 + 거래 감소인데 안 팔았다"
    assert vwap_fade(10_100, 1e9, 1e5, 4e6, 5e6) is None, "VWAP 위인데 팔았다"
    assert vwap_fade(9_900, 1e9, 1e5, 6e6, 5e6) is None, "거래가 평소 이상인데 팔았다"
    assert vwap_fade(9_900, 1e9, 1e5, 4e6, None) is None, "기준선을 모르는데 팔았다"
    assert vwap_fade(9_950, 1e9, 1e5, 4e6, 5e6, below=0.01) is None, "VWAP 1% 아래 조건 무시"

    saved = {k: getattr(s, k) for k in ("VWAP_EXIT", "VWAP_EXIT_WINDOW", "VWAP_EXIT_START_MIN",
                                        "VWAP_EXIT_POLL_SEC")}
    try:
        s.VWAP_EXIT, s.VWAP_EXIT_WINDOW, s.VWAP_EXIT_START_MIN, s.VWAP_EXIT_POLL_SEC = True, 10, 10, 0.0
        eng = TradingEngine(s, enable_dashboard=False)
        eng._vwap_profile = {"AAA": [5e6] * 380}     # 평소 직전 10분 5백만
        cum = {"v": 9.0e8}

        class _B:
            async def get_turnovers(self, symbols):
                # VWAP = 누적 9억 / 9만주 = 1만원 → 9,900원은 VWAP 아래
                return {"AAA": {"price": 9_900.0, "acml_vol": 9e4, "turnover": cum["v"]}}
        eng.broker = _B()
        eng.portfolio.positions["AAA"] = Position(symbol="AAA", qty=10, avg_price=10_000.0,
                                                  last_price=10_000.0)
        sold = []

        async def _submit(symbol, side, qty, price, reason):
            sold.append((symbol, qty, reason))
            return True
        eng._submit = _submit
        t0 = dt.datetime(2026, 10, 15, 10, 0)
        await eng._vwap_exits(t0)
        assert not sold, "W분 전 표본도 없이 팔았다"
        cum["v"] += 3e6                                # 10분 동안 3백만(평소 5백만보다 적다)
        await eng._vwap_exits(t0 + dt.timedelta(minutes=10))
        assert sold and sold[0][0] == "AAA", f"VWAP 아래 + 거래 감소인데 안 팔았다: {sold}"
        sold.clear()
        eng._vwap_hist.clear()
        eng._vwap_hist["AAA"] = deque([(t0.timestamp(), 9.0e8)])
        cum["v"] = 9.0e8 + 8e6                         # 10분 8백만(평소 이상) → 팔지 않는다
        await eng._vwap_exits(t0 + dt.timedelta(minutes=10))
        assert not sold, "거래가 평소 이상인데 팔았다"
        await eng._vwap_exits(dt.datetime(2026, 10, 15, 9, 5))
        assert not sold, "09:10 전에 판정했다"
    finally:
        for k, v in saved.items():
            setattr(s, k, v)

    # --- 장중 추가 매수: 첫 발동만 · 배수 순 · 빈자리만 · 오늘 판 종목 제외 ---
    saved = {k: getattr(s, k) for k in ("SLIDE_ENTRY", "SLIDE_WINDOW", "SLIDE_MULT", "MAX_POSITIONS",
                                        "SLIDE_MIN_CHASE")}
    try:
        s.SLIDE_ENTRY, s.SLIDE_WINDOW, s.SLIDE_MULT, s.MAX_POSITIONS = True, 5, 3.0, 2
        s.SLIDE_MIN_CHASE = 0.0        # `.env`(0.03)가 새면 전일 종가가 없어 전부 막힌다 — 하한은 아래에서 따로 본다
        eng = TradingEngine(s, enable_dashboard=False)
        A, B, C = list(s.UNIVERSE[:3])     # 실제 종목(대회 규정 관문을 통과해야 한다)
        eng.universe = [A, B, C]
        eng._slide_profile = {c: [1e6] * 380 for c in eng.universe}   # 평소 직전 5분 1백만
        day = "20261015"
        eng._opening_done_day = day
        cum = {A: 1e8, B: 1e8, C: 1e8}

        class _B2:
            async def get_turnovers(self, symbols):
                return {c: {"price": 10_000.0, "acml_vol": 1e4, "turnover": cum[c]} for c in symbols}

            async def inquire_price(self, symbol):
                return {"output": {"stck_prpr": "10000", "iscd_stat_cls_code": "55", "marg_rate": "40.00"}}
        eng.broker = _B2()
        bought = []

        async def _submit2(symbol, side, qty, price, reason):
            bought.append(symbol)
            return True
        eng._submit = _submit2
        t1 = dt.datetime(2026, 10, 15, 10, 0)
        assert eng._slide_ready(t1), "장 시작 진입 뒤인데 장중 추가 매수를 보지 않는다"
        await eng._slide_poll(t1)
        cum[A] += 5e6          # 5분 5백만 = 평소 x5 → 발동
        cum[B] += 2e6          # x2 → 미발동
        cum[C] += 8e6          # x8 → 발동(배수 1위)
        await eng._slide_poll(t1 + dt.timedelta(minutes=5))
        assert [sy for _, sy in eng._slide_queue] == [C, A], f"배수 순 아님: {eng._slide_queue}"
        eng.portfolio.positions["ZZZ"] = Position(symbol="ZZZ", qty=1, avg_price=1.0, last_price=1.0)
        await eng._slide_entries(t1 + dt.timedelta(minutes=5), {})
        assert bought == [C], f"빈자리 1개를 배수 1위로 채우지 않았다: {bought}"
        board = {r["symbol"]: r["outcome"] for r in eng._board["slide"]}
        assert board == {C: "매수 주문", A: "자리 없음"}, f"관제 화면 장중 발동 결과가 다르다: {board}"
        cum[C] += 9e6
        await eng._slide_poll(t1 + dt.timedelta(minutes=10))
        assert all(sy != C for _, sy in eng._slide_queue), "같은 종목을 하루에 두 번 발동시켰다"
        eng._sold_today[A] = day
        eng._slide_queue = [(5.0, A)]
        eng.portfolio.positions.pop("ZZZ")
        bought.clear()
        await eng._slide_entries(t1 + dt.timedelta(minutes=10), {})
        assert A not in bought, "오늘 판 종목을 다시 샀다"
        # 전일 대비 하한(SLIDE_MIN_CHASE) — 전일 종가 1만원, 지금 1만원(+0%)이면 +3% 하한에 막힌다
        s.SLIDE_MIN_CHASE = 0.03
        eng._prev_daily = {A: (10_000.0, 1e5, 1e9), B: (9_500.0, 1e5, 1e9)}
        eng._sold_today = {}
        bought.clear()
        eng._slide_queue = [(5.0, A), (4.0, B)]
        await eng._slide_entries(t1 + dt.timedelta(minutes=11), {})
        assert bought == [B], f"전일 대비 +3% 하한이 안 걸렸다(A +0% 는 막고 B +5.3% 는 사야 한다): {bought}"
        s.SLIDE_MIN_CHASE = 0.0
        # 체결통보(WS)가 미체결 추적을 지운다 — 모의투자는 미체결 조회가 없어 이것뿐이다.
        # 안 지우면 판 종목이 자리를 차지해 빈자리가 0이 된다(10/16 실측)
        eng.portfolio.positions.clear()
        eng.portfolio.positions["ZZZ"] = Position(symbol="ZZZ", qty=1, avg_price=1.0, last_price=1.0)
        eng.open_orders = {"0000016101": {"symbol": "YYY", "side": "SELL", "qty": 10, "reason": "VWAP"}}
        eng.bus.publish = (lambda *_a, **_k: asyncio.sleep(0))
        notice = {"cntg_yn": "2", "seln_byov_cls": "01", "stck_shrn_iscd": "YYY", "oder_no": "0000016101"}
        await eng._on_ws_notice({**notice, "cntg_qty": "4", "cntg_unpr": "100"})
        assert eng.open_orders.get("0000016101", {}).get("qty") == 6, f"부분체결 잔량: {eng.open_orders}"
        await eng._on_ws_notice({**notice, "cntg_qty": "6", "cntg_unpr": "100"})
        assert not eng.open_orders, f"전량 체결통보 뒤에도 미체결 추적이 남았다: {eng.open_orders}"
        bought.clear()
        eng._sold_today = {}
        eng._slide_queue = [(5.0, A)]
        await eng._slide_entries(t1 + dt.timedelta(minutes=12), {})
        assert bought == [A], f"체결된 매도가 자리를 막았다: {bought}"
        # 날짜가 바뀌면 전날 미체결 추적·보유하지 않은 요건 표시를 버린다(모의투자는 미체결 조회가 없다)
        eng._save_state = lambda: None
        eng.open_orders = {"old": {"symbol": "YYY", "side": "BUY", "qty": 3, "ts": "2026-10-15T09:05:00"},
                           "new": {"symbol": "XXX", "side": "BUY", "qty": 3, "ts": "2026-10-16T00:00:10"}}
        eng._compliance_syms = {"ZZZ": "20261015", "WWW": "20261015"}
        eng._drop_stale_day_state("20261016")
        assert list(eng.open_orders) == ["new"], f"전날 미체결 추적이 남았다: {eng.open_orders}"
        assert eng._compliance_syms == {"ZZZ": "20261015"}, \
            f"보유 요건 포지션은 남기고 보유 안 한 표시만 버려야 한다: {eng._compliance_syms}"
        # 목표 도달 잠금(LOCK_RETURN) — 60초 확인 뒤 잠그고 전략 포지션만 판다(요건 포지션 제외)
        from strategy.phase import phase_config as _pc
        saved_lock = (s.LOCK_RETURN, s.COMP_START, s.COMP_END, s.PHASE1_END)
        try:
            s.LOCK_RETURN = 1.0
            s.COMP_START, s.COMP_END, s.PHASE1_END = "2026-10-12", "2026-11-20", "2026-10-16"
            eng.phase = _pc(settings=s, now=dt.date(2026, 10, 19))
            eng.open_orders = {}
            eng._compliance_syms = {"CCC": "20261019"}
            eng.portfolio.positions["CCC"] = Position(symbol="CCC", qty=1, avg_price=1.0, last_price=1.0)
            sells = []

            async def _submit_lock(symbol, side, qty, price, reason):
                sells.append((symbol, side.value, reason.split("|")[0]))
                return True
            eng._submit = _submit_lock
            base_cap = eng.portfolio.initial_capital
            eng.portfolio.initial_capital = eng.portfolio.equity / 2.5      # 누적 +150%
            t_l = dt.datetime(2026, 10, 19, 10, 0)
            await eng._lock_check(t_l, {})
            await eng._lock_check(t_l + dt.timedelta(seconds=30), {})
            assert not eng._locked and not sells, "확인 대기(60초) 전에 잠갔다"
            await eng._lock_check(t_l + dt.timedelta(seconds=61), {})
            assert eng._locked, "목표를 60초 넘게 유지했는데 잠그지 않았다"
            assert sells == [("ZZZ", "SELL", "LOCK")], f"전략 포지션만 팔아야 한다(요건 포지션 제외): {sells}"
            eng.portfolio.initial_capital = base_cap
            eng._locked, eng._lock_seen = False, None
            s.LOCK_RETURN = 0.0
            await eng._lock_check(t_l + dt.timedelta(seconds=200), {})
            assert not eng._locked, "LOCK_RETURN=0 인데 잠갔다"
            eng.portfolio.positions.pop("CCC")
            eng._compliance_syms = {}
        finally:
            s.LOCK_RETURN, s.COMP_START, s.COMP_END, s.PHASE1_END = saved_lock
        eng._opening_done_day = ""
        eng._opening_mode = lambda: True               # opening 모드에서는 장 시작 진입이 먼저다
        assert not eng._slide_ready(t1), "장 시작 진입 전에 장중 추가 매수를 봤다"
        # 가격제한폭 밖 가격은 평가·청산에 쓰지 않는다(10/20 최고 자산 5.34억 사건)
        eng._prev_daily = {"PX1": (10_000.0, 1.0, 1.0), "PX2": (10_000.0, 1.0, 1.0)}
        eng.latest_prices = {"PX1": 108_000.0, "PX2": 12_900.0, "PX3": 5_000.0}
        sp = eng._sane_prices()
        assert "PX1" not in sp and sp.get("PX2") == 12_900.0, f"제한폭 밖 가격을 걸러야 한다: {sp}"
        assert sp.get("PX3") == 5_000.0, "전일 종가를 모르면 막지 않는다"

        # 매수 지정가(+2%)는 상한가로 자른다(10/27 050110 `40270000 상/하한가 오류`).
        # 기준이 지금 가격과 안 맞으면(병합 직후 등) 자르지 않는다.
        import types
        sent: List[float] = []

        async def _submit_order(req):
            sent.append(req.price)
            return types.SimpleNamespace(success=False, msg_cd="TEST", msg="selftest")
        saved_broker = eng.broker
        eng.broker = types.SimpleNamespace(submit_order=_submit_order)
        eng._prev_daily["LIM"] = (10_000.0, 1.0, 1.0)
        await type(eng)._submit(eng, "LIM", Side.BUY, 1, 12_900.0, "selftest")
        await type(eng)._submit(eng, "LIM", Side.BUY, 1, 50_000.0, "selftest")
        assert sent == [13_000.0, 51_000.0], f"상한가 클립: {sent}"

        # 오래 걸린 매도는 취소하고 잔고를 다시 맞춘다 — 11/02 09:00 매도가 09:05까지 안 팔려
        # 장 시작 판정 자리를 막았다. 취소 실패 + 잔고에 없음 = 다 팔림 → 추적 삭제.
        t_r = dt.datetime(2026, 11, 2, 9, 1, 30)
        old, new = (t_r - dt.timedelta(seconds=90)).isoformat(), (t_r - dt.timedelta(seconds=10)).isoformat()
        eng.open_orders = {
            "S1": {"symbol": "RP1", "side": "SELL", "qty": 10, "price": 100, "org_no": "1", "ts": old},
            "S2": {"symbol": "RP2", "side": "SELL", "qty": 10, "price": 100, "org_no": "1", "ts": new},
            "S3": {"symbol": "RP3", "side": "SELL", "qty": 10, "price": 100, "org_no": "1", "ts": old},
            "S4": {"symbol": "RP4", "side": "SELL", "qty": 10, "price": 100, "org_no": "1", "ts": old},
            "B1": {"symbol": "RP5", "side": "BUY", "qty": 10, "price": 100, "org_no": "1", "ts": old},
        }
        eng.portfolio.positions["RP4"] = Position(symbol="RP4", qty=10, avg_price=100.0)
        cancels: List[str] = []

        async def _cancel(org_no, order_no, qty=0, all_qty=True):
            cancels.append(order_no)
            return {"rt_cd": "0" if order_no == "S1" else "1", "msg1": "selftest"}
        synced: List[int] = []

        async def _sync(initial=False):
            synced.append(1)
        eng.broker = types.SimpleNamespace(cancel_order=_cancel)
        saved_sync, eng._sync_portfolio = eng._sync_portfolio, _sync
        try:
            await eng._reprice_stale_sells(t_r)
        finally:
            eng.broker, eng._sync_portfolio = saved_broker, saved_sync
        assert cancels == ["S1", "S3", "S4"], f"60초 넘은 매도만 취소: {cancels}"
        assert synced, "취소 뒤 잔고를 다시 맞춰야 한다"
        assert set(eng.open_orders) == {"S2", "S4", "B1"}, f"추적 정리: {sorted(eng.open_orders)}"
        assert eng.open_orders["S4"]["ts"] == t_r.isoformat(), "취소 실패·보유 중이면 한 주기 뒤에 다시 본다"
        eng.open_orders = {}
        eng.portfolio.positions.pop("RP4", None)

        # 대회 마지막 날 — FINAL_FLATTEN_HHMM부터 요건 포지션까지 전부 팔고 그날·종료 뒤에는 사지 않는다
        saved_final = (s.COMP_END, s.FINAL_FLATTEN, s.FINAL_FLATTEN_HHMM)
        saved_pos = eng.portfolio.positions
        try:
            s.COMP_END, s.FINAL_FLATTEN, s.FINAL_FLATTEN_HHMM = "2026-11-20", True, "1510"
            d = dt.datetime
            assert not eng._entries_closed(d(2026, 11, 19, 9, 5)), "마지막 날 전에 매수를 막았다"
            assert eng._entries_closed(d(2026, 11, 20, 9, 5)), "마지막 날(전량 매도 켬)에 샀다"
            assert eng._entries_closed(d(2026, 11, 23, 9, 5)), "대회 종료 뒤에 샀다"
            eng.portfolio.positions = {}      # 앞 검증이 남긴 포지션과 섞지 않는다
            eng.portfolio.positions["FN1"] = Position(symbol="FN1", qty=7, avg_price=100.0)
            eng.portfolio.positions["FN2"] = Position(symbol="FN2", qty=3, avg_price=100.0)
            eng._compliance_syms = {"FN2": "20261120"}
            final_sells: List[Tuple[str, int, str]] = []

            async def _submit_final(symbol, side, qty, price, reason):
                final_sells.append((symbol, qty, reason.split("|")[0]))
                return True
            saved_submit, eng._submit = eng._submit, _submit_final
            await eng._final_flatten(d(2026, 11, 19, 15, 10), {})
            await eng._final_flatten(d(2026, 11, 20, 15, 9), {})
            assert not final_sells, f"마지막 날 매도 시각 전에 팔았다: {final_sells}"
            await eng._final_flatten(d(2026, 11, 20, 15, 10), {})
            assert sorted(final_sells) == [("FN1", 7, "FINAL"), ("FN2", 3, "FINAL")], \
                f"요건 포지션까지 전부 팔아야 한다: {final_sells}"
            s.FINAL_FLATTEN = False
            final_sells.clear()
            await eng._final_flatten(d(2026, 11, 20, 15, 10), {})
            assert not final_sells and not eng._entries_closed(d(2026, 11, 20, 9, 5)), "FINAL_FLATTEN=false 인데 동작했다"
            assert eng._entries_closed(d(2026, 11, 23, 9, 5)), "종료 뒤 매수 금지는 설정과 무관해야 한다"
            eng._submit = saved_submit
        finally:
            s.COMP_END, s.FINAL_FLATTEN, s.FINAL_FLATTEN_HHMM = saved_final
            eng.portfolio.positions = saved_pos
            eng._compliance_syms = {}

        # WS 체결가 — 한 건 필드 수를 머리의 건수로 나눈다(모의서버가 46개가 아니라 두 번째 건부터
        # 어긋나 가짜 종목코드·가격이 들어왔다, 10/20~29). 구독하지 않은 종목은 버린다.
        from core.websocket_client import WS_TR_TRADE, KISWebSocketClient, Subscription

        def _rec(sym, px, n=47):
            f = [""] * n
            f[0], f[2], f[5], f[13] = sym, str(px), "1.0", "1000"
            for i in (7, 8, 9, 10, 11):
                f[i] = str(px)
            return f
        body = "^".join(_rec("005930", 70_000) + _rec("000660", 200_000))
        qs = KISWebSocketClient._parse_trade(body, 2)
        assert [(q.symbol, q.price) for q in qs] == [("005930", 70_000.0), ("000660", 200_000.0)], \
            f"여러 건 메시지 파싱: {[(q.symbol, q.price) for q in qs]}"
        ws = KISWebSocketClient(s)
        ws._subs[(WS_TR_TRADE, "005930")] = Subscription(WS_TR_TRADE, "005930", 1)
        got: List[str] = []

        async def _on_trade(q):
            got.append(q.symbol)
        ws.on_trade = _on_trade
        await ws._handle_realtime(f"0|{WS_TR_TRADE}|002|{body}")
        assert got == ["005930"], f"구독한 종목만 받아야 한다: {got}"

        # 연결된 채 메시지가 끊기면(10/29 09:50~) 다시 연결한다
        class _Stalled:
            async def recv(self):
                await asyncio.sleep(3600)

            async def send(self, m):
                pass

            async def close(self):
                pass
        opens: List[int] = []

        async def _open():
            opens.append(1)
            return _Stalled()
        saved_silence = s.WS_SILENCE_RECONNECT_SEC
        s.WS_SILENCE_RECONNECT_SEC = 0.05
        ws2 = KISWebSocketClient(s)
        kst = dt.timezone(dt.timedelta(hours=9))
        assert ws2._silence_limit(dt.datetime(2026, 11, 4, 10, 0, tzinfo=kst)) == 0.05, "장중 무음 기준"
        assert ws2._silence_limit(dt.datetime(2026, 11, 4, 2, 0, tzinfo=kst)) == 1800.0, "장 밖은 30분"
        assert ws2._silence_limit(dt.datetime(2026, 11, 7, 10, 0, tzinfo=kst)) == 1800.0, "주말은 30분"
        ws2._silence_limit = lambda now=None: 0.05     # 실행 시각과 무관하게 재연결을 본다
        ws2._open = _open
        ws2.auth._approval_key = "SELFTEST"
        try:
            ws2.start()
            await asyncio.sleep(1.4)
        finally:
            await ws2.stop()
            s.WS_SILENCE_RECONNECT_SEC = saved_silence
        assert len(opens) >= 2, f"멈춘 연결을 다시 열지 않았다: {len(opens)}회"
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
    return ("주문 읽기 타임아웃·5xx 재시도 안 함(결과 불명) · 접속 실패 재시도 · 조회 재시도 · "
            "무손실 체결 채널 · 히스토그램 분위수·Prometheus · Date 헤더 시계 오차 · "
            "장중 VWAP 청산(판정·배관) · 장중 추가 매수(첫 발동·배수 순·빈자리·당일 매도 제외·전일 대비 하한) · 체결통보로 미체결 추적 정리 · 날짜 변경 시 전날 미체결·요건 표시 정리 · 목표 도달 잠금(확인 대기·요건 제외) · 제한폭 밖 가격 버림 · 매수 지정가 상한가 클립 · "
            "미체결 매도 취소·재대조 · 대회 마지막 날 전량 매도·종료 뒤 매수 금지 · WS 여러 건 체결가 파싱·미구독 종목 버림·멈춘 연결 재연결 모두 작동")


@check("27. 직접 매매 스위치 · 지수 종목 순위판")
async def test_rank_board():
    """직접 매매 기간(2026-11-18~) — 엔진 매수 차단 스위치와 관제 화면 순위판.

    보는 것: (1) 순위판은 지수 종목만, 거래대금 큰 순, 전일 종가를 모르면 등락률 없음(0%로 쓰지 않는다),
    소형(시총 1조 이하)·보유 표시 (2) 관심 종목에 거래대금·등락률 순위가 붙고 지수 밖 관심 종목도 보인다
    (3) `state/NO_BUY` 파일·`ENGINE_NO_BUY`가 신규 매수를 막고, 지우면 풀린다.
    """
    import datetime as dt

    from main import TradingEngine
    from monitor.ranking import build_rank_rows, pick_rows

    snap = {
        "A": {"price": 11000.0, "turnover": 5e10, "prev_close": 10000.0, "name": "에이"},
        "B": {"price": 9000.0, "turnover": 9e10, "prev_close": 0.0, "name": "비"},     # 전일 종가는 엔진 값
        "C": {"price": 5000.0, "turnover": 1e11, "prev_close": 0.0, "name": "씨"},     # 전일 종가 모름
        "X": {"price": 3000.0, "turnover": 9e12, "prev_close": 2000.0, "name": "지수밖"},
        "Z": {"price": 0.0, "turnover": 1e9, "prev_close": 100.0},                     # 시세 없음
    }
    index_of = {"A": "KOSDAQ150", "B": "KOSPI200", "C": "KOSPI200", "Z": "KOSPI200"}
    rows = build_rank_rows(snap, index_of, market_cap={"A": 5e11, "B": 3e12},
                           prev_close={"B": 10000.0}, held={"B"})
    assert [r["symbol"] for r in rows] == ["C", "B", "A"], f"지수 밖·시세 없는 종목 제외 · 거래대금 순 실패: {rows}"
    by = {r["symbol"]: r for r in rows}
    assert by["A"]["chg_pct"] == 10.0 and by["B"]["chg_pct"] == -10.0, "등락률(복수시세 전일 종가 → 엔진 값 순)"
    assert by["C"]["chg_pct"] is None, "전일 종가를 모르면 등락률은 None이어야 한다(0%면 순위가 거짓)"
    assert by["A"]["small"] and not by["B"]["small"] and not by["C"]["small"], "소형주(시총 1조 이하) 표시"
    assert by["B"]["held"] and not by["A"]["held"], "보유 표시"
    pk = pick_rows(rows, ["A", "Q"])
    assert pk[0]["tv_rank"] == 3 and pk[0]["chg_rank"] == 1, f"관심 종목 순위: {pk[0]}"
    assert pk[1]["symbol"] == "Q" and pk[1]["tv_rank"] is None, "시세 없는 관심 종목은 빈 행"

    s = make_settings()
    saved = {k: getattr(s, k) for k in ("ENGINE_NO_BUY", "COMP_END", "FINAL_FLATTEN", "DASH_PICKS_FILE")}
    try:
        s.ENGINE_NO_BUY, s.COMP_END, s.FINAL_FLATTEN = False, "2026-11-20", True
        Path(s.STATE_DIR).mkdir(parents=True, exist_ok=True)
        flag = Path(s.STATE_DIR) / "NO_BUY"
        flag.unlink(missing_ok=True)
        eng = TradingEngine(s, enable_dashboard=False)
        now = dt.datetime(2026, 11, 19, 9, 5)
        assert not eng._entries_closed(now), "스위치가 꺼져 있는데 매수가 막혔다"
        flag.write_text("", encoding="utf-8")
        assert eng._entries_closed(now), "state/NO_BUY 파일이 매수를 막지 못했다"
        flag.unlink()
        assert not eng._entries_closed(now), "파일을 지웠는데 매수가 계속 막혔다(재시작 없이 풀려야 한다)"
        s.ENGINE_NO_BUY = True
        assert eng._entries_closed(now), "ENGINE_NO_BUY 가 매수를 막지 못했다"
        s.ENGINE_NO_BUY = False
        assert eng._entries_closed(dt.datetime(2026, 11, 20, 9, 5)), "마지막 날 매수 금지가 깨졌다"

        # --- 엔진 배선: 복수시세 → 순위판(지수 종목만 + 지수 밖 관심 종목) ---
        members = s.index_members()
        mem = sorted(x for lst in members.values() for x in lst)[:3]
        assert len(mem) == 3, "지수 멤버십 정보가 없다(UNIVERSE_FILE 확인)"
        eng._dash_picks = [mem[0], "999999"]
        eng._prev_daily = {mem[1]: (1000.0, 1.0, 1.0)}

        class _Broker:
            async def get_turnovers(self, syms):
                assert "999999" in syms, "관심 종목이 조회 대상에 없다"
                return {mem[0]: {"price": 1100.0, "turnover": 3e9, "prev_close": 1000.0, "name": "가"},
                        mem[1]: {"price": 900.0, "turnover": 5e9, "prev_close": 0.0, "name": "나"},
                        "999999": {"price": 500.0, "turnover": 1e8, "prev_close": 400.0, "name": "밖"}}
        eng.broker = _Broker()
        await eng._rank_poll(now)
        rk = eng._rank
        assert [r["symbol"] for r in rk["rows"]] == [mem[1], mem[0]], f"순위판 행: {rk['rows']}"
        assert rk["rows"][0]["chg_pct"] == -10.0, "엔진 전일 종가로 등락률을 구하지 못했다"
        assert [p["symbol"] for p in rk["picks"]] == [mem[0], "999999"], "관심 종목 순서"
        assert rk["picks"][0]["tv_rank"] == 2 and rk["picks"][1]["chg_pct"] == 25.0, f"관심 종목 값: {rk['picks']}"
        assert eng.state_snapshot()["rank"] is rk, "상태 스냅샷에 순위판이 없다"
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
        (Path(s.STATE_DIR) / "NO_BUY").unlink(missing_ok=True)
    return ("순위판(지수 종목만·거래대금 순·전일 종가 모르면 등락률 없음·소형·보유 표시) · 관심 종목 순위·지수 밖 표시 · "
            "state/NO_BUY·ENGINE_NO_BUY 매수 차단과 해제 · 마지막 날 매수 금지 유지 모두 작동")


# ============================================================ 실행

async def run_all() -> int:
    setup_logging(str(_SELFTEST_TMP / "logs"), "WARNING")   # 실거래 로그와 섞지 않는다
    tests = [
        test_settings, test_market_rules, test_indicators,
        test_rate_limiter, test_priority, test_mock_broker,
        test_risk_manager, test_features, test_backtest, test_event_bus,
        test_live_client_offline, test_validation, test_weights_and_bets,
        test_competition_rules, test_selection_and_phase, test_panel_min_bars,
        test_intraday, test_intraday_book, test_decay_exits,
        test_orderbook_recorder, test_burst_wiring, test_opening_entry,
        test_sliding, test_sliding_wiring, test_compliance_trades,
        test_server_reliability, test_rank_board,
    ]
    print("\n" + "=" * 86)
    print(" KIS AI Quant Trader — 셀프 테스트")
    print("=" * 86)
    for test in tests:
        await test()
        name, ok, msg, elapsed = RESULTS[-1]
        mark = "PASS" if ok else "FAIL"
        print(f" [{mark}] {name:<34} {elapsed:6.2f}s  {msg}")

    passed = sum(1 for _, ok, _, _ in RESULTS if ok)
    total = len(RESULTS)
    print("=" * 86)
    print(f" 결과: {passed}/{total} 통과" + ("  — 실전 연동 준비 완료" if passed == total else ""))
    print("=" * 86 + "\n")
    if passed != total:
        for name, ok, msg, _ in RESULTS:
            if not ok:
                print(f"  ✗ {name}: {msg}")
        print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(run_all()))
