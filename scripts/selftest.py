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
    far = round_to_tick(price * 0.7)
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
    return (f"매수/매도 판정 · 한도 {MAX_SINGLE_WEIGHT:.0%} 클립 · "
            f"지수 멤버십 {len(members)}종목 · 요건 4항목 집계")


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

    # 초기화는 마지막 날에만 권고하고, 이익이면 권고하지 않는다
    loss = Portfolio(cash=90_000_000.0, initial_capital=100_000_000.0)
    gain = Portfolio(cash=108_000_000.0, initial_capital=100_000_000.0)
    assert not reset_recommendation(loss, dt.date(2026, 10, 13), s).should_reset
    assert reset_recommendation(loss, dt.date(2026, 10, 16), s).should_reset
    assert not reset_recommendation(gain, dt.date(2026, 10, 16), s).should_reset
    assert not reset_recommendation(loss, dt.date(2026, 10, 19), s).should_reset
    return "레짐 단일 적용 · 잠금/교체 · 국면 6일자 · 초기화 권고 모두 작동"


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
    return ("거래량 감소 · 상승 감쇠(손절선 전달) · 급락 · EXIT_RULES 순서 반영 · "
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


# ============================================================ 실행

async def run_all() -> int:
    setup_logging("logs", "WARNING")
    tests = [
        test_settings, test_market_rules, test_indicators,
        test_rate_limiter, test_priority, test_mock_broker,
        test_risk_manager, test_features, test_backtest, test_event_bus,
        test_live_client_offline, test_validation, test_weights_and_bets,
        test_competition_rules, test_selection_and_phase, test_panel_min_bars,
        test_intraday, test_intraday_book, test_decay_exits,
        test_orderbook_recorder,
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
