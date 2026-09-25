"""전 모듈 스모크 테스트 — 계좌 발급 전 로직 검증용.

    python scripts/selftest.py

검증 항목
  1. 설정/TR_ID 환경 분기          6. Mock 체결 엔진(부분체결·거부·취소)
  2. 호가단위·가격제한폭 규칙      7. 리스크 매니저(사이징·손절·킬스위치)
  3. 기술적 지표 정확성/인과성     8. 피처·라벨 파이프라인
  4. 유량 제어기(실측 RPS)         9. 백테스트 엔진 + 성과지표
  5. 우선순위 큐(주문 우선)       10. 이벤트 버스
                                  11. 실서버 클라이언트(오프라인)

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

    # 변동성이 크면 수량이 줄어야 한다
    calm = Signal(symbol="A", score=0.6, price=70_000, atr=700, volatility=0.15)
    wild = Signal(symbol="B", score=0.6, price=70_000, atr=4_000, volatility=0.75)
    q_calm = risk.evaluate_entry(calm, pf).qty
    q_wild = risk.evaluate_entry(wild, pf).qty
    assert q_calm > q_wild, f"변동성 타게팅 미작동 ({q_calm} vs {q_wild})"

    # 최대 보유 종목 수 제한
    for i, code in enumerate(["100000", "100001", "100002"]):
        pos = Position(symbol=code, qty=100, avg_price=10_000, last_price=10_000,
                       sector=f"S{i}")
        pf.positions[code] = pos
    pf.cash = capital - 3_000_000
    blocked = risk.evaluate_entry(Signal(symbol="200000", score=0.9, price=10_000,
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

    # 익절 트리거
    exits = risk3.evaluate_exits(pf3, {"005930": 78_000}, {"005930": 1_400})
    assert exits and exits[0].reason in (ExitReason.TAKE_PROFIT, ExitReason.TRAILING_STOP), exits

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
    return f"사이징({q_calm}주 vs {q_wild}주) · 손절 · 익절 · 킬스위치 · 디리스킹 모두 작동"


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

    tail = df[FEATURE_COLUMNS].iloc[-50:]
    nan_ratio = float(tail.isna().to_numpy().mean())
    assert nan_ratio < 0.05, f"최근 구간 결측률 과다 {nan_ratio:.1%}"

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
    return f"피처 {len(FEATURE_COLUMNS)}개, 학습샘플 {len(X):,}, 양성비율 {pos_rate:.1%}"


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


# ============================================================ 실행
async def run_all() -> int:
    setup_logging("logs", "WARNING")
    tests = [
        test_settings, test_market_rules, test_indicators,
        test_rate_limiter, test_priority, test_mock_broker,
        test_risk_manager, test_features, test_backtest, test_event_bus,
        test_live_client_offline,
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
