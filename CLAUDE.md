# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python scripts/selftest.py                    # 전체 검증 (18개 항목) — 변경 후 필수로 실행
python scripts/train_model.py --days 1500     # AI 모델 학습 → models/alpha_gbdt.pkl
python scripts/train_model.py --source data/cache --no-flow   # 수급 피처 제외(측정용)
python scripts/run_backtest.py --days 1000    # 백테스트 → reports/
python scripts/run_backtest.py --no-model     # AI 기여도 비교(기술적 팩터만)
python main.py --duration 120                 # 짧게 실시간 루프 구동(스모크)
python main.py --health-only                  # 연결/인증/계좌 점검만
python scripts/validate_strategy.py           # CPCV + PSR/DSR 과적합 검증 (약 7분)
python scripts/walkforward.py                 # 워크포워드 재학습 시뮬레이션 (약 8분)
python scripts/collect_data.py --probe        # 토스 API 연결·필드명 확인
python -m compileall -q config core api strategy backtest utils monitor scripts main.py
```

단일 항목만 검증하려면 `scripts/selftest.py`의 `run_all()` 안 `tests` 리스트를 줄이거나,
해당 테스트 함수(`test_risk_manager` 등)를 직접 `asyncio.run()`으로 호출한다. pytest는 쓰지 않는다.

실행에는 신규 패키지 설치가 필요 없다. `lightgbm`/`loguru`는 **선택**이며, 설치되어 있으면
각각 GBDT 백엔드와 로거가 자동으로 그쪽으로 승격된다(`HAS_LGB`, `HAS_LOGURU` 분기).

## Architecture

한국투자증권 Open API 기반 자동매매 시스템. 대회 계좌 발급 전후를 **같은 코드**로 커버하는 것이
전체 설계의 축이다.

### 1. Mock ↔ 실서버 전환은 `.env` 한 줄

`api/factory.py::create_broker()`가 `USE_MOCK_MODE`를 보고 `MockKISClient` 또는 `KISClient`를
반환한다. 둘 다 `api/base.py::AbstractBroker`를 구현하고 **동일한 KIS JSON 스키마**
(`rt_cd`/`msg_cd`/`msg1`/`output*`)를 반환하므로 상위 레이어는 어느 쪽인지 모른다.

→ 브로커에 메서드를 추가할 때는 **반드시 Mock과 실클라이언트 양쪽에 동시에** 구현하고,
   Mock은 실제 KIS 응답 필드명을 그대로 흉내내야 한다(`base.py`의 파서가 양쪽에 공용).

`config/settings.py::Settings.tr(name)`이 TR_ID를 환경별로 해석한다
(`order_buy` → 모의 `VTTC0802U` / 실전 `TTTC0802U`). 호출부에 TR_ID 문자열을 하드코딩하지 말 것.

### 2. 리스크 매니저는 조언자가 아니라 관문

`strategy/risk_manager.py::RiskManager`가 **모든 주문의 수량을 최종 결정**한다. 전략은 점수만
내고, 진입은 `evaluate_entry()`가 `RiskDecision(approved, qty, ...)`을 승인해야만 나간다.
새 매수 경로를 추가한다면 이 관문을 우회해선 안 된다.

- 사이징: `변동성타게팅(TARGET_VOLATILITY/자산변동성) × 켈리배수 × 신호강도 × 레짐`,
  그 뒤 종목/섹터/총익스포저 상한과 가용현금으로 클립.
  **켈리는 상한이 아니라 배수다** — 상한으로 쓰면 표본이 적을 때 상수가 항상 이겨
  변동성 타게팅이 무력화된다(과거에 실제로 발생한 버그).
  **단 익스포저를 묶는 것은 사이징이 아니다.** 사이징 배수를 7배 올려도 익스포저는
  1.2배만 움직이고(38.4%→46.6%), 종목당 상한을 2.5배 올리면 익스포저는 그대로인 채
  집중도만 오른다(측정 완료). 실제 제약은 **`SIGNAL_BUY_THRESHOLD` 절대 임계값을
  넘는 종목 수**이며, 8슬롯 중 1~2개만 찬다. 익스포저를 올리려면 `USE_TOP_K`로
  슬롯을 채워야 한다 — 사이징 파라미터를 건드리는 것은 효과가 없다.
- 상태 기계: `NORMAL → DERISK`(소프트 MDD: 신규 진입 차단 + 익스포저 축소)
  `→ HALT`(하드 MDD 또는 일일 손실 한도: 전량 청산 + 쿨다운). `update()`를 매 루프 1회 호출해야
  날짜 롤오버(`Portfolio.start_new_day()` → `bars_held` 증가)가 동작한다.
- 청산 우선순위: 킬스위치 > 급락 > 손절 > 트레일링 > 익절 > 시간손절 > 신호반전 > 디리스킹.

### 3. 유량 제어는 우회 불가 경로

`core/rate_limiter.py::PriorityRateLimiter`를 **Mock까지 포함해 모든 REST 호출이** 통과한다
(`MockKISClient._throttle()`). 그래야 실서버 전환 시 타이밍이 달라지지 않는다.
우선순위는 주문 P0 > 계좌 P1 > 시세 P2 — 시세 폴링이 몰려도 손절 주문이 밀리지 않게 한다.
`EGW00201`/HTTP 429를 받으면 `notify_rate_limit_error()`가 레이트를 일시 감축한다.

KIS 유량: 실전 20건/초, **모의투자 2건/초**. 기본값은 모의투자 기준(2)이다.

### 4. 백테스트와 실매매는 같은 객체를 공유

`backtest/engine.py::BacktestEngine`이 실매매와 **동일한** `EnsembleAlphaStrategy`/`RiskManager`
인스턴스를 주입받는다. 전략·리스크 로직을 고치면 양쪽에 동시에 반영되므로, 백테스트-실거래
불일치가 구조적으로 생기지 않는다. 대신 다음 두 경로는 갈린다.

- 신호 생성: 백테스트는 `precompute()` + `signals_at(i)` (피처 전구간 1회 계산 → O(n)),
  실시간은 `generate_signals()` (매 루프 마지막 봉). 새 피처/점수를 추가하면
  `_make_signal()` 한 곳만 고치면 양쪽에 반영된다.
- 레짐: 백테스트는 `regime_series()`(봉별 배열), 실시간은 `detect_regime()`(현재 시점).
  판정 규칙은 `_classify_regime()` 하나를 공유한다.

### 5. 룩어헤드 차단이 전제

- `utils/indicators.py`의 모든 지표는 인과적이다. 전구간에서 한 번 계산한 값과 시점마다
  잘라서 계산한 값이 같아야 하며, `selftest.py` 3번 항목이 이를 검증한다.
  **이 성질이 깨지면 백테스트의 사전계산 최적화가 곧바로 룩어헤드 버그가 된다.**
- 백테스트는 봉 t 종가로 판단하고 봉 t+1 시가에 체결한다. 갭하락은 손절가가 아니라 시가 체결.
- `backtest/data.py::MarketSimulator`는 커서 이후 데이터를 노출하지 않는다(Mock 브로커의 시간축).

### 6. 실시간 루프의 상태 관리 (`main.py::TradingEngine`)

- `_now()`: Mock은 가상 시계(**봉 1개 = 1거래일**, 시뮬레이터 커서 기반), 실서버는 KST 실시각.
- 손절선은 **주문 접수가 아니라 체결 시점**에 평단가 기준으로 설정한다(`_on_fill_event`).
  접수 시점에는 포지션이 없어 설정이 누락된다.
- `open_orders`는 부분체결 시 잔량만 차감하고, 잔량이 0이 될 때만 제거한다. 먼저 제거하면
  같은 종목에 중복 진입 주문이 나간다.
- 주기 작업: `BAR_REFRESH_LOOPS`(일봉 재적재) / `RECONCILE_LOOPS`(잔고·미체결 대조) /
  `FILL_POLL_LOOPS`(체결 폴링). Mock은 `_refresh_bars_from_sim()`으로 REST를 건너뛴다.
- WebSocket이 건강하면(`ws.is_healthy()`) REST 시세 폴링을 생략하고, 끊기면 자동으로 폴백한다.

### 7. 성과 수치는 반드시 교차 검증한다

단일 백테스트 Sharpe는 실제의 두 배까지 부풀려진다(측정: 1.71 vs CPCV 0.78 / 워크포워드 0.91).
파라미터나 전략을 바꿨으면 `scripts/validate_strategy.py`와 `scripts/walkforward.py`를 돌린다.

- **다른 유니버스의 AUC를 비교하지 않는다.** 종목 집합·표본 수·삼중배리어 양성비율이
  달라져 라벨 모집단 자체가 다르다. 유효한 비교는 같은 유니버스에서 피처 축 하나만
  움직이는 것이고, 유니버스 효과는 같은 실행 내 수익률로만 판정한다.
  `.env`가 `UNIVERSE_FILE`을 주입하므로 기본 40종목으로 돌리려면 `UNIVERSE_FILE=`을
  앞에 붙여야 한다(`--symbols 40`은 284 중 앞 40개를 집어 교집합이 6개뿐이다).
- DSR의 `trial_sharpes`에는 **서로 다른 설정**의 Sharpe만 넣는다. CPCV 분할별 Sharpe는
  같은 설정의 기간별 변동이므로 넣으면 안 된다(기준선만 부풀려 DSR이 무의미해진다).
- 방법론을 바꿀 때는 **한 번에 하나씩** 넣고 백테스트로 확인한다. 여러 개를 동시에 넣으면
  무엇이 기여했는지 알 수 없다(실제로 3개를 한 번에 넣어 Sharpe가 0.105로 붕괴한 적이 있다).
- 기본 OFF인 것들에는 이유가 있다: `USE_META_MODEL=false`(정밀도는 오르나 신호의 53%를 막아
  Sharpe 하락), `AlphaModel.fit(calibrate=False)`(확률 압축이 베팅 사이징과 충돌).

### 8. 베팅 사이징의 중립점은 기저확률이다

`bet_size(p, threshold)`의 중립점은 0.5가 아니라 라벨의 양성 비율(`AlphaModel.base_rate`)이다.
0.5로 두면 정상 신호가 전부 음수로 읽혀 익스포저가 4%까지 붕괴한다.
`bet == 0.0`은 "진입 금지"라는 뜻이며, `_MIN_BET` 바닥값으로 덮지 않는다.

## 문서

- `docs/ARCHITECTURE.md` — 코드 구조·비동기 설계·데이터 계약. 새 모듈을 넣기 전에 읽는다.
- `docs/매매전략.md` — 피처·라벨링·앙상블·사이징·청산 규칙, 그리고 12장의 성과 해석.
  전략 파라미터를 바꾸기 전에 읽는다.
- 전략 로직을 고쳤으면 `docs/매매전략.md`를, 구조를 고쳤으면 `docs/ARCHITECTURE.md`를 같이 고친다.

## Conventions

- 코드 주석·로그·문서는 한국어. 식별자는 영어.
- 금액은 원(KRW) 단위 float, 수량은 int. 주문 가격은 반드시
  `utils/market_rules.py::round_to_tick()`으로 호가단위에 정렬한다(미정렬 시 KIS가 거부).
- OHLCV DataFrame 스키마는 `ts, open, high, low, close, volume` (ts는 문자열 `YYYYMMDD`,
  **과거→최신** 정렬). KIS 응답은 최신→과거이므로 `base.py::_chart_rows_to_frame()`이 뒤집는다.
- 새 설정값은 `config/settings.py`에 필드로 추가하고 `.env.example`에도 같이 적는다.
- 매매 판단·체결은 `utils/logger.py::JsonLedger`로 `logs/ledger.jsonl`에 남긴다(대회 근거자료).
- Windows 콘솔 인코딩 문제 때문에 stdout에 출력하는 스크립트는 `enable_utf8_console()`을 먼저
  호출한다.
