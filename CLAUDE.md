# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python scripts/selftest.py                    # 전체 검증 (22개 항목) — 변경 후 필수로 실행
python scripts/train_model.py --days 1500     # AI 모델 학습 → models/alpha_gbdt.pkl
python scripts/train_model.py --source data/cache --no-flow   # 수급 피처 제외(측정용)
python scripts/run_backtest.py --days 1000    # 백테스트 → reports/
python scripts/run_backtest.py --no-model     # AI 기여도 비교(기술적 팩터만)
python main.py --duration 120                 # 짧게 실시간 루프 구동(스모크)
python main.py --health-only                  # 연결/인증/계좌 점검만
python scripts/validate_strategy.py           # CPCV + PSR/DSR 과적합 검증 (약 7분)
python scripts/walkforward.py                 # 워크포워드 재학습 시뮬레이션 (약 8분)
python scripts/collect_data.py --probe        # 토스 API 연결·필드명 확인
python scripts/collect_minute.py              # 급등일 1분봉 수집(토스, 중단 후 이어받기)
python scripts/replay_intraday.py --days 250  # 분봉 리플레이(장중 전략 검증·측정)
python scripts/minute_exit_grid.py            # 분봉 청산 규칙 격자
python scripts/minute_skill.py                # 분봉 모델 스킬(AUC·상위사분위 적중률)
python scripts/surge_study.py                 # 급등 포착 조건별 선행 수익률 분포
python scripts/tail_optimizer.py              # P(+100%) 최대 구조 탐색(조건 1종)
python scripts/tail_grid.py                   # 꼬리 구조 전수 탐색(조건 6종 x 종목수 x 회전)
python scripts/ablate_exits.py --set catalyst # catalyst 진입 구조를 실엔진으로 확인
python scripts/intraday_entry_study.py        # 장중 진입 vs T+1 진입(다일 보유)
#   --burst-mult / --hold / --since 로 파인튜닝. --since 는 커버리지 편향 방지용
python scripts/collect_minute.py --select turnover  # 거래대금 폭증일 분봉 수집
python scripts/preflight.py                   # 대회 직전 점검 (--live 로 실서버까지 요구)
python scripts/slot_sim.py                    # 달력 시간순 2슬롯 평가(라운드 근사 대신 — 보유일수가 다른 규칙 비교용)
python scripts/collect_opening.py             # 모든 거래일 09:00~09:30 분봉(토스 · 수집 전 현재 IP 확인)
python scripts/opening_study.py               # 장 시작 구간 진입 격자 vs 현재 규칙(수집 완료 후)
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
  `MOCK_MINUTES_PER_LOOP > 0`이면 Mock이 분 단위로 전진한다(시각 기반 로직 검증용, §9 참고).
- 손절선은 **주문 접수가 아니라 체결 시점**에 평단가 기준으로 설정한다(`_on_fill_event`).
  접수 시점에는 포지션이 없어 설정이 누락된다.
- `open_orders`는 부분체결 시 잔량만 차감하고, 잔량이 0이 될 때만 제거한다. 먼저 제거하면
  같은 종목에 중복 진입 주문이 나간다.
- 주기 작업: `BAR_REFRESH_LOOPS`(일봉 재적재) / `RECONCILE_LOOPS`(잔고·미체결 대조) /
  `FILL_POLL_LOOPS`(체결 폴링). Mock은 `_refresh_bars_from_sim()`으로 REST를 건너뛴다.
- WebSocket이 건강하면(`ws.is_healthy()`) REST 시세 폴링을 생략하고, 끊기면 자동으로 폴백한다.
  **단 장중 모드는 예외다** — 폭증 감시 대상이 유니버스 전체(284)인데 WS는 40종목이라,
  관심목록 밖 종목을 REST로 돌아가며 본다(`_refresh_quotes`). `UNIVERSE_SIZE`가
  유니버스보다 작으면 `_build_universe`가 거래대금 상위만 남겨 감시 범위가 줄어든다
  (`.env`가 20이라 대형주만 보던 것을 2026-10-09에 400으로 고쳤다. preflight가 막는다).

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

### 9. 장중 분봉 전략이 현재 주 전략이다 (`USE_INTRADAY=true`)

목적함수가 **절대수익 top10**이므로 최적화 대상은 `E[R]`이 아니라 `P(R ≥ 100%)`다.
그래서 일봉 스윙에서 **장중 돌파 → 당일 청산**으로 전환했다. 두 경로는 배타적이다 —
`USE_INTRADAY=true`면 `_tick()`이 일봉 진입 블록을 건너뛴다. 같이 돌면 익스포저가
의도의 두 배가 되고 측정한 분포와 달라진다.

```
① 아침 스캔 (09:05)  전 후보 상승률을 훑어 '오늘 움직이는' 40종목 구독
                     — universe.json은 생성 시점의 테마다
② 폭증 감시 (burst)  당일 누적 거래대금 ≥ 10 × 전일까지 20봉 중앙값
                     (breakout 모드는 가격 +3% AND 분당 거래량 12배 — 지금은 쓰지 않는다)
③ 명부 → 선정        발동을 명부에 올리고 신선도 상위부터 빈 슬롯을 채운다
④ 사이징             고정 2종목 × 45% (`SIZING_MODE=fixed`)
⑤ 보유·청산          `INTRADAY_FLATTEN=false` → 다일 보유. 청산은 일봉 경로
                     (`vol_alive` + 손절 25% + 30봉 상한, §9-1)
```

**opening 모드(구현만, 측정 전 — 2026-10-09):** `INTRADAY_ENTRY_MODE=opening`이면 ②가
"장 시작 N분 거래대금 ≥ k x 평소(20봉 중앙) N/390 몫"으로 바뀐다. 09:0N에 KIS 당일분봉
(`get_minute_bars`)으로 한 번 판정하고 09:(N+`OPENING_LAG_MIN`)에 한 번 진입한다
(`main.py::_opening_scan`·`_opening_entries`). 판정은 `strategy/opening.py` 순수 함수이고
**측정(`scripts/opening_study.py`)이 같은 함수를 import한다** — 셀프테스트 22번이 함수 객체
동일성까지 본다. 측정 전이므로 preflight가 치명으로 막는다. KIS 분봉의 시각 표기(시작/끝)는
실서버 첫날 `[장중·개장] 분봉 시각 확인` 로그로 확인해야 한다.

**규칙 진입의 점수는 `RULE_TRIGGER_SCORE`(1.0)다.** 리스크 관문의 `SIGNAL_BUY_THRESHOLD`
(0.22)는 일봉 모델용이다. burst 경로가 "가격 상승률 x 10"을 점수로 넣어 **+2.2% 미만에서 터진
폭증이 전부 거부되고 있었다**(측정에 없는 가격 필터, 2026-10-09 수정).

(2026-10-08 전에는 ⑤가 "15:15 마감 정리, 오버나이트 없음"이었다. 당일 청산은 모든
레버에서 `P(+100%)` 0.00%로 측정돼 내렸다. 아래 세 항목 중 마감 정리 관련 내용은
`INTRADAY_FLATTEN=true`일 때만 해당한다.)

**순수 로직은 `strategy/intraday.py`에 있고 주문을 내지 않는다.** "지금 사라/팔라"만
돌려주고 실제 주문은 `TradingEngine`이 `RiskManager`를 통과시킨 뒤 낸다(2번 관문
원칙 유지). 덕분에 셀프테스트에서 네트워크 없이 검증된다(17·18번 항목).

측정은 `scripts/replay_intraday.py`가 **같은 `IntradayWatcher`를 그대로 구동**한다.
재구현하면 검증이 아니라 다른 코드를 테스트하는 것이 된다.

알아야 할 세 가지.

- **선정이 전략의 거의 전부다.** 하루 돌파가 중앙 14건인데 슬롯은 2개다. 27건을
  버리는 선택이 `P(+100%)`를 1.04%(무작위)에서 6.17%(스킬 20%)까지 바꾼다.
  지금은 **신선도 고정 공식**을 쓴다(`_intraday_score`) — 분봉 모델이 부분 데이터에서
  무작위보다 나빴기 때문이고, 공정 비교 후에 교체한다.
- **주문 지정가는 비대칭이다.** 매수 2.0% / 매도 0.4%. 매수 0.4%로는 체결률이
  61.5%뿐이고 못 사는 쪽이 하필 가장 좋은 집단(빠른 급등)이다.
- **시각 기반 로직은 `MOCK_MINUTES_PER_LOOP`로 검증한다.** Mock 가상시계는 기본적으로
  일 단위만 더해 가상 시각이 항상 09:00이고, 아침 스캔·마감 정리·신선도 경과분이
  **한 번도 실행되지 않는다.** 리플레이는 `IntradayWatcher`만 구동하므로 `main.py`의
  배선을 보지 않는다. 이 설정(기본 0, 테스트 전용)을 0보다 크게 주면 `_now()`가
  루프 수 x 분으로 전진해 그 경로가 돈다.

  ```bash
  MOCK_MINUTES_PER_LOOP=30 python main.py --duration 380
  ```

  **단 자정을 그대로 통과한다.** `should_flatten()`은 `HHMM` 문자열 비교이므로 00:00이
  되면 진입이 다시 열린다. 마감 정리를 보려면 진입 이후 15:15까지 갈 만큼
  `--duration`을 길게 줘야 한다(루프당 약 3.5초, 30분 스텝에서 31루프). `--duration 200`
  으로 돌려 `청산 거래 0회`를 보고 코드 버그로 오해한 적이 있다.
- **매도는 체결 대기 중인 잔량을 빼고 낸다**(`_sellable_qty`). 청산 판정은 포지션
  수량을 보는데 주문 직후에도 수량이 남아 있어, 가드가 없으면 매 루프 같은 수량을
  재접수해 `40320000 주문가능수량을 초과했습니다`가 쌓인다(매도 4건에 거부 15건).
  대신 미체결을 영원히 기다리면 오버나이트가 되므로, 마감 정리 구간에서는
  `_sync_open_orders`를 매 루프 `FLATTEN_ORDER_TTL=30`초로 돌려 취소·재접수한다.

### 9-1. 청산 규칙은 `strategy/exit_rules.py`에 있다

청산은 **두 경로**가 따로 있다.

- 일봉/catalyst: `strategy/risk_manager.py::evaluate_exits()`
- 장중 분봉: `strategy/intraday.py::exit_reason()` (15:15 마감 정리 포함)

감쇠 기반 규칙 세 개는 모듈로 분리했다. **순수 로직이라 주문도 포트폴리오도
모르고**, 가격·거래대금·ATR을 받아 사유만 돌려준다(셀프테스트 19번).

| 규칙 | 보는 것 | 설정 | 측정 |
|---|---|---|---|
| **`vol_alive`** | 거래대금이 **20봉 중앙값**의 배수 아래 | `EXIT_VOL_ALIVE_MULT` | **채택 4.06%** |
| `crash` | 최근 `CRASH_LOOKBACK`봉 고점 대비 하락 | `CRASH_DROP_PCT` | 기각 1.84% |
| `trailing` | 고점 − `TRAILING_ATR_MULT` x ATR | `TRAILING_ACTIVATE_PCT` | 기각 1.88% |
| `vol_decay` | 거래대금이 **보유 중 고점**의 비율 아래 | `EXIT_VOL_DECAY_RATIO` | 기각 1.01% |

**`vol_alive`와 `vol_decay`는 분모만 다르고 결과가 4배 갈린다.** `vol_decay`는
"보유 중 고점"을 분모로 쓰는데 **폭증일 자신이 그 고점**이라 다음 날부터 거의 항상
깨진다(발동률 80.9%). `vol_alive`는 진입 조건과 **같은 분모**(20봉 중앙값)를 쓴다.

**`vol_alive`가 `MAX_HOLDING_BARS`를 대체한다.** 기간을 미리 정하지 않고 죽은
종목은 일찍 내보내고 살아 있는 종목은 더 들고 간다 — 중앙 보유일수는 고정 6봉과
같은 6일인데 단일 2배 확률이 0.64% → 0.86%(2배 사건 25 → 34건), 포트 `P(+100%)`가
2.21% → **4.06%**다(라운드 5회 고정). 그래서 `MAX_HOLDING_BARS=30`은 주 청산이
아니라 **안전 상한**이고, 6봉으로 두면 상태 기반이 무력화된다.

**기준을 올려도 좋아지지 않는다.** x1.5→x5에서 포트 수치는 오르지만 **단일 2배
비율은 0.94% → 0.69%로 떨어진다** — 상승분은 라운드 환산(30 ÷ 중앙 보유일수)이
4 → 10으로 늘어난 측정 근사의 artifact다. 더 올리면 보유가 1일로 수렴하는데
그것이 당일 청산이고 0.00%로 측정된 바로 그것이다.

**수급·상한가를 생존 조건에 넣으면 나빠진다** — 수급은 1거래일 지연이라 보유가
2일로 짧아지고(0.52%), 상한가는 드물어 1일이 된다(2.66%).

`EXIT_RULES`가 **순서 있는 목록**이다(`crash,trailing,vol_decay`). 같은 봉에서
둘이 동시에 충족될 때 앞쪽 규칙으로 기록되므로 **순서가 사유별 통계를 바꾼다.**

알아야 할 네 가지.

- **우선순위가 바뀌었다.** 분리 전에는 급락이 손절보다 먼저였고 지금은
  `킬스위치 > 손절 > 감쇠 > 익절 > 시간손절 > 신호반전`이다. 급락 임계값이
  손절보다 크면 손절이 먼저 걸려 사유 라벨이 바뀐다(청산 자체는 일어난다).
- **`check()`는 매 봉 한 번만 부른다.** 고점·거래대금 고점·가격 버퍼를 그 안에서
  갱신하므로, 건너뛰면 되돌림 기준이 멈추고 두 번 부르면 룩백 창이 두 배로 흐른다.
- **미보유 종목의 상태를 반드시 버린다.** 남기면 과거 폭증일의 거래대금 고점이
  살아 있어 같은 종목 **재진입 첫 봉에 `vol_decay`가 바로 걸린다.**
  `evaluate_exits()`가 매 호출마다 정리한다.
- **`turnover_map`과 `burst_map`을 백테스트와 실거래 양쪽에서 넘겨야 한다.**
  한쪽만 넘기면 그 규칙이 조용히 꺼지고 두 경로가 갈린다. 정의는 양쪽 모두
  **가격 x 거래량**이며 `catalyst_features.py`의 `turnover`와 같다. 실거래는
  `Quote.volume`(KIS `acml_vol`, 당일 누적)을 쓴다 — 패널 마지막 봉의 거래량은
  `_apply_latest_prices_to_panel()`이 갱신하지 않는다.
- **거래대금 중앙값의 창이 둘이고 섞으면 안 된다.** 일봉 진입
  (`turnover_burst_20`)은 `rolling(20)`으로 **당일 포함**(종가 확정 후 판정이라
  괜찮다), 장중 진입·`burst_map`·수집·측정은 **전일까지**(09:00에 알아야 한다).
  수집기에서 `.shift(1)`을 빼먹어 커버리지가 100%가 아니라 **83.1%**였다.
- **배수를 못 받으면(0) `vol_alive`는 판정하지 않는다.** 0을 "거래 소멸"로 읽으면
  배관이 끊겼을 때 전 종목이 첫 봉에 청산된다.
- **실거래 `vol_alive`는 확정 일봉으로 하루 한 번 판정한다**(`daily_alive_burst`).
  백테스트·측정은 "봉 t 종가에 하루치로 판정 → t+1 시가 매도"다. 장중 누적
  거래대금(`acml_vol`)을 넘기면 다음 날 09:00에 누적이 0에 가까워 **살아 있는 종목도
  매일 아침 팔린다** — 실제로 그렇게 배선돼 있었고(2026-10-09 수정), Mock은
  `acml_vol`에 일봉 거래량 전체를 넣어 스모크에 안 보였다. **진입 당일은 판정하지
  않는다**(넘기면 폭증 전날 거래대금으로 읽혀 사자마자 판다). 진입일은
  `state/engine_state.json`에 저장되고, 날짜가 바뀌면 `_roll_daily_reference()`가
  일봉을 다시 받아 기준값을 새로 만든다(`run()`은 `startup()`을 한 번만 부른다).
- **대회 1단계에도 `vol_alive`는 돈다.** `PHASE1_DISABLE_RISK`가 끄는 것은 손절·MDD
  게이트뿐이다. 예전에는 감쇠 청산까지 꺼져 1주차가 "사서 30봉 보유"였다 — 백테스트에
  대회 날짜가 없어 측정에 안 나타난 것으로, §11과 같은 종류의 실수다.
- **프로파일은 덮어쓰기이지 초기화가 아니다.** 모드를 결정하는 키
  (`USE_INTRADAY`·`ENTRY_MODE`·`EXIT_RULES`·`MAX_HOLDING_BARS`)를 프로파일마다
  전부 명시한다 — `profile-swing.env`에 `ENTRY_MODE`가 없어서 `.env`의 `catalyst`가
  살아남아 preflight가 치명 8건을 냈다.

### 10. 측정 전에 데이터 지문을 확인한다

`scripts/ablate_exits.py`와 `scripts/replay_intraday.py`가 실행마다 패널 지문
(정렬된 종가의 MD5 + 종목·봉 수)을 찍는다. **지문이 다른 실행의 수치는 비교하지
않는다.**

두 번 크게 당했다. 일봉에서 같은 설정이 Sharpe 0.873/+33.2%와 0.263/+7.0%로 갈렸고
(원인 미확정, 학습 구간 데이터 변화로 추정), 분봉에서 같은 코드가 +0.557%와
−0.442%로 갈렸다(수집이 돌아가는 중에 측정해 후보 풀이 달라졌다 — 두 실행 모두
"178종목 · 20250917~20260929"로 보여서 코드 차이를 몇 시간 뒤졌다).

**수집이 진행 중일 때는 측정하지 않는다.** 완주 후 한 번에 돌려 같은 지문에서
비교한다.

### 11. 반복된 실수 네 가지 — 같은 종류다

이 프로젝트에서 같은 모양의 실수가 네 번 났다. **전부 "측정이 실거래를 반영하지
않는다"는 한 가지 문제다.**

**측정에 없는 규칙은 측정이 틀렸다는 뜻이다.** 리플레이에 리스크 매니저가 없어서
`risk.evaluate_exits()`의 규칙 **여섯 개**가 측정에 나타나지 않았다. 익절 9%가
상승을 자르고, 일일 손실 한도 3%가 보유 평균 −3.6%에서 전량 청산을 걸고 있었다.
전수 점검으로만 찾았다.

**측정과 실거래가 같은 코드를 쓰는지 확인한다.** 진입 지정가가 매수 0.4%로
체결률이 61.5%였는데, 리플레이는 "고른 것은 반드시 산다"고 가정해 몰랐다.
못 사는 38.5%가 하필 가장 좋은 집단(빠른 급등)이었다.

**지문이 다른 수치를 비교하지 않는다.** 같은 코드가 +0.557%와 −0.442%로 갈렸고
코드 차이를 몇 시간 뒤졌다. 원인은 수집이 돌아가는 중에 측정한 것이었다.
두 실행 모두 "178종목 · 같은 기간"으로 보여서 알아채지 못했다.

**진단은 모집단과 선택된 표본을 구분해서 찍는다.** `trig_minute`가 전부 1분으로
나와 "신선도가 무력화됐다"고 판단했는데, 진입한 표본만 보고 있었다. 사건 전체로는
36.8분 vs 74.9분으로 **차이 50.9%의 강한 신호**였다.

그리고 **설정을 확인할 때는 필드가 실제로 있는지 먼저 본다.** 존재하지 않는
필드명(`SOFT_MDD_LIMIT`)을 조회해 `None`을 받고 "MDD가 꺼져 있다"고 판단했다.
실제는 `MDD_SOFT_LIMIT=0.07`로 살아 있었다.

### 12. 셀프테스트는 환경이 아니라 코드를 검증한다

`make_settings()`가 `.env`를 읽어서 두 번 깨졌다 — `SIZING_MODE=fixed`로 바꾸자
변동성 타게팅 검증이, 분봉용으로 리스크 청산을 해제하자 킬스위치·국면 검증이
깨졌다. **둘 다 코드 버그가 아니라 설정 변경이었다.**

테스트가 단정하는 값 17개를 `Settings.model_fields[...].default`로 고정했다.
각 모드·규칙의 on/off 동작은 해당 테스트 안에서 따로 검증한다.

## 문서

- `docs/ARCHITECTURE.md` — 코드 구조·비동기 설계·데이터 계약. 새 모듈을 넣기 전에 읽는다.
- `docs/TODO-대회준비.md` — 측정 결과와 남은 일. **파라미터를 바꾸기 전에 읽는다.**
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

## 작업 방식

- **큰 작업에 들어가기 전에 먼저 묻는다.** 전략 파라미터 변경, 긴 측정(수십 분 이상),
  데이터 재수집, 구조 리팩터링, 되돌리기 어려운 변경이 여기 해당한다.
  질문은 **중요한 순으로 하나씩**, 선택지마다 **무엇이 바뀌고 무엇을 포기하는지** 적어서 낸다.
  "진행해도 되나요" 같은 빈 확인은 묻지 않는다 — 답이 작업을 실제로 줄이는 것만 묻는다.
- 작은 작업(버그 수정, 로그 추가, 문서 최신화, 셀프테스트 보강)은 묻지 않고 한다.
- **구현을 전부 끝낸 뒤에 측정한다.** 시스템을 다 만들어 놓고 파인튜닝하는 순서다.
  측정과 구현을 번갈아 하면 지문이 갈리고(§10) 무엇이 기여했는지 알 수 없다.
- **파인튜닝 대상은 모듈로 분리한다.** 튜닝할 값이 관문 코드 안에 있으면 값을
  바꿀 때마다 모든 주문의 최종 결정권자를 건드려야 한다. 실제로 청산 규칙이
  `risk_manager.evaluate_exits()` 안에 하드코딩돼 있어서 `strategy/exit_rules.py`로
  분리했다. 순수 로직만 두면 네트워크 없이 셀프테스트로 검증된다.
