# KIS AI Quant Trader

제16회 뱅키스 대학생 모의투자대회(AI 부문)용 한국투자증권 Open API 기반 자동매매 시스템.

**핵심 설계**: `.env`의 `USE_MOCK_MODE` 한 줄로 내부 가상 체결 엔진 ↔ 실제 KIS 서버를 전환한다.
계좌 발급 전에는 Mock으로 전체 파이프라인을 100% 검증하고, 발급 후에는 코드 수정 없이 실서버에 붙는다.

---

## 1. 빠른 시작 (계좌 발급 전)

```bash
copy .env.example .env          # 기본값이 USE_MOCK_MODE=true

python scripts/selftest.py      # 전 모듈 스모크 테스트 (11개 항목)
python scripts/train_model.py   # AI 모델 학습 → models/alpha_gbdt.pkl
python scripts/run_backtest.py  # 백테스트 → reports/ 에 지표·차트 저장
python main.py                  # Mock 실시간 매매 루프 + 대시보드
```

대시보드: <http://127.0.0.1:8500>

추가 옵션

```bash
python main.py --duration 120      # 120초만 구동(스모크)
python main.py --no-dashboard
python main.py --health-only       # 연결/인증/계좌 점검만
python scripts/run_backtest.py --no-model   # AI 기여도 비교(기술적 팩터만)
```

## 2. 문서

| 문서 | 내용 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 코드 구조, 비동기 설계, 유량 제어, 데이터 계약, 코드 읽는 순서 |
| [docs/매매전략.md](docs/매매전략.md) | 피처 44종, 삼중배리어 라벨링, 앙상블, 포지션 사이징, 청산 규칙 |
| [docs/architecture.html](docs/architecture.html) | 인터랙티브 구조 다이어그램 |

## 3. 아키텍처

```
                    ┌──────────────── config/settings.py (.env) ───────────────┐
                    │  USE_MOCK_MODE / KIS_ENV → 도메인·TR_ID 자동 분기          │
                    └──────────────────────────────────────────────────────────┘
                                             │
  [WS 실시간 체결/호가] ──┐                    ▼
                         ├──→ MarketData ──→ strategy/features.py (44개 피처)
  [REST 시세 폴링] ───────┘                    │
                                              ├─ Alpha1: GBDT P(상승)       ─┐
                                              ├─ Alpha2: 기술적 팩터 스코어  ─┼─ 앙상블
                                              ├─ Alpha3: 횡단면 랭크        ─┤   ×
                                              └─ Regime Filter(0.15~1.0)   ─┘ 레짐
                                                            │
                                                            ▼
                                       ┌─────────────────────────────────────┐
                                       │  strategy/risk_manager.py           │
                                       │  ★ 모든 주문의 단일 관문 (거부권)    │
                                       │  · 변동성타게팅 × Kelly 배수         │
                                       │  · 손절/트레일링/익절/시간손절       │
                                       │  · MDD 2단계 (DERISK → HALT)        │
                                       │  · 섹터·종목·총익스포저 상한         │
                                       └─────────────────────────────────────┘
                                                            │
                                                            ▼
                              core/rate_limiter.py (토큰버킷 + 우선순위 큐)
                                        주문 P0 > 계좌 P1 > 시세 P2
                                                            │
                          ┌─────────────────────────────────┴──────────────────┐
                          ▼                                                    ▼
              api/mock_client.py (가상 체결 엔진)                api/client.py (실 KIS 서버)
                          └─────────────────────────────────┬──────────────────┘
                                                            ▼
                                            체결 → Portfolio → 원장(JSONL) → 대시보드
```

### 기술 선택 이유

| 영역 | 선택 | 이유 |
|---|---|---|
| 비동기 | `asyncio` + `httpx` | 유량 제한이 병목이므로 단일 루프 + 큐가 제어에 유리 |
| 지표 | 자체 numpy 구현 | TA-Lib의 Windows C 빌드 의존성 제거, 백테스트/실시간 동일 코드 |
| AI | sklearn HistGBDT ↔ LightGBM | 동일 계열 알고리즘. lightgbm 설치 시 자동 승격 |
| 검증 | PurgedKFold + embargo | 삼중배리어 라벨의 구간 중첩으로 인한 정보 누수 차단 |
| 로깅 | loguru ↔ stdlib shim | loguru 미설치 환경에서도 동일 API로 동작 |
| 암복호 | `cryptography` | 체결통보 AES-256-CBC 복호화 |

## 4. 계좌 발급 후 실전 전환

1. **`.env` 수정** — 이것이 전부다.
   ```ini
   USE_MOCK_MODE=false
   KIS_ENV=vts
   KIS_APP_KEY=발급받은키
   KIS_APP_SECRET=발급받은시크릿
   KIS_ACCOUNT_NO=12345678-01
   KIS_HTS_ID=본인HTS아이디
   REST_RATE_LIMIT=2          # 모의투자 계정은 초당 2건
   ```
2. **연결 점검** — `python main.py --health-only`
   → 토큰 발급 / 시세 조회 / 잔고 조회 3단계가 모두 `ok: true`여야 한다.
3. **무주문 리허설** — `.env`에 `DRY_RUN=true` 후 `python main.py --duration 600`
   → 주문 직전까지 전 과정이 돌고 전송만 생략된다. 로그의 `[DRY_RUN]` 라인으로 주문 내용 확인.
4. **실주문 시작** — `DRY_RUN=false`로 되돌리고 `python main.py`
5. **실데이터 재학습** (권장) — 실서버에서 받은 일봉을 `data/cache/<종목코드>.csv`로 저장 후
   `python scripts/train_model.py --source data/cache`

### 전환 시 주의사항

- **유량**: 모의투자 계정은 초당 2건이다. `REST_RATE_LIMIT`을 올리면 `EGW00201`이 발생하고
  최악의 경우 손절 주문이 거부된다.
- **토큰**: 접근토큰은 24시간 유효하고 짧은 간격의 재발급이 거부된다. `.tokens/`에 캐싱되므로
  이 디렉토리를 지우고 반복 재시작하지 말 것.
- **모의투자 도메인 제약**: 일부 순위/조건검색 API가 지원되지 않는다. 유니버스는
  `config/settings.py`의 `DEFAULT_UNIVERSE`에서 일봉 거래대금 랭킹으로 선정한다.
- **체결통보**: `KIS_HTS_ID`가 없으면 WebSocket 체결통보 구독을 건너뛰고 REST 폴링으로 동작한다
  (체결 반영이 수 초 늦어질 뿐 매매는 정상).

## 5. 디렉토리

```
config/settings.py        .env 로딩, 도메인·TR_ID 환경 분기
core/auth.py              OAuth2 토큰 발급·디스크 캐시·선제 재발급
core/rate_limiter.py      토큰 버킷 + 우선순위 큐 + 적응형 백오프
core/websocket_client.py  실시간 체결/호가/체결통보(AES 복호화)
core/events.py            이벤트 정의 + Portfolio/Position + EventBus
api/base.py               AbstractBroker(공통 인터페이스·응답 파싱)
api/mock_client.py        가상 호가/체결 엔진 (KIS와 동일 JSON)
api/client.py             실 KIS REST 클라이언트
strategy/features.py      44개 피처 파이프라인(종목 + 횡단면)
strategy/ai_model.py      삼중배리어 라벨·PurgedKFold·GBDT·앙상블 전략
strategy/risk_manager.py  사이징·손절·MDD 방어 (단일 관문)
backtest/engine.py        이벤트 기반 백테스터(룩어헤드 차단·비용 반영)
backtest/metrics.py       Sharpe/Sortino/MDD/Calmar/PF
backtest/data.py          합성 시세 생성기 + 로컬 로더
monitor/app.py            FastAPI 대시보드 (/ /health /api/state /metrics)
utils/                    지표·로거·KRX 캘린더·시장 규칙
scripts/                  selftest / train_model / run_backtest
main.py                   엔트리포인트
```

## 6. 파라미터 튜닝 가이드

대회 심사 항목별로 조정할 `.env` 값:

| 목표 | 조정 항목 |
|---|---|
| Sharpe ↑ | `TARGET_VOLATILITY` ↓, `SIGNAL_BUY_THRESHOLD` ↑(신호 선별), `MAX_POSITIONS` ↑(분산) |
| MDD ↓ | `MDD_SOFT_LIMIT`·`MDD_HARD_LIMIT` ↓, `STOP_LOSS_PCT` ↓, `MAX_GROSS_EXPOSURE` ↓ |
| 예측 정확도 ↑ | `LABEL_HORIZON`·`LABEL_PT_ATR`/`LABEL_SL_ATR` 조정 후 재학습, 학습 기간 확대 |
| 회전율 ↓ (비용 절감) | `TAKE_PROFIT_PCT` ↑, `MAX_HOLDING_BARS` ↑, `SIGNAL_BUY_THRESHOLD` ↑ |

변경 후에는 반드시 `python scripts/run_backtest.py`로 Sharpe/MDD를 재확인한다.
자세한 튜닝 지침과 각 파라미터의 의미는 [docs/매매전략.md](docs/매매전략.md)에 있다.
