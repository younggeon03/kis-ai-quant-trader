# KIS AI Quant Trader

제16회 뱅키스 대학생 모의투자대회(AI 부문)용 한국투자증권 Open API 기반 자동매매 시스템.

> ## 면책
>
> **이 코드는 모의투자 대회 참가용이며 투자 조언이 아니다.** 실제 주문을 내는
> 자동매매 시스템이므로, 실계좌에 연결해 발생하는 금전적 손실을 포함한 모든
> 결과는 사용자 책임이다. 저자는 어떠한 보증도 하지 않는다(MIT License 참고).
>
> 저장소의 성과 수치는 **과거 데이터에 대한 백테스트·리플레이 결과**이며 미래
> 수익을 보장하지 않는다. 측정 과정에서 드러난 한계를 문서에 그대로 남겨 두었다
> — 생존 편향(유니버스가 특정 시점 거래대금 상위로 구성됨), 파라미터 선택편향
> (여러 값을 시험해 최고를 고름), 표본 부족(30거래일 구간에서는 운과 실력을
> 구분할 수 없음). 수치를 읽을 때 `docs/매매전략.md`의 단서를 함께 보기를 권한다.
>
> API 키·계좌번호는 `.env`에만 두고 절대 커밋하지 않는다(`.gitignore`로 차단함).
> `.env.example`을 복사해 쓴다.

**핵심 설계**: `.env`의 `USE_MOCK_MODE` 한 줄로 내부 가상 체결 엔진 ↔ 실제 KIS 서버를 전환한다.
계좌 발급 전에는 Mock으로 전체 파이프라인을 100% 검증하고, 발급 후에는 코드 수정 없이 실서버에 붙는다.

## 대회 조건

| 구분 | 내용 |
|---|---|
| 기간 | **2026-10-12 ~ 11-20 = 거래일 30일** |
| 심사 | **절대수익률 top10** (지난 대회 10위권 +100%) |
| 초기자본 | 1억원 |
| 매매 가능 | **KOSPI·KOSDAQ 전 개별 종목** (지수 제한 없음) |
| **단일 종목 한도** | **자산의 50%** |
| 매수 불가 | 현재가 1,000원 미만 / 발행주식수 10만주 미만 / 관리종목·거래정지 |
| 금지 | 신용거래·공매도 (현금 매수/매도만) |
| 최소 요건 | OpenAPI 5회 · 매매금액 5억원 · 매매일수 5영업일 · 코스피200·코스닥150 중 5종목 |
| 수익률 초기화 | 10/12~10/16 중 **단 1회** |

규정 판정은 `strategy/eligibility.py`, 요건 추적은 `strategy/compliance.py`,
국면 전환과 초기화 권고는 `strategy/phase.py`가 담당한다.
**초기화는 코드가 실행하지 않는다** — 되돌릴 수 없는 1회성 행위라 권고만 띄운다.

---

## 1. 빠른 시작 (계좌 발급 전)

```bash
copy .env.example .env          # 기본값이 USE_MOCK_MODE=true

python scripts/selftest.py      # 전 모듈 스모크 테스트 (16개 항목)
python scripts/train_model.py   # AI 모델 학습 → models/alpha_gbdt.pkl
python scripts/run_backtest.py  # 백테스트 → reports/ 에 지표·차트 저장
python main.py                  # Mock 실시간 매매 루프 + 대시보드
```

**단일 백테스트 숫자를 믿지 않는다.** 대회 기간이 30거래일이므로 **30봉 구간**으로 평가한다.

```bash
# 설정군별 절제 실험 — 구간당 학습 1회로 여러 설정을 동시 평가 (약 30분)
python scripts/ablate_exits.py --source data/cache --set exits --step 30
python scripts/ablate_exits.py --source data/cache --set sizing --step 30

python scripts/validate_strategy.py   # CPCV 경로 분포 + PSR/DSR
python scripts/walkforward.py         # 재학습하며 과거만 보고 평가
```

실데이터 21구간 out-of-sample 측정 결과 (데이터 지문 `30d0cafd15bc1a5f`, 수급 미사용):

| 설정 | Sharpe | 누적수익 | MDD | 익스포저 |
|---|---|---|---|---|
| 동일가중 매수보유(벤치마크) | 0.869 | +61.8% | 22.2% | — |
| 원 설정(손절 3.5% / 익절 9%) | −0.147 | −5.2% | 13.9% | 29.7% |
| **손절 12%만 ← 현재 설정** | **0.425** | **+12.1%** | 16.4% | 35.8% |
| 손절12 + 익절제거 | 0.263 | +7.0% | 17.9% | 38.4% |

**손절을 넓히는 것이 검증된 단일 최대 개선이다.** 손절 3.5%가 유니버스 일간 변동성
3.3%와 거의 같아 무작위 진입에서도 10봉 내 62.6%가 노이즈로 손절됐다.

반면 **익절·트레일링·보유기간은 건드리지 않는다** — 익절 제거는 단독으로도(−0.163)
조합에서도(0.425 → 0.263) 나빴고, 트레일링 제거 −0.089, 보유 10봉 단축 −0.560이었다.

> **절대 수치는 데이터 지문마다 다르다.** 같은 설정이 데이터 재수집 전후로 Sharpe
> 0.873과 0.263으로 갈린 적이 있다. 코드·시드·평가 구간 가격이 모두 동일함을 확인했고
> 학습 구간 데이터만 남는 설명이었다. 그래서 `ablate_exits.py`가 실행마다 패널 지문을
> 기록한다. **지문이 다른 실행의 수치는 비교하지 않는다.**

**아직 절대수익이 벤치마크에 밀린다**(+12.1% vs +61.8%). 대회 심사가 절대수익이므로
이 격차가 본질이며, 진행 상황은 [docs/TODO-대회준비.md](docs/TODO-대회준비.md)에 있다.

## 유니버스 갱신

지수 구성종목은 정기변경이 6월·12월 두 번뿐인 참조 데이터이므로 런타임에 받지 않는다.
`data.krx.co.kr`에서 CSV를 내려 재생성한다(pykrx 1.2.9의 목록성 조회는 KRX 계정을 요구한다).

```bash
python scripts/refresh_universe.py     --all-stocks data/krx/<전종목시세>.csv     --kospi200   data/krx/<코스피200>.csv     --kosdaq150   data/krx/<코스닥150>.csv     --min-trading-value 3e9
python scripts/collect_data.py          # 토스에서 일봉·수급 수집
```

전종목 2,872 → 규정 필터 통과 2,429 → 거래대금 30억 이상 **284종목**
(지수 소속 173 + 지수 외 111). 결과는 `config/universe.json`에 `as_of`·필터 조건·
제외 사유와 함께 저장된다.

대시보드: <http://127.0.0.1:8500>

추가 옵션

```bash
python main.py --duration 120      # 120초만 구동(스모크)
python main.py --no-dashboard
python main.py --health-only       # 연결/인증/계좌 점검만
python scripts/run_backtest.py --no-model   # AI 기여도 비교(기술적 팩터만)
```

### 실데이터 수집 (토스증권 API)

```bash
# .env 에 TOSS_CLIENT_ID / TOSS_CLIENT_SECRET 를 채운 뒤
python scripts/collect_data.py --probe     # 연결·응답 필드명 확인 (먼저 이것부터)
python scripts/collect_data.py             # data/cache/ 에 일봉·수급 저장
python scripts/train_model.py --source data/cache
```

토스는 **학습 데이터 전용**이다. 장중 시세와 주문은 KIS로 간다(대회 계좌가 KIS이므로).

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
