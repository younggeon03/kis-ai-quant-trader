# 이 저장소의 기술노트 규칙

tech-note 스킬의 공통 규칙 위에 이 저장소에만 해당하는 것을 적는다. 공통 규칙과 다르면 이 파일이 이긴다.

## 독자

공통 기본값(파이썬 문법은 알지만 실무 용어는 처음인 주니어)을 따른다. 매매 전략(피처·라벨·청산 규칙의 근거)은
이 노트에서 다루지 않고 `docs/매매전략.md` 로 링크한다. 이 노트는 그 전략을 돌리는 **서버 쪽 기술**만 다룬다.

## 바뀐 곳 → 고칠 장

| 바뀐 곳 | 고칠 장 |
|---|---|
| `api/client.py`, `api/base.py`, `core/auth.py`, `api/factory.py` | 2장 외부 API 연동, 1.3절 |
| `core/rate_limiter.py`, `.env` 의 `REST_RATE_LIMIT`·`REST_BURST` | 3장 유량 제어 |
| `core/websocket_client.py` | 4장 WebSocket |
| `main.py` 의 `run`·`_tick`·`shutdown`·`_sync_*`·상태 저장, `core/events.py` | 5장 비동기 엔진과 상태 |
| `strategy/risk_manager.py`, `main.py::_submit`, `utils/market_rules.py`, `utils/logger.py::JsonLedger` | 6장 주문과 장부 |
| `monitor/app.py`, `core/latency.py`, `KISClient._measure_clock_skew` | 7장 관측 |
| `deploy/oracle/*` | 8장 배포와 서버 운영 |
| `scripts/collect_opening.py`, `api/toss_data.py`, 수집 실행 방식(터널·systemd-run) | 9장 데이터 수집 배치 |
| `scripts/selftest.py`(항목 수), `scripts/preflight.py` | 10장 검증 |
| 진입 시각·유니버스 크기(`OPENING_WINDOW_MIN`, `UNIVERSE_FILE`) | 1.1절, 2.4절, 5.1절의 숫자 |

## 장 첫머리 지도의 부분 이름

| part | 뜻 |
|---|---|
| `kis` | KIS REST·WebSocket(바깥) |
| `toss` | 토스 API(바깥) |
| `limiter` | 유량 제어기 |
| `api` | REST 클라이언트 |
| `ws` | WebSocket 클라이언트 |
| `engine` | 매매 엔진(`main.py::TradingEngine`) |
| `risk` | 리스크 관문 |
| `ledger` | 원장·상태 파일 |
| `monitor` | 관측(`/metrics`) |
| `server` | systemd·서버 설정 |
| `batch` | 수집 배치(노트북) |

## 노트와 같이 고칠 다른 문서·숫자

- 셀프테스트 항목 수: `CLAUDE.md` 의 Commands 주석과 10.1절
- 포트폴리오 문서(`docs/포트폴리오-자기소개서.md`, 저장소 밖 · git 제외)와 숫자가 겹치는 곳: 3.4절(유량 실측), 2.4절(305.0초 → 19.8초), 8.3절(디스크), 9.2·9.3절(터널·cgroup)
- 원격 저장소는 비공개지만 HTML 판은 Artifact 로 올라간다. 서버 공인 IP·계좌·키는 노트와 오른쪽 칸에 넣지 않는다(본문은 "서버"로 쓴다)

## 사실 뽑기

`.tech-note/facts.js` 는 아직 없다. 셀프테스트 수는 `grep -c '^@check("' scripts/selftest.py`, 파일 줄 수는 `wc -l` 로 센다.
