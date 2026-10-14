// 기술노트 HTML 판의 오른쪽 칸. 단락(### 4.1 …)마다 그 내용을 돕는 실제 코드·설정·그림을 붙인다.
// 키는 단락 번호("4.1"). 단락이 없는 장은 장 번호("7").
//
// 코드는 빌드할 때 저장소 파일에서 직접 잘라 온다. 코드가 바뀌면 다시 빌드할 때 따라온다.
//   { file: 저장소 기준 경로, from: "이 글자가 처음 나오는 줄부터", lines: 몇 줄, lang: 강조 언어, note: 왜 보는지 한 문장 }
//   to: "글자" 를 주면 lines 대신 그 글자가 나오는 줄까지 자른다
//   exact: true 면 줄 전체(앞뒤 공백 제외)가 from 과 같을 때만 고른다
//   from 을 못 찾으면 빌드가 멈춘다(코드가 바뀌어 위치가 사라졌다는 뜻이다. 새 위치를 찾아 고친다)
// 그림: { diagram: "diagrams.js 의 이름", note }   예시: { code: "...", lang: "bash", title: "명령 순서" }
// 비밀값이 든 파일(.env 등)은 가리키지 않는다.

module.exports = {
    "1.1": [
        { file: "main.py", from: "async def run(self, duration", to: "await self.shutdown()", lang: "python",
          note: "엔진 전체가 이 반복 하나다. _tick 한 번 → 남은 시간만큼 _stop 을 기다림 → 다시 _tick. 예외가 나도 루프는 죽지 않는다." },
    ],
    "1.2": [
        { code: "api/        증권사·데이터 API 클라이언트\ncore/       유량 제어 · 인증 · WebSocket · 지표 · 이벤트 버스\nstrategy/   판단(순수 함수) · 리스크 관문 · 청산 규칙\nmonitor/    관제 화면 · /metrics\nutils/      로거·원장 · 거래소 달력 · 호가 단위\ndeploy/     서버 설치·배포 스크립트\nscripts/    셀프테스트 · 프리플라이트 · 수집 · 측정\nmain.py     TradingEngine — 부품을 묶어 루프를 돈다",
          lang: "text", title: "폴더 한눈에", note: "판단은 strategy/, 바깥과 이야기하는 것은 api/·core/ 로 나뉜다. 판단 코드는 네트워크를 모른다." },
    ],
    "1.3": [
        { file: "api/factory.py", from: "def create_broker(", to: "return KISClient", lang: "python",
          note: ".env 한 줄(USE_MOCK_MODE)로 가짜와 진짜를 바꿔 끼운다. 둘 다 같은 limiter 를 받는다." },
    ],
    "2.1": [
        { file: "api/client.py", from: "for attempt in range(1, self.settings.HTTP_MAX_RETRY + 1):", to: "return last_payload", lang: "python",
          note: "limiter.acquire → 전송 → 지연 기록 → 429·EGW00201(속도 줄이고 재시도) · 토큰 오류(재발급 1회) · 통신 오류(두 배씩 대기) 순서로 갈린다." },
    ],
    "2.2": [
        { file: "core/auth.py", from: "REFRESH_MARGIN = dt.timedelta", lines: 2, lang: "python",
          note: "만료 30분 전에 미리 바꾸고, 재발급은 70초 간격을 지킨다." },
        { file: "core/auth.py", from: "async def get_token(self, force: bool = False)", lines: 15, lang: "python",
          note: "async with self._lock — 여러 코루틴이 동시에 불러도 발급은 한 번만 일어난다." },
    ],
    "2.3": [
        { file: "api/client.py", from: "not_sent = isinstance(exc, (httpx.ConnectError", lines: 8, lang: "python",
          note: "접속 단계 실패만 '안 나갔다'로 본다. 주문(idempotent=False)이 그 밖의 실패를 만나면 재시도하지 않고 LOCAL_UNKNOWN 을 돌려준다." },
        { file: "main.py", from: "if ack.msg_cd == UNKNOWN_RESULT_CODE:", lines: 10, lang: "python",
          note: "결과 불명이면 다시 보내지 않고 미체결·잔고를 즉시 조회해 실제로 접수됐는지 확인한다." },
    ],
    "2.4": [
        { file: "api/base.py", from: "async def get_turnovers(self, symbols", to: "return out", lang: "python",
          note: "range(0, len(symbols), 30) — 30종목씩 묶어 한 번에 묻는다. 494종목이면 17호출이다." },
    ],
    "3.1": [
        { file: "core/rate_limiter.py", from: "def _refill(self) -> None:", lines: 6, exact: false, lang: "python",
          note: "지난 시간 × 속도만큼 채우고 바구니 크기에서 자른다. 시간은 time.monotonic() 이라 시스템 시각이 바뀌어도 튀지 않는다." },
    ],
    "3.2": [
        { file: "core/rate_limiter.py", from: "PRIORITY_ORDER = 0", lines: 4, lang: "python",
          note: "숫자가 작을수록 급하다. 주문이 0 이다." },
        { file: "core/rate_limiter.py", from: "async def _run(self) -> None:", lines: 20, lang: "python",
          note: "디스패처 하나가 토큰이 생기면 heappop 으로 가장 급한 대기자 하나를 깨운다. 취소된 대기자는 토큰을 쓰지 않는다." },
    ],
    "3.3": [
        { file: "core/rate_limiter.py", from: "def notify_rate_limit_error(self) -> float:", lines: 14, lang: "python",
          note: "거절이면 토큰을 0 으로 비우고, 페널티 중 또 거절이면 두 배(최대 60초)로 늘린다." },
    ],
    "3.4": [
        { code: "# .env (유량 — 2026-10-13 실측으로 정함)\nREST_RATE_LIMIT=1\nREST_BURST=1\n\n# 30건씩 보낸 실측\n# 초당 1.8 → 거절 13 · 1.5 → 10 · 1.2 → 5 · 1.0 → 0",
          lang: "bash", title: "설정과 근거", note: "버스트 1 이 핵심이다. 페널티가 끝난 순간 한꺼번에 나가는 요청이 사라진다." },
    ],
    "4.1": [
        { file: "config/settings.py", from: "WS_TR_TRADE = \"H0STCNT0\"", lines: 4, lang: "python",
          note: "구독하는 실시간 TR. 체결 통보는 모의·실전 TR 이 다르다." },
    ],
    "4.2": [
        { file: "core/websocket_client.py", from: "async def _run(self) -> None:", to: "delay = min(self.settings.WS_RECONNECT_MAX_DELAY", lang: "python",
          note: "연결 → 성공하면 delay=1 로 되돌리고 _resubscribe_all → 예외면 delay 초 기다려 다시. 실패가 이어지면 두 배씩, 최대 30초." },
        { file: "core/websocket_client.py", from: "async def _resubscribe_all(self) -> None:", lines: 11, lang: "python",
          note: "새 연결은 백지라 구독을 우선순위 순서로 다시 등록한다. 등록 사이 30ms 쉰다." },
    ],
    "4.3": [
        { file: "core/websocket_client.py", from: "def is_healthy(self, max_silence: float = 60.0)", lines: 8, lang: "python",
          note: "연결이 없거나 60초 넘게 조용하면 비정상. 엔진은 이때 REST 로 시세를 조회한다." },
    ],
    "5.1": [
        { file: "main.py", from: "if (self._opening_mode() and self._opening_day != now.strftime", lines: 5, lang: "python",
          note: "오래 걸리는 판정은 create_task 로 떼어 내고 루프는 계속 돈다. 결과는 다음 루프가 _opening_picks 에서 읽는다." },
    ],
    "5.2": [
        { file: "core/events.py", from: "LOSSLESS_TOPICS = frozenset", lines: 30, lang: "python",
          note: "fill·order 는 크기 제한 없는 큐로, 나머지는 가득 차면 가장 오래된 것을 버리는 큐로 간다." },
    ],
    "5.3": [
        { file: "main.py", from: "async def _sync_open_orders(self, max_age_seconds", to: "self.open_orders = live", lang: "python",
          note: "증권사 미체결 목록으로 open_orders 를 다시 만들고, 180초 넘은 미체결은 취소한다." },
        { file: "api/base.py", from: "# 현금은 D+2 예수금(`prvs_rcdl_excc_amt`)을 쓴다", lines: 9, lang: "python",
          note: "어느 필드를 믿느냐 — 예수금 총액은 당일 매수가 빠지지 않아 자산이 두 배로 읽혔다." },
    ],
    "5.4": [
        { file: "main.py", from: "def _install_signal_handlers(engine", lines: 17, lang: "python",
          note: "SIGINT·SIGTERM 이 오면 _stop 만 켠다. 정리는 루프가 끝난 뒤 shutdown 이 한다." },
        { file: "main.py", from: "# 1) 미체결 주문 취소", lines: 16, lang: "python",
          note: "미체결 취소 → 남은 체결 수집 → 상태 저장 순서. 하나가 실패해도 다음으로 간다." },
    ],
    "6.1": [
        { file: "strategy/risk_manager.py", from: "def evaluate_entry(", to: "자산 0 이하", lang: "python",
          note: "모든 매수가 이 함수의 RiskDecision 을 받아야 나간다. 전략은 수량을 정하지 않는다." },
    ],
    "6.2": [
        { file: "main.py", from: "band = self.s.BUY_LIMIT_BAND if side is Side.BUY", lines: 3, lang: "python",
          note: "매수 +2.0% · 매도 −0.4% 를 곱한 뒤 round_to_tick 으로 호가 단위에 맞춘다." },
        { file: "utils/market_rules.py", from: "def round_to_tick(", lines: 15, lang: "python",
          note: "가격대별 호가 단위로 내림·올림·반올림한다. 안 맞으면 증권사가 거절한다." },
    ],
    "6.3": [
        { file: "utils/logger.py", from: "class JsonLedger:", to: "fp.write(line + \"\\n\")", lang: "python",
          note: "한 줄에 사건 하나, 덧붙이기(\"a\")만 한다. 락으로 줄이 섞이지 않게 한다." },
    ],
    "7.1": [
        { file: "monitor/app.py", from: "async def metrics() -> str:", lines: 26, lang: "python",
          note: "이름 값 한 줄씩. HELP·TYPE 줄이 Prometheus 형식의 약속이다." },
    ],
    "7.2": [
        { file: "core/latency.py", from: "BUCKETS: List[float] =", lines: 1, lang: "python",
          note: "경계 12개. 각 구간에 들어온 개수만 센다." },
        { file: "core/latency.py", from: "def quantile(self, q: float) -> float:", lines: 13, lang: "python",
          note: "누적 개수가 q 비율을 넘는 첫 구간의 경계값을 분위수로 쓴다." },
    ],
    "7.3": [
        { file: "api/client.py", from: "def _measure_clock_skew(self, date_header", lines: 14, lang: "python",
          note: "Date 헤더 − (지금 − 왕복/2). 2초를 넘으면 경고한다." },
    ],
    "8.1": [
        { file: "deploy/oracle/setup.sh", from: "[Service]", to: "StandardError=", lang: "ini",
          note: "Restart=always · TimeoutStopSec=60 · MemoryMax=2G 가 각각 '다시 띄움 · 정리 시간 · 메모리 상한'이다." },
    ],
    "8.2": [
        { file: "deploy/oracle/deploy.sh", from: "push_data() {", lines: 12, lang: "bash",
          note: "tar 종료 코드 1(읽는 중 바뀜)은 경고로 넘기고 2 이상만 멈춘다." },
        { file: "deploy/oracle/setup.sh", from: "[ -x .venv/bin/python ] || uv venv", lines: 1, lang: "bash",
          note: "있으면 건너뛴다 — 몇 번을 다시 돌려도 같은 결과(멱등)." },
    ],
    "8.3": [
        { file: "deploy/oracle/setup.sh", from: "sudo tee /etc/logrotate.d/bankis", lines: 11, lang: "bash",
          note: "14일 보관 · 압축 · copytruncate. 원장(ledger.jsonl)은 *.log 가 아니라 대상이 아니다." },
        { file: "deploy/oracle/setup.sh", from: "SystemMaxUse=200M", lines: 1, lang: "bash",
          note: "시스템 로그 상한 200MB." },
    ],
    "8.4": [
        { file: "deploy/oracle/setup.sh", from: "sudo tee /etc/sysctl.d/90-bankis.conf", lines: 6, lang: "bash",
          note: "60초 조용하면 확인 시작, 10초 간격 6번 — 죽은 연결을 약 2분 안에 정리한다." },
    ],
    "9.1": [
        { file: "api/toss_data.py", from: "if resp.status_code == 429:", lines: 6, lang: "python",
          note: "Retry-After 헤더만큼(없으면 1초) 쉬고 다시 보낸다." },
    ],
    "9.2": [
        { code: "# 노트북 — 로컬 1080 포트에 SOCKS5 프록시, 출구는 서버\nssh -N -D 127.0.0.1:1080 \\\n    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\\n    -o ExitOnForwardFailure=yes root@<서버>\n\n# 수집기는 환경변수만 받는다(httpx 가 읽는다)\nALL_PROXY=socks5://127.0.0.1:1080 python scripts/collect_opening.py ...\n\n# 출발 IP 확인\ncurl --socks5-hostname 127.0.0.1:1080 https://api.ipify.org",
          lang: "bash", title: "터널과 실행", note: "코드 변경 없이 환경변수 하나로 토스 요청만 서버를 거친다." },
    ],
    "9.3": [
        { code: [
            "# 엔진과 다른 cgroup 의 임시 유닛으로 띄운다",
            "systemd-run --unit=bankis-collect --nice=19 \\",
            "    -p MemoryMax=1G -p CPUQuota=60% \\",
            "    -E UNIVERSE_FILE=config/universe_10b.json \\",
            "    .venv/bin/python scripts/collect_opening.py --concurrency 3",
            "",
            "# 결과 · 자원 사용",
            "journalctl -u bankis-collect | tail",
            "systemctl status bankis-collect   # Memory · CPU 누적",
          ].join("\n"),
          lang: "bash", title: "격리 실행", note: "nice 는 CPU 만 양보한다. MemoryMax 가 OOM 을 이 유닛 안에 가둔다." },
    ],
    "10.1": [
        { file: "scripts/selftest.py", from: "_SELFTEST_TMP = Path(_tempfile.mkdtemp", lines: 1, lang: "python",
          note: "테스트의 원장·로그·상태는 임시 폴더에 쓴다. 실거래 원장을 건드리지 않는다." },
        { file: "scripts/selftest.py", from: "@check(\"5. 우선순위 큐(주문 최우선)\")", lines: 24, lang: "python",
          note: "시세 요청을 잔뜩 걸어 둔 뒤 주문을 넣고, 주문이 먼저 나가는지 본다." },
    ],
    "10.2": [
        { file: "scripts/preflight.py", from: "add(FATAL, \"폭증 감시 범위\"", lines: 4, lang: "python",
          note: "UNIVERSE_SIZE 가 유니버스보다 작으면 일부 종목을 조용히 안 본다. 치명으로 막는다." },
    ],
    "10.3": [
        { file: "scripts/selftest.py", from: "# 인과성: 앞부분만 잘라 계산해도", lines: 7, lang: "python",
          note: "전 구간 한 번 계산한 값과 시점마다 잘라 계산한 값이 같아야 한다. 다르면 미래를 보고 있다." },
    ],
};
