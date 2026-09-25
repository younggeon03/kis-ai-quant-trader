"""환경설정 (.env 기반, pydantic-settings v2).

`USE_MOCK_MODE` 하나로 Mock 엔진 ↔ 실제 KIS 서버를 전환한다.
TR_ID는 `settings.tr("order_buy")` 형태로 조회하며 모의(VTS)/실전(PROD) 환경에 따라
자동으로 다른 코드가 반환되므로 호출부 코드는 전혀 바뀌지 않는다.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# KIS 도메인
REST_DOMAIN_PROD = "https://openapi.koreainvestment.com:9443"
REST_DOMAIN_VTS = "https://openapivts.koreainvestment.com:29443"
WS_DOMAIN_PROD = "ws://ops.koreainvestment.com:21000"
WS_DOMAIN_VTS = "ws://ops.koreainvestment.com:31000"

# TR_ID 매핑: (실전, 모의). 시세계 TR은 양쪽 공통이라 동일 값을 둔다.
TR_MAP: Dict[str, Tuple[str, str]] = {
    # --- 주문/계좌 (환경별로 다름) ---
    "order_buy": ("TTTC0802U", "VTTC0802U"),
    "order_sell": ("TTTC0801U", "VTTC0801U"),
    "order_revise_cancel": ("TTTC0803U", "VTTC0803U"),
    "inquire_balance": ("TTTC8434R", "VTTC8434R"),
    "inquire_psbl_order": ("TTTC8908R", "VTTC8908R"),
    "inquire_daily_ccld": ("TTTC8001R", "VTTC8001R"),
    "inquire_psbl_rvsecncl": ("TTTC8036R", "VTTC8036R"),
    "inquire_balance_pl": ("TTTC8494R", "VTTC8494R"),
    # --- 시세 (공통) ---
    "inquire_price": ("FHKST01010100", "FHKST01010100"),
    "inquire_asking_price": ("FHKST01010200", "FHKST01010200"),
    "inquire_daily_chart": ("FHKST03010100", "FHKST03010100"),
    "inquire_minute_chart": ("FHKST03010200", "FHKST03010200"),
    "inquire_investor": ("FHKST01010900", "FHKST01010900"),
}

# WebSocket TR_ID
WS_TR_TRADE = "H0STCNT0"        # 실시간 체결가
WS_TR_ORDERBOOK = "H0STASP0"    # 실시간 호가
WS_TR_NOTICE_PROD = "H0STCNI0"  # 실시간 체결통보(실전)
WS_TR_NOTICE_VTS = "H0STCNI9"   # 실시간 체결통보(모의)

# 기본 유니버스: KOSPI200 대형주 + 유동성 상위 종목 (모의투자 도메인은 순위/조건검색
# API가 제한되므로 사전 정의 풀에서 일봉 기반 랭킹으로 추립다)
DEFAULT_UNIVERSE: List[str] = [
    "005930", "000660", "373220", "207940", "005380", "005490", "000270", "068270",
    "035420", "051910", "006400", "035720", "105560", "055550", "012330", "028260",
    "066570", "003670", "096770", "017670", "015760", "034730", "018260", "010130",
    "011200", "009150", "032830", "086790", "316140", "138040", "323410", "377300",
    "251270", "036570", "259960", "263750", "293490", "112040", "041510", "352820",
]

# 종목 → 섹터 (분산 제약용). 미등록 종목은 "ETC"로 처리된다.
SECTOR_MAP: Dict[str, str] = {
    "005930": "IT", "000660": "IT", "042700": "IT", "009150": "IT", "011070": "IT",
    "373220": "BATTERY", "051910": "BATTERY", "006400": "BATTERY", "247540": "BATTERY",
    "005380": "AUTO", "000270": "AUTO", "012330": "AUTO", "204320": "AUTO",
    "005490": "MATERIAL", "010130": "MATERIAL", "011200": "MATERIAL", "010620": "MATERIAL",
    "207940": "BIO", "068270": "BIO", "302440": "BIO", "128940": "BIO", "326030": "BIO",
    "035420": "PLATFORM", "035720": "PLATFORM", "036570": "GAME", "251270": "GAME",
    "259960": "GAME", "263750": "GAME", "293490": "GAME", "112040": "GAME",
    "105560": "FINANCE", "055550": "FINANCE", "086790": "FINANCE", "316140": "FINANCE",
    "138040": "FINANCE", "323410": "FINANCE", "377300": "FINANCE", "024110": "FINANCE",
    "015760": "UTILITY", "017670": "TELECOM", "030200": "TELECOM", "032640": "TELECOM",
    "034730": "HOLDING", "003550": "HOLDING", "018260": "IT", "030000": "MEDIA",
    "028260": "HOLDING", "066570": "IT", "003670": "MATERIAL", "096770": "ENERGY",
    "010950": "ENERGY", "267250": "INDUSTRIAL", "009540": "INDUSTRIAL",
    "032830": "FINANCE", "041510": "MEDIA", "352820": "MEDIA", "035900": "MEDIA",
}


class Settings(BaseSettings):
    """전체 시스템 설정. 모든 항목은 .env로 덮어쓸 수 있다."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ 실행 모드
    USE_MOCK_MODE: bool = Field(default=True, description="True면 KIS 서버 호출 없이 내부 가상 체결 엔진 사용")
    KIS_ENV: Literal["vts", "prod"] = Field(default="vts", description="vts=모의투자 도메인, prod=실전 도메인")
    DRY_RUN: bool = Field(default=False, description="True면 주문 직전에 실제 전송을 생략(실서버 안전 점검용)")

    # ------------------------------------------------------------------ 인증 정보
    KIS_APP_KEY: str = Field(default="")
    KIS_APP_SECRET: str = Field(default="")
    KIS_ACCOUNT_NO: str = Field(default="00000000-01", description="종합계좌번호 8자리-상품코드 2자리")
    KIS_HTS_ID: str = Field(default="", description="실시간 체결통보 구독용 HTS ID")
    TOKEN_CACHE_DIR: str = Field(default=str(PROJECT_ROOT / ".tokens"))

    # ------------------------------------------------------------------ Rate Limit
    # KIS 유량: 실전 계정 초당 20건, 모의투자 계정 초당 2건(계정 단위).
    # 기본값은 안전하게 모의투자 기준으로 두고, 실전 전환 시 .env에서 올린다.
    REST_RATE_LIMIT: float = Field(default=2.0, description="초당 REST 요청 허용량")
    REST_BURST: int = Field(default=2, description="토큰 버킷 최대 버스트")
    RATE_LIMIT_SAFETY: float = Field(default=0.9, description="설정 유량 대비 실제 사용 비율(마진)")
    HTTP_TIMEOUT: float = Field(default=10.0)
    HTTP_MAX_RETRY: int = Field(default=3)
    HTTP_BACKOFF_BASE: float = Field(default=0.6)
    WS_MAX_SUBSCRIPTIONS: int = Field(default=40, description="세션당 실시간 등록 한도(체결+호가 합산, KIS 41건)")
    WS_RECONNECT_MAX_DELAY: float = Field(default=30.0)

    # ------------------------------------------------------------------ 토스증권 (학습 데이터 전용)
    # 과거 일봉과 수급 데이터를 받아 모델 학습·백테스트에 쓴다. 장중 시세와 주문은 KIS로 간다.
    # 유량(TPS): 차트 20 · 수급 10 · 인증 5. 기본값은 여유를 둔 절반 수준.
    TOSS_CLIENT_ID: str = Field(default="")
    TOSS_CLIENT_SECRET: str = Field(default="")
    TOSS_BASE_URL: str = Field(default="https://openapi.tossinvest.com")
    TOSS_CHART_TPS: float = Field(default=10.0, description="캔들 조회 초당 요청 수")
    TOSS_FLOW_TPS: float = Field(default=5.0, description="수급 조회 초당 요청 수")
    TOSS_TIMEOUT: float = Field(default=15.0)
    TOSS_MAX_RETRY: int = Field(default=3)
    DATA_CACHE_DIR: str = Field(default=str(PROJECT_ROOT / "data" / "cache"))
    COLLECT_BARS: int = Field(default=1500, description="종목당 수집할 일봉 수")
    COLLECT_FLOW_DAYS: int = Field(default=1500, description="종목당 수집할 수급 일수")

    # ------------------------------------------------------------------ 유니버스
    UNIVERSE: List[str] = Field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    UNIVERSE_SIZE: int = Field(default=20, description="일봉 랭킹으로 추릴 실제 매매 대상 종목 수")
    MIN_TRADING_VALUE: float = Field(default=3_000_000_000, description="최소 20일 평균 거래대금(원)")

    # ------------------------------------------------------------------ 전략/AI
    MODEL_DIR: str = Field(default=str(PROJECT_ROOT / "models"))
    MODEL_NAME: str = Field(default="alpha_gbdt")
    SIGNAL_BUY_THRESHOLD: float = Field(default=0.22, description="앙상블 점수(-1~1) 매수 진입 임계값")
    SIGNAL_SELL_THRESHOLD: float = Field(default=-0.15, description="보유 종목 청산 임계값")
    W_MODEL: float = Field(default=0.50, description="AI 모델 가중치")
    W_TECHNICAL: float = Field(default=0.30, description="기술적 팩터 가중치")
    W_CROSS_SECTION: float = Field(default=0.20, description="횡단면 랭크 가중치")
    LABEL_HORIZON: int = Field(default=5, description="라벨 예측 지평(봉 수)")
    LABEL_PT_ATR: float = Field(default=1.5, description="삼중배리어 익절 배리어(ATR 배수)")
    LABEL_SL_ATR: float = Field(default=1.0, description="삼중배리어 손절 배리어(ATR 배수)")
    CV_SPLITS: int = Field(default=5)
    CV_EMBARGO: int = Field(default=5, description="PurgedKFold embargo 봉 수")

    # ------------------------------------------------------------------ 리스크
    MAX_POSITIONS: int = Field(default=8, description="최대 동시 보유 종목 수")
    MAX_POSITION_WEIGHT: float = Field(default=0.18, description="단일 종목 최대 비중")
    MAX_SECTOR_WEIGHT: float = Field(default=0.35, description="섹터별 최대 비중")
    MAX_GROSS_EXPOSURE: float = Field(default=0.90, description="총 익스포저 상한(현금 제외 비중)")
    TARGET_VOLATILITY: float = Field(default=0.18, description="연율 목표 변동성(포지션 사이징 기준)")
    KELLY_FRACTION: float = Field(default=0.25, description="Fractional Kelly 계수(1/4 Kelly)")
    KELLY_MAX_WEIGHT: float = Field(default=0.20)
    STOP_LOSS_PCT: float = Field(default=0.035, description="종목 고정 손절 비율")
    TAKE_PROFIT_PCT: float = Field(default=0.09, description="종목 익절 목표")
    TRAILING_ATR_MULT: float = Field(default=2.2, description="ATR 트레일링 스톱 배수")
    TRAILING_ACTIVATE_PCT: float = Field(default=0.025, description="트레일링 스톱 활성화 수익률")
    MAX_HOLDING_BARS: int = Field(default=30, description="최대 보유 봉 수(시간 손절)")
    DAILY_LOSS_LIMIT: float = Field(default=0.03, description="일중 손실 한도(당일 시작 자산 대비)")
    MDD_SOFT_LIMIT: float = Field(default=0.07, description="소프트 MDD: 신규 진입 차단 + 익스포저 축소")
    MDD_HARD_LIMIT: float = Field(default=0.12, description="하드 MDD: 전량 청산 + 쿨다운")
    DERISK_RATIO: float = Field(default=0.5, description="소프트 트리거 시 유지할 익스포저 비율")
    COOLDOWN_MINUTES: int = Field(default=60, description="킬스위치 이후 신규 진입 금지 시간(분)")
    CRASH_DROP_PCT: float = Field(default=0.045, description="단기 급락 감지 임계(전량 청산 트리거)")
    CRASH_LOOKBACK: int = Field(default=10, description="급락 감지 룩백(봉 수)")

    # ------------------------------------------------------------------ 비용 모델
    FEE_BPS: float = Field(default=1.4, description="매매 수수료(bps). 편도")
    TAX_BPS: float = Field(default=15.0, description="증권거래세(bps). 매도시에만")
    SLIPPAGE_BPS: float = Field(default=5.0, description="슬리피지 가정(bps)")

    # ------------------------------------------------------------------ 운영
    INITIAL_CAPITAL: float = Field(default=100_000_000, description="Mock/백테스트 시작 자산")
    LOOP_INTERVAL: float = Field(default=3.0, description="메인 루프 주기(초)")
    LOG_LEVEL: str = Field(default="INFO")
    LOG_DIR: str = Field(default=str(PROJECT_ROOT / "logs"))
    STATE_DIR: str = Field(default=str(PROJECT_ROOT / "state"))
    DATA_DIR: str = Field(default=str(PROJECT_ROOT / "data"))
    LEDGER_PATH: str = Field(default=str(PROJECT_ROOT / "logs" / "ledger.jsonl"))

    # ------------------------------------------------------------------ Mock 설정
    MOCK_SPEED: float = Field(default=60.0, description="Mock 가속 시계 배속(1초 = N초)")
    MOCK_SEED: int = Field(default=20261012, description="합성 시세 재현성 시드")
    MOCK_HISTORY_BARS: int = Field(default=400, description="Mock 시작 시 사전 생성할 과거 봉 수")
    MOCK_FUTURE_BARS: int = Field(default=1200, description="Mock 진행 가능한 미래 봉 수")
    MOCK_PARTIAL_FILL_PROB: float = Field(default=0.25, description="부분체결 발생 확률")
    MOCK_REJECT_PROB: float = Field(default=0.01, description="임의 주문거부 확률(예외 경로 검증용)")

    # ------------------------------------------------------------------ 대시보드
    DASHBOARD_ENABLED: bool = Field(default=True)
    DASHBOARD_HOST: str = Field(default="127.0.0.1")
    DASHBOARD_PORT: int = Field(default=8500)

    # ------------------------------------------------------------------ validators
    @field_validator("KIS_ACCOUNT_NO")
    @classmethod
    def _check_account(cls, v: str) -> str:
        v = v.strip()
        if "-" not in v:
            # 10자리 연속 입력도 허용
            digits = "".join(ch for ch in v if ch.isdigit())
            if len(digits) == 10:
                return f"{digits[:8]}-{digits[8:]}"
            raise ValueError("KIS_ACCOUNT_NO 형식은 '12345678-01' 이어야 합니다")
        return v

    @field_validator("UNIVERSE", mode="before")
    @classmethod
    def _parse_universe(cls, v):
        if isinstance(v, str):
            items = [s.strip() for s in v.replace(";", ",").split(",")]
            return [s.zfill(6) for s in items if s]
        return v

    # ------------------------------------------------------------------ 파생 속성
    @property
    def cano(self) -> str:
        """종합계좌번호 앞 8자리."""
        return self.KIS_ACCOUNT_NO.split("-")[0]

    @property
    def acnt_prdt_cd(self) -> str:
        """계좌상품코드 2자리."""
        return self.KIS_ACCOUNT_NO.split("-")[1]

    @property
    def is_prod(self) -> bool:
        return self.KIS_ENV == "prod"

    @property
    def rest_base_url(self) -> str:
        return REST_DOMAIN_PROD if self.is_prod else REST_DOMAIN_VTS

    @property
    def ws_url(self) -> str:
        return WS_DOMAIN_PROD if self.is_prod else WS_DOMAIN_VTS

    @property
    def ws_notice_tr(self) -> str:
        return WS_TR_NOTICE_PROD if self.is_prod else WS_TR_NOTICE_VTS

    @property
    def effective_rate(self) -> float:
        """안전 마진을 적용한 실제 사용 유량."""
        return max(0.2, self.REST_RATE_LIMIT * self.RATE_LIMIT_SAFETY)

    @property
    def round_trip_cost(self) -> float:
        """왕복 거래비용(비율). 라벨링/임계값 산정의 기준."""
        return (2 * self.FEE_BPS + self.TAX_BPS + 2 * self.SLIPPAGE_BPS) / 10_000.0

    def tr(self, name: str) -> str:
        """논리적 이름 → 환경별 TR_ID."""
        try:
            prod, vts = TR_MAP[name]
        except KeyError as exc:  # pragma: no cover - 개발 중 오타 방지
            raise KeyError(f"알 수 없는 TR 이름: {name}") from exc
        return prod if self.is_prod else vts

    def sector_of(self, symbol: str) -> str:
        return SECTOR_MAP.get(symbol, "ETC")

    def ensure_dirs(self) -> None:
        for d in (self.LOG_DIR, self.STATE_DIR, self.DATA_DIR, self.MODEL_DIR,
                  self.TOKEN_CACHE_DIR, self.DATA_CACHE_DIR, self.flow_cache_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    def credentials_ready(self) -> bool:
        return bool(self.KIS_APP_KEY and self.KIS_APP_SECRET and not self.cano.startswith("0000"))

    def toss_ready(self) -> bool:
        return bool(self.TOSS_CLIENT_ID and self.TOSS_CLIENT_SECRET)

    @property
    def flow_cache_dir(self) -> str:
        return str(Path(self.DATA_CACHE_DIR) / "flow")

    def describe(self) -> str:
        mode = "MOCK(가상체결)" if self.USE_MOCK_MODE else ("REAL-PROD" if self.is_prod else "KIS 모의투자")
        return (
            f"mode={mode} | domain={self.rest_base_url} | account={self.cano[:4]}****-{self.acnt_prdt_cd} | "
            f"rate={self.effective_rate:.1f}/s | universe={len(self.UNIVERSE)} | dry_run={self.DRY_RUN}"
        )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """싱글턴 설정 인스턴스."""
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
