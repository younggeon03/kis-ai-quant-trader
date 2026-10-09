"""환경설정 (.env 기반, pydantic-settings v2).

`USE_MOCK_MODE` 하나로 Mock 엔진 ↔ 실제 KIS 서버를 전환한다.
TR_ID는 `settings.tr("order_buy")` 형태로 조회하며 모의(VTS)/실전(PROD) 환경에 따라
자동으로 다른 코드가 반환되므로 호출부 코드는 전혀 바뀌지 않는다.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from pydantic import Field, field_validator, model_validator
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
    UNIVERSE_FILE: str = Field(
        default="",
        description="코스피200·코스닥150 구성종목 JSON 경로(`scripts/refresh_universe.py` 산출물). "
                    "비어 있거나 파일이 없으면 DEFAULT_UNIVERSE(40종목)를 쓴다. 지수 구성종목은 "
                    "정기변경이 6월·12월 두 번뿐인 참조 데이터이므로 런타임에 받지 않는다")
    UNIVERSE_SIZE: int = Field(
        default=120,
        description="일봉 랭킹으로 추릴 신호 계산 대상 종목 수. 20이면 거래대금 상위 "
                    "대형주만 남아 소형주 꼬리를 노릴 수 없다(백테스트는 전체를 보는데 "
                    "실거래만 20종목이면 두 경로가 갈라진다). 실시간 시세는 별도로 "
                    "WS_MAX_SUBSCRIPTIONS 만큼만 구독한다(main.py::_refresh_watchlist)")
    MIN_TRADING_VALUE: float = Field(default=3_000_000_000, description="최소 20일 평균 거래대금(원)")
    MIN_HISTORY_BARS: int = Field(
        default=1350,
        description="이 봉 수 미만의 이력을 가진 종목은 패널에서 제외한다. align_panel이 "
                    "공통 날짜 교집합을 잡으므로 상장이 늦은 종목 하나가 전체 구간을 "
                    "잘라낸다(신규 상장이 섞이면 수십 봉으로 붕괴). 실제 적용 하한은 "
                    "min(이 값, 중앙값×0.9)이라 짧은 패널은 보호된다. 284종목 실측 "
                    "트레이드오프: 하한 750이면 239종목이지만 교집합이 764봉(30봉 구간 8개), "
                    "1350이면 219종목에 1353봉(구간 28개)이다. 늦게 상장한 소수 종목이 "
                    "전체 구간을 잘라내므로 하한을 높이는 쪽이 유리하다")

    # ------------------------------------------------------------------ 전략/AI
    MODEL_DIR: str = Field(default=str(PROJECT_ROOT / "models"))
    MODEL_NAME: str = Field(default="alpha_gbdt")
    SIGNAL_BUY_THRESHOLD: float = Field(default=0.22, description="앙상블 점수(-1~1) 매수 진입 임계값")

    # 진입 방식 — "model"(앙상블 점수 임계값) / "catalyst"(거래대금 폭증 조건)
    #
    # **`catalyst`는 모델을 쓰지 않는다.** 측정된 선정 스킬이 0에 가깝기 때문이다:
    # 분봉 모델 AUC 0.5585(표본을 2.6배로 늘리자 0.6063→0.5661→0.5585로 하락),
    # 분봉 선정은 **무작위보다 나빴다**(선택 −0.366% vs 전체 −0.213%), 일봉 전략
    # 승구간 11/28 vs 동일가중 매수보유 21/28.
    #
    # 대신 꼬리 확률이 가장 큰 **조건**을 쓴다. `scripts/tail_grid.py` 전수 탐색
    # (지문 22d557320d0e640d · 284종목 · 조건 6종 x 종목수 4 x 회전 5 · 시행 8만):
    #
    # | 조건 | 구조 | 30일 P(+100%) @스킬0 | P(-50%) |
    # |---|---|---|---|
    # | **거래대금 10배** | **2종목 x 5라운드(6봉)** | **1.67%** | **0.03%** |
    # | 거래대금 10배 | 2종목 x 3라운드(10봉) | 1.57% | 0.00% |
    # | 일간 +10% | 2종목 x 1라운드(30봉) | 1.15% | 0.06% |
    # | 일간 +15% | 2종목 x 5라운드(6봉) | 1.06% | 0.89% |
    # | (기준선) 스윙 8종목 분산 | — | 0.96% | — |
    # | (기준선) 분봉 단타 | — | **0.00%** | — |
    #
    # 거래대금 조건은 꼬리가 더 두꺼운데 파산 확률이 거의 0이다 — 급등률 조건보다
    # 양쪽이 다 낫다. 그리고 조건 없이(`무조건`) 30봉 단일 종목 P(+100%)는 0.95%인데
    # 조건을 걸면 3.04~4.06%로 **3배**가 된다.
    ENTRY_MODE: str = Field(
        default="model",
        description="진입 방식: model=앙상블 점수 임계값 / catalyst=거래대금 폭증 조건")
    CATALYST_BURST_MULT: float = Field(
        default=10.0,
        description="catalyst 진입의 거래대금 폭증 하한(20봉 중앙값 대비 배수). "
                    "turnover_burst_20 >= log1p(이 값) 을 요구한다")
    SIGNAL_SELL_THRESHOLD: float = Field(default=-0.15, description="보유 종목 청산 임계값")
    W_MODEL: float = Field(default=0.50, description="AI 모델 가중치")
    W_TECHNICAL: float = Field(default=0.30, description="기술적 팩터 가중치")
    W_CROSS_SECTION: float = Field(default=0.20, description="횡단면 랭크 가중치")
    LABEL_HORIZON: int = Field(default=5, description="라벨 예측 지평(봉 수)")

    # --- 라벨 방식 ---
    # "triple_barrier": 방향 — 상단/하단 배리어 중 어디에 먼저 닿는가 (기본)
    # "magnitude":      크기 — 30봉 안에 +50% 이상 오르는가 (대회 꼬리 전략)
    #
    # 방향 라벨은 상승확률 0.7인 +5% 종목을 0.5인 +150% 종목보다 높게 평가한다.
    # 심사가 절대수익 top10이면 그 선택이 반대다. 급등 포착 연구에서 모든 소재
    # 조건의 중앙값이 음수인데 P(+100%)는 1.6~3.1배였다 — 평균 기준으로 배우면
    # 쓸 만한 조건을 전부 버린다(strategy/ai_model.py::make_magnitude_labels).
    LABEL_MODE: str = Field(
        default="triple_barrier",
        description="라벨 방식 — triple_barrier(방향) | magnitude(크기)")
    LABEL_MAGNITUDE_HORIZON: int = Field(
        default=30, description="크기 라벨 지평(봉). 대회 거래일 수와 맞춘다")
    LABEL_MAGNITUDE_THRESHOLD: float = Field(
        default=0.50,
        description="크기 라벨 문턱. 실측 양성비율 — +50%에서 3.9%, +100%에서 1.0%")
    LABEL_PT_ATR: float = Field(default=1.5, description="삼중배리어 익절 배리어(ATR 배수)")
    LABEL_SL_ATR: float = Field(default=1.0, description="삼중배리어 손절 배리어(ATR 배수)")
    USE_META_MODEL: bool = Field(
        default=False,
        description="2차 모델(메타라벨링) 사용 여부. 정밀도는 오르지만 신호의 절반을 막아 "
                    "합성 데이터에서는 Sharpe가 떨어졌다(1.744→0.814). 실데이터에서 재평가할 것")
    FLOW_LAG_DAYS: int = Field(
        default=1,
        description="수급 피처를 며칠 늦춰 쓸지. 토스 응답의 updatedAt을 보면 T일 수급은 "
                    "T일 저녁~T+4에 확정된다. T일 종가로 판단하는 시점에는 존재하지 않으므로 "
                    "최소 1일은 늦춰야 한다. 0으로 두면 룩어헤드가 생긴다")
    CV_SPLITS: int = Field(default=5)
    CV_EMBARGO: int = Field(default=5, description="PurgedKFold embargo 봉 수")

    # ------------------------------------------------------------------ 리스크
    MAX_POSITIONS: int = Field(default=8, description="최대 동시 보유 종목 수")
    MAX_POSITION_WEIGHT: float = Field(default=0.18, description="단일 종목 최대 비중")
    MAX_SECTOR_WEIGHT: float = Field(default=0.35, description="섹터별 최대 비중")
    MAX_GROSS_EXPOSURE: float = Field(default=0.90, description="총 익스포저 상한(현금 제외 비중)")
    TARGET_VOLATILITY: float = Field(default=0.18, description="연율 목표 변동성(포지션 사이징 기준)")

    # --- 사이징 모드 ---
    # "vol_target": 변동성 타게팅 × 켈리 × 신호 × 레짐 (위험 분산 설계, 기본값)
    # "fixed":      종목당 FIXED_POSITION_WEIGHT 고정 (꼬리 전략)
    #
    # 대회 심사는 절대수익 top10이고 보상이 계단함수다. 그때 최적화 대상은 E[R]이
    # 아니라 P(R > 임계)이고, 두 목적의 최적해가 다르다. 실측에서 Top-K 2종목이
    # 평균 익스포저 5.8%였는데 역산하면 vol_w(0.34) × kelly(0.40) × 신호(0.3) ×
    # 레짐(0.8) ≈ 3.3%로 공식 자체가 만든 값이었다(상한 18%는 닿지도 않았다).
    # 파라미터로 90%에 도달하려면 TARGET_VOLATILITY가 2.5 같은 값이 되어야 하므로
    # 모드를 분리한다. `strategy/risk_manager.py::evaluate_entry()`가 분기한다.
    # --- 주문 지정가 폭 ---
    # 매수와 매도가 비대칭이다. 매수는 올라가는 가격을 추격해야 하고(실측 +0.4%에서
    # 체결률 61.5%), 매도는 내려가는 가격에 양보하는 쪽이다(−0.4%에서 99.71%).
    # 근거 표는 `main.py::_submit()` 주석에 있다.
    BUY_LIMIT_BAND: float = Field(
        default=0.02,
        description="매수 지정가 상한 폭. 체결률 85.7%·실제 체결가 중앙 +0.313%")
    SELL_LIMIT_BAND: float = Field(
        default=0.004,
        description="매도 지정가 하한 폭. 0.4%에서 체결률 99.71%로 충분하다")

    # --- Mock 가상시각 분 단위 전진 (테스트 전용) ---
    #
    # Mock은 `_mock_epoch`(2025-01-02 09:00)에 **일 단위만** 더하므로 가상 시각이
    # 항상 09:00이다. 그래서 `main.py`의 시각 기반 배선 — 아침 스캔(09:05),
    # 마감 정리(15:15), 신선도 경과분 — 이 **한 번도 실행되지 않는다.**
    #
    # 리플레이는 `IntradayWatcher`만 검증한다. 마감 정리가 실거래에서 안 돌면
    # 오버나이트를 들고 가고 하방이 밤새 열리는데, 그 배선을 10/12 장중에 처음
    # 보게 되는 것은 받아들일 수 없다.
    #
    # 0보다 크면 **날짜를 고정하고 루프마다 이 분수만큼 전진**시킨다. 30으로 두면
    # 13루프에 09:00~15:30을 지나므로 스캔·마감 정리가 모두 발동한다.
    # **기본값 0은 기존 동작(일 단위)이다** — 켜면 일봉 경로의 봉 전진이 멈추므로
    # 장중 배선 검증에만 쓴다.
    MOCK_MINUTES_PER_LOOP: int = Field(
        default=0,
        description="Mock 가상시각을 루프당 N분 전진(0=기존 일 단위). 장중 시각 "
                    "로직 검증 전용")

    # --- 장중 분봉 전략 (strategy/intraday.py) ---
    # 일봉 엔진과 별개 경로다. 장중에 조건이 충족되는 순간 진입하고 그날 안에 청산한다.
    # 기본 OFF — 켜면 일봉 신호 대신 장중 돌파가 진입을 결정한다.
    USE_INTRADAY: bool = Field(
        default=False, description="장중 돌파 전략 사용(분봉). OFF면 일봉 신호로 진입")
    INTRADAY_TRIGGER_PCT: float = Field(
        default=0.03, description="장중 돌파 기준 — 현재가/전일종가 − 1")
    # 장중 진입 판정 방식.
    #
    #   breakout : 현재가 >= 전일종가 x (1+INTRADAY_TRIGGER_PCT) AND 분당 거래량 하한
    #   burst    : **당일 누적 거래대금** >= 20봉 중앙값 x INTRADAY_BURST_MULT
    #
    # `burst`는 일봉 catalyst와 **같은 조건을 장중 누적으로** 본다. 일봉 조건은
    # `turnover = 종가 x 거래량`이라 장 마감 후에야 확정되고, 그래서 T+1 시가에
    # 들어가며 폭증일을 통째로 버린다 — 거래대금 10배 사건 4,902건 실측에서
    # 폭증일 시가→종가가 평균 **+5.29%**(승률 71.8%)인데 현재 먹는 구간
    # (T+1시가→T+6종가)은 중앙 **−1.50%**다. 버리는 구간이 평균 +6.2%다.
    #
    # **`burst` 모드는 가격 조건을 걸지 않는다.** "거래대금은 10배인데 가격은 안
    # 오른 날"이 분별 대상의 일부이고, 가격 조건을 같이 걸면 그 집단이 사라진다
    # (분봉 수집을 가격 기준으로만 해서 커버리지가 87.2%였던 것과 같은 문제다).
    # ------------------------------------------------- 호가·체결 전방 수집
    #
    # **이번 대회에는 쓸 수 없다.** 호가·체결강도는 과거 이력을 받을 방법이 없고
    # (REST에 날짜 파라미터가 없고 WS는 구독 시점부터만), 10/9 휴장이라 대회 전
    # 수집 가능일이 10/8 하루뿐이다. 하루치로는 학습도 검증도 안 된다.
    # **다음을 위해 지금부터 쌓는다.**
    #
    # 구독 한도 41건을 체결과 나눠 쓰므로 호가까지 켜면 실질 감시 종목이 절반이
    # 된다. 다만 catalyst는 2종목만 보므로 슬롯이 남아 사실상 공짜다.
    # 기록 실패는 매매를 멈추지 않는다(`utils/orderbook_log.py`).
    COLLECT_ORDERBOOK: bool = Field(
        default=False,
        description="호가 10단·체결 테이프를 data/cache/{orderbook,tape}/ 에 쌓는다. "
                    "다음 대회용 전방 수집이며 이번 대회 전략에는 쓰이지 않는다")
    ORDERBOOK_SAMPLE_SEC: float = Field(
        default=5.0,
        description="호가·테이프 샘플링 간격(초). 틱마다 저장하면 하루 수백만 행이고 "
                    "대부분 직전과 같다")
    INTRADAY_ENTRY_MODE: str = Field(
        default="breakout",
        description="장중 진입 판정: breakout=가격 돌파+분당 거래량 / "
                    "burst=당일 누적 거래대금 폭증")
    INTRADAY_BURST_MULT: float = Field(
        default=10.0,
        description="burst 모드의 거래대금 폭증 배수(20봉 중앙값 대비). "
                    "일봉 CATALYST_BURST_MULT 와 같은 값을 쓰는 것이 기본이다")
    # 15:15 마감 정리를 끌 수 있다. 끄면 포지션이 **일봉 청산 경로**
    # (`risk_manager`: 손절 + 감쇠 3종 + 시간손절)로 넘어가 다일 보유가 된다.
    #
    # **오버나이트 0건은 지금까지 설계 불변 조건이었다.** 끄는 근거: 당일 청산은
    # 변동을 하루로 묶고 꼬리는 변동에서 나온다(1회 표준편차 분봉 5.3% vs 일봉
    # 6봉 보유는 훨씬 크다). 분봉 단타가 모든 레버에서 `P(+100%)` 0.00%였던 원인이
    # **진입이 아니라 청산 쪽**일 수 있고, 둘을 분리해 측정한 적이 없다.
    INTRADAY_FLATTEN: bool = Field(
        default=True,
        description="15:15 전량 청산. false 면 일봉 청산 경로로 넘겨 다일 보유한다")
    INTRADAY_VOL_MULT: float = Field(
        default=12.0, description="돌파 분의 거래량 ÷ 전일 분당 평균 하한")
    # 왜 12배인가 (분봉 리플레이 250거래일 · 거래 약 500건)
    #
    # | 하한 | 승률 | 평균/회 | 30일 중앙 | P(+100%) | P(−50%) |
    # |---|---|---|---|---|---|
    # | 1.5 | 42.4% | **−0.436%** | −13.3% | 0.08% | 1.84% |
    # | 8   | 45.1% | +0.264% | +3.6% | 0.39% | 0.17% |
    # | 12  | 46.5% | **+0.557%** | **+11.7%** | **1.18%** | 0.12% |
    # | 20  | 49.1% | +0.507% | +10.2% | 0.89% | 0.14% |
    #
    # 하한이 낮으면 **장 시작 직후 노이즈가 전부 통과한다.** 실측 장중 프로파일이
    # U자형이라 10분 경과 시점에 이미 하루 거래량의 9.7%(균등 가정의 3.8배)가
    # 돌아간다. 1.5배는 그 구간에서 사실상 조건이 없는 것과 같다.
    #
    # **단서 둘.** 5개 값을 시험해 최고를 골랐으므로 선택편향이 있고 1.18%는
    # 부풀려진 값이다(이 프로젝트가 DSR로 보정하는 상황이다). 그리고 1.5→12가
    # 단조롭지 않다 — 3배가 1.5배보다 나빴다(−0.738%). 단조롭지 않은 파라미터
    # 반응은 노이즈를 맞추고 있다는 신호일 수 있다.
    #
    # 다만 **1.5배가 명확히 나쁘다는 것**은 신뢰할 만하다. 500건에서 평균이 음수이고
    # 대안 전부가 양수다.
    # 청산 기본값은 '규칙 없음 + 마감 정리'다. 부분 데이터 격자에서 종가 청산이
    # 꼬리가 가장 두꺼웠다 — 모든 청산 규칙이 분산을 깎고 분산이 꼬리를 만든다.
    # 전 표본 격자 결과가 나오면 이 값을 갱신한다.
    INTRADAY_TARGET_PCT: float = Field(default=0.0, description="장중 익절(0=미적용)")
    INTRADAY_STOP_PCT: float = Field(default=0.0, description="장중 손절(0=미적용)")
    INTRADAY_TRAIL_PCT: float = Field(default=0.0, description="장중 트레일링 폭(0=미적용)")
    INTRADAY_TRAIL_ACTIVATE_PCT: float = Field(
        default=0.0, description="트레일링 발동 수익률")
    # 아침 후보 스캔 — 유니버스 스냅샷에 의존하지 않기 위한 장치.
    # `config/universe.json`은 생성 시점(2026-09-29)에 거래대금 상위였던 종목이고,
    # 거래대금 하한은 측정에서 **"그날 뭔가 돌고 있던 종목"을 남기는 필터**로
    # 작동했다(하한을 낮추면 p95 변동성이 12.80%→5.32%로 떨어진다). 즉 그 목록은
    # 생성 시점의 테마다. 대회 기간엔 다른 종목이 돈다.
    #
    # 그래서 매일 아침 전 후보의 상승률·거래량을 한 번 훑어 **오늘 움직이는** 종목을
    # 구독한다. 구독 한도가 41건이라 상시 전 종목 감시는 불가능하다.
    # 후보 명부와 교체 — 선택이 실제로 일어나게 하는 장치.
    #
    # 처음에는 발동 신호를 매 분 비워서 **같은 분에 동시 발동한 것들 사이에서만**
    # 골랐다. 실측에서 하루 돌파가 중앙 14건인데 경과분이 중앙 15분·p75 82분·
    # p90 253분으로 시간에 흩어져 있고, 2번째 돌파가 5분 안에 나는 날이 76.9%였다.
    # 슬롯 2개가 09:00~09:01에 차고 나머지 12건은 선택 기회조차 없이 버려졌다 —
    # 사실상 "하루의 첫 2건"을 사는 전략이었고 모델도 신선도도 쓰이지 않았다.
    INTRADAY_CANDIDATE_TTL: int = Field(
        default=1,
        description="후보 명부 보관 시간(분). 같은 분의 넘침만 처리한다")
    # 왜 1분인가 — 30분으로 두면 낡은 후보를 현재가로 사게 된다 (리플레이 250거래일)
    #
    # | TTL | 평균/회 | P(+100%) |
    # |---|---|---|
    # | 명부 없음(기존) | **+0.557%** | **1.18%** |
    # | 30분 (교체 OFF) | +0.236% | 0.36% |
    #
    # 교체를 껐는데도 나빠졌다. 원인은 **점수와 체결가의 시점이 어긋나는 것**이다.
    # 09:00에 3건이 발동해 2건만 사면 3번째가 명부에 남는다. 신선도는 "일찍 온
    # 돌파가 좋다"를 내장하므로(경과분 항목) 그 낡은 후보가 20분 뒤에 온 신선한
    # 후보를 **점수로 이긴다.** 그런데 진입가는 20분 뒤의 가격이다. 점수는 과거
    # 시점의 것이고 체결은 현재 시점이라, 명부가 길수록 그 괴리가 커진다.
    #
    # 1분이면 같은 분에 동시 발동한 것들의 넘침만 처리하고, 그 경우 점수와 체결가의
    # 시점이 거의 같다. 명부를 아예 없애지 않는 이유는 장중에 슬롯이 비면(손절·익절을
    # 켤 경우) 직전 분의 후보로 채울 수 있기 때문이다.
    INTRADAY_SWAP_MARGIN: float = Field(
        default=0.10,
        description="보유 종목과 교체하려면 점수가 이만큼 높아야 한다. "
                    "교체는 왕복비용 0.278%를 즉시 지불하므로 여유가 필요하다")
    INTRADAY_MAX_SWAPS: int = Field(
        default=0, description="하루 최대 교체 횟수. 0이면 교체하지 않는다(선착순)")
    # 왜 0인가 — 교체는 신선도 점수와 구조적으로 충돌한다 (리플레이 250거래일)
    #
    # | | 교체 없음 | 교체 허용(한도 2회) |
    # |---|---|---|
    # | 평균/회 | **+0.557%** | +0.104% |
    # | P(+100%) | **1.18%** | 0.33% |
    # | 실제 교체 | — | 18건 (가능 500건의 3.6%) |
    #
    # 신선도는 "일찍 온 돌파가 좋다"를 내장한다(경과분 항목). 09:00 돌파는 그
    # 항목이 만점이라 나중에 온 후보가 이길 수 없고, 그래서 교체가 250일에 18건
    # 뿐이었다. 그 18건은 **왕복비용을 즉시 지불하고 신선도가 낮은 종목으로
    # 갈아탄 것**이라 양쪽으로 불리했고, 평균을 +0.557%에서 +0.104%로 끌어내렸다.
    #
    # 교체는 정의상 "나중에 온 것으로 바꾼다"인데 점수는 "먼저 온 것이 좋다"고
    # 한다. 둘을 같이 쓸 수 없다. 실측이 이른 돌파를 지지하므로(상위 25% 36.8분
    # vs 하위 75% 74.9분) 선착순을 받아들인다.
    #
    # 명부(`INTRADAY_CANDIDATE_TTL`)는 남겨 둔다 — 장중에 슬롯이 비면 최근 후보로
    # 채울 수 있고, 교체가 꺼진 상태에서는 기존 동작과 같다.

    INTRADAY_SCAN_HHMM: str = Field(
        default="0905", description="아침 후보 스캔 시각(HHMM). 이 시각 이후 첫 루프에 1회")
    INTRADAY_SCAN_TOP: int = Field(
        default=40, description="스캔 결과 상위 몇 종목을 구독할지(WS 한도 이하)")

    INTRADAY_FLATTEN_HHMM: str = Field(
        default="1515", description="마감 정리 시각(HHMM). 오버나이트를 만들지 않는다")

    SIZING_MODE: str = Field(
        default="vol_target",
        description="포지션 사이징 방식 — vol_target(변동성 타게팅) | fixed(고정 비중)")
    FIXED_POSITION_WEIGHT: float = Field(
        default=0.42,
        description="고정 비중 모드의 종목당 목표 비중")
    # 왜 45%가 아니라 42%인가 (실측)
    #
    # 측정 최적은 2종목 x 45%였는데, 그대로 두면 45+45=90%가 `MAX_GROSS_EXPOSURE`
    # (0.9)와 **정확히 같아** 수수료·슬리피지만 얹혀도 넘친다. 여기에 부분 체결이
    # 겹치면 더 나쁘다 — 브로커 가용금액은 `cash - reserved`이고, 첫 주문이 체결을
    # 마치기 전에는 예약금이 잡혀 있어 두 번째 진입이 "주문가능금액 부족"으로
    # 거부된다(Mock 스모크에서 실제로 발생).
    #
    # 거부가 나면 **조용히 1종목 구조가 된다.** `tail_optimizer` 기준 P(+100%)가
    # 2종목 1.04% → 1종목 0.50%로 반토막이고, 측정한 분포와 실거래가 갈라지는
    # 가장 나쁜 형태다. 그래서 42+42=84%로 6%p 여유를 둔다. 종목당 노출이 3%p
    # 줄어 꼬리가 약간 얇아지는 것이 그 대가다.
    KELLY_FRACTION: float = Field(default=0.25, description="Fractional Kelly 계수(1/4 Kelly)")
    KELLY_MAX_WEIGHT: float = Field(default=0.20)
    STOP_LOSS_PCT: float = Field(
        default=0.12,
        description="종목 고정 손절 비율. 3.5%는 유니버스 중앙값 일간 변동성(3.3%)과 거의 "
                    "같아 무작위 진입에서도 10봉 내 62.6%가 노이즈로 손절됐다. 실데이터 "
                    "21구간에서 12%로 넓히자 Sharpe -0.147 → 0.425, 누적 -5.2% → +12.1%. "
                    "**이 프로젝트에서 검증된 단일 최대 개선이다**")
    TAKE_PROFIT_PCT: float = Field(
        default=0.09,
        description="종목 익절 목표. 0이면 익절하지 않는다. **제거하지 않는다** — 실데이터 "
                    "21구간 재측정에서 손절 12%만 적용한 쪽(Sharpe 0.425 / +12.1%)이 "
                    "익절까지 제거한 쪽(0.263 / +7.0%)보다 나았고, 익절 제거 단독은 "
                    "원 설정보다도 나빴다(-0.147 → -0.163)")
    TRAILING_ATR_MULT: float = Field(default=2.2, description="ATR 트레일링 스톱 배수")
    # 트레일링과 MAX_HOLDING_BARS는 절제 실험에서 현행 유지가 더 좋았다.
    # 단독 효과(원 설정 -0.147 대비): 트레일링 제거 -0.089, 보유 10봉 단축 -0.560.
    # (0으로 두면 각각 비활성화된다 — risk_manager.evaluate_exits 참고)
    TRAILING_ACTIVATE_PCT: float = Field(default=0.025, description="트레일링 스톱 활성화 수익률")
    MAX_HOLDING_BARS: int = Field(default=30, description="최대 보유 봉 수(시간 손절)")
    DAILY_LOSS_LIMIT: float = Field(default=0.03, description="일중 손실 한도(당일 시작 자산 대비)")
    # **`USE_INTRADAY=true`로 쓸 때는 반드시 올려야 한다.**
    #
    # `risk_manager.py`가 이것을 **하드 트리거와 동일 취급**해 HALT로 전이하고
    # 전량 청산 + 쿨다운을 건다. 분봉 전략은 2종목 × 42% = 익스포저 84%이고
    # 급등주의 일간 변동이 크므로, 보유 종목이 평균 3.6%만 내려도 포트폴리오가
    # −3%가 되어 **그날 거래가 끝난다.**
    #
    # 세션 중에 실제로 놓쳤던 자리다. MDD 한도(SOFT/HARD)는 꺼 뒀는데 이 게이트만
    # 살아 있었고, **리플레이가 리스크 매니저를 모사하지 않아 측정에 나타나지
    # 않았다.** 측정과 실거래가 갈라지는 전형적인 지점이다.
    #
    # 0.03은 일봉 스윙 기준의 기본값이다. 분봉으로 돌릴 때는 `.env`에서 1.0으로
    # 사실상 해제한다(킬스위치는 남겨 단일 종목 붕괴만 막는다).
    MDD_SOFT_LIMIT: float = Field(default=0.07, description="소프트 MDD: 신규 진입 차단 + 익스포저 축소")
    # **`USE_INTRADAY=true`로 쓸 때는 MDD·익절·트레일링·급락을 전부 해제해야 한다.**
    #
    # 분봉 설계는 "청산 없음 + 15:15 마감 정리"인데 `risk.evaluate_exits()`가 분봉
    # 포지션에도 적용되고, PHASE2(10/19~11/20, 대회의 25일)에서는 손절류가 모두
    # 켜진다(PHASE1은 `use_stop_loss=False`로 건너뛴다).
    #
    # | 규칙 | 기본값 | 분봉에 미치는 영향 |
    # |---|---|---|
    # | `TAKE_PROFIT_PCT` | 0.09 | **상승을 9%에서 자른다.** 측정에서 익절은 꼬리를 구조적으로 없앤다(익절 3%는 스킬 30%에서도 P(+100%)=0) |
    # | `TRAILING_ACTIVATE_PCT` | 0.025 | +2.5%만 올라도 발동해 되돌림에 바로 잘린다 |
    # | `MDD_SOFT_LIMIT` / `HARD` | 0.07 / 0.12 | 익스포저 84%에서 쉽게 닿는다. 소프트는 신규 진입 차단, 하드는 전량 청산 + 쿨다운 |
    # | `CRASH_DROP_PCT` | 0.045 | 급등주의 정상 변동폭이다 |
    # | `DAILY_LOSS_LIMIT` | 0.03 | 보유 종목이 평균 3.6%만 내려도 그날 거래가 끝난다 |
    #
    # `STOP_LOSS_PCT`(0.12)는 남긴다 — 단일 종목 붕괴를 막는 마지막 선이고
    # 측정에서 수익을 깎지 않았다(−0.147 → 0.425).
    #
    # **세션 중에 한 번 잘못 봤다.** "MDD 한도가 꺼져 있다"고 판단했는데, 존재하지
    # 않는 필드명(`SOFT_MDD_LIMIT`/`HARD_MDD_LIMIT`)을 조회해 `None`을 받은 것이었다.
    # 설정값을 확인할 때는 **필드가 실제로 있는지** 먼저 봐야 한다.
    MDD_HARD_LIMIT: float = Field(default=0.12, description="하드 MDD: 전량 청산 + 쿨다운")
    DERISK_RATIO: float = Field(default=0.5, description="소프트 트리거 시 유지할 익스포저 비율")
    COOLDOWN_MINUTES: int = Field(default=60, description="킬스위치 이후 신규 진입 금지 시간(분)")
    CRASH_DROP_PCT: float = Field(default=0.045, description="단기 급락 감지 임계(전량 청산 트리거)")
    CRASH_LOOKBACK: int = Field(default=10, description="급락 감지 룩백(봉 수)")

    # ------------------------------------------------- 감쇠 기반 청산 (exit_rules)
    #
    # 규칙 구현은 `strategy/exit_rules.py`에 있다. 청산 규칙이 관문 코드
    # (`risk_manager.evaluate_exits`) 안에 하드코딩돼 있어서, 파인튜닝 대상인데
    # 가장 조심해야 하는 곳을 건드려야 했다. 그래서 분리했다.
    #
    # `EXIT_RULES`는 **순서 있는 목록**이다. 같은 봉에서 둘이 동시에 충족될 때
    # 어느 쪽으로 기록되느냐가 달라지므로 순서도 설정으로 노출한다.
    #
    # | 규칙 | 보는 것 | 근거 |
    # |---|---|---|
    # | `crash` | 최근 `CRASH_LOOKBACK`봉 고점 대비 `CRASH_DROP_PCT` 하락 | 급등주 붕괴는 단발로 온다 |
    # | `trailing` | 고점 − `TRAILING_ATR_MULT` x ATR | 절제 실험에서 **트레일링 제거가 두 실행 모두 악화**(−0.255 / −0.089) |
    # | `vol_decay` | 거래대금이 보유 중 고점의 `EXIT_VOL_DECAY_RATIO` 아래 | 신규. 폭증이 식으면 상승 동력이 사라진다 |
    #
    # 기본값을 `crash,trailing`으로 둔 것은 **기존 동작을 보존**하기 위해서다
    # (분리 전 `evaluate_exits`의 순서가 급락 → 트레일링이었다). `vol_decay`는
    # 측정 전이라 기본 OFF다(`EXIT_VOL_DECAY_RATIO=0`).
    EXIT_RULES: str = Field(
        default="crash,trailing",
        description="감쇠 청산 규칙의 순서 있는 목록(쉼표 구분). "
                    "가능: crash, trailing, vol_decay / 빈 문자열이면 전부 끈다")
    # `vol_alive` — **기간이 아니라 상태가 보유를 정한다.**
    #
    # 거래대금이 **20봉 중앙값**(진입 조건과 같은 분모) 대비 이 배수 아래로
    # 떨어지면 청산한다. `vol_decay`와 분모만 다르고 그것이 전부다 —
    # `vol_decay`는 "보유 중 고점"을 분모로 써서 폭증일 자신이 고점이 되어
    # 발동률 80.9%가 됐다.
    #
    # 측정(3,932건 · 라운드 5회 고정): 중앙 보유일수는 고정 6봉과 같은 6일인데
    # 단일 2배 확률 0.64% → **0.86%**(2배 사건 25 → 34건), 포트 P(+100%)
    # 2.21% → **4.06%**. 죽은 종목은 일찍 내보내고 살아 있는 종목은 더 들고 간다.
    #
    # **기준을 올려도 좋아지지 않는다.** x1.5/x2/x3/x5에서 단일 2배 비율은
    # 0.94% → 0.69%로 떨어지고, 포트 수치가 오르는 것은 라운드 환산이 4 → 10으로
    # 늘어난 측정 근사의 artifact다. x2가 라운드 5회로 검증된 기준선과 같아
    # 왜곡이 없는 유일한 지점이다. 쌍별 검정에서 기준 간 차이는 전부 구분 불가다.
    EXIT_VOL_ALIVE_MULT: float = Field(
        default=0.0,
        description="거래대금이 20봉 중앙값의 이 배수 아래면 청산(0=끔). "
                    "측정 최적 2.0 — 올려도 좋아지지 않는다")
    EXIT_VOL_ALIVE_MIN_BARS: int = Field(
        default=1,
        description="vol_alive 의 최소 보유 봉 수")
    EXIT_VOL_DECAY_RATIO: float = Field(
        default=0.0,
        description="거래대금이 보유 중 고점의 이 비율 아래로 줄면 청산(0=끔). "
                    "예: 0.3 이면 고점의 30% 미만")
    EXIT_VOL_DECAY_MIN_BARS: int = Field(
        default=2,
        description="거래량 감소 청산의 최소 보유 봉 수. 진입 직후 거래가 적다고 "
                    "바로 털리지 않게 한다")
    USE_TOP_K: bool = Field(
        default=False,
        description="점수 상위 K개를 골라 목표 익스포저를 채운다. False면 기존 절대 "
                    "임계값(SIGNAL_BUY_THRESHOLD) 방식을 쓴다. 임계값 방식은 넘는 "
                    "종목이 없으면 현금을 들기 때문에 상승장에서 참여를 못 한다")
    MIN_HOLDING_BARS: int = Field(
        default=0,
        description="최소 보유 봉 수. 이 기간에는 순위가 밀려도 교체하지 않는다. "
                    "순위는 매 봉 흔들리므로, 없으면 왕복비용 0.278%가 알파를 먹는다")
    KELLY_COLD_START_MULT: float = Field(
        default=0.40,
        description="거래 이력이 부족할 때 쓰는 Kelly 배수. 대회처럼 기간이 짧으면 "
                    "표본이 모이지 않으므로 낮은 값은 노이즈에 맞추는 것과 같다")

    # ------------------------------------------------------------------ 대회
    # 날짜가 비어 있으면 대회 로직 전체가 꺼지고 평소 동작을 유지한다.
    COMP_START: str = Field(default="", description="대회 시작일 YYYY-MM-DD")
    COMP_END: str = Field(default="", description="대회 종료일 YYYY-MM-DD")
    PHASE1_END: str = Field(
        default="",
        description="수익률 초기화 가능 마지막 날 YYYY-MM-DD. 이 날까지가 1단계다")
    PHASE1_TOP_K: int = Field(
        default=2,
        description="1단계 보유 종목 수(꼬리 베팅). 대회 규정 단일 종목 한도가 50%%이므로 "
                    "1종목이면 현금 50%%가 남는다. 2종목 x 50%%가 규정 안에서 가능한 "
                    "최대 집중이며, 1종목 100%% 몰빵은 규정 위반이다")
    PHASE2_TOP_K: int = Field(default=4, description="2단계 보유 종목 수(본 전략)")
    PHASE1_DISABLE_RISK: bool = Field(
        default=True,
        description="1단계에서 손절·MDD 게이트를 끈다. 초기화 옵션이 하방을 덮기 "
                    "때문이며, 이 논리는 1단계가 끝나면 성립하지 않는다. 감쇠 청산"
                    "(EXIT_RULES)은 전략의 기본 청산이라 끄지 않는다")

    # 대회 최소 요건 — 못 채우면 수익률과 무관하게 실격이다(모두 체결 기준).
    MIN_TRADE_COUNT: int = Field(default=0, description="최소 거래 횟수")
    MIN_TRADE_AMOUNT: float = Field(default=0.0, description="최소 매매금액(원)")
    MIN_TRADE_DAYS: int = Field(default=0, description="최소 매매일수(영업일)")
    MIN_TRADE_SYMBOLS: int = Field(default=0, description="최소 매매 종목 수")

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
        """섹터. 코드에 박힌 `SECTOR_MAP`을 우선하고, 없으면 유니버스 파일의 값을 쓴다."""
        hit = SECTOR_MAP.get(symbol)
        if hit:
            return hit
        return self._universe_blob().get("sectors", {}).get(symbol, "ETC")

    def _universe_blob(self) -> Dict[str, Any]:
        """`UNIVERSE_FILE`을 한 번만 읽어 캐시한다. 없거나 깨졌으면 빈 dict."""
        cached = getattr(self, "_universe_cache", None)
        if cached is not None:
            return cached
        blob: Dict[str, Any] = {}
        path = (self.UNIVERSE_FILE or "").strip()
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            try:
                blob = json.loads(p.read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass
            except (json.JSONDecodeError, OSError):
                # 조용히 넘기면 40종목으로 도는 이유를 찾기 어렵다.
                print(f"[Settings] 경고: UNIVERSE_FILE을 읽을 수 없습니다 ({p}) "
                      f"→ DEFAULT_UNIVERSE로 폴백")
        object.__setattr__(self, "_universe_cache", blob)
        return blob

    def resolved_universe(self) -> List[str]:
        """실제로 쓸 종목 목록.

        `UNIVERSE_FILE`이 지정되고 읽히면 그 안의 `tradable`(토스 시세가 확인된 종목)을
        쓰고, 아니면 `UNIVERSE`(기본 40종목)를 쓴다. 이 함수를 거치므로 매매 런타임은
        pykrx를 import하지 않는다.

        `UNIVERSE`는 설정 로드 시점에 이미 이 값으로 치환되므로 보통은 같다
        (`_apply_universe_file`). 명시적으로 확인할 때 쓴다.
        """
        blob = self._universe_blob()
        syms = [str(x) for x in (blob.get("tradable") or []) if str(x).strip()]
        return syms or list(self.UNIVERSE)

    def market_cap_of(self, symbol: str) -> float:
        """상장 시가총액(**원**). 유니버스 파일에 없으면 0.

        `scripts/refresh_universe.py`가 원 단위로 정규화해 저장한다(지수 CSV의
        `상장시가총액`은 백만원, 전종목시세의 `시가총액`은 원 단위로 서로 다르다).
        """
        return self._universe_num("market_cap", symbol)

    def shares_of(self, symbol: str) -> float:
        """상장주식수(주). 대회 규정의 10만주 하한 판정에 쓴다."""
        return self._universe_num("shares", symbol)

    def trading_value_of(self, symbol: str) -> float:
        """유니버스 기준일의 거래대금(원). 유동성 판정에 쓴다."""
        return self._universe_num("trading_value", symbol)

    def index_members(self) -> Dict[str, List[str]]:
        """지수별 구성종목. 최소 자격 요건("코스피200·코스닥150 중 5종목") 판정용이다.

        **매매 제한이 아니다.** 대회 규정상 KOSPI·KOSDAQ 전 개별 종목을 매매할 수 있다.
        """
        blob = self._universe_blob().get("indices") or {}
        return {str(k): [str(x) for x in (v or [])] for k, v in blob.items()}

    def _universe_num(self, key: str, symbol: str) -> float:
        try:
            return float(self._universe_blob().get(key, {}).get(symbol, 0.0))
        except (TypeError, ValueError):
            return 0.0

    @model_validator(mode="after")
    def _apply_universe_file(self) -> "Settings":
        """`UNIVERSE_FILE`이 있으면 `UNIVERSE`를 거기서 읽은 목록으로 치환한다.

        호출부(`s.UNIVERSE`를 쓰는 12군데)를 하나하나 고치면 누락이 생긴다.
        설정 로드 시점에 한 번 바꿔 두면 백테스트·수집·실매매가 자동으로 같은
        유니버스를 본다.
        """
        if (self.UNIVERSE_FILE or "").strip():
            syms = [str(x) for x in (self._universe_blob().get("tradable") or [])
                    if str(x).strip()]
            if syms:
                object.__setattr__(self, "UNIVERSE", syms)
        return self

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
