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
    # --- 장중 분봉 전략 (strategy/intraday.py) ---
    # 일봉 엔진과 별개 경로다. 장중에 조건이 충족되는 순간 진입하고 그날 안에 청산한다.
    # 기본 OFF — 켜면 일봉 신호 대신 장중 돌파가 진입을 결정한다.
    USE_INTRADAY: bool = Field(
        default=False, description="장중 돌파 전략 사용(분봉). OFF면 일봉 신호로 진입")
    INTRADAY_TRIGGER_PCT: float = Field(
        default=0.03, description="장중 돌파 기준 — 현재가/전일종가 − 1")
    INTRADAY_VOL_MULT: float = Field(
        default=1.5, description="돌파 분의 거래량 ÷ 전일 분당 평균 하한")
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
    MDD_SOFT_LIMIT: float = Field(default=0.07, description="소프트 MDD: 신규 진입 차단 + 익스포저 축소")
    MDD_HARD_LIMIT: float = Field(default=0.12, description="하드 MDD: 전량 청산 + 쿨다운")
    DERISK_RATIO: float = Field(default=0.5, description="소프트 트리거 시 유지할 익스포저 비율")
    COOLDOWN_MINUTES: int = Field(default=60, description="킬스위치 이후 신규 진입 금지 시간(분)")
    CRASH_DROP_PCT: float = Field(default=0.045, description="단기 급락 감지 임계(전량 청산 트리거)")
    CRASH_LOOKBACK: int = Field(default=10, description="급락 감지 룩백(봉 수)")
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
                    "때문이며, 이 논리는 1단계가 끝나면 성립하지 않는다")

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
