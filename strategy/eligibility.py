"""대회 규정 매매 가능 여부 판정 — 위반 주문은 거부되거나 실격 사유가 된다.

## 규정 (제16회 뱅키스 대학생 모의투자대회)

| 항목 | 기준 | 적용 |
|---|---|---|
| 매수 불가 | 현재가 **1,000원 미만** | 매수만. 보유 종목이 아래로 내려가면 매도는 된다 |
| 거래 불가 | 발행주식수 **100,000주 미만** | 매수·매도 모두 |
| 제외 | 거래정지·관리종목·거래 미발생 | 매수 |
| 시장 | KOSPI·KOSDAQ 개별 종목 | KONEX 제외 |
| 단일 종목 한도 | 자산의 **50%** | 매수 수량 산정 |
| 금지 | 신용거래·공매도 | 구조적으로 롱 온리라 해당 없음 |

## 왜 별도 모듈인가

`RiskManager`가 모든 주문의 최종 결정권자라는 기존 구조를 깨지 않으려면, 규정 판정을
리스크 매니저가 호출하는 순수 함수로 두는 것이 맞다. 전략이 아무리 좋은 점수를 내도
규정에 걸리면 주문이 나가서는 안 된다.

**판정 근거는 `config/universe.json`이다**(`scripts/refresh_universe.py`가 KRX 전종목시세로
만든다). 런타임에 KRX를 조회하지 않는다 — 발행주식수·시장구분·소속부는 정기변경 때만
바뀌는 참조 데이터다.

## 한계

**보유 중에 발생하는 거래정지는 막을 수 없다.** 진입 시점의 스냅샷으로만 판정하기
때문이다. 소형주 비중이 커지면 이 위험이 실질적이며, 30거래일 대회에서 정지는
회복 시간이 없다는 뜻이다. 유니버스를 갱신하면(`refresh_universe.py` 재실행) 다음
판정부터 반영된다.
"""

from __future__ import annotations

from typing import Optional, Tuple

from config.settings import Settings, get_settings

# 대회 규정 상수. `scripts/refresh_universe.py`의 필터와 같은 값을 쓴다.
MIN_BUY_PRICE = 1_000.0        # 현재가 1,000원 미만 매수 불가(동전주)
MIN_SHARES = 100_000.0         # 발행주식수 10만주 미만 거래 불가
MAX_SINGLE_WEIGHT = 0.50       # 단일 종목 최대 투자 한도

BLOCKED_SECTIONS = ("관리종목", "투자주의환기", "SPAC", "외국기업")
ALLOWED_MARKETS = ("KOSPI", "KOSDAQ", "KOSDAQ GLOBAL")

# 모의투자 서버가 주문을 거부하는 종목(`40070000 매매불가 종목`). 2026-10-14 실측: 거부된 5종목이
# 전부 증거금률 100%(종목상태 57, 그중 하나는 투자경고 53)였고 체결된 2종목은 40%(55)였다.
# 종목상태(`iscd_stat_cls_code`): 51 관리 · 52 투자위험 · 53 투자경고 · 58 거래정지 · 59 단기과열.
VTS_BLOCKED_STATUS = frozenset({"51", "52", "53", "58", "59"})


def vts_tradable(status: Optional[dict]) -> bool:
    """현재가 응답의 상태 필드로 모의투자에서 살 수 있는지 판정한다.

    상태를 모르면(조회 실패·새 종목) 막지 않는다 — 거부되면 엔진이 그날 다시 시도하지 않는다.
    """
    if not status:
        return True
    if str(status.get("iscd_stat_cls_code", "")) in VTS_BLOCKED_STATUS:
        return False
    try:
        return float(status.get("marg_rate") or 0.0) < 100.0
    except (TypeError, ValueError):
        return True


def is_buyable(symbol: str, price: float,
               settings: Optional[Settings] = None) -> Tuple[bool, str]:
    """이 종목을 **매수**할 수 있는가. (가능여부, 사유)

    유니버스 파일이 없으면 판정할 근거가 없으므로 가격 조건만 보고 통과시킨다.
    기존 40종목 기본 유니버스로 돌 때 이 모듈이 전부 막아버리면 안 되기 때문이다.
    """
    s = settings or get_settings()

    if price > 0 and price < MIN_BUY_PRICE:
        return False, f"현재가 {price:,.0f}원 < {MIN_BUY_PRICE:,.0f}원 (동전주 매수 불가)"

    blob = s._universe_blob()
    if not blob:
        return True, "유니버스 파일 없음 — 가격 조건만 확인"

    shares = s.shares_of(symbol)
    if shares and shares < MIN_SHARES:
        return False, f"발행주식수 {shares:,.0f}주 < {MIN_SHARES:,.0f}주 (거래 불가)"

    market = str(blob.get("market", {}).get(symbol, ""))
    if market and market not in ALLOWED_MARKETS:
        return False, f"대상 외 시장({market})"

    section = str(blob.get("section", {}).get(symbol, ""))
    for bad in BLOCKED_SECTIONS:
        if bad in section:
            return False, f"{bad} — 매매 제외 대상"

    tradable = blob.get("tradable") or []
    if tradable and symbol not in tradable:
        return False, "유니버스 제외 종목(규정·유동성 필터 미통과)"

    return True, "매수 가능"


def is_sellable(symbol: str, settings: Optional[Settings] = None) -> Tuple[bool, str]:
    """이 종목을 **매도**할 수 있는가.

    1,000원 미만은 **매수** 제한이므로 매도에는 적용하지 않는다. 보유 종목이 1,000원
    아래로 내려갔을 때 팔 수 없다면 손절 자체가 불가능해지고, 그것이 더 위험하다.
    """
    s = settings or get_settings()
    blob = s._universe_blob()
    if not blob:
        return True, "유니버스 파일 없음"
    shares = s.shares_of(symbol)
    if shares and shares < MIN_SHARES:
        return False, f"발행주식수 {shares:,.0f}주 < {MIN_SHARES:,.0f}주 (거래 불가)"
    return True, "매도 가능"


def position_weight_cap(settings: Optional[Settings] = None) -> float:
    """단일 종목 비중 상한. 규정 한도와 설정값 중 **더 작은 쪽**을 쓴다.

    설정을 실수로 50% 넘게 올려도 규정을 넘지 않게 하는 안전장치다. 체결 슬리피지로
    의도한 비중을 살짝 넘길 수 있으므로 설정값 자체는 여유를 두는 것이 좋다.
    """
    s = settings or get_settings()
    return min(float(s.MAX_POSITION_WEIGHT), MAX_SINGLE_WEIGHT)


def index_membership_count(symbols, settings: Optional[Settings] = None) -> int:
    """주어진 종목들 중 코스피200·코스닥150에 속한 개수.

    대회 최소 자격 요건("코스피200·코스닥150 중 5종목 이상")을 판정한다. 유니버스가
    284종목으로 넓어지면 지수 외 종목만 거래하다 이 요건을 놓칠 수 있다.
    """
    s = settings or get_settings()
    members = set()
    for lst in s.index_members().values():
        members.update(lst)
    if not members:
        return 0
    return len({str(x) for x in symbols} & members)
