"""브로커 추상 인터페이스.

Mock 클라이언트와 실서버 클라이언트가 **완전히 동일한 JSON 스키마**를 반환하도록
강제하는 것이 이 모듈의 목적이다. 상위 레이어(전략/리스크/엔진)는 어느 쪽이
연결되어 있는지 전혀 알 필요가 없다.

KIS 응답 공통 규약
    rt_cd : "0"이면 성공, 그 외는 실패
    msg_cd: 메시지 코드 (예: EGW00201 = 초당 거래건수 초과)
    msg1  : 메시지 본문
    output / output1 / output2 : 실제 데이터
"""

from __future__ import annotations

import abc
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.events import Fill, OrderAck, OrderRequest, OrderType, Position, Quote, Side

# 유량 초과 관련 메시지 코드
RATE_LIMIT_CODES = {"EGW00201", "EGW00202"}
TOKEN_ERROR_CODES = {"EGW00121", "EGW00123", "EGW00133"}


class KISAPIError(RuntimeError):
    def __init__(self, message: str, code: str = "", raw: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.raw = raw or {}


class RateLimitExceeded(KISAPIError):
    pass


class TokenExpiredError(KISAPIError):
    pass


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    return int(to_float(value, default))


def is_ok(payload: Dict[str, Any]) -> bool:
    return str(payload.get("rt_cd", "")) == "0"


def make_response(
    output: Any = None,
    rt_cd: str = "0",
    msg_cd: str = "MCA00000",
    msg1: str = "정상처리 되었습니다.",
    **extra: Any,
) -> Dict[str, Any]:
    """KIS 규약과 동일한 형태의 응답 dict를 생성한다."""
    resp: Dict[str, Any] = {"rt_cd": rt_cd, "msg_cd": msg_cd, "msg1": msg1}
    if output is not None:
        resp["output"] = output
    resp.update(extra)
    return resp


class AbstractBroker(abc.ABC):
    """모든 브로커 구현체의 공통 인터페이스."""

    #: Mock 여부 (엔진이 가속 시계 등 부가기능 사용 여부를 판단)
    is_mock: bool = False

    # ------------------------------------------------------------------ 수명주기
    @abc.abstractmethod
    async def connect(self) -> None:
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        ...

    @abc.abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """연결/인증/계좌 접근 가능 여부 점검."""

    # ------------------------------------------------------------------ 시세
    @abc.abstractmethod
    async def inquire_price(self, symbol: str) -> Dict[str, Any]:
        """주식현재가 시세 (TR: FHKST01010100)."""

    @abc.abstractmethod
    async def inquire_asking_price(self, symbol: str) -> Dict[str, Any]:
        """주식현재가 호가/예상체결 (TR: FHKST01010200)."""

    @abc.abstractmethod
    async def inquire_daily_chart(
        self, symbol: str, start: str = "", end: str = "", period: str = "D", adjust: bool = True
    ) -> Dict[str, Any]:
        """국내주식 기간별 시세(일/주/월) (TR: FHKST03010100)."""

    @abc.abstractmethod
    async def inquire_minute_chart(self, symbol: str, hour: str = "", past: bool = True) -> Dict[str, Any]:
        """주식당일분봉조회 (TR: FHKST03010200)."""

    # ------------------------------------------------------------------ 계좌
    @abc.abstractmethod
    async def inquire_balance(self) -> Dict[str, Any]:
        """주식잔고조회 (TR: TTTC8434R / VTTC8434R)."""

    @abc.abstractmethod
    async def inquire_psbl_order(self, symbol: str, price: float = 0.0) -> Dict[str, Any]:
        """매수가능조회 (TR: TTTC8908R / VTTC8908R)."""

    @abc.abstractmethod
    async def inquire_daily_ccld(self, start: str = "", end: str = "") -> Dict[str, Any]:
        """주식일별주문체결조회 (TR: TTTC8001R / VTTC8001R)."""

    # ------------------------------------------------------------------ 주문
    @abc.abstractmethod
    async def order_cash(
        self,
        symbol: str,
        side: Side,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Dict[str, Any]:
        """주식주문(현금) (TR: TTTC0802U/VTTC0802U 매수, TTTC0801U/VTTC0801U 매도)."""

    @abc.abstractmethod
    async def cancel_order(
        self, org_no: str, order_no: str, qty: int = 0, all_qty: bool = True
    ) -> Dict[str, Any]:
        """주식주문(정정취소) (TR: TTTC0803U / VTTC0803U)."""

    @abc.abstractmethod
    async def inquire_open_orders(self) -> Dict[str, Any]:
        """정정취소가능주문조회 (미체결 잔량)."""

    # ------------------------------------------------------------------ 고수준 헬퍼
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """현재가 조회 결과를 `Quote` 도메인 객체로 변환."""
        payload = await self.inquire_price(symbol)
        if not is_ok(payload):
            return None
        out = payload.get("output") or {}
        price = to_float(out.get("stck_prpr"))
        if price <= 0:
            return None
        return Quote(
            symbol=symbol,
            price=price,
            open=to_float(out.get("stck_oprc")),
            high=to_float(out.get("stck_hgpr")),
            low=to_float(out.get("stck_lwpr")),
            volume=to_float(out.get("acml_vol")),
            change_pct=to_float(out.get("prdy_ctrt")),
        )

    @staticmethod
    def _chart_rows_to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """KIS 차트 응답(최신→과거)을 시간 오름차순 DataFrame으로 변환."""
        records = []
        for row in rows:
            date_raw = row.get("stck_bsop_date") or row.get("stck_cntg_hour") or ""
            close = to_float(row.get("stck_clpr") or row.get("stck_prpr"))
            if close <= 0:
                continue
            records.append(
                {
                    "ts": str(date_raw),
                    "open": to_float(row.get("stck_oprc"), close),
                    "high": to_float(row.get("stck_hgpr"), close),
                    "low": to_float(row.get("stck_lwpr"), close),
                    "close": close,
                    "volume": to_float(row.get("acml_vol") or row.get("cntg_vol")),
                }
            )
        if not records:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(records)
        df = df.iloc[::-1].reset_index(drop=True)  # KIS는 최신순이므로 뒤집는다
        return df

    async def get_bars(self, symbol: str, count: int = 300, period: str = "D") -> pd.DataFrame:
        """OHLCV DataFrame(과거→최신)을 반환한다."""
        if period == "M":
            payload = await self.inquire_minute_chart(symbol)
            rows = payload.get("output2") or payload.get("output") or []
        else:
            end = dt.date.today().strftime("%Y%m%d")
            start = (dt.date.today() - dt.timedelta(days=int(count * 1.8) + 30)).strftime("%Y%m%d")
            payload = await self.inquire_daily_chart(symbol, start=start, end=end, period=period)
            rows = payload.get("output2") or []
        if not is_ok(payload):
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        df = self._chart_rows_to_frame(rows if isinstance(rows, list) else [])
        return df.tail(count).reset_index(drop=True)

    async def get_portfolio_state(self) -> Tuple[float, Dict[str, Position]]:
        """(주문가능현금, 보유포지션) 조회."""
        payload = await self.inquire_balance()
        if not is_ok(payload):
            raise KISAPIError(payload.get("msg1", "잔고조회 실패"), payload.get("msg_cd", ""), payload)
        positions: Dict[str, Position] = {}
        for row in payload.get("output1") or []:
            qty = to_int(row.get("hldg_qty"))
            if qty <= 0:
                continue
            symbol = str(row.get("pdno", "")).zfill(6)
            pos = Position(
                symbol=symbol,
                qty=qty,
                avg_price=to_float(row.get("pchs_avg_pric")),
                last_price=to_float(row.get("prpr")),
            )
            pos.peak_price = max(pos.avg_price, pos.last_price)
            positions[symbol] = pos
        summary = (payload.get("output2") or [{}])
        cash = to_float(summary[0].get("dnca_tot_amt")) if summary else 0.0
        # 예수금이 0으로 오는 계좌는 D+2 예수금을 사용
        if cash <= 0 and summary:
            cash = to_float(summary[0].get("prvs_rcdl_excc_amt"))
        return cash, positions

    async def submit_order(self, request: OrderRequest) -> OrderAck:
        """`OrderRequest`를 전송하고 `OrderAck`로 정규화해 반환."""
        payload = await self.order_cash(
            symbol=request.symbol,
            side=request.side,
            qty=request.qty,
            price=request.price,
            order_type=request.order_type,
        )
        out = payload.get("output") or {}
        return OrderAck(
            success=is_ok(payload),
            order_no=str(out.get("ODNO", "")),
            org_no=str(out.get("KRX_FWDG_ORD_ORGNO", "")),
            symbol=request.symbol,
            side=request.side,
            qty=request.qty,
            price=request.price,
            msg_cd=str(payload.get("msg_cd", "")),
            msg=str(payload.get("msg1", "")),
            raw=payload,
        )

    # 체결 폴링(실서버). Mock은 즉시 체결 이벤트를 만들어 오버라이드한다.
    async def poll_fills(self) -> List[Fill]:
        return []
