"""KIS Open API REST 클라이언트 (실전/모의투자 공용).

설계 원칙
 1. **모든 요청은 PriorityRateLimiter를 통과한다.** 주문은 P0이라 시세 폴링이
    아무리 몰려도 손절 주문이 밀리지 않는다.
 2. `EGW00201`(초당 거래건수 초과) 응답을 받으면 레이트를 자동 감축하고 재시도한다.
 3. 토큰 오류(EGW00121 등)는 강제 재발급 후 1회 재시도한다.
 4. 반환 스키마는 `api/mock_client.py`와 완전히 동일하므로 상위 코드는 불변이다.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from typing import Any, Dict, List, Optional

from api.base import (
    RATE_LIMIT_CODES,
    TOKEN_ERROR_CODES,
    UNKNOWN_RESULT_CODE,
    AbstractBroker,
    is_ok,
    make_response,
    to_float,
    to_int,
)
from config.settings import Settings
from core.auth import TokenManager
from core.events import Fill, OrderType, Side
from core.latency import LatencyStats
from core.rate_limiter import (
    PRIORITY_ACCOUNT,
    PRIORITY_ORDER,
    PRIORITY_QUOTE,
    PriorityRateLimiter,
)
from utils.krx_calendar import KST
from utils.logger import logger
from utils.market_rules import round_to_tick


class KISClient(AbstractBroker):
    """실제 KIS 서버와 통신하는 프로덕션 클라이언트."""

    is_mock = False

    def __init__(
        self,
        settings: Settings,
        token_manager: Optional[TokenManager] = None,
        rate_limiter: Optional[PriorityRateLimiter] = None,
    ):
        self.settings = settings
        self.limiter = rate_limiter or PriorityRateLimiter(
            rate=settings.effective_rate, burst=settings.REST_BURST
        )
        self.auth = token_manager or TokenManager(settings)
        self._client = None
        self._seen_fills: set[str] = set()
        # API(TR)별 지연·결과 지표 — `/metrics`로 내보낸다(`core/latency.py`)
        self.http_stats = LatencyStats()
        # 증권사 서버 시각 - 로컬 시각(초). 첫 응답의 `Date` 헤더로 잰다(1초 해상도).
        # 09:10 판정·진입이 로컬 시계로 돌기 때문에 몇 초만 틀려도 판정이 이르거나 늦다.
        self.clock_skew_s: Optional[float] = None
        self._connected = False

    # ------------------------------------------------------------------ 수명주기
    async def connect(self) -> None:
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.rest_base_url,
                timeout=httpx.Timeout(self.settings.HTTP_TIMEOUT),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers={
                    "content-type": "application/json; charset=utf-8",
                    "User-Agent": "kis-ai-quant-trader/1.0",
                },
            )
        await self.auth.get_token()
        self.auth.start_auto_refresh()
        self._connected = True
        logger.success(f"[KISClient] 연결 완료 | {self.settings.describe()}")

    async def close(self) -> None:
        self._connected = False
        await self.auth.close()
        await self.limiter.close()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> Dict[str, Any]:
        """토큰 발급 → 시세 조회 → 잔고 조회 순으로 실제 연결을 검증한다."""
        result: Dict[str, Any] = {"ok": False, "mode": self.settings.KIS_ENV, "steps": {}}
        try:
            token = await self.auth.get_token()
            result["steps"]["token"] = {"ok": True, "seconds_left": round(token.seconds_left)}
        except Exception as exc:
            result["steps"]["token"] = {"ok": False, "error": str(exc)}
            return result

        probe = self.settings.UNIVERSE[0] if self.settings.UNIVERSE else "005930"
        try:
            quote = await self.inquire_price(probe)
            result["steps"]["quote"] = {"ok": is_ok(quote), "symbol": probe,
                                        "price": (quote.get("output") or {}).get("stck_prpr")}
        except Exception as exc:
            result["steps"]["quote"] = {"ok": False, "error": str(exc)}

        try:
            balance = await self.inquire_balance()
            summary = (balance.get("output2") or [{}])[0]
            result["steps"]["balance"] = {
                "ok": is_ok(balance),
                "cash": summary.get("dnca_tot_amt"),
                "total": summary.get("tot_evlu_amt"),
                "msg": balance.get("msg1"),
            }
        except Exception as exc:
            result["steps"]["balance"] = {"ok": False, "error": str(exc)}

        result["ok"] = all(step.get("ok") for step in result["steps"].values())
        return result

    # ------------------------------------------------------------------ 요청 엔진
    async def _headers(self, tr_id: str, hashkey: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": await self.auth.get_authorization(),
            "appkey": self.settings.KIS_APP_KEY,
            "appsecret": self.settings.KIS_APP_SECRET,
            "tr_id": tr_id,
            "custtype": "P",
        }
        if hashkey:
            headers["hashkey"] = hashkey
        return headers

    async def _hashkey(self, body: Dict[str, Any]) -> Optional[str]:
        """주문 위변조 방지용 hashkey. 실패해도 주문은 진행한다(선택 헤더)."""
        try:
            resp = await self._client.post(
                "/uapi/hashkey",
                json=body,
                headers={
                    "content-type": "application/json; charset=utf-8",
                    "appkey": self.settings.KIS_APP_KEY,
                    "appsecret": self.settings.KIS_APP_SECRET,
                },
            )
            if resp.status_code == 200:
                return resp.json().get("HASH")
        except Exception as exc:
            logger.debug(f"[KISClient] hashkey 생성 실패(무시): {exc}")
        return None

    async def _request(
        self,
        method: str,
        path: str,
        tr_id: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        priority: int = PRIORITY_QUOTE,
        use_hashkey: bool = False,
        idempotent: bool = True,
    ) -> Dict[str, Any]:
        """KIS REST 공용 경로 — 유량 제어 · 재시도 · 토큰 갱신 · 지연 지표.

        ## 주문은 멱등하지 않다 (`idempotent=False`)

        응답을 받지 못한 실패(읽기 타임아웃 · 연결 중 끊김 · 5xx)는 **요청이 증권사에 도착해
        처리됐을 수 있다.** 그대로 재시도하면 같은 매수가 두 번 나간다. 그래서 주문은
        "요청이 나가지 않은 것이 확실한" 실패(접속 실패 · 접속 타임아웃 · 커넥션 풀 대기)만
        재시도하고, 나머지는 `LOCAL_UNKNOWN`으로 돌려준다 — 엔진이 미체결·잔고 조회로 확인한다.
        429·`EGW00201`은 게이트웨이가 **거절한** 것이라(실행 안 됨) 주문도 재시도한다.
        """
        import httpx

        if self._client is None:
            await self.connect()

        last_payload: Dict[str, Any] = make_response(rt_cd="1", msg_cd="LOCAL999", msg1="요청 실패")
        token_retried = False

        for attempt in range(1, self.settings.HTTP_MAX_RETRY + 1):
            hashkey = await self._hashkey(body) if (use_hashkey and body) else None
            headers = await self._headers(tr_id, hashkey)

            await self.limiter.acquire(priority)
            t0 = time.perf_counter()
            try:
                if method == "GET":
                    resp = await self._client.get(path, params=params, headers=headers)
                else:
                    resp = await self._client.post(path, json=body, headers=headers)
            except Exception as exc:
                self.http_stats.observe(tr_id, time.perf_counter() - t0, "error")
                not_sent = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                                            httpx.PoolTimeout))
                if not idempotent and not not_sent:
                    logger.error(f"[KISClient] 주문 결과 불명({type(exc).__name__}: {exc}) — "
                                 "재시도하지 않는다(이중 주문 방지). 미체결·잔고로 확인한다")
                    return make_response(rt_cd="1", msg_cd=UNKNOWN_RESULT_CODE,
                                         msg1=f"주문 결과 불명: {type(exc).__name__}")
                backoff = self.settings.HTTP_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(f"[KISClient] 통신 오류({path}) {attempt}회차: {exc} → {backoff:.1f}s 후 재시도")
                last_payload = make_response(rt_cd="1", msg_cd="LOCAL001", msg1=f"통신 오류: {exc}")
                await asyncio.sleep(backoff)
                continue

            self.http_stats.observe(tr_id, time.perf_counter() - t0,
                                    "ok" if resp.status_code == 200 else str(resp.status_code))
            if self.clock_skew_s is None:
                self._measure_clock_skew(resp.headers.get("date"), time.perf_counter() - t0)
            if resp.status_code == 429:
                delay = self.limiter.notify_rate_limit_error()
                logger.warning(f"[KISClient] HTTP 429 유량 초과 → {delay:.1f}s 감축 후 재시도")
                await asyncio.sleep(min(delay, 5.0))
                continue

            try:
                payload = resp.json()
            except json.JSONDecodeError:
                payload = make_response(rt_cd="1", msg_cd="LOCAL002", msg1=f"JSON 파싱 실패: {resp.text[:200]}")

            msg_cd = str(payload.get("msg_cd", ""))

            if msg_cd in RATE_LIMIT_CODES:
                delay = self.limiter.notify_rate_limit_error()
                logger.warning(f"[KISClient] {msg_cd} 초당 유량 초과 → 레이트 감축({delay:.1f}s) 후 재시도")
                await asyncio.sleep(min(delay, 5.0))
                last_payload = payload
                continue

            if msg_cd in TOKEN_ERROR_CODES and not token_retried:
                logger.warning(f"[KISClient] 토큰 오류({msg_cd}) → 강제 재발급 후 재시도")
                token_retried = True
                await self.auth.get_token(force=True)
                continue

            if resp.status_code != 200:
                last_payload = payload if isinstance(payload, dict) else {}
                last_payload.setdefault("rt_cd", "1")
                logger.warning(f"[KISClient] HTTP {resp.status_code} {path} :: {payload}")
                if not idempotent and resp.status_code >= 500:
                    # 5xx 는 처리 도중 실패일 수 있다 — 주문은 결과를 모른다
                    last_payload["msg_cd"] = UNKNOWN_RESULT_CODE
                    return last_payload
                await asyncio.sleep(self.settings.HTTP_BACKOFF_BASE * attempt)
                continue

            self.limiter.notify_success()
            return payload

        return last_payload

    def _measure_clock_skew(self, date_header: Optional[str], rtt: float) -> None:
        """`Date` 헤더(서버 시각, 1초 해상도)와 로컬 시각의 차이. 왕복 절반을 보정한다."""
        if not date_header:
            return
        try:
            from email.utils import parsedate_to_datetime
            server = parsedate_to_datetime(date_header)
        except (TypeError, ValueError):
            return
        local = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=rtt / 2)
        self.clock_skew_s = round((server - local).total_seconds(), 2)
        level = logger.warning if abs(self.clock_skew_s) >= 2.0 else logger.info
        level(f"[KISClient] 시계 오차 {self.clock_skew_s:+.2f}s (증권사 서버 - 로컬)"
              + (" — 2초 이상이다. 시간 동기화(w32tm/chrony)를 확인하라" if abs(self.clock_skew_s) >= 2.0 else ""))

    # ------------------------------------------------------------------ 시세 API
    async def inquire_price(self, symbol: str) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            self.settings.tr("inquire_price"),
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
            priority=PRIORITY_QUOTE,
        )

    async def inquire_asking_price(self, symbol: str) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            self.settings.tr("inquire_asking_price"),
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
            priority=PRIORITY_QUOTE,
        )

    async def inquire_daily_chart(
        self, symbol: str, start: str = "", end: str = "", period: str = "D", adjust: bool = True
    ) -> Dict[str, Any]:
        today = dt.datetime.now(tz=KST).date()
        end = end or today.strftime("%Y%m%d")
        start = start or (today - dt.timedelta(days=400)).strftime("%Y%m%d")
        return await self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            self.settings.tr("inquire_daily_chart"),
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start,
                "FID_INPUT_DATE_2": end,
                "FID_PERIOD_DIV_CODE": period,
                "FID_ORG_ADJ_PRC": "0" if adjust else "1",
            },
            priority=PRIORITY_QUOTE,
        )

    async def inquire_multi_price(self, symbols: List[str]) -> Dict[str, Any]:
        params: Dict[str, str] = {}
        for i, code in enumerate(list(symbols)[:30], 1):
            params[f"FID_COND_MRKT_DIV_CODE_{i}"] = "J"
            params[f"FID_INPUT_ISCD_{i}"] = code
        return await self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/intstock-multprice",
            self.settings.tr("inquire_multi_price"),
            params=params,
            priority=PRIORITY_QUOTE,
        )

    async def inquire_minute_chart(self, symbol: str, hour: str = "", past: bool = True) -> Dict[str, Any]:
        hour = hour or dt.datetime.now(tz=KST).strftime("%H%M%S")
        return await self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
            self.settings.tr("inquire_minute_chart"),
            params={
                "FID_ETC_CLS_CODE": "",
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_HOUR_1": hour,
                "FID_PW_DATA_INCU_YN": "Y" if past else "N",
            },
            priority=PRIORITY_QUOTE,
        )

    # ------------------------------------------------------------------ 계좌 API
    async def inquire_balance(self) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            self.settings.tr("inquire_balance"),
            params={
                "CANO": self.settings.cano,
                "ACNT_PRDT_CD": self.settings.acnt_prdt_cd,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
            priority=PRIORITY_ACCOUNT,
        )

    async def inquire_psbl_order(self, symbol: str, price: float = 0.0) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            self.settings.tr("inquire_psbl_order"),
            params={
                "CANO": self.settings.cano,
                "ACNT_PRDT_CD": self.settings.acnt_prdt_cd,
                "PDNO": symbol,
                "ORD_UNPR": str(int(price)) if price > 0 else "0",
                "ORD_DVSN": "00" if price > 0 else "01",
                "CMA_EVLU_AMT_ICLD_YN": "N",
                "OVRS_ICLD_YN": "N",
            },
            priority=PRIORITY_ACCOUNT,
        )

    async def inquire_daily_ccld(self, start: str = "", end: str = "") -> Dict[str, Any]:
        today = dt.datetime.now(tz=KST).strftime("%Y%m%d")
        return await self._request(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            self.settings.tr("inquire_daily_ccld"),
            params={
                "CANO": self.settings.cano,
                "ACNT_PRDT_CD": self.settings.acnt_prdt_cd,
                "INQR_STRT_DT": start or today,
                "INQR_END_DT": end or today,
                "SLL_BUY_DVSN_CD": "00",
                "INQR_DVSN": "00",
                "PDNO": "",
                "CCLD_DVSN": "01",   # 01: 체결만
                "ORD_GNO_BRNO": "",
                "ODNO": "",
                "INQR_DVSN_3": "00",
                "INQR_DVSN_1": "",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
            priority=PRIORITY_ACCOUNT,
        )

    async def inquire_open_orders(self) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl",
            self.settings.tr("inquire_psbl_rvsecncl"),
            params={
                "CANO": self.settings.cano,
                "ACNT_PRDT_CD": self.settings.acnt_prdt_cd,
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
                "INQR_DVSN_1": "0",
                "INQR_DVSN_2": "0",
            },
            priority=PRIORITY_ACCOUNT,
        )

    # ------------------------------------------------------------------ 주문 API
    async def order_cash(
        self,
        symbol: str,
        side: Side,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Dict[str, Any]:
        if qty <= 0:
            return make_response(rt_cd="1", msg_cd="LOCAL010", msg1="주문수량이 0입니다.")

        ord_unpr = "0" if order_type is OrderType.MARKET else str(int(round_to_tick(price)))
        body = {
            "CANO": self.settings.cano,
            "ACNT_PRDT_CD": self.settings.acnt_prdt_cd,
            "PDNO": symbol,
            "ORD_DVSN": order_type.value,
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": ord_unpr,
        }
        tr_id = self.settings.tr("order_buy") if side is Side.BUY else self.settings.tr("order_sell")

        if self.settings.DRY_RUN:
            logger.warning(f"[DRY_RUN] 주문 전송 생략 :: {tr_id} {side.value} {symbol} {qty}주 @{ord_unpr}")
            return make_response(
                {"KRX_FWDG_ORD_ORGNO": "DRYRUN", "ODNO": "0000000000",
                 "ORD_TMD": dt.datetime.now(tz=KST).strftime("%H%M%S")},
                msg_cd="DRYRUN", msg1="DRY_RUN 모드로 실제 전송하지 않았습니다.",
            )

        payload = await self._request(
            "POST",
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            body=body,
            priority=PRIORITY_ORDER,
            use_hashkey=True,
            idempotent=False,          # 이중 주문 방지 — `_request` 참고
        )
        if is_ok(payload):
            out = payload.get("output") or {}
            logger.success(
                f"[주문접수] {side.value} {symbol} {qty}주 @{ord_unpr} → ODNO={out.get('ODNO')}"
            )
        else:
            logger.error(
                f"[주문실패] {side.value} {symbol} {qty}주 @{ord_unpr} :: "
                f"{payload.get('msg_cd')} {payload.get('msg1')}"
            )
        return payload

    async def cancel_order(
        self, org_no: str, order_no: str, qty: int = 0, all_qty: bool = True
    ) -> Dict[str, Any]:
        body = {
            "CANO": self.settings.cano,
            "ACNT_PRDT_CD": self.settings.acnt_prdt_cd,
            "KRX_FWDG_ORD_ORGNO": org_no,
            "ORGN_ODNO": order_no,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 02: 취소
            "ORD_QTY": str(int(qty)) if not all_qty else "0",
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y" if all_qty else "N",
        }
        return await self._request(
            "POST",
            "/uapi/domestic-stock/v1/trading/order-rvsecncl",
            self.settings.tr("order_revise_cancel"),
            body=body,
            priority=PRIORITY_ORDER,
            use_hashkey=True,
        )

    # ------------------------------------------------------------------ 체결 폴링
    async def poll_fills(self) -> List[Fill]:
        """당일 체결내역을 조회해 신규 체결만 `Fill`로 변환한다.

        WebSocket 체결통보가 살아 있으면 그쪽이 우선이며, 이 폴링은 **이중 안전망**이다.
        (`_seen_fills`로 중복 방지)
        """
        payload = await self.inquire_daily_ccld()
        if not is_ok(payload):
            return []
        fills: List[Fill] = []
        for row in payload.get("output1") or []:
            filled = to_int(row.get("tot_ccld_qty"))
            if filled <= 0:
                continue
            order_no = str(row.get("odno", ""))
            key = f"{order_no}:{filled}"
            if key in self._seen_fills:
                continue
            self._seen_fills.add(key)
            side = Side.BUY if str(row.get("sll_buy_dvsn_cd")) == "02" else Side.SELL
            price = to_float(row.get("avg_prvs")) or to_float(row.get("ord_unpr"))
            fee = filled * price * self.settings.FEE_BPS / 10_000.0
            tax = filled * price * self.settings.TAX_BPS / 10_000.0 if side is Side.SELL else 0.0
            fills.append(
                Fill(
                    symbol=str(row.get("pdno", "")).zfill(6),
                    side=side,
                    qty=filled,
                    price=price,
                    fee=fee,
                    tax=tax,
                    order_no=order_no,
                    reason="poll",
                )
            )
        return fills

    async def cancel_all_open_orders(self) -> int:
        """미체결 주문 전량 취소 (종료 시퀀스/킬스위치에서 사용)."""
        payload = await self.inquire_open_orders()
        if not is_ok(payload):
            return 0
        count = 0
        for row in payload.get("output1") or []:
            remaining = to_int(row.get("rmn_qty"))
            if remaining <= 0:
                continue
            resp = await self.cancel_order(
                org_no=str(row.get("ord_gno_brno") or row.get("orgn_odno") or ""),
                order_no=str(row.get("odno", "")),
                all_qty=True,
            )
            if is_ok(resp):
                count += 1
        if count:
            logger.warning(f"[KISClient] 미체결 주문 {count}건 취소 완료")
        return count
