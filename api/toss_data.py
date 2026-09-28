"""토스증권 Open API 데이터 클라이언트 — 학습·백테스트용 과거 데이터 수집 전용.

이 모듈은 **주문을 내지 않는다.** 대회 계좌가 KIS이므로 장중 시세와 매매는 KIS로 가고,
토스는 KIS 모의투자 도메인이 주지 않는 두 가지를 공급하는 데만 쓴다.

  1. 충분히 긴 과거 일봉 (수정주가 적용)
  2. 수급 데이터 — 투자자별 매매동향, 공매도, 신용, 대차, 프로그램매매

API 요약 (developers.tossinvest.com)
  인증   POST /oauth2/token  (client_credentials, form-urlencoded)
  일봉   GET  /api/v1/candles?symbol=&interval=1d&count=200&before=&adjusted=true
         → 최신순 정렬, 요청당 최대 200봉, `before`로 과거로 페이지네이션
  수급   GET  /api/v1/stocks/{symbol}/{investor-trading|short-selling|credit-trades|
              securities-lending|program-trades}?count=100&until=YYYY-MM-DD
         → 최신순, 응답의 `nextUntil`을 다음 `until`로 넘겨 페이지네이션
  유량   차트 20 TPS · 수급 10 TPS · 인증 5 TPS, 429 응답에 `Retry-After` 헤더

응답 필드명이 공개 문서에 전부 명시되어 있지 않아, 파서는 **후보 이름을 순서대로 시도**하고
찾지 못한 값은 NaN으로 둔다. 수급 원본 컬럼은 접두어를 붙여 통째로 보존하므로,
실제 필드명을 확인한 뒤에도 재수집 없이 피처를 늘릴 수 있다.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from config.settings import Settings, get_settings
from core.rate_limiter import AsyncTokenBucket
from utils.logger import logger

OHLCV_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

# 수급 엔드포인트: (경로 조각, 컬럼 접두어)
FLOW_ENDPOINTS: Dict[str, str] = {
    "investor-trading": "inv",
    "short-selling": "short",
    "credit-trades": "credit",
    "securities-lending": "lend",
    "program-trades": "prog",
}


class TossAPIError(RuntimeError):
    def __init__(self, message: str, status: int = 0, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


# --------------------------------------------------------------------------------------
# 응답 파싱 헬퍼 — 필드명이 문서에 없으므로 후보를 순서대로 시도한다
# --------------------------------------------------------------------------------------
def _rows_of(payload: Any) -> List[Dict[str, Any]]:
    """응답에서 레코드 리스트를 찾아낸다. 봉투 키 이름이 무엇이든 동작한다."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("result", "data", "items", "candles", "content", "list", "results"):
        value = payload.get(key)
        if isinstance(value, list) and (not value or isinstance(value[0], dict)):
            return value
        if isinstance(value, dict):
            nested = _rows_of(value)
            if nested:
                return nested
    # 마지막 수단: dict 값 중 첫 번째 레코드 리스트
    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return []


def _next_cursor(payload: Any) -> Optional[str]:
    """`nextUntil` 등 다음 페이지 커서를 찾는다."""
    if not isinstance(payload, dict):
        return None
    for key in ("nextUntil", "next_until", "nextCursor", "next"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for value in payload.values():
        if isinstance(value, dict):
            found = _next_cursor(value)
            if found:
                return found
    return None


def _pick(row: Dict[str, Any], *names: str) -> Any:
    """후보 키 이름을 순서대로 시도한다. 대소문자와 밑줄 차이를 흡수한다."""
    for name in names:
        if name in row:
            return row[name]
    lowered = {k.lower().replace("_", ""): v for k, v in row.items()}
    for name in names:
        key = name.lower().replace("_", "")
        if key in lowered:
            return lowered[key]
    return None


def _flatten(obj: Any, prefix: str = "", out: Optional[Dict[str, float]] = None
             ) -> Dict[str, float]:
    """중첩 dict를 평탄화해 숫자 필드만 뽑는다.

    토스 수급 응답은 중첩 구조다.
        {"institution": {"netBuyVolume": ..., "breakdown": {"pensionFund": {...}}}}
    → institution_netBuyVolume, institution_breakdown_pensionFund_netBuyVolume

    기관 세부 7종(연기금·금융투자·투신·보험·사모·은행·기타)까지 이렇게 살아남는다.
    """
    out = {} if out is None else out
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("date", "updatedAt", "currency"):
                continue
            _flatten(value, f"{prefix}_{key}" if prefix else str(key), out)
    else:
        num = _num(obj)
        if num == num and prefix:      # NaN이 아니면 숫자 필드
            out[prefix] = num
    return out


def _num(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        if isinstance(value, bool):
            return float(value)
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return float("nan")


def _to_yyyymmdd(value: Any) -> str:
    """ISO 8601 / epoch millis / YYYY-MM-DD 무엇이 와도 YYYYMMDD로 정규화."""
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # epoch 초 또는 밀리초
        seconds = float(value) / 1000.0 if float(value) > 1e11 else float(value)
        try:
            return dt.datetime.fromtimestamp(seconds).strftime("%Y%m%d")
        except (OSError, OverflowError, ValueError):
            return ""
    text = str(value).strip()
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return text


class TossDataClient:
    """토스 Open API에서 과거 데이터를 받아오는 비동기 클라이언트."""

    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or get_settings()
        self._client = None
        self._token: Optional[str] = None
        self._token_expires: Optional[dt.datetime] = None
        self._token_lock = asyncio.Lock()
        # 토스는 엔드포인트 그룹별로 유량이 다르다
        self._chart_bucket = AsyncTokenBucket(rate=self.s.TOSS_CHART_TPS,
                                              capacity=max(1.0, self.s.TOSS_CHART_TPS))
        self._flow_bucket = AsyncTokenBucket(rate=self.s.TOSS_FLOW_TPS,
                                             capacity=max(1.0, self.s.TOSS_FLOW_TPS))
        self.requests = 0

    # ------------------------------------------------------------------ 수명주기
    async def connect(self) -> None:
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.s.TOSS_BASE_URL,
                timeout=httpx.Timeout(self.s.TOSS_TIMEOUT),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                headers={"accept": "application/json", "User-Agent": "kis-ai-quant-trader/1.0"},
            )
        await self._access_token()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "TossDataClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ------------------------------------------------------------------ 인증
    async def _access_token(self) -> str:
        async with self._token_lock:
            now = dt.datetime.now()
            if self._token and self._token_expires and now < self._token_expires:
                return self._token
            if not self.s.toss_ready():
                raise TossAPIError(
                    "TOSS_CLIENT_ID / TOSS_CLIENT_SECRET 이 비어 있습니다. .env를 확인하세요."
                )
            resp = await self._client.post(
                "/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.s.TOSS_CLIENT_ID,
                    "client_secret": self.s.TOSS_CLIENT_SECRET,
                },
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                raise TossAPIError(f"토스 토큰 발급 실패 [{resp.status_code}] {resp.text[:300]}",
                                   resp.status_code)
            data = resp.json()
            token = _pick(data, "access_token", "accessToken")
            if not token:
                raise TossAPIError(f"토큰 응답에 access_token이 없습니다: {data}")
            expires_in = int(_num(_pick(data, "expires_in", "expiresIn")) or 3600)
            self._token = str(token)
            # 만료 2분 전에 갱신한다
            self._token_expires = now + dt.timedelta(seconds=max(60, expires_in - 120))
            logger.info(f"[Toss] 토큰 발급 완료 (유효 {expires_in}s)")
            return self._token

    # ------------------------------------------------------------------ 요청
    async def _get(self, path: str, params: Dict[str, Any], bucket: AsyncTokenBucket) -> Any:
        """유량 제어 + 429 백오프 + 토큰 만료 복구를 모두 처리하는 단일 경로."""
        last_error: Optional[str] = None
        for attempt in range(1, self.s.TOSS_MAX_RETRY + 1):
            await bucket.acquire()
            token = await self._access_token()
            try:
                resp = await self._client.get(
                    path, params=params, headers={"authorization": f"Bearer {token}"}
                )
            except Exception as exc:
                last_error = f"통신 오류: {exc}"
                await asyncio.sleep(0.5 * attempt)
                continue

            self.requests += 1

            if resp.status_code == 429:
                retry_after = _num(resp.headers.get("Retry-After"))
                delay = retry_after if retry_after == retry_after else 1.0  # NaN 체크
                logger.warning(f"[Toss] 429 유량 초과 → {delay:.1f}s 대기 ({path})")
                await asyncio.sleep(min(max(delay, 0.5), 10.0))
                continue

            if resp.status_code in (401, 403):
                # 토큰 만료로 보고 강제 재발급 후 1회 재시도
                self._token = None
                self._token_expires = None
                last_error = f"인증 실패 [{resp.status_code}]"
                continue

            if resp.status_code != 200:
                last_error = f"[{resp.status_code}] {resp.text[:200]}"
                await asyncio.sleep(0.4 * attempt)
                continue

            try:
                return resp.json()
            except ValueError:
                last_error = f"JSON 파싱 실패: {resp.text[:200]}"
                break

        raise TossAPIError(f"{path} 요청 실패: {last_error}")

    # ------------------------------------------------------------------ 일봉
    async def fetch_candles(self, symbol: str, count: int = 1500,
                            interval: str = "1d") -> pd.DataFrame:
        """과거 일봉을 `count`개까지 모아 과거→최신 순 DataFrame으로 반환한다.

        토스는 요청당 최대 200개를 최신순으로 주므로, `before` 커서를 과거로 밀며 반복한다.
        """
        collected: List[Dict[str, Any]] = []
        before: Optional[str] = None
        seen: set[str] = set()

        while len(collected) < count:
            params: Dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "count": min(200, count - len(collected)),
                "adjusted": "true",   # 수정주가 — 학습 데이터에는 필수
            }
            if before:
                params["before"] = before

            payload = await self._get("/api/v1/candles", params, self._chart_bucket)
            rows = _rows_of(payload)
            if not rows:
                break

            page: List[Dict[str, Any]] = []
            oldest_raw: Optional[str] = None
            for row in rows:
                raw_ts = _pick(row, "timestamp", "dateTime", "datetime", "time",
                               "date", "tradeDate", "candleDateTime")
                ts = _to_yyyymmdd(raw_ts)
                close = _num(_pick(row, "close", "closePrice", "c", "trade_price"))
                if not ts or close != close or close <= 0:
                    continue
                # `before` 커서는 원본 타임스탬프 형식 그대로여야 한다.
                # YYYY-MM-DD로 자르면 토스가 typeMismatch로 거부한다.
                if isinstance(raw_ts, str) and (oldest_raw is None or raw_ts < oldest_raw):
                    oldest_raw = raw_ts
                if ts in seen:
                    continue
                seen.add(ts)
                page.append({
                    "ts": ts,
                    "open": _num(_pick(row, "open", "openPrice", "o")),
                    "high": _num(_pick(row, "high", "highPrice", "h")),
                    "low": _num(_pick(row, "low", "lowPrice", "l")),
                    "close": close,
                    "volume": _num(_pick(row, "volume", "accTradeVolume", "tradeVolume", "v")),
                })

            if not page:
                break
            collected.extend(page)

            # 다음 페이지 커서: 응답이 주면 그것을, 없으면 이번 페이지에서 가장 오래된
            # 봉의 **원본 타임스탬프**를 그대로 쓴다
            cursor = _next_cursor(payload)
            next_before = cursor or oldest_raw
            if not next_before or next_before == before:
                break   # 커서가 진전되지 않으면 더 받을 것이 없다
            before = next_before
            # 종료 판정은 **중복 제거 전 원본 개수**로 한다.
            # `before` 커서가 경계 봉을 다시 포함시키므로 page는 보통 1개 적다.
            # 이를 "더 받을 게 없다"로 오인하면 200봉에서 수집이 멈춘다.
            if len(rows) < params["count"]:
                break   # 더 이상 과거 데이터가 없다

        if not collected:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.DataFrame(collected).drop_duplicates(subset="ts")
        df = df.sort_values("ts").reset_index(drop=True)   # 과거 → 최신
        # 결측 보정: 시가/고가/저가가 비면 종가로 채운다
        for col in ("open", "high", "low"):
            df[col] = df[col].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0.0)
        return df[OHLCV_COLUMNS].tail(count).reset_index(drop=True)

    # ------------------------------------------------------------------ 수급
    async def _fetch_flow_endpoint(self, symbol: str, endpoint: str, prefix: str,
                                   days: int) -> pd.DataFrame:
        """수급 엔드포인트 하나를 페이지네이션으로 수집한다.

        필드명을 모두 알 수 없으므로 **숫자형 필드를 통째로 접두어와 함께 보존**한다.
        나중에 실제 필드명을 확인해도 재수집 없이 피처를 만들 수 있다.
        """
        rows: List[Dict[str, Any]] = []
        until: Optional[str] = None
        seen: set[str] = set()

        while len(rows) < days:
            params: Dict[str, Any] = {"count": min(100, days - len(rows))}
            if until:
                params["until"] = until
            try:
                payload = await self._get(
                    f"/api/v1/stocks/{symbol}/{endpoint}", params, self._flow_bucket
                )
            except TossAPIError as exc:
                logger.debug(f"[Toss] {symbol} {endpoint} 조회 중단: {exc}")
                break

            page = _rows_of(payload)
            if not page:
                break

            added = 0
            for row in page:
                ts = _to_yyyymmdd(_pick(row, "date", "tradeDate", "baseDate", "timestamp", "dt"))
                if not ts or ts in seen:
                    continue
                seen.add(ts)
                record: Dict[str, Any] = {"ts": ts}
                # 응답이 중첩 dict이므로 평탄화해서 숫자 필드를 모두 살린다
                for flat_key, num in _flatten(row).items():
                    record[f"{prefix}_{flat_key}"] = num
                rows.append(record)
                added += 1

            if added == 0:
                break

            cursor = _next_cursor(payload)
            oldest = min(r["ts"] for r in rows[-added:])
            until = cursor or f"{oldest[:4]}-{oldest[4:6]}-{oldest[6:8]}"
            if len(page) < params["count"]:
                break

        if not rows:
            return pd.DataFrame(columns=["ts"])
        df = pd.DataFrame(rows).drop_duplicates(subset="ts")
        return df.sort_values("ts").reset_index(drop=True)

    async def fetch_flow(self, symbol: str, days: int = 1500,
                         endpoints: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """수급 5종을 모아 `ts` 기준으로 병합한 하나의 DataFrame을 반환한다."""
        targets = {k: v for k, v in FLOW_ENDPOINTS.items()
                   if endpoints is None or k in endpoints}
        frames: List[pd.DataFrame] = []
        for endpoint, prefix in targets.items():
            frame = await self._fetch_flow_endpoint(symbol, endpoint, prefix, days)
            if not frame.empty and len(frame.columns) > 1:
                frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=["ts"])

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="ts", how="outer")
        return merged.sort_values("ts").reset_index(drop=True)

    # ------------------------------------------------------------------ 진단
    async def probe(self, symbol: str = "005930") -> Dict[str, Any]:
        """연결과 응답 형태를 한 번에 확인한다. 필드명 파악용.

        실제 응답의 키 목록을 그대로 돌려주므로, 파서가 놓친 필드가 있으면 여기서 보인다.
        """
        out: Dict[str, Any] = {"ok": False}
        try:
            await self._access_token()
            out["token"] = True
        except Exception as exc:
            out["token"] = False
            out["error"] = str(exc)
            return out

        try:
            payload = await self._get("/api/v1/candles",
                                      {"symbol": symbol, "interval": "1d", "count": 3,
                                       "adjusted": "true"},
                                      self._chart_bucket)
            rows = _rows_of(payload)
            out["candle_envelope"] = list(payload.keys()) if isinstance(payload, dict) else "list"
            out["candle_fields"] = sorted(rows[0].keys()) if rows else []
            out["candle_sample"] = rows[0] if rows else None
        except Exception as exc:
            out["candle_error"] = str(exc)

        for endpoint, prefix in FLOW_ENDPOINTS.items():
            try:
                payload = await self._get(f"/api/v1/stocks/{symbol}/{endpoint}",
                                          {"count": 2}, self._flow_bucket)
                rows = _rows_of(payload)
                out[f"{prefix}_fields"] = sorted(_flatten(rows[0]).keys()) if rows else []
            except Exception as exc:
                out[f"{prefix}_error"] = str(exc)[:120]

        out["ok"] = bool(out.get("candle_fields"))
        out["requests"] = self.requests
        return out
