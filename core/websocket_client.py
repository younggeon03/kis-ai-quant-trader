"""KIS 실시간 시세 WebSocket 엔진.

지원 TR
 · H0STCNT0 : 주식 체결가 (실시간 현재가/거래량)
 · H0STASP0 : 주식 호가 (매수/매도 1~10호가)
 · H0STCNI0 / H0STCNI9 : 체결통보(실전/모의) — AES-256-CBC 복호화 필요

운영 고려사항
 · 세션당 실시간 등록 건수 제한(41건)을 슬롯 관리로 방어한다.
 · PINGPONG 프레임에 즉시 응답하지 않으면 서버가 연결을 끊는다.
 · 끊기면 지수 백오프로 재접속하고 **구독 목록을 자동 복원**한다.
 · 연결이 불안정하면 상위 엔진은 REST 폴링으로 폴백하므로 매매가 멈추지 않는다.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from config.settings import (
    WS_TR_ORDERBOOK,
    WS_TR_TRADE,
    Settings,
)
from core.auth import TokenManager
from core.events import Quote
from utils.krx_calendar import KST
from utils.logger import logger

TradeCallback = Callable[[Quote], Awaitable[None]]
OrderbookCallback = Callable[[Dict[str, Any]], Awaitable[None]]
NoticeCallback = Callable[[Dict[str, Any]], Awaitable[None]]


def _aes_cbc_decrypt(key: str, iv: str, cipher_b64: str) -> str:
    """체결통보 복호화(AES-256-CBC + PKCS7). `cryptography` 사용."""
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    cipher = Cipher(algorithms.AES(key.encode("utf-8")), modes.CBC(iv.encode("utf-8")))
    decryptor = cipher.decryptor()
    raw = decryptor.update(base64.b64decode(cipher_b64)) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    return (unpadder.update(raw) + unpadder.finalize()).decode("utf-8")


@dataclass
class Subscription:
    tr_id: str
    tr_key: str
    priority: int = 5  # 값이 작을수록 유지 우선순위 높음

    @property
    def slot_key(self) -> Tuple[str, str]:
        return (self.tr_id, self.tr_key)


@dataclass
class WSStats:
    connects: int = 0
    disconnects: int = 0
    messages: int = 0
    trades: int = 0
    orderbooks: int = 0
    notices: int = 0
    errors: int = 0
    last_message_at: Optional[dt.datetime] = None
    subscriptions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connects": self.connects,
            "disconnects": self.disconnects,
            "messages": self.messages,
            "trades": self.trades,
            "orderbooks": self.orderbooks,
            "notices": self.notices,
            "errors": self.errors,
            "subscriptions": self.subscriptions,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
        }


class KISWebSocketClient:
    """실시간 시세 스트리밍 클라이언트."""

    def __init__(
        self,
        settings: Settings,
        token_manager: Optional[TokenManager] = None,
        on_trade: Optional[TradeCallback] = None,
        on_orderbook: Optional[OrderbookCallback] = None,
        on_notice: Optional[NoticeCallback] = None,
    ):
        self.settings = settings
        self.auth = token_manager or TokenManager(settings)
        self.on_trade = on_trade
        self.on_orderbook = on_orderbook
        self.on_notice = on_notice

        self._ws = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._subs: Dict[Tuple[str, str], Subscription] = {}
        self._active: Set[Tuple[str, str]] = set()
        self._aes_key: Optional[str] = None
        self._aes_iv: Optional[str] = None
        self.stats = WSStats()
        self.connected = False

    # ------------------------------------------------------------------ 연결
    async def _open(self):
        try:
            from websockets.asyncio.client import connect as ws_connect
        except ImportError:  # pragma: no cover - 구버전 websockets 호환
            from websockets import connect as ws_connect  # type: ignore

        return await ws_connect(
            self.settings.ws_url,
            ping_interval=None,   # KIS는 자체 PINGPONG 프로토콜을 사용
            close_timeout=5,
            max_size=2 ** 22,
        )

    def _build_message(self, sub: Subscription, subscribe: bool, approval_key: str) -> str:
        return json.dumps(
            {
                "header": {
                    "approval_key": approval_key,
                    "custtype": "P",
                    "tr_type": "1" if subscribe else "2",
                    "content-type": "utf-8",
                },
                "body": {"input": {"tr_id": sub.tr_id, "tr_key": sub.tr_key}},
            }
        )

    # ------------------------------------------------------------------ 구독 관리
    @property
    def slots_used(self) -> int:
        return len(self._active)

    @property
    def slots_available(self) -> int:
        return max(0, self.settings.WS_MAX_SUBSCRIPTIONS - self.slots_used)

    async def subscribe(self, tr_id: str, tr_key: str, priority: int = 5) -> bool:
        """실시간 등록. 슬롯이 부족하면 우선순위가 낮은 구독을 해지하고 자리를 만든다."""
        sub = Subscription(tr_id, tr_key, priority)
        if sub.slot_key in self._subs:
            return True

        if self.slots_available <= 0:
            victim = self._lowest_priority_sub(exclude_priority=priority)
            if victim is None:
                logger.warning(f"[WS] 구독 슬롯 부족으로 등록 실패: {tr_id}/{tr_key}")
                return False
            await self.unsubscribe(victim.tr_id, victim.tr_key)

        self._subs[sub.slot_key] = sub
        if self.connected and self._ws is not None:
            try:
                approval = await self.auth.get_approval_key()
                await self._ws.send(self._build_message(sub, True, approval))
                self._active.add(sub.slot_key)
            except Exception as exc:
                logger.warning(f"[WS] 구독 전송 실패 {tr_id}/{tr_key}: {exc}")
                return False
        self.stats.subscriptions = len(self._subs)
        return True

    def _lowest_priority_sub(self, exclude_priority: int) -> Optional[Subscription]:
        candidates = [s for s in self._subs.values() if s.priority > exclude_priority]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.priority)

    async def unsubscribe(self, tr_id: str, tr_key: str) -> None:
        key = (tr_id, tr_key)
        sub = self._subs.pop(key, None)
        self._active.discard(key)
        self.stats.subscriptions = len(self._subs)
        if sub and self.connected and self._ws is not None:
            try:
                approval = await self.auth.get_approval_key()
                await self._ws.send(self._build_message(sub, False, approval))
            except Exception as exc:
                logger.debug(f"[WS] 구독 해지 전송 실패: {exc}")

    async def subscribe_symbols(self, symbols: List[str], orderbook: bool = False) -> None:
        """종목 리스트를 일괄 구독(체결가 우선, 여유가 있으면 호가까지)."""
        for i, symbol in enumerate(symbols):
            await self.subscribe(WS_TR_TRADE, symbol, priority=1 + i // 10)
        if orderbook:
            for i, symbol in enumerate(symbols):
                if self.slots_available <= 2:
                    break
                await self.subscribe(WS_TR_ORDERBOOK, symbol, priority=6 + i // 10)

    async def subscribe_notice(self) -> bool:
        """체결통보 구독(HTS ID 필요)."""
        if not self.settings.KIS_HTS_ID:
            logger.warning("[WS] KIS_HTS_ID가 없어 체결통보 구독을 건너뜁니다.")
            return False
        return await self.subscribe(self.settings.ws_notice_tr, self.settings.KIS_HTS_ID, priority=0)

    async def _resubscribe_all(self) -> None:
        approval = await self.auth.get_approval_key()
        self._active.clear()
        for sub in sorted(self._subs.values(), key=lambda s: s.priority):
            try:
                await self._ws.send(self._build_message(sub, True, approval))
                self._active.add(sub.slot_key)
                await asyncio.sleep(0.03)  # 연속 등록 시 서버 부하 방지
            except Exception as exc:
                logger.warning(f"[WS] 재구독 실패 {sub.tr_id}/{sub.tr_key}: {exc}")
        logger.info(f"[WS] 구독 복원 완료 ({len(self._active)}건)")

    # ------------------------------------------------------------------ 메시지 파싱
    @staticmethod
    def _parse_trade(body: str, count: int = 0) -> List[Quote]:
        """H0STCNT0 체결 데이터 파싱 (여러 건이 ^로 이어져 올 수 있음).

        **한 건의 필드 수는 머리의 건수(`count`)로 나눠 구한다.** 문서상 46개라 고정했더니
        모의서버 응답에서 여러 건이 묶인 메시지의 두 번째 건부터 어긋나 `000020` 같은
        가짜 종목코드에 20원·1,615억원 같은 가격이 들어왔다(10/20~10/29 원장 `bad_price`
        23,890건, 10/20 최고 자산 5.34억 사건의 출처로 추정).
        """
        fields = body.split("^")
        quotes: List[Quote] = []
        stride = 46
        if count > 1 and len(fields) >= count * 14:
            stride = len(fields) // count
        count = max(1, len(fields) // stride)
        for i in range(count):
            chunk = fields[i * stride: (i + 1) * stride]
            if len(chunk) < 14:
                continue
            try:
                quotes.append(
                    Quote(
                        symbol=chunk[0].strip().zfill(6),
                        price=float(chunk[2] or 0),
                        change_pct=float(chunk[5] or 0),
                        open=float(chunk[7] or 0),
                        high=float(chunk[8] or 0),
                        low=float(chunk[9] or 0),
                        ask=float(chunk[10] or 0),
                        bid=float(chunk[11] or 0),
                        volume=float(chunk[13] or 0),
                        ts=dt.datetime.now(tz=KST),
                    )
                )
            except (ValueError, IndexError):
                continue
        return quotes

    @staticmethod
    def _parse_orderbook(body: str) -> Optional[Dict[str, Any]]:
        """H0STASP0 호가 데이터 파싱."""
        f = body.split("^")
        if len(f) < 43:
            return None
        try:
            return {
                "symbol": f[0].strip().zfill(6),
                "time": f[1],
                "ask": [float(f[3 + i] or 0) for i in range(10)],
                "bid": [float(f[13 + i] or 0) for i in range(10)],
                "ask_qty": [float(f[23 + i] or 0) for i in range(10)],
                "bid_qty": [float(f[33 + i] or 0) for i in range(10)],
            }
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _parse_notice(plain: str) -> Dict[str, Any]:
        """체결통보 평문 파싱."""
        f = plain.split("^")
        keys = [
            "cust_id", "acnt_no", "oder_no", "ooder_no", "seln_byov_cls", "rctf_cls",
            "oder_kind", "oder_cond", "stck_shrn_iscd", "cntg_qty", "cntg_unpr",
            "stck_cntg_hour", "rfus_yn", "cntg_yn", "acpt_yn", "brnc_no", "oder_qty",
        ]
        data = {k: (f[i] if i < len(f) else "") for i, k in enumerate(keys)}
        return data

    async def _handle_realtime(self, raw: str) -> None:
        parts = raw.split("|")
        if len(parts) < 4:
            return
        encrypted, tr_id, _count, body = parts[0], parts[1], parts[2], parts[3]

        if tr_id in (self.settings.ws_notice_tr,):
            if encrypted == "1" and self._aes_key and self._aes_iv:
                try:
                    body = _aes_cbc_decrypt(self._aes_key, self._aes_iv, body)
                except Exception as exc:
                    self.stats.errors += 1
                    logger.warning(f"[WS] 체결통보 복호화 실패: {exc}")
                    return
            self.stats.notices += 1
            if self.on_notice:
                await self.on_notice(self._parse_notice(body))
            return

        if tr_id == WS_TR_TRADE:
            self.stats.trades += 1
            if self.on_trade:
                try:
                    n = int(_count)
                except ValueError:
                    n = 0
                for quote in self._parse_trade(body, n):
                    # 구독한 종목만 받는다 — 필드가 어긋나면 엉뚱한 코드에 엉뚱한 가격이 붙는다
                    if (WS_TR_TRADE, quote.symbol) not in self._subs:
                        self.stats.errors += 1
                        continue
                    await self.on_trade(quote)
        elif tr_id == WS_TR_ORDERBOOK:
            self.stats.orderbooks += 1
            parsed = self._parse_orderbook(body)
            if parsed and self.on_orderbook:
                await self.on_orderbook(parsed)

    async def _handle_control(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        header = msg.get("header", {})
        tr_id = header.get("tr_id", "")

        if tr_id == "PINGPONG":
            # 서버 핑에 즉시 응답하지 않으면 연결이 종료된다.
            await self._ws.send(raw)
            return

        body = msg.get("body") or {}
        output = body.get("output") or {}
        if output.get("key") and output.get("iv"):
            # 구독 응답마다 같은 키가 와서(41건) 바뀔 때만 남긴다
            if (output["key"], output["iv"]) != (self._aes_key, self._aes_iv):
                logger.info("[WS] 체결통보 복호화 키 수신 완료")
            self._aes_key = output["key"]
            self._aes_iv = output["iv"]

        rt_cd = str(body.get("rt_cd", ""))
        if rt_cd not in ("", "0"):
            self.stats.errors += 1
            logger.warning(f"[WS] 구독 응답 오류 {tr_id}: {body.get('msg1')}")

    # ------------------------------------------------------------------ 루프
    async def _run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            try:
                self._ws = await self._open()
                self.connected = True
                self.stats.connects += 1
                delay = 1.0
                logger.success(f"[WS] 연결 성공 {self.settings.ws_url}")
                await self._resubscribe_all()

                # **조용히 멈춘 연결을 끊는다.** `ping_interval=None`이라 상대가 사라져도
                # 오류가 안 난다 — 10/29 09:50 뒤 "연결됨" 상태로 메시지가 0건이었고,
                # 체결통보가 끊겨 원장에 체결이 안 남고 판 종목이 자리를 막았다(11/04 확인).
                while True:
                    silence = self._silence_limit()
                    try:
                        raw = await asyncio.wait_for(self._ws.recv(), timeout=silence)
                    except asyncio.TimeoutError:
                        raise ConnectionError(f"{silence:.0f}초 동안 메시지 없음 — 멈춘 연결로 보고 다시 연결")
                    self.stats.messages += 1
                    self.stats.last_message_at = dt.datetime.now(tz=KST)
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")
                    if raw.startswith("0") or raw.startswith("1"):
                        await self._handle_realtime(raw)
                    else:
                        await self._handle_control(raw)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.stats.errors += 1
                logger.warning(f"[WS] 연결 오류: {exc}")
            finally:
                self.connected = False
                self.stats.disconnects += 1
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None

            if self._stop.is_set():
                break
            logger.info(f"[WS] {delay:.1f}초 후 재연결을 시도합니다.")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass
            delay = min(self.settings.WS_RECONNECT_MAX_DELAY, delay * 2)

    def _silence_limit(self, now: Optional[dt.datetime] = None) -> float:
        """멈춘 연결로 볼 무음 시간(초). 체결이 끊임없이 오는 평일 08:00~16:00만 짧게 잡는다.

        장 밖에는 KIS PINGPONG 간격을 몰라(확인 안 됨) 짧게 잡으면 밤새 재연결·재구독이 반복될 수 있다.
        """
        now = now or dt.datetime.now(tz=KST)
        if now.weekday() < 5 and 8 <= now.hour < 16:
            return float(self.settings.WS_SILENCE_RECONNECT_SEC)
        return max(float(self.settings.WS_SILENCE_RECONNECT_SEC), 1800.0)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="ws-client")
            logger.info("[WS] 스트리밍 태스크 시작")

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self.connected = False

    def is_healthy(self, max_silence: float = 60.0) -> bool:
        """일정 시간 메시지가 없으면 비정상으로 간주(REST 폴백 판단 기준)."""
        if not self.connected:
            return False
        if self.stats.last_message_at is None:
            return True
        silence = (dt.datetime.now(tz=KST) - self.stats.last_message_at).total_seconds()
        return silence <= max_silence

    def snapshot(self) -> Dict[str, Any]:
        return {"connected": self.connected, "slots": f"{self.slots_used}/{self.settings.WS_MAX_SUBSCRIPTIONS}",
                **self.stats.to_dict()}
