"""KIS OAuth 2.0 인증 관리자.

핵심 고려사항
 1. 접근토큰은 발급 후 24시간 유효하지만, KIS는 **짧은 간격의 재발급을 거부**한다
    (EGW00133 등). 따라서 디스크에 캐싱해 프로세스를 재시작해도 재사용한다.
 2. 만료 직전에 토큰이 끊기면 청산 주문이 실패하므로, 백그라운드 태스크가
    만료 `REFRESH_MARGIN` 전에 **선제적으로** 재발급한다.
 3. WebSocket은 별도의 `approval_key`를 사용한다(접근토큰과 다름).
 4. Mock 모드에서는 네트워크 호출 없이 동일한 인터페이스로 가짜 토큰을 준다.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.settings import Settings
from utils.logger import logger

REFRESH_MARGIN = dt.timedelta(minutes=30)   # 만료 30분 전 재발급
MIN_REISSUE_INTERVAL = dt.timedelta(seconds=70)  # 재발급 최소 간격(서버 제한 회피)


@dataclass
class Token:
    access_token: str
    expires_at: dt.datetime
    issued_at: dt.datetime
    token_type: str = "Bearer"

    @property
    def is_expired(self) -> bool:
        return dt.datetime.now(dt.timezone.utc) >= self.expires_at

    @property
    def needs_refresh(self) -> bool:
        return dt.datetime.now(dt.timezone.utc) >= (self.expires_at - REFRESH_MARGIN)

    @property
    def seconds_left(self) -> float:
        return max(0.0, (self.expires_at - dt.datetime.now(dt.timezone.utc)).total_seconds())

    @property
    def header(self) -> str:
        return f"Bearer {self.access_token}"

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "expires_at": self.expires_at.isoformat(),
            "issued_at": self.issued_at.isoformat(),
            "token_type": self.token_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Token":
        return cls(
            access_token=data["access_token"],
            expires_at=dt.datetime.fromisoformat(data["expires_at"]),
            issued_at=dt.datetime.fromisoformat(data["issued_at"]),
            token_type=data.get("token_type", "Bearer"),
        )


class AuthError(RuntimeError):
    pass


class TokenManager:
    """접근토큰 + WebSocket approval_key 수명주기 관리."""

    def __init__(self, settings: Settings, http_client=None):
        self.settings = settings
        self._token: Optional[Token] = None
        self._approval_key: Optional[str] = None
        self._lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None
        self._external_client = http_client
        self._client = None
        self._last_issue_attempt: Optional[dt.datetime] = None
        self._cache_path = Path(settings.TOKEN_CACHE_DIR) / f"token_{settings.KIS_ENV}.json"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ HTTP
    async def _get_client(self):
        if self._external_client is not None:
            return self._external_client
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.settings.rest_base_url,
                timeout=self.settings.HTTP_TIMEOUT,
                headers={"content-type": "application/json; charset=utf-8"},
            )
        return self._client

    # ------------------------------------------------------------------ 캐시
    def _load_cache(self) -> Optional[Token]:
        if not self._cache_path.exists():
            return None
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            token = Token.from_dict(data)
            if token.is_expired:
                logger.info("[Auth] 캐시된 토큰이 만료되어 폐기합니다.")
                return None
            logger.info(f"[Auth] 캐시 토큰 재사용 (남은시간 {token.seconds_left / 3600:.1f}h)")
            return token
        except Exception as exc:
            logger.warning(f"[Auth] 토큰 캐시 로드 실패: {exc}")
            return None

    def _save_cache(self, token: Token) -> None:
        try:
            self._cache_path.write_text(
                json.dumps(token.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            try:
                os.chmod(self._cache_path, 0o600)
            except OSError:
                pass  # Windows에서는 무시
        except Exception as exc:
            logger.warning(f"[Auth] 토큰 캐시 저장 실패: {exc}")

    # ------------------------------------------------------------------ 발급
    def _mock_token(self) -> Token:
        now = dt.datetime.now(dt.timezone.utc)
        return Token(
            access_token="MOCK-ACCESS-TOKEN-" + now.strftime("%Y%m%d%H%M%S"),
            expires_at=now + dt.timedelta(hours=24),
            issued_at=now,
        )

    async def _issue_token(self) -> Token:
        if self.settings.USE_MOCK_MODE:
            return self._mock_token()

        if not self.settings.KIS_APP_KEY or not self.settings.KIS_APP_SECRET:
            raise AuthError("KIS_APP_KEY / KIS_APP_SECRET 이 설정되지 않았습니다 (.env 확인)")

        # 재발급 최소 간격 보호
        now = dt.datetime.now(dt.timezone.utc)
        if self._last_issue_attempt and (now - self._last_issue_attempt) < MIN_REISSUE_INTERVAL:
            wait = (MIN_REISSUE_INTERVAL - (now - self._last_issue_attempt)).total_seconds()
            logger.warning(f"[Auth] 재발급 간격 보호로 {wait:.0f}초 대기합니다.")
            await asyncio.sleep(wait)
        self._last_issue_attempt = dt.datetime.now(dt.timezone.utc)

        client = await self._get_client()
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.settings.KIS_APP_KEY,
            "appsecret": self.settings.KIS_APP_SECRET,
        }
        last_error: Optional[Exception] = None
        for attempt in range(1, self.settings.HTTP_MAX_RETRY + 1):
            try:
                resp = await client.post("/oauth2/tokenP", json=payload)
                data = resp.json()
                if resp.status_code != 200 or "access_token" not in data:
                    raise AuthError(f"토큰 발급 실패 [{resp.status_code}] {data}")
                issued = dt.datetime.now(dt.timezone.utc)
                expires_in = int(data.get("expires_in", 86400))
                token = Token(
                    access_token=data["access_token"],
                    expires_at=issued + dt.timedelta(seconds=expires_in),
                    issued_at=issued,
                    token_type=data.get("token_type", "Bearer"),
                )
                logger.success(f"[Auth] 접근토큰 발급 완료 (유효 {expires_in / 3600:.1f}h)")
                return token
            except Exception as exc:
                last_error = exc
                backoff = self.settings.HTTP_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(f"[Auth] 토큰 발급 재시도 {attempt}/{self.settings.HTTP_MAX_RETRY}: {exc}")
                if attempt < self.settings.HTTP_MAX_RETRY:
                    await asyncio.sleep(backoff)
        raise AuthError(f"토큰 발급 최종 실패: {last_error}")

    async def get_token(self, force: bool = False) -> Token:
        """유효한 토큰을 반환한다(필요 시 발급). 동시 호출은 락으로 1회만 수행."""
        async with self._lock:
            if not force and self._token and not self._token.needs_refresh:
                return self._token
            if not force and self._token is None:
                cached = self._load_cache()
                if cached and not cached.needs_refresh:
                    self._token = cached
                    return cached
            token = await self._issue_token()
            self._token = token
            if not self.settings.USE_MOCK_MODE:
                self._save_cache(token)
            return token

    async def get_authorization(self) -> str:
        token = await self.get_token()
        return token.header

    # ------------------------------------------------------------------ approval_key
    async def get_approval_key(self, force: bool = False) -> str:
        """WebSocket 접속용 approval_key."""
        if self._approval_key and not force:
            return self._approval_key
        if self.settings.USE_MOCK_MODE:
            self._approval_key = "MOCK-APPROVAL-KEY-0000000000000000"
            return self._approval_key

        client = await self._get_client()
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.settings.KIS_APP_KEY,
            "secretkey": self.settings.KIS_APP_SECRET,
        }
        resp = await client.post("/oauth2/Approval", json=payload)
        data = resp.json()
        if resp.status_code != 200 or "approval_key" not in data:
            raise AuthError(f"approval_key 발급 실패 [{resp.status_code}] {data}")
        self._approval_key = data["approval_key"]
        logger.success("[Auth] WebSocket approval_key 발급 완료")
        return self._approval_key

    # ------------------------------------------------------------------ 백그라운드
    async def _refresh_loop(self) -> None:
        """만료 전 선제 재발급 루프."""
        while True:
            try:
                token = await self.get_token()
                sleep_for = max(60.0, token.seconds_left - REFRESH_MARGIN.total_seconds())
                logger.debug(f"[Auth] 다음 토큰 점검까지 {sleep_for / 60:.1f}분")
                await asyncio.sleep(sleep_for)
                await self.get_token(force=True)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[Auth] 토큰 갱신 루프 오류: {exc}. 60초 후 재시도")
                await asyncio.sleep(60)

    def start_auto_refresh(self) -> None:
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh_loop(), name="token-refresh")
            logger.info("[Auth] 토큰 자동 갱신 태스크 시작")

    async def close(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except (asyncio.CancelledError, Exception):
                pass
            self._refresh_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def snapshot(self) -> dict:
        return {
            "has_token": self._token is not None,
            "seconds_left": round(self._token.seconds_left, 0) if self._token else 0,
            "has_approval_key": self._approval_key is not None,
            "env": self.settings.KIS_ENV,
            "mock": self.settings.USE_MOCK_MODE,
        }
