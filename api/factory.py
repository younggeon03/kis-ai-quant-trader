"""브로커 팩토리.

`USE_MOCK_MODE` 값 하나로 Mock 엔진과 실서버 클라이언트를 교체한다.
상위 레이어는 `AbstractBroker` 타입만 알면 되므로 전환 비용이 0이다.
"""

from __future__ import annotations

from typing import Optional

from api.base import AbstractBroker
from config.settings import Settings, get_settings
from core.rate_limiter import PriorityRateLimiter
from utils.logger import logger


def create_broker(
    settings: Optional[Settings] = None,
    rate_limiter: Optional[PriorityRateLimiter] = None,
    simulator=None,
) -> AbstractBroker:
    settings = settings or get_settings()
    limiter = rate_limiter or PriorityRateLimiter(
        rate=settings.effective_rate, burst=settings.REST_BURST
    )
    if settings.USE_MOCK_MODE:
        from api.mock_client import MockKISClient

        logger.info("[Factory] MOCK 모드 - 내부 가상 체결 엔진을 사용합니다 (서버 호출 없음)")
        return MockKISClient(settings, simulator=simulator, rate_limiter=limiter)

    from api.client import KISClient

    logger.info(f"[Factory] 실서버 모드 - {settings.rest_base_url}")
    if not settings.credentials_ready():
        logger.error("[Factory] APP_KEY/SECRET/계좌번호가 비어 있습니다. .env를 확인하세요.")
    return KISClient(settings, rate_limiter=limiter)
