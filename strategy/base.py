"""전략 추상 클래스.

백테스트 엔진과 실시간 엔진이 **같은 전략 객체**를 그대로 사용한다.
따라서 "백테스트에서는 잘 되는데 실매매에서 다르게 동작하는" 불일치가 구조적으로 없다.
"""

from __future__ import annotations

import abc
import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd

from core.events import Signal


class AbstractStrategy(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def generate_signals(
        self,
        panel: Dict[str, pd.DataFrame],
        holdings: Optional[Dict[str, Any]] = None,
        ts: Optional[dt.datetime] = None,
    ) -> List[Signal]:
        """종목별 OHLCV 패널(과거→현재)을 받아 신호 리스트를 반환한다.

        Args:
            panel: {종목코드: OHLCV DataFrame}. 마지막 행이 현재 봉이다.
            holdings: 현재 보유 종목(청산 신호 판단에 사용).
            ts: 현재 시각.
        """

    def on_fill(self, fill) -> None:  # pragma: no cover - 선택적 훅
        """체결 이벤트 훅(온라인 학습/상태 갱신 등)."""

    def warmup_bars(self) -> int:
        """신호 생성에 필요한 최소 봉 수."""
        return 80

    def snapshot(self) -> Dict[str, Any]:
        return {"name": self.name}
