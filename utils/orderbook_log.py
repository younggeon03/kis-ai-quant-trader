"""호가·체결 테이프 전방 수집 — 다음 대회를 위해 지금부터 쌓는다.

## 왜 전방 수집밖에 없는가

**호가와 체결강도는 과거 이력을 받을 방법이 없다.**

| 경로 | 과거 조회 | 근거 |
|---|---|---|
| KIS REST `inquire-asking-price` | **불가** | 파라미터가 `FID_COND_MRKT_DIV_CODE`, `FID_INPUT_ISCD` 둘뿐 — 날짜 인자가 없다 |
| KIS WebSocket `H0STASP0` | **불가** | 구독 시점부터만 흐른다 |
| 체결강도 | **불가** | 조회·계산 코드가 전무하고 TR_MAP에 체결 틱 TR이 없다 |

그래서 이번 대회에는 쓸 수 없다(10/9 휴장이라 대회 전 수집 가능일이 10/8 하루뿐이고,
하루치로는 학습도 검증도 안 된다). **다음을 위해 지금부터 쌓는 것이 이 모듈의 전부다.**

## 체결강도를 여기서 계산하지 않는 이유

체결을 매수/매도 주도로 분류하는 규칙이 하나가 아니다(`가격 >= 매도1호가`냐
`가격 > 중간가`냐에 따라 결과가 달라진다). 규칙을 지금 박아 넣으면 나중에 다른
규칙으로 재계산할 수 없다. 그래서 **원재료만 저장한다** — 가격, 누적거래량,
매수1·매도1호가. 분류는 분석 시점에 고른다.

## 샘플링하는 이유

호가는 틱마다 흐른다. 40종목 x 390분을 전부 저장하면 하루에 수백만 행이 되고,
대부분이 직전 행과 거의 같다. `interval_sec`(기본 5초)마다 종목당 한 행만 남긴다
— 40종목 x 390분 x 12회 = 약 19만 행/일로 묶인다.

## 저장 형식

```
data/cache/orderbook/<종목>.csv   ts, ask1..5, bid1..5, ask_qty1..5, bid_qty1..5,
                                  ask_qty_sum, bid_qty_sum      (10단 합)
data/cache/tape/<종목>.csv        ts, price, cum_volume, bid1, ask1
```

`ts`는 `YYYYMMDDHHMMSS` 문자열이다(프로젝트 규약: 일봉 `YYYYMMDD`,
분봉 `YYYYMMDDHHMM`). **과거→최신 정렬**이며 append만 한다.

## 실패해도 매매를 멈추지 않는다

수집은 부가 기능이다. 디스크 오류·권한 문제로 기록이 실패해도 **예외를 바깥으로
던지지 않는다** — 던지면 호가 콜백이 WebSocket 수신 루프를 깨고 시세가 끊긴다.
실패는 세어서 한 번만 경고한다.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import logger

LEVELS = 5          # 저장할 호가 단수(10단 중 상위 5단 + 합계)

OB_COLUMNS: List[str] = (
    ["ts"]
    + [f"ask{i + 1}" for i in range(LEVELS)]
    + [f"bid{i + 1}" for i in range(LEVELS)]
    + [f"ask_qty{i + 1}" for i in range(LEVELS)]
    + [f"bid_qty{i + 1}" for i in range(LEVELS)]
    + ["ask_qty_sum", "bid_qty_sum"]
)
TAPE_COLUMNS: List[str] = ["ts", "price", "cum_volume", "bid1", "ask1"]


def _fmt(ts: dt.datetime) -> str:
    return ts.strftime("%Y%m%d%H%M%S")


class OrderbookRecorder:
    """호가·체결 테이프를 종목별 CSV로 append 한다.

    `observe_*`는 샘플링 간격 안이면 아무것도 하지 않고 `False`를 돌려준다.
    버퍼가 `flush_rows`를 넘으면 디스크에 쓴다 — 틱마다 파일을 열면 느리다.
    """

    def __init__(self, out_dir: str | Path, interval_sec: float = 5.0,
                 flush_rows: int = 400) -> None:
        self.root = Path(out_dir)
        self.interval = float(interval_sec)
        self.flush_rows = int(flush_rows)
        self._ob: Dict[str, List[List[Any]]] = {}
        self._tape: Dict[str, List[List[Any]]] = {}
        self._last_ob: Dict[str, dt.datetime] = {}
        self._last_tape: Dict[str, dt.datetime] = {}
        self._rows = 0
        self._errors = 0
        self._warned = False

    # ------------------------------------------------------------ 수신
    def observe_orderbook(self, ob: Dict[str, Any],
                          ts: Optional[dt.datetime] = None) -> bool:
        """`WebSocketClient._parse_orderbook()`의 출력을 받는다."""
        sym = str(ob.get("symbol") or "")
        if not sym:
            return False
        now = ts or dt.datetime.now()
        if not self._due(self._last_ob, sym, now):
            return False
        ask = list(ob.get("ask") or [])
        bid = list(ob.get("bid") or [])
        aq = list(ob.get("ask_qty") or [])
        bq = list(ob.get("bid_qty") or [])
        row: List[Any] = [_fmt(now)]
        for src in (ask, bid, aq, bq):
            row.extend(float(src[i]) if i < len(src) else 0.0 for i in range(LEVELS))
        row.append(float(sum(aq)))      # 10단 전체 합 — 상위 5단만 저장하므로 따로 남긴다
        row.append(float(sum(bq)))
        self._ob.setdefault(sym, []).append(row)
        self._bump()
        return True

    def observe_tape(self, symbol: str, price: float, cum_volume: float,
                     bid: float = 0.0, ask: float = 0.0,
                     ts: Optional[dt.datetime] = None) -> bool:
        """체결 틱의 원재료. 체결강도 분류는 **분석 시점에** 고른다."""
        if not symbol or price <= 0:
            return False
        now = ts or dt.datetime.now()
        if not self._due(self._last_tape, symbol, now):
            return False
        self._tape.setdefault(symbol, []).append(
            [_fmt(now), float(price), float(cum_volume), float(bid), float(ask)])
        self._bump()
        return True

    def _due(self, last: Dict[str, dt.datetime], sym: str,
             now: dt.datetime) -> bool:
        prev = last.get(sym)
        if prev is not None and (now - prev).total_seconds() < self.interval:
            return False
        last[sym] = now
        return True

    def _bump(self) -> None:
        self._rows += 1
        if self._rows >= self.flush_rows:
            self.flush()

    # ------------------------------------------------------------ 기록
    def flush(self) -> int:
        """버퍼를 디스크에 쓴다. 쓴 행 수를 돌려준다.

        **예외를 바깥으로 던지지 않는다** — 호가 콜백에서 불리므로 던지면
        WebSocket 수신 루프가 깨지고 시세가 끊긴다.
        """
        written = 0
        for sub, buf, cols in (("orderbook", self._ob, OB_COLUMNS),
                               ("tape", self._tape, TAPE_COLUMNS)):
            if not buf:
                continue
            d = self.root / sub
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                self._fail()
                buf.clear()
                continue
            for sym, rows in list(buf.items()):
                if not rows:
                    continue
                path = d / f"{sym}.csv"
                try:
                    new = not path.exists()
                    with path.open("a", encoding="utf-8", newline="") as fh:
                        if new:
                            fh.write(",".join(cols) + "\n")
                        for r in rows:
                            fh.write(",".join(str(x) for x in r) + "\n")
                    written += len(rows)
                except Exception:
                    self._fail()
                finally:
                    rows.clear()
        self._rows = 0
        return written

    def _fail(self) -> None:
        self._errors += 1
        if not self._warned:
            self._warned = True
            logger.warning("[호가수집] 기록 실패 — 수집은 부가 기능이므로 "
                           "매매는 계속한다. 디스크·권한을 확인하십시오")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "buffered": self._rows,
            "symbols_ob": len(self._ob),
            "symbols_tape": len(self._tape),
            "errors": self._errors,
            "interval_sec": self.interval,
        }
