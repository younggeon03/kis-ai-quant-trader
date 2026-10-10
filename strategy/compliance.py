"""대회 최소 요건 추적 — 못 채우면 수익률과 무관하게 실격이다.

## 요건 (제16회 뱅키스 대학생 모의투자대회 AI 부문, 모두 체결 기준)

| 요건 | 기준 | 설정값 |
|---|---|---|
| OpenAPI 거래 횟수 | 5회 이상 | `MIN_TRADE_COUNT` |
| 매매금액 | 5억원 이상 | `MIN_TRADE_AMOUNT` |
| 매매일수 | 5영업일 이상 | `MIN_TRADE_DAYS` |
| 매매 종목수 | 코스피200·코스닥150 중 5종목 이상 | `MIN_TRADE_SYMBOLS` |

**주문이 아니라 체결이 기준이다.** 거부되거나 미체결로 남은 주문은 세지 않으므로
`logs/ledger.jsonl`의 `fill` 이벤트만 읽는다.

## 왜 추적이 필요한가

산수로는 여유가 있다. 1억원을 5종목에 나눠 담고 3라운드 회전하면 매수 3억 + 매도
3억 = 6억으로 5억 요건을 넘긴다. 그런데 이 전략은 1단계에서 1종목을 5영업일
보유하므로 그 구간의 회전이 거의 없다. 2단계에서 몰아서 채워야 하고, 레짐이
나빠 보유 종목 수가 줄면 종목 요건이 빠듯해진다.

**가정하지 않고 센다.** 수익률이 좋아도 요건 하나를 놓치면 전부 무의미하다.

## 초기화와의 관계

수익률 초기화가 거래 실적까지 되돌리는지는 **대회 주최 측에 확인이 필요하다.**
되돌린다면 1주차 거래가 요건에 안 잡히고 남은 25영업일로 전부 채워야 한다.
그래서 `since`로 집계 시작일을 받는다 — 확인 결과에 따라 초기화일을 넣으면 된다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import Settings, get_settings
from utils.logger import JsonLedger


@dataclass(frozen=True)
class Requirement:
    """요건 하나의 충족 현황."""

    key: str
    label: str
    current: float
    required: float
    unit: str = ""

    @property
    def met(self) -> bool:
        return self.required <= 0 or self.current >= self.required

    @property
    def shortfall(self) -> float:
        return max(0.0, self.required - self.current)

    @property
    def ratio(self) -> float:
        if self.required <= 0:
            return 1.0
        return min(1.0, self.current / self.required)

    def describe(self) -> str:
        mark = "충족" if self.met else "미충족"
        if self.unit == "원":
            return (f"{self.label}: {self.current / 1e8:.2f}억 / "
                    f"{self.required / 1e8:.2f}억 [{mark}]")
        return f"{self.label}: {self.current:,.0f}{self.unit} / {self.required:,.0f}{self.unit} [{mark}]"


@dataclass
class ComplianceStatus:
    """전체 요건 현황."""

    requirements: List[Requirement] = field(default_factory=list)
    days_remaining: int = 0
    fills: int = 0

    @property
    def all_met(self) -> bool:
        return all(r.met for r in self.requirements)

    @property
    def unmet(self) -> List[Requirement]:
        return [r for r in self.requirements if not r.met]

    @property
    def urgent(self) -> bool:
        """남은 영업일이 부족한 요건이 있는가.

        매매일수는 하루에 하나씩만 쌓이므로 남은 영업일보다 부족분이 크면 이미
        불가능하다. 다른 요건은 하루에 여러 건을 채울 수 있으므로 여유를 둔다.
        """
        for r in self.unmet:
            if r.key == "trade_days" and r.shortfall >= self.days_remaining:
                return True
            if self.days_remaining <= 5:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_met": self.all_met,
            "urgent": self.urgent,
            "days_remaining": self.days_remaining,
            "fills": self.fills,
            "requirements": [
                {"key": r.key, "label": r.label, "current": r.current,
                 "required": r.required, "unit": r.unit, "met": r.met,
                 "shortfall": r.shortfall, "ratio": round(r.ratio, 4)}
                for r in self.requirements
            ],
        }

    def describe(self) -> str:
        lines = [f"대회 요건 — 체결 {self.fills}건, 남은 {self.days_remaining}영업일"]
        for r in self.requirements:
            lines.append(f"  {r.describe()}")
        if self.all_met:
            lines.append("  → 모든 요건 충족")
        else:
            names = ", ".join(r.label for r in self.unmet)
            flag = " [긴급]" if self.urgent else ""
            lines.append(f"  → 미충족: {names}{flag}")
        return "\n".join(lines)


def _fill_date(row: Dict[str, Any]) -> Optional[dt.date]:
    raw = str(row.get("ts") or "")
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw).date()
    except ValueError:
        # 백테스트 원장은 YYYYMMDD 문자열을 쓸 수 있다.
        digits = raw[:8]
        if len(digits) == 8 and digits.isdigit():
            return dt.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    return None


def read_fills(ledger_path: str | Path,
               since: Optional[dt.date] = None,
               until: Optional[dt.date] = None) -> List[Dict[str, Any]]:
    """원장에서 체결(`fill`) 기록만 읽는다. 주문·거부는 세지 않는다."""
    rows = JsonLedger(ledger_path).read_all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("event") != "fill":
            continue
        day = _fill_date(row)
        if since is not None and (day is None or day < since):
            continue
        if until is not None and (day is None or day > until):
            continue
        out.append(row)
    return out


def evaluate(fills: List[Dict[str, Any]],
             settings: Optional[Settings] = None,
             days_remaining: int = 0) -> ComplianceStatus:
    """체결 목록에서 요건 충족 현황을 계산한다."""
    s = settings or get_settings()

    amount = 0.0
    days: set = set()
    symbols: set = set()
    for row in fills:
        try:
            qty = abs(float(row.get("qty") or 0))
            price = float(row.get("price") or 0)
        except (TypeError, ValueError):
            continue
        # 매매금액은 매수·매도를 모두 합산한 체결대금이다.
        amount += qty * price
        day = _fill_date(row)
        if day is not None:
            days.add(day)
        sym = str(row.get("symbol") or "")
        if sym:
            symbols.add(sym)

    # 요건은 "**코스피200·코스닥150 중** 5종목 이상"이다. 전체 거래 종목 수가 아니다.
    # 유니버스가 지수 외 종목까지 넓어졌으므로(대회 규정상 KOSPI·KOSDAQ 전 종목 매매 가능)
    # 지수 외 종목만 거래하다 이 요건을 놓칠 수 있다. 그래서 멤버십으로 센다.
    from strategy.eligibility import index_membership_count

    index_hits = index_membership_count(symbols, s)
    members = {x for lst in s.index_members().values() for x in lst}
    # 지수 멤버십 정보가 없으면(유니버스 파일 미설정) 전체 종목 수로 폴백한다.
    symbol_count = float(index_hits if members else len(symbols))
    symbol_label = "지수 소속 종목수" if members else "매매 종목수"

    reqs = [
        Requirement("trade_count", "거래 횟수", float(len(fills)),
                    float(s.MIN_TRADE_COUNT), "회"),
        Requirement("trade_amount", "매매금액", amount,
                    float(s.MIN_TRADE_AMOUNT), "원"),
        Requirement("trade_days", "매매일수", float(len(days)),
                    float(s.MIN_TRADE_DAYS), "일"),
        Requirement("trade_symbols", symbol_label, symbol_count,
                    float(s.MIN_TRADE_SYMBOLS), "종목"),
    ]
    return ComplianceStatus(requirements=reqs, days_remaining=days_remaining,
                            fills=len(fills))


def holding_bars_override(status: ComplianceStatus,
                          settings: Optional[Settings] = None) -> Optional[int]:
    """요건이 빠듯하면 최소 보유 기간을 줄여 회전을 늘린다.

    회전율 통제(`MIN_HOLDING_BARS`)는 거래비용을 아끼기 위한 것이지만, 요건을
    놓치면 수익률 자체가 무의미해진다. 그래서 요건이 우선한다. 충족되면 `None`을
    돌려주고 호출자는 원래 값을 쓴다.
    """
    s = settings or get_settings()
    if status.all_met or not status.urgent:
        return None
    return 0 if status.days_remaining <= 3 else max(0, int(s.MIN_HOLDING_BARS) // 2)


def status_from_ledger(ledger_path: str | Path,
                       settings: Optional[Settings] = None,
                       since: Optional[dt.date] = None,
                       now: Optional[dt.datetime | dt.date] = None) -> ComplianceStatus:
    """원장 경로에서 바로 현황을 만든다(대시보드·운영 루프용)."""
    s = settings or get_settings()
    from strategy.phase import trading_days_remaining

    if since is None:
        raw = (s.COMP_START or "").strip()
        if raw:
            try:
                since = dt.date.fromisoformat(raw)
            except ValueError:
                since = None
    fills = read_fills(ledger_path, since=since)
    return evaluate(fills, s, days_remaining=trading_days_remaining(now, s))


def index_symbols_traded(fills: List[Dict[str, Any]], members: set) -> set:
    """체결 기록에서 **지수 소속** 매매 종목 집합."""
    return {str(r.get("symbol") or "") for r in fills} & set(members)


def compliance_picks(traded: set, members: set, ranked: List[str], required: int,
                     exclude: Optional[set] = None, spare: int = 1) -> List[str]:
    """지수 종목 요건을 채우려면 오늘 무엇을 살지 — 모자란 수 + 여유 `spare`개.

    `ranked`는 우선순위(거래대금이 큰 순)다. 이미 매매했거나(`traded`) 제외(`exclude`, 보유·
    미체결) 종목은 건너뛴다. 모자라지 않으면 빈 목록이다. 여유분을 두는 이유는 거부·미체결이
    나도 그날 안에 채우기 위해서다(체결 기준이라 주문만으로는 세지 않는다).
    """
    need = int(required) - len(set(traded) & set(members))
    if need <= 0:
        return []
    skip = set(traded) | set(exclude or set())
    out = [s_ for s_ in ranked if s_ in members and s_ not in skip]
    return out[: need + max(0, int(spare))]
