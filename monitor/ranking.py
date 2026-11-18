"""관제 화면 순위판 — 지수(코스피200·코스닥150) 종목의 거래대금·등락률 순위. 순수 로직.

직접 매매(2026-11-18~, 사용자 결정)를 돕는 화면 전용 계산이다. **매매에는 쓰지 않는다.**
엔진(`main.py::_rank_poll`)이 복수시세(`get_turnovers`) 한 바퀴를 넘기면 표 한 장을 만든다.

- 거래대금은 KIS 누적거래대금(`acml_tr_pbmn`) — 장 시작부터 지금까지의 실제 체결대금이다
- 등락률은 전일 종가 대비 — 복수시세의 전일 종가(`inter2_prdy_clpr`)가 없으면 엔진이 가진 전일 종가로 대신한다
- 시가총액은 유니버스 파일 기준일(9/29) 값이라 **소형주 표시의 기준으로만** 쓴다
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

SMALL_CAP_KRW = 1e12          # 화면의 "소형주" 기준 — 시가총액 1조 원 이하


def build_rank_rows(snap: Mapping[str, Mapping[str, Any]],
                    index_of: Mapping[str, str],
                    names: Optional[Mapping[str, str]] = None,
                    market_cap: Optional[Mapping[str, float]] = None,
                    prev_close: Optional[Mapping[str, float]] = None,
                    held: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """복수시세 결과 → 지수 종목 행 목록(거래대금 큰 순).

    `index_of`(종목 → 지수 이름)에 없는 종목은 버린다 — 순위는 지수 안에서만 본다.
    시세가 없거나(0원) 전일 종가를 모르는 종목의 등락률은 `None`이다(0%로 쓰면 순위가 거짓이 된다).
    """
    names = names or {}
    market_cap = market_cap or {}
    prev_close = prev_close or {}
    held = set(held)
    rows: List[Dict[str, Any]] = []
    for symbol, q in snap.items():
        index = index_of.get(symbol)
        if not index:
            continue
        price = float(q.get("price") or 0.0)
        if price <= 0:
            continue
        pc = float(q.get("prev_close") or 0.0) or float(prev_close.get(symbol) or 0.0)
        chg = (price / pc - 1.0) * 100.0 if pc > 0 else None
        mcap = float(market_cap.get(symbol) or 0.0)
        rows.append({
            "symbol": symbol,
            "name": str(q.get("name") or names.get(symbol) or ""),
            "index": index,
            "price": price,
            "chg_pct": round(chg, 2) if chg is not None else None,
            "turnover": float(q.get("turnover") or 0.0),
            "mcap": mcap,
            "small": 0 < mcap <= SMALL_CAP_KRW,
            "held": symbol in held,
        })
    rows.sort(key=lambda r: (-r["turnover"], r["symbol"]))
    return rows


def pick_rows(rows: List[Dict[str, Any]], picks: Iterable[str]) -> List[Dict[str, Any]]:
    """관심 종목 행 — 관심 목록 순서대로, 거래대금·등락률 순위를 붙인다. 시세가 없는 종목은 빈 행."""
    by_sym = {r["symbol"]: r for r in rows}
    tv_rank = {r["symbol"]: i + 1 for i, r in enumerate(rows)}
    ranked_chg = sorted((r for r in rows if r["chg_pct"] is not None),
                        key=lambda r: (-r["chg_pct"], r["symbol"]))
    chg_rank = {r["symbol"]: i + 1 for i, r in enumerate(ranked_chg)}
    out: List[Dict[str, Any]] = []
    for symbol in picks:
        row = dict(by_sym.get(symbol) or {"symbol": symbol, "name": "", "index": "", "price": 0.0,
                                          "chg_pct": None, "turnover": 0.0, "mcap": 0.0,
                                          "small": False, "held": False})
        row["tv_rank"] = tv_rank.get(symbol)
        row["chg_rank"] = chg_rank.get(symbol)
        out.append(row)
    return out


def cross_rows(rows: List[Dict[str, Any]], k: int = 10, key: str = "turnover") -> List[Dict[str, Any]]:
    """거래대금 상위 ∩ 등락률 상위 k종목(11/18 사용자 요청).

    두 순위 중 나쁜 쪽(`max(거래대금 순위, 등락률 순위)`)이 작은 순으로 k개 — "상위 N 두 목록의 교집합이 k개가 되는
    가장 작은 N"의 교집합과 같다. 같으면 두 순위 합, 그다음 거래대금 순위. 등락률이 0 이하(하락·보합)는 넣지 않는다.
    `key` 로 거래대금 기준을 고른다 — "turnover"(09:00부터 누적) 또는 "tv_win"(직전 N분, `recent_turnover`).
    """
    by_tv = sorted(rows, key=lambda r: (-float(r.get(key) or 0.0), r["symbol"]))
    tv_rank = {r["symbol"]: i + 1 for i, r in enumerate(by_tv)}
    ranked_chg = sorted((r for r in rows if r["chg_pct"] is not None),
                        key=lambda r: (-r["chg_pct"], r["symbol"]))
    chg_rank = {r["symbol"]: i + 1 for i, r in enumerate(ranked_chg)}
    cand = [r for r in ranked_chg if r["chg_pct"] > 0]
    cand.sort(key=lambda r: (max(tv_rank[r["symbol"]], chg_rank[r["symbol"]]),
                             tv_rank[r["symbol"]] + chg_rank[r["symbol"]], tv_rank[r["symbol"]]))
    return [dict(r, tv_rank=tv_rank[r["symbol"]], chg_rank=chg_rank[r["symbol"]]) for r in cand[:k]]


def recent_turnover(history: List[Tuple[float, Dict[str, float]]], now: float, cur: Mapping[str, float],
                    win_sec: float, open_ts: Optional[float] = None) -> Tuple[Dict[str, float], float]:
    """직전 `win_sec`초 거래대금(순간 거래대금) = 지금 누적 − `win_sec`초 전 누적(11/18 사용자 요청: 누적 대신 "그 당시").

    `history`는 (시각, {종목: 누적 거래대금}) 과거→최신. 기준은 `now - win_sec` 이전의 가장 최근 스냅숏이다.
    그런 스냅숏이 없으면: 장 시작(`open_ts`) 뒤 `win_sec`초가 안 지났으면 0(=누적이 곧 창), 아니면 가장 오래된 스냅숏(창이 짧다).
    돌려주는 두 번째 값은 실제 창 길이(초) — 화면에 "몇 분 기준"인지 쓴다.
    """
    base: Optional[Dict[str, float]] = None
    span = 0.0
    for t, snap in history:
        if t <= now - win_sec:
            base, span = snap, now - t
    if base is None:
        if open_ts is not None and now - open_ts <= win_sec:
            base, span = {}, max(0.0, now - open_ts)
        elif history:
            base, span = history[0][1], now - history[0][0]
        else:
            base, span = {}, 0.0
    return {s_: max(0.0, float(v or 0.0) - float(base.get(s_, 0.0) or 0.0)) for s_, v in cur.items()}, span
