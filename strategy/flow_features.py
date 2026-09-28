"""수급 피처 — 국내 시장에서 실제로 알파가 나오는 축.

## 왜 수급인가

국내 주식은 투자자 주체별 매매 동향이 공개된다. 외국인과 기관이 며칠째 순매수하는 종목,
공매도 비중이 급증한 종목, 신용잔고가 쌓인 종목은 가격·거래량만 보는 피처로는 잡히지 않는
정보를 갖는다. KIS 모의투자 도메인은 이 데이터를 주지 않지만 토스 Open API는 준다.

## 필드명을 모르는 상태에서 만드는 방법

토스 공개 문서에 수급 응답의 필드명이 전부 나와 있지 않다. 그래서 수집기
(`api/toss_data.py`)는 **숫자 필드를 접두어와 함께 통째로 보존**한다.

    inv_*     투자자별 매매동향   short_*  공매도
    credit_*  신용거래            lend_*   대차
    prog_*    프로그램매매

이 모듈은 그 원본 컬럼에서 **개념별 후보 이름**으로 컬럼을 찾아낸다. 예를 들어 "외국인
순매수"는 `foreign`/`frgn`/`외국인` 중 하나와 `net` 중 하나를 포함하는 컬럼을 찾는다.
실제 필드명이 무엇이든 대부분 걸린다. 못 찾은 피처는 NaN으로 남고 나머지는 정상 동작한다.

`describe_flow_columns()`로 어떤 컬럼이 어떤 개념에 매칭됐는지 확인할 수 있다.
수집 후 이 함수를 한 번 돌려 매칭이 의도대로 됐는지 보는 것을 권한다.

## 수준(level)을 그대로 쓰지 않는 이유

외국인 보유비중이나 신용잔고 비율 같은 값은 **종목마다 수준이 다르고 시간에 따라 거의
변하지 않는다.** 측정해 보면 외국인 보유비중은 전체 분산의 93%, 신용잔고 비율은 71%가
종목 간 차이였다. 이런 피처를 넣으면 모델이 "이 값이 0.46이면 사는 종목"이라는 식으로
**종목을 외운다.** 백테스트에서는 그 종목들이 실제로 올랐으므로 성과가 폭발하지만,
그것은 알파가 아니라 과거를 외운 것이다(실측: 수준 포함 시 Sharpe 3.79).

그래서 수준 피처는 전부 **자기 시계열 대비 z-score**로 바꿨다. "평소보다 외국인이
들어왔는가"는 종목과 무관한 질문이므로 종목을 식별하지 못한다.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.indicators import safe_div, zscore

# 수급 피처 (16종). 실제 토스 응답 필드에 맞춰 정의했다.
FLOW_FEATURES: List[str] = [
    # --- 주체별 순매수 (거래량 기준, 20일 z-score) ---
    "foreign_net_z",       # 외국인
    "inst_net_z",          # 기관 합계
    "indiv_net_z",         # 개인 (국내에서는 역지표로 쓰인다)
    "corp_net_z",          # 기타법인
    # --- 기관 세부 (국내 수급 분석의 핵심) ---
    "pension_net_z",       # 연기금 — 장기 자금, 추세 지속성이 높다
    "fininv_net_z",        # 금융투자(증권사 자기매매) — 단기 변동성이 크다
    "trust_net_z",         # 투신
    # --- 외국인 보유 ---
    "foreign_hold_chg",    # 보유 비중 변화(일간)
    "foreign_hold_z",      # 보유 비중의 **자기 시계열 대비** z-score (수준이 아니라 이탈도)
    "foreign_streak",      # 외국인 연속 순매수 일수(부호 포함, ±5 클립)
    "flow_concord",        # 외국인·기관 동시 매수(+1) / 동시 매도(−1)
    # --- 공매도·신용·대차 ---
    "short_ratio_z",       # 공매도 비중 z-score (급증 감지)
    "credit_bal_z",        # 신용융자 잔고 비율의 자기 시계열 대비 z-score
    "credit_bal_chg",      # 신용융자 잔고 변화율
    "lend_bal_chg",        # 대차잔고 변화율 — 늘면 공매도 대기 물량이 쌓인다
    # --- 프로그램 ---
    "prog_net_z",          # 프로그램 순매수(차익+비차익)
]

ZWINDOW = 20   # 순매수 표준화 창

# 개념 → (접두어, 반드시 포함할 후보군들, 제외어)
# 후보군은 AND로 묶이고 각 군 안에서는 OR다. 실제 필드명을 1순위로 두되,
# 토스가 필드명을 바꾸더라도 걸리도록 일반 후보를 함께 넣었다.
_CONCEPTS: Dict[str, tuple] = {
    "foreign_net":  ("inv", [["foreigner", "foreign", "frgn"], ["netbuy", "net"]],
                     ["holding", "hold"]),
    "inst_net":     ("inv", [["institution"], ["netbuy", "net"]], ["breakdown"]),
    "indiv_net":    ("inv", [["individual", "indiv", "person"], ["netbuy", "net"]], []),
    "corp_net":     ("inv", [["othercorporation", "corporation", "corp"], ["netbuy", "net"]], []),
    "pension_net":  ("inv", [["pensionfund", "pension"], ["netbuy", "net"]], []),
    "fininv_net":   ("inv", [["financialinvestment", "fininv"], ["netbuy", "net"]], []),
    "trust_net":    ("inv", [["trust"], ["netbuy", "net"]], []),
    "foreign_hold": ("inv", [["foreignerholding", "foreignhold"], ["rate", "ratio"]],
                     ["limit"]),
    "short_ratio":  ("short", [["volumerate", "rate"]], ["amount"]),
    "short_volume": ("short", [["volume"]], ["rate"]),
    "credit_rate":  ("credit", [["marginloan", "margin"], ["balancerate"]], ["stock"]),
    "credit_bal":   ("credit", [["marginloan", "margin"], ["balancequantity"]], ["stock"]),
    "lend_bal":     ("lend", [["balancequantity"]], ["amount"]),
    "prog_arb":     ("prog", [["arbitrage"], ["netbuy", "net"]], ["nonarbitrage"]),
    "prog_nonarb":  ("prog", [["nonarbitrage"], ["netbuy", "net"]], []),
}


def _normalize(name: str) -> str:
    """컬럼명을 비교용으로 정규화한다. 접두어·구분자·대소문자 차이를 흡수한다."""
    return re.sub(r"[^a-z0-9가-힣]", "", name.lower())


def find_column(columns: Sequence[str], prefix: str, groups: Sequence[Sequence[str]],
                exclude: Sequence[str] = ()) -> Optional[str]:
    """개념에 해당하는 컬럼을 찾는다.

    `groups`의 각 군에서 최소 하나씩을 모두 포함하고, `exclude`의 어떤 것도 포함하지 않는
    컬럼을 고른다. 후보가 여럿이면 **이름이 가장 짧은 것**을 택한다.
    (긴 이름은 보통 세부 분류라 우리가 원하는 합계가 아니다.)
    """
    matches: List[str] = []
    for col in columns:
        if not col.startswith(f"{prefix}_"):
            continue
        norm = _normalize(col[len(prefix) + 1:])
        if any(_normalize(x) in norm for x in exclude):
            continue
        if all(any(_normalize(x) in norm for x in group) for group in groups):
            matches.append(col)
    if not matches:
        return None
    return min(matches, key=lambda c: (len(c), c))


def resolve_columns(flow: pd.DataFrame) -> Dict[str, Optional[str]]:
    """개념 → 실제 컬럼명 매핑을 만든다."""
    cols = [c for c in flow.columns if c != "ts"]
    return {concept: find_column(cols, prefix, groups, exclude)
            for concept, (prefix, groups, exclude) in _CONCEPTS.items()}


def describe_flow_columns(flow: pd.DataFrame) -> str:
    """어떤 컬럼이 어떤 개념에 매칭됐는지 사람이 읽을 수 있게 출력한다.

    수집 직후 한 번 돌려 매칭이 의도대로 됐는지 확인한다. 엉뚱한 컬럼이 잡혔거나
    못 찾은 개념이 있으면 `_CONCEPTS`의 후보군에 이름을 추가하면 된다.
    """
    resolved = resolve_columns(flow)
    available = [c for c in flow.columns if c != "ts"]
    lines = [f"수급 원본 컬럼 {len(available)}개 → 개념 매칭 결과", "-" * 62]
    for concept, col in resolved.items():
        mark = "○" if col else "×"
        lines.append(f" {mark} {concept:<14} {col or '(찾지 못함)'}")
    missing = [c for c, v in resolved.items() if v is None]
    if missing:
        lines += ["", f"매칭 실패 {len(missing)}개 — 해당 피처는 NaN으로 남습니다.",
                  "원본 컬럼 목록:", "  " + ", ".join(available[:40])]
        if len(available) > 40:
            lines.append(f"  ... 외 {len(available) - 40}개")
    return "\n".join(lines)


def _series(flow: pd.DataFrame, col: Optional[str]) -> Optional[np.ndarray]:
    if col is None or col not in flow.columns:
        return None
    return flow[col].to_numpy(dtype=float)


def _pct_change(values: np.ndarray) -> np.ndarray:
    """전일 대비 변화율. 0으로 나누는 경우는 NaN으로 둔다."""
    prev = np.roll(values, 1)
    prev[0] = np.nan
    return safe_div(values - prev, np.abs(prev), fill=np.nan)


def _streak(values: np.ndarray, limit: int = 5) -> np.ndarray:
    """연속 동일 부호 일수. 양수 연속이면 +, 음수 연속이면 −."""
    out = np.zeros(len(values), dtype=float)
    run = 0
    prev_sign = 0
    for i, v in enumerate(values):
        sign = 0 if not np.isfinite(v) or v == 0 else (1 if v > 0 else -1)
        if sign != 0 and sign == prev_sign:
            run += 1
        elif sign != 0:
            run = 1
        else:
            run = 0
        prev_sign = sign
        out[i] = float(np.clip(run * sign, -limit, limit))
    return out


def build_flow_features(flow: Optional[pd.DataFrame]) -> pd.DataFrame:
    """수급 원본 → 피처 DataFrame (`ts` + FLOW_FEATURES).

    순매수 거래량은 종목 규모에 따라 절대값이 크게 다르므로 **20일 z-score로 표준화**한다.
    비율 계열(보유비중, 공매도 비중, 신용잔고 비율)은 수준 자체가 의미를 가지므로 그대로
    쓰되, 급변을 잡기 위해 변화량이나 z-score를 함께 만든다.
    """
    if flow is None or flow.empty or "ts" not in flow.columns:
        return pd.DataFrame(columns=["ts"] + FLOW_FEATURES)

    flow = flow.sort_values("ts").reset_index(drop=True)
    col = resolve_columns(flow)
    out = pd.DataFrame({"ts": flow["ts"].astype(str)})
    for name in FLOW_FEATURES:
        out[name] = np.nan

    def zs(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if values is None:
            return None
        return np.clip(zscore(values, ZWINDOW), -5, 5)

    foreign = _series(flow, col["foreign_net"])
    inst = _series(flow, col["inst_net"])

    # --- 주체별 순매수 ---
    for feature, concept in (
        ("foreign_net_z", "foreign_net"),
        ("inst_net_z", "inst_net"),
        ("indiv_net_z", "indiv_net"),
        ("corp_net_z", "corp_net"),
        ("pension_net_z", "pension_net"),
        ("fininv_net_z", "fininv_net"),
        ("trust_net_z", "trust_net"),
    ):
        values = zs(_series(flow, col[concept]))
        if values is not None:
            out[feature] = values

    # --- 외국인 보유 ---
    hold = _series(flow, col["foreign_hold"])
    if hold is not None:
        out["foreign_hold_chg"] = np.concatenate([[np.nan], np.diff(hold)])
        # 수준을 그대로 쓰면 종목 이름표가 된다(분산의 93%가 종목 간 차이).
        # 자기 시계열 대비 이탈도로 바꾸면 "평소보다 외국인이 들어왔는가"가 된다.
        out["foreign_hold_z"] = np.clip(zscore(hold, 60), -5, 5)

    if foreign is not None:
        out["foreign_streak"] = _streak(foreign)

    # 외국인·기관 동시 순매수 — 국내에서 가장 널리 쓰이는 수급 신호
    if foreign is not None and inst is not None:
        out["flow_concord"] = np.where((foreign > 0) & (inst > 0), 1.0,
                                       np.where((foreign < 0) & (inst < 0), -1.0, 0.0))

    # --- 공매도 ---
    ratio = _series(flow, col["short_ratio"])
    if ratio is not None:
        out["short_ratio_z"] = np.clip(zscore(ratio, ZWINDOW), -5, 5)

    # --- 신용융자 ---
    credit_rate = _series(flow, col["credit_rate"])
    if credit_rate is not None:
        out["credit_bal_z"] = np.clip(zscore(credit_rate, 60), -5, 5)
    credit_bal = _series(flow, col["credit_bal"])
    if credit_bal is not None:
        out["credit_bal_chg"] = _pct_change(credit_bal)

    # --- 대차잔고 ---
    lend_bal = _series(flow, col["lend_bal"])
    if lend_bal is not None:
        out["lend_bal_chg"] = _pct_change(lend_bal)

    # --- 프로그램 (차익 + 비차익) ---
    arb = _series(flow, col["prog_arb"])
    nonarb = _series(flow, col["prog_nonarb"])
    parts = [x for x in (arb, nonarb) if x is not None]
    if parts:
        total = parts[0] if len(parts) == 1 else np.nansum(np.vstack(parts), axis=0)
        out["prog_net_z"] = np.clip(zscore(total, ZWINDOW), -5, 5)

    return out


def merge_flow(price: pd.DataFrame, flow: Optional[pd.DataFrame],
               lag: Optional[int] = None) -> pd.DataFrame:
    """OHLCV에 수급 피처를 `ts` 기준으로 붙이되 **`lag` 거래일만큼 늦춘다.**

    ## 왜 늦춰야 하는가

    토스 응답의 `updatedAt`을 보면 T일 수급 데이터의 확정 시점은 이렇다.

        투자자별 매매동향   T일 저녁 ~ T+4
        신용거래           T+1 새벽
        대차잔고           T일 저녁 ~ T+3
        공매도             T일 장마감 후

    우리는 **T일 종가로 판단**하므로, 그 시점에 T일 수급은 아직 공시되지 않았다.
    지연 없이 붙이면 미래 정보를 쓰는 것이고, 백테스트 Sharpe가 실제와 무관하게 부풀려진다
    (실측: 지연 0일 → 3.588, 지연 1일 → 아래 참조).

    기본값 1일은 **최소한**의 보수적 처리다. 투자자별 매매동향처럼 T+4에 확정되는 항목이
    있으므로, 실전 운용에서는 2일 이상을 고려할 만하다.

    수급이 없으면 FLOW_FEATURES 컬럼을 NaN으로 만들어 스키마만 맞춘다.
    이렇게 해야 수급이 있는 종목과 없는 종목을 같은 학습 행렬에 넣을 수 있다.
    """
    if lag is None:
        from config.settings import get_settings

        lag = get_settings().FLOW_LAG_DAYS

    price = price.copy()
    features = build_flow_features(flow)
    if features.empty or len(features.columns) <= 1:
        for name in FLOW_FEATURES:
            price[name] = np.nan
        return price

    price["ts"] = price["ts"].astype(str)
    merged = price.merge(features, on="ts", how="left")
    for name in FLOW_FEATURES:
        if name not in merged.columns:
            merged[name] = np.nan

    # 가격 시계열 기준으로 lag 거래일만큼 밀어 공시 시차를 반영한다
    if lag and lag > 0:
        merged[FLOW_FEATURES] = merged[FLOW_FEATURES].shift(lag)
    return merged


def load_flow_dir(path, symbols: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """`data/cache/flow/<종목코드>.csv` 들을 읽는다."""
    from pathlib import Path

    root = Path(path)
    out: Dict[str, pd.DataFrame] = {}
    if not root.exists():
        return out
    wanted = set(symbols) if symbols else None
    for file in sorted(root.glob("*.csv")):
        symbol = file.stem.zfill(6)
        if wanted and symbol not in wanted:
            continue
        try:
            df = pd.read_csv(file, dtype={"ts": str})
        except Exception:
            continue
        if "ts" in df.columns and len(df.columns) > 1:
            out[symbol] = df
    return out
