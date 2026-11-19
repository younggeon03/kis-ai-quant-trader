"""미시구조·거래량 에이전트 최종 후보(11/18). g = 그날 day_df(09:03까지 피처), 반환 = 종목코드 또는 None.
점수 = 그날 후보 안 백분위 순위 평균(동점은 배수 큰 순 → 종목코드). 파라미터는 문헌·둥근 값으로 미리 정했다."""
from typing import List, Optional, Tuple

import pandas as pd


def _best(c: pd.DataFrame, cols: List[Tuple[str, int]]) -> Optional[str]:
    if c.empty:
        return None
    c = c.copy()
    c["_s"] = sum(c[k].rank(pct=True, ascending=(w > 0)).fillna(0.0) for k, w in cols)
    return c.sort_values(["_s", "ratio", "sym"], ascending=[False, False, True])["sym"].iloc[0]


def candidate_1(g: pd.DataFrame) -> Optional[str]:
    """엔진 판정 통과 중 ATR10(변동성 군집) + 09:03 전일 대비 상승률(장중 모멘텀, Gao–Han–Li–Zhou 2018) 순위 평균."""
    return _best(g[g["engine_pass"]], [("atr10", 1), ("chg", 1)])


def candidate_2(g: pd.DataFrame) -> Optional[str]:
    """엔진 판정 통과 중 ATR10 + 전일 회전율(거래량/상장주식, 고거래량 프리미엄 GKM 2001) 순위 평균."""
    return _best(g[g["engine_pass"]], [("atr10", 1), ("turn1", 1)])


def candidate_3(g: pd.DataFrame) -> Optional[str]:
    """판정 배수 임계 3 → 5(더 강한 비정상 거래량)로 올리고 나머지 조건(전일 종가 위·VWAP 위·3일 상승<20%)은 같게, ATR10 큰 순."""
    c = g[(g["ratio"] >= 5.0) & (g["px0903"] > g["prev_close"]) & (g["vwap_dev"] >= 0) & ~(g["run3"] >= 0.20)]
    return _best(c, [("atr10", 1)])
