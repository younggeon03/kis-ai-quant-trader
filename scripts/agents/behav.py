"""행동재무·이상현상·테마 에이전트 최종 후보(11/18). 각 함수: 그날 행(g) → 종목코드 또는 None.

점수 = 그날 후보 안 백분위 순위 평균(동점은 종목코드 순). 결과 열은 읽지 않는다.
- candidate_1: 엔진 판정 통과 중 ATR10 + 업종 대비 고유 상승(chg − ind_chg). 업종 동반 강세 종목이 일관되게 나빴던
  결과(H4, 5조각 모두 기준 아래)를 뒤집은 '종목 고유 뉴스' 가설 — 1차 결과를 보고 만든 사후 설정이다.
- candidate_2: 엔진 판정 통과 중 ATR10 + 전일 |수익률|(Barber–Odean 주의 집중). 평균·손실 확률이 가장 좋다.
- candidate_3: 전 지수 종목 중 당일 상승(chg>0) 종목에서 로또 합성(MAX20·vol20·ATR10, Bali–Cakici–Whitelaw).
  P는 높지만 평균이 음수이고 조각 최소가 낮다 — 꼬리만 노리는 공격형.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _rank_pick(c: pd.DataFrame, cols: List[Tuple[str, int]]) -> Optional[str]:
    if c.empty:
        return None
    s = np.zeros(len(c))
    for col, sgn in cols:
        s += (c[col] * sgn).rank(pct=True, na_option="bottom").fillna(0).to_numpy()
    c = c.assign(_s=s).sort_values(["_s", "sym"], ascending=[False, True])
    return str(c["sym"].iloc[0])


def candidate_1(g: pd.DataFrame) -> Optional[str]:
    c = g[g["engine_pass"]]
    c = c.assign(_idio=c["chg"] - c["ind_chg"].fillna(0.0))
    return _rank_pick(c, [("atr10", 1), ("_idio", 1)])


def candidate_2(g: pd.DataFrame) -> Optional[str]:
    return _rank_pick(g[g["engine_pass"]], [("atr10", 1), ("absr1", 1)])


def candidate_3(g: pd.DataFrame) -> Optional[str]:
    return _rank_pick(g[g["chg"] > 0], [("max20", 1), ("vol20", 1), ("atr10", 1)])


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.agent_eval import evaluate
    for f in (candidate_1, candidate_2, candidate_3):
        evaluate(f, f.__name__)
