"""에이전트 팀 공통 평가기(11/18) — 메인 지표 P(이틀 누적 ≥ +10%), 시간 순 5조각.

    from scripts.agent_eval import load_dev, evaluate, load_paths, simulate_exit

- `load_dev()`  개발 구간(2026-07-01 전 시작 에피소드)만 돌려준다. **최종 시험 구간(2026-07-01~)은 에이전트가 보지 않는다.**
- `evaluate(pick_fn, name)`  pick_fn(day_df) → 종목코드 또는 None(그날 현금). day_df = 그날 모든 지수 종목 행(09:03까지의 피처).
  그날 고른 종목의 결과 = 행의 `rB`(09:04 시가 매수 → 순수익 +10% 지정가 → 아니면 다음 날 15:10). None 이면 0.
  돌려주는 값: 전체 P(≥+10%) · 조각별 P · 평균 · 손실 확률 · 매매일 수.
- 매도 규칙을 바꾸려면 `load_paths()` + `simulate_exit(row, exit_fn)` — 판정 통과(engine_pass) 행만 분 경로가 있다.
  exit_fn(o, h, l, c, entry, day_idx, minute) → None(보유) | 매도가. 분 인덱스 0 = 09:00, 4 = 매수, 370 = 15:10.

결과 표기를 다른 에이전트와 맞추려고 모든 평가는 이 함수를 쓴다. pick_fn 안에서 `rB`·`hit10`·`r_d2c`·`r_d1c`·`up_*`·`dn_*`
(결과 열)를 읽으면 룩어헤드다 — `evaluate`가 그 열을 지운 뒤 넘긴다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "agent_data"
HOLDOUT_START = "20260701"
K = 5
OUTCOME_COLS = ["rB", "hit10", "r_d2c", "r_d1c", "up_max", "dn_min", "up_d1", "dn_d1"]
COST = 0.00278


def _rows() -> pd.DataFrame:
    return pd.read_pickle(DATA / "rows.pkl")


def load_dev() -> pd.DataFrame:
    df = _rows()
    return df[df["day"] < HOLDOUT_START].reset_index(drop=True)


def _load_holdout_for_final_review_only() -> pd.DataFrame:      # 에이전트 사용 금지 — 리더가 최종 1회만
    df = _rows()
    return df[df["day"] >= HOLDOUT_START].reset_index(drop=True)


def evaluate(pick_fn: Callable[[pd.DataFrame], Optional[str]], name: str = "", df: Optional[pd.DataFrame] = None,
             verbose: bool = True, outcome: str = "rB") -> Dict[str, float]:
    df = load_dev() if df is None else df
    days = sorted(df["day"].unique())
    res = np.zeros(len(days))
    traded = np.zeros(len(days), bool)
    for i, (d, g) in enumerate(df.groupby("day", sort=True)):
        feat = g.drop(columns=[c for c in OUTCOME_COLS if c in g.columns])
        sym = pick_fn(feat)
        if sym is None:
            continue
        hit = g[g["sym"] == sym]
        if hit.empty:
            continue
        res[i] = float(hit[outcome].iloc[0])
        traded[i] = True
    edges = np.linspace(0, len(days), K + 1).astype(int)
    folds = [float((res[edges[f]:edges[f + 1]] >= 0.10 - 1e-9).mean()) for f in range(K)]
    out = {"name": name, "p10": float((res >= 0.10 - 1e-9).mean()), "folds": folds, "mean": float(res.mean()),
           "p_loss": float((res < 0).mean()), "trade_days": int(traded.sum()), "days": len(days),
           "fold_min": float(min(folds))}
    if verbose:
        print(f"{name:<40} P(≥+10%) {out['p10']:.1%} | 조각 " + " · ".join(f"{x:.1%}" for x in folds)
              + f" | 평균 {out['mean']:+.2%} | 손실 {out['p_loss']:.0%} | 매매 {out['trade_days']}/{out['days']}일")
    return out


def load_paths() -> np.ndarray:
    return np.load(DATA / "paths.npz")["paths"]


def simulate_exit(row: pd.Series, paths: np.ndarray, exit_fn, cost: float = COST) -> Optional[float]:
    """판정 통과 행의 분 경로로 매도 규칙을 돌린다(15:10 d2 까지 안 팔면 그 시가). 결과 = 순수익."""
    pi = int(row["path_idx"])
    if pi < 0:
        return None
    p = paths[pi]
    entry = float(row["entry"])
    for dd in (0, 1):
        m0 = 4 if dd == 0 else 0
        for m in range(m0, 370):
            o, h, lo, c = (float(x) for x in p[dd, :, m])
            if dd == 0 and m == 4:
                o = entry
            if not np.isfinite(h):
                continue
            px = exit_fn(o, h, lo, c, entry, dd, m)
            if px is not None:
                return float(px) / entry * (1 - cost) - 1.0
    o = float(p[1, 0, 370])
    if not np.isfinite(o):
        o = float(p[1, 3, 369])
    return o / entry * (1 - cost) - 1.0


# 기준선 두 개 — 지금 서버(ATR10 순)와 그 전(배수 순)
def pick_engine_ratio(g: pd.DataFrame) -> Optional[str]:
    c = g[g["engine_pass"]]
    return None if c.empty else c.sort_values(["ratio", "sym"], ascending=[False, True])["sym"].iloc[0]


def pick_engine_atr10(g: pd.DataFrame) -> Optional[str]:
    c = g[g["engine_pass"]].copy()
    if c.empty:
        return None
    c["_a"] = c["atr10"].fillna(-1.0)
    return c.sort_values(["_a", "ratio", "sym"], ascending=[False, False, True])["sym"].iloc[0]


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    evaluate(pick_engine_ratio, "기준: 엔진 판정 · 배수 순")
    evaluate(pick_engine_atr10, "기준: 엔진 판정 · ATR10 순(지금 서버)")
