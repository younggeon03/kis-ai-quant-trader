"""행동재무·이상현상·테마 가설 시험(11/18, 개발 구간만). 설정은 미리 정한 목록이며 데이터로 깎지 않았다."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scripts.research.agent_eval import load_dev, evaluate, pick_engine_ratio, pick_engine_atr10

def rank_pick(cols, pool="pass", filt=None):
    """cols: [(열, +1 큰 순 / -1 작은 순)] — 그날 후보 안 백분위 순위 평균이 가장 큰 종목."""
    def f(g):
        c = g[g["engine_pass"]] if pool == "pass" else g
        if filt is not None:
            c = c[filt(c)]
        if c.empty:
            return None
        s = np.zeros(len(c))
        for col, sgn in cols:
            s += (c[col] * sgn).rank(pct=True, na_option="bottom").fillna(0).to_numpy()
        c = c.assign(_s=s).sort_values(["_s", "sym"], ascending=[False, True])
        return c["sym"].iloc[0]
    return f

def cond(good, bad, test):
    def f(g):
        return good(g) if test(g) else bad(g)
    return f

A = [("atr10", 1)]
specs = {
 # 기준
 "B1 배수 순": pick_engine_ratio,
 "B2 ATR10 순": pick_engine_atr10,
 # 주의 집중(Barber–Odean)
 "H1a 갭 큰 순": rank_pick([("gap", 1)]),
 "H1b 전일 회전율 큰 순": rank_pick([("turn1", 1)]),
 "H1c 전일 |수익률| 큰 순": rank_pick([("absr1", 1)]),
 "H1d 주의 합성(배수·갭·회전율·|r1|)": rank_pick([("ratio", 1), ("gap", 1), ("turn1", 1), ("absr1", 1)]),
 # 로또·MAX·현저성
 "H2a MAX20 큰 순": rank_pick([("max20", 1)]),
 "H2b 변동성20 큰 순": rank_pick([("vol20", 1)]),
 "H2c 로또 합성(MAX·vol20·ATR10)": rank_pick([("max20", 1), ("vol20", 1), ("atr10", 1)]),
 "H2d 전일 고저폭 큰 순(현저성)": rank_pick([("rng1", 1)]),
 # 단기 반전 vs 지속
 "H3a run5 작은 순(반전)": rank_pick([("run5", -1)]),
 "H3b run5 큰 순(지속)": rank_pick([("run5", 1)]),
 "H3c mom20 큰 순": rank_pick([("mom20", 1)]),
 "H3d 장 시작 4분 수익 ret4 큰 순(일중 지속)": rank_pick([("ret4", 1)]),
 # 업종·테마
 "H4a 업종 동반 상승률 ind_chg 큰 순": rank_pick([("ind_chg", 1)]),
 "H4b 같은 업종 판정 통과 수 큰 순": rank_pick([("ind_pass", 1), ("ind_up", 1)]),
 "H4c ATR10 + ind_chg": rank_pick(A + [("ind_chg", 1)]),
 # 소형·코스닥
 "H5a 시총 작은 순": rank_pick([("mcap", -1)]),
 "H5b ATR10 + 시총 작은 순": rank_pick(A + [("mcap", -1)]),
 "H5c 코스닥만 ATR10": rank_pick(A, filt=lambda c: c["kosdaq"] == 1),
 # 결합
 "H6a ATR10 + MAX20": rank_pick(A + [("max20", 1)]),
 "H6b ATR10 + 배수": rank_pick(A + [("ratio", 1)]),
 "H6c ATR10 + 갭": rank_pick(A + [("gap", 1)]),
 # 국면 조건부: 시장 좋으면(폭≥0.5·mkt20>0) 주의 합성, 아니면 로또 합성
 "H7a 국면: 좋음→주의 / 나쁨→ATR10": cond(rank_pick([("ratio", 1), ("gap", 1), ("turn1", 1), ("absr1", 1)]),
                                     pick_engine_atr10, lambda g: g["breadth"].iloc[0] >= 0.5 and g["mkt20"].iloc[0] > 0),
 "H7b 국면: 좋음→ATR10 / 나쁨→ATR10+ind_chg": cond(pick_engine_atr10, rank_pick(A + [("ind_chg", 1)]),
                                     lambda g: g["breadth"].iloc[0] >= 0.5),
 # 전 지수 종목(판정 무시) 로또형 — 당일 상승(chg>0)만
 "H8a 전 종목 chg>0 ATR10": rank_pick(A, pool="all", filt=lambda c: c["chg"] > 0),
 "H8b 전 종목 chg>0 로또 합성": rank_pick([("max20", 1), ("vol20", 1), ("atr10", 1)], pool="all", filt=lambda c: c["chg"] > 0),
}

if __name__ == "__main__":
    df = load_dev()
    rows = []
    for k, f in specs.items():
        r = evaluate(f, k, df=df)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_pickle(ROOT / "reports" / "agents" / "behav_results.pkl")
    print("시험 설정 수", len(specs) - 2)
