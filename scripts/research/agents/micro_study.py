"""미시구조·거래량 가설 시험(11/18 에이전트 micro). 설정은 미리 정한 목록 그대로 — 데이터로 깎지 않는다."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.agent_eval import evaluate, load_dev  # noqa


def _illiq(c):
    tv = c["prev_close"] * c["prev_volume"]
    return np.where(tv > 0, c["absr1"] / tv, np.nan)


def pick_rank(cols, pool="engine", extra=None):
    """cols = [(열, +1/-1)]. 그날 후보 안 백분위 순위 평균이 가장 큰 종목."""
    def fn(g):
        c = g[g["engine_pass"]] if pool == "engine" else g
        if extra is not None:
            c = c[extra(c)]
        if c.empty:
            return None
        c = c.copy()
        if any(k == "illiq" for k, _ in cols):
            c["illiq"] = _illiq(c)
        s = sum(c[k].rank(pct=True, ascending=(w > 0)).fillna(0.0) for k, w in cols)
        c["_s"] = s
        return c.sort_values(["_s", "ratio", "sym"], ascending=[False, False, True])["sym"].iloc[0]
    return fn


def gate(th):      # 엔진 판정에서 배수 임계만 바꾼다
    return lambda c: (c["ratio"] >= th) & (c["px0903"] > c["prev_close"]) & (c["vwap_dev"] >= 0) & ~(c["run3"] >= 0.20)


SETS = [
    # 기준
    ("기준 ATR10 [엔진]", pick_rank([("atr10", 1)])),
    # 1) 고거래량 프리미엄(GKM 2001) · 거래량-수익 동학(LMSW 2002)
    ("H1a GKM 배수 순 [엔진]", pick_rank([("ratio", 1)])),
    ("H1b GKM 전일 회전율 순 [엔진]", pick_rank([("turn1", 1)])),
    ("H1c LMSW ATR10+배수 [엔진]", pick_rank([("atr10", 1), ("ratio", 1)])),
    ("H1d LMSW ATR10+배수+소형 [엔진]", pick_rank([("atr10", 1), ("ratio", 1), ("mcap", -1)])),
    # 2) 개장 구간 돌파(Crabel·Zarattini–Aziz)
    ("H2a ORB 09:00~03 상승 순 [엔진]", pick_rank([("ret4", 1)])),
    ("H2b ORB ATR10+상승+VWAP 위 [엔진]", pick_rank([("atr10", 1), ("ret4", 1), ("vwap_dev", 1)])),
    ("H2c ORB ATR10+개장폭 [엔진]", pick_rank([("atr10", 1), ("range4", 1)])),
    # 3) 장중 모멘텀(Gao 2018) · 갭(Lou–Polk–Skouras 2019)
    ("H3a 장중모멘텀 전일비 순 [엔진]", pick_rank([("chg", 1)])),
    ("H3b 갭 작은 순(갭 반전 회피) ATR10+(-갭) [엔진]", pick_rank([("atr10", 1), ("gap", -1)])),
    ("H3c 갭 큰 순(갭 지속) ATR10+갭 [엔진]", pick_rank([("atr10", 1), ("gap", 1)])),
    # 4) 비유동성(Amihud 2002)
    ("H4a Amihud 전일 비유동성 순 [엔진]", pick_rank([("illiq", 1)])),
    ("H4b ATR10+Amihud [엔진]", pick_rank([("atr10", 1), ("illiq", 1)])),
    # 5) 거래대금 절대 규모 vs 배수
    ("H5a 09:00~03 거래대금 큰 순 [엔진]", pick_rank([("tv4", 1)])),
    ("H5b ATR10+거래대금(-) [엔진]", pick_rank([("atr10", 1), ("tv4", -1)])),
    # 6) 판정 자체 바꾸기(배수 임계)
    ("H6a 배수 ≥2 판정 · ATR10 [전체]", pick_rank([("atr10", 1)], pool="all", extra=gate(2.0))),
    ("H6b 배수 ≥5 판정 · ATR10 [전체]", pick_rank([("atr10", 1)], pool="all", extra=gate(5.0))),
    ("H6c 배수 ≥10 판정 · ATR10 [전체]", pick_rank([("atr10", 1)], pool="all", extra=gate(10.0))),
    ("H6d 판정 없이 ATR10 순 [전체 · 전일비>0]", pick_rank([("atr10", 1)], pool="all", extra=lambda c: c["chg"] > 0)),
]

if __name__ == "__main__":
    df = load_dev()
    rows = []
    for name, fn in SETS:
        r = evaluate(fn, name, df=df)
        rows.append(r)
    print(f"\n시험 설정 수 {len(SETS) - 1}(기준 제외)")
    pd.DataFrame(rows).to_pickle(ROOT / "reports" / "agents" / "micro_results.pkl")
