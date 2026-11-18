"""이틀 +10% 확률 교차검증(11/18 사용자 요청) — 메인 지표 P(이틀 누적 ≥ +10%).

후보 = 엔진 장 시작 판정 통과 종목(`reports/filter_cands.pkl`, filter_study.py 가 만든다) + 두 해 모두 좋았던 필터 4개의
**모든 조합(16)**(배수 ≥ 10은 매매일이 너무 줄어 뺀다 — 사용자 결정) x 이틀 매매 방식(two_day_study.episode). 선택 = 필터 통과 중 배수 1위.
교차검증 = 시간 순 5조각. 4조각으로 P(≥+10%) 최고 설정을 고르고 남은 1조각에서 잰다(경계 하루씩 비움 — 이틀 매매가 걸친다).
같은 (방식, 1일차 종목, 2일차 종목, 날짜) 결과는 한 번만 계산한다.

    PYTHONIOENCODING=utf-8 python scripts/two_day_cv.py
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import get_settings  # noqa: E402
import scripts.hold_study as hs  # noqa: E402
import scripts.two_day_study as td  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

FILTERS = {
    "52주80%": lambda f: f["near52"] >= 0.80,
    "MAX15%": lambda f: f["max20"] < 0.15,
    "갭5%": lambda f: f["gap"] < 0.05,
    "모멘텀6M": lambda f: f["mom120"] > 0,
}
# 점수제(11/18 사용자 제안): 조건으로 거르지 않고 그날 후보 안 백분위 순위의 평균으로 1위를 고른다(후보가 있으면 늘 산다).
# 방향은 문헌대로(+ 클수록 좋음, − 작을수록 좋음). vol20 + 는 "+10% 도달 확률"이 목표라 변동성이 클수록 유리하다는 가정.
SCORES = {
    "점수(배수·52주·MAX·갭·모멘텀)": [("ratio", 1), ("near52", 1), ("max20", -1), ("gap", -1), ("mom120", 1)],
    "점수(52주·MAX·갭·모멘텀)": [("near52", 1), ("max20", -1), ("gap", -1), ("mom120", 1)],
    "점수(배수·52주·MAX·갭·모멘텀·변동성)": [("ratio", 1), ("near52", 1), ("max20", -1), ("gap", -1), ("mom120", 1), ("vol20", 1)],
    "점수(배수·52주·변동성)": [("ratio", 1), ("near52", 1), ("vol20", 1)],
    "점수(변동성)": [("vol20", 1)],
    # 11/18 2차 — 장벽 도달 확률 ∝ 변동성(반사 원리). 변동성 예측 피처: 변동성 군집(Engle 1982) · 거래량-변동성(Clark 1973) · 규모 · 회전율
    "점수(ATR10)": [("atr10", 1)],
    "점수(전일 고저폭)": [("rng1", 1)],
    "점수(09:00~09:03 고저폭)": [("range4", 1)],
    "점수(시총 작은 순)": [("mcap", -1)],
    "점수(회전율)": [("turn1", 1)],
    "점수(장중 상승폭)": [("ret4", 1)],
    "점수(변동성 예측: 20일·ATR·전일폭·개장폭)": [("vol20", 1), ("atr10", 1), ("rng1", 1), ("range4", 1)],
    "점수(변동성 예측 + 배수)": [("vol20", 1), ("atr10", 1), ("rng1", 1), ("range4", 1), ("ratio", 1)],
    "점수(변동성 예측 + 시총 + 회전율)": [("vol20", 1), ("atr10", 1), ("rng1", 1), ("range4", 1), ("mcap", -1), ("turn1", 1)],
    "점수(변동성 예측 + 장중 상승)": [("vol20", 1), ("atr10", 1), ("rng1", 1), ("range4", 1), ("ret4", 1)],
    "점수(변동성 예측 + MAX 회피 아님 + 52주)": [("vol20", 1), ("atr10", 1), ("near52", 1)],
    # 업종(Moskowitz·Grinblatt 1999 업종 모멘텀 · 같은 날 업종 동반 상승 · 같은 업종 동시 발동 = 테마)
    "점수(업종 모멘텀)": [("ind_mom20", 1)],
    "점수(업종 동반 상승)": [("ind_up", 1)],
    "점수(업종 동시 발동)": [("ind_trig", 1)],
    "점수(변동성 예측 + 업종 3종)": [("vol20", 1), ("atr10", 1), ("rng1", 1), ("range4", 1),
                                ("ind_mom20", 1), ("ind_up", 1), ("ind_trig", 1)],
}
MODES = [("A", "none"), ("A", "cs0"), ("B", "none"), ("B", "cs0"), ("D", "none"), ("D", "cs0")]
MODE_NAME = {"A": "이틀 보유", "B": "이틀 보유·+10% 매도", "D": "+5% → 재매수 남은 목표"}
K = 5


def main() -> int:
    enable_utf8_console()
    s = get_settings()
    cost = s.round_trip_cost
    syms_all = sorted({x for v in s.index_members().values() for x in v})
    t0 = time.time()
    data, pcs, fp = hs.load_all(syms_all)
    days = sorted({d for per in data.values() for d in per})
    cd = pd.read_pickle(ROOT / "reports" / "filter_cands.pkl")
    print(f"지문 {fp} · {len(data)}종목 · {len(days)}일 · 후보 {len(cd)} · 적재 {time.time() - t0:.0f}s", flush=True)

    names = list(FILTERS)
    combos: List[Tuple[str, ...]] = [()]
    for r in range(1, len(names) + 1):
        combos += list(combinations(names, r))
    masks = {n: FILTERS[n](cd).fillna(False).to_numpy() for n in names}
    cache: Dict[tuple, float] = {}

    tops: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for combo in combos:
        m = np.ones(len(cd), bool)
        for n in combo:
            m &= masks[n]
        sub = cd[m].sort_values(["day", "ratio"], ascending=[True, False])
        tops[combo] = sub.groupby("day")["sym"].first().to_dict()
    for sname, comps in SCORES.items():
        sc = np.zeros(len(cd))
        for col, sign in comps:
            pr = cd.groupby("day")[col].rank(pct=True, ascending=sign > 0).fillna(0.5).to_numpy()
            sc += pr
        tmp = cd.assign(_s=sc).sort_values(["day", "_s"], ascending=[True, False])
        tops[(sname,)] = tmp.groupby("day")["sym"].first().to_dict()

    # 설정별 에피소드(연속 두 날) 결과 — 길이 len(days)-1, 계산 못 하면 nan
    res: Dict[tuple, np.ndarray] = {}
    for ci, (combo, top) in enumerate(tops.items()):
        ep = {d: [x] for d, x in top.items()}
        for mode, stop in MODES:
            arr = np.full(len(days) - 1, np.nan)
            for j in range(len(days) - 1):
                d1, d2 = days[j], days[j + 1]
                key = (mode, stop, d1, top.get(d1), top.get(d2))
                if key not in cache:
                    cache[key] = td.episode(data, ep, d1, d2, mode, stop, cost)
                v = cache[key]
                arr[j] = np.nan if v is None else v
            res[(combo, mode, stop)] = arr
        if (ci + 1) % 8 == 0:
            print(f"  조합 {ci + 1}/{len(tops)} · 캐시 {len(cache)} ({time.time() - t0:.0f}s)", flush=True)

    def p10(a):
        a = a[np.isfinite(a)]
        return float((a >= 0.10).mean()) if len(a) else np.nan

    def lab(k):
        combo, mode, stop = k
        return f"{'+'.join(combo) if combo else '필터 없음'} · {MODE_NAME[mode]}{' · 첫날 종가 손절' if stop == 'cs0' else ''}"

    # 전체 기간 상위
    full = sorted(res, key=lambda k: -p10(res[k]))
    print("\n[전체 기간 P(이틀 ≥ +10%) 상위 15]  (참고: 같은 데이터로 고른 값이라 낙관적)")
    for k in full[:15]:
        a = res[k]
        f = a[np.isfinite(a)]
        y25 = a[[d.startswith("2025") for d in days[:-1]]]
        y26 = a[[d.startswith("2026") for d in days[:-1]]]
        print(f"  {p10(a):.1%} (25 {p10(y25):.1%} · 26 {p10(y26):.1%}) | 평균 {f.mean():+.2%} · 손실 {(f < 0).mean():.0%} · "
              f"매매 없는 날 {(f == 0).mean():.0%} | {lab(k)}")
    base = ((), "A", "none")
    print(f"  기준(필터 없음·이틀 보유): {p10(res[base]):.1%}")

    # 교차검증
    n = len(days) - 1
    edges = np.linspace(0, n, K + 1).astype(int)
    oof = np.full(n, np.nan)
    oof_base = np.full(n, np.nan)
    print("\n[시간 순 5조각 교차검증 — 4조각으로 고르고 1조각에서 잰다]")
    picks = []
    for f_ in range(K):
        lo, hi = edges[f_], edges[f_ + 1]
        test = np.zeros(n, bool)
        test[lo:hi] = True
        train = ~test
        train[max(0, lo - 1):min(n, hi + 1)] = False      # 경계 하루 비움
        train &= ~test
        best = max(res, key=lambda k: (p10(res[k][train]), np.nanmean(res[k][train])))
        picks.append(lab(best))
        oof[test] = res[best][test]
        oof_base[test] = res[base][test]
        print(f"  조각 {f_ + 1} ({days[lo]}~{days[hi - 1]}): 고른 설정 {lab(best)} | 학습 {p10(res[best][train]):.1%} → "
              f"검증 {p10(res[best][test]):.1%} (기준 {p10(res[base][test]):.1%})")
    # 고정 설정끼리 조각별 비교(11/18) — 배수 1위 vs 변동성 점수
    fixed = [((), "B", "none"), (("점수(ATR10)",), "B", "none"), (("점수(변동성 예측 + 시총 + 회전율)",), "B", "none"),
             (("점수(회전율)",), "B", "none"), (("점수(변동성 예측: 20일·ATR·전일폭·개장폭)",), "B", "none")]
    print()
    print("[고정 설정 조각별 P(≥+10%)]")
    for k in fixed:
        if k not in res:
            continue
        vals = [p10(res[k][edges[f_]:edges[f_ + 1]]) for f_ in range(K)]
        means = [float(np.nanmean(res[k][edges[f_]:edges[f_ + 1]])) for f_ in range(K)]
        print(f"  {lab(k)} | " + " · ".join(f"{v:.1%}" for v in vals) + f" | 조각 평균 수익 " + " · ".join(f"{m:+.1%}" for m in means))
    fin = np.isfinite(oof)
    print(f"\n  교차검증 합계 P(≥+10%): 고르는 방식 {p10(oof):.1%} vs 기준 {p10(oof_base):.1%} · "
          f"평균 {np.nanmean(oof):+.2%} vs {np.nanmean(oof_base):+.2%} · 손실 {(oof[fin] < 0).mean():.0%} vs {(oof_base[np.isfinite(oof_base)] < 0).mean():.0%}")

    out = {"fingerprint": fp, "top": [{"cfg": lab(k), "p10": p10(res[k]), "mean": float(np.nanmean(res[k]))} for k in full[:30]],
           "cv_picks": picks, "cv_p10": p10(oof), "cv_base_p10": p10(oof_base)}
    (ROOT / "reports" / "two_day_cv.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
