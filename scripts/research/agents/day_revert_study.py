"""평균 회귀 규칙 탐색 — 피처 표에서 그날 1종목을 고르고 day_eval.simulate 로 잰다(개발 구간만)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.day_eval import simulate, stats, Ctx, dev_days, K

SCR = Path(sys.argv[1])
F = pd.read_pickle(SCR / "feats.pkl")
DAYS = dev_days()
EDGES = np.linspace(0, len(DAYS), K + 1).astype(int)
POS = {d: n for n, d in enumerate(DAYS)}
N_CFG = [0]
RESULTS = []

def vwap_exit(i, j, m, band=0.0):
    """진입 전 누적으로 시작해 분마다 VWAP 갱신. 종가가 VWAP*(1-band) 이상이면 다음 분 시가에 판다."""
    b = Ctx(i).bars(m)[j]
    tp_ = (b[1] + b[2] + b[3]) / 3; v = np.nan_to_num(b[4])
    st = {"pv": float(np.nansum(np.nan_to_num(tp_) * v)), "sv": float(v.sum()), "go": False}
    def f(o, h, l, c, vol, entry, k):
        if st["go"] and np.isfinite(o):
            return o
        if np.isfinite(c) and np.isfinite(vol):
            st["pv"] += (h + l + c) / 3 * vol; st["sv"] += vol
            if st["sv"] > 0 and c >= st["pv"] / st["sv"] * (1 - band):
                st["go"] = True
        return None
    return f

def run(name, picks, rule_maker, show=True):
    """picks: DataFrame(i,j,m) 하루 1행."""
    N_CFG[0] += 1
    tr, folds = [], [[] for _ in range(K)]
    for i, j, m in picks[["i", "j", "m"]].itertuples(index=False):
        rule = rule_maker(i, j, m)
        r = simulate(int(i), int(j), int(m), rule)
        if r is None:
            continue
        tr.append(r)
        f = int(np.searchsorted(EDGES, POS[i], side="right") - 1)
        folds[min(f, K - 1)].append(r)
    st = stats(tr, len(DAYS), folds); st["name"] = name
    RESULTS.append(st)
    if show and st["trades"]:
        print(f"{name:<52} n={st['trades']:>3} 승 {st['win']:.0%} 손 {st['loss']:.0%} +5↑ {st['p5']:.0%} 평 {st['mean']:+.2%} "
              f"중 {st['median']:+.2%} −3↓ {st['p_le_m3']:.0%} q05 {st['q05']:+.1%} 조각 " + " ".join(f"{x:.0%}" for x in st["fold_win"]), flush=True)
    elif show:
        print(name, "0건")
    return st

def pick(df, score, asc=False):
    d = df.dropna(subset=[score]).sort_values(score, ascending=asc)
    return d.groupby("i", as_index=False).head(1)

EXITS = {
    "1510": lambda i, j, m: {},
    "tp1": lambda i, j, m: {"tp": 0.01},
    "tp2": lambda i, j, m: {"tp": 0.02},
    "tp3": lambda i, j, m: {"tp": 0.03},
    "tp2sl5": lambda i, j, m: {"tp": 0.02, "sl": 0.05},
    "vwap": lambda i, j, m: {"exit_fn": vwap_exit(i, j, m)},
}

stage = sys.argv[2] if len(sys.argv) > 2 else "1"
if stage == "1":
    # 유동성: 전일까지 20일 평균 거래대금 상위 100 (대형·유동)
    F["liqrank"] = F.groupby(["i", "m"])["adv20"].rank(ascending=False)
    for m in [10, 30]:
        for X in [0.03, 0.05]:
            base = F[(F.m == m) & (F.liqrank <= 100) & (F.r_pc <= -X)]
            p = pick(base, "adv20")              # 가장 유동성 큰 종목
            for en, ef in EXITS.items():
                run(f"A 전일비≤-{X:.0%} @{m}분 유동1위 | {en}", p, ef)
    for m in [10, 30, 60]:
        for k in [0.01, 0.02, 0.03]:
            base = F[(F.m == m) & (F.liqrank <= 100) & (F.vdev <= -k)]
            p = pick(base, "adv20")
            for en in ["1510", "tp1", "tp2", "vwap"]:
                run(f"B VWAP-{k:.0%} @{m}분 유동1위 | {en}", p, EXITS[en])
    print("설정 수", N_CFG[0])
    pd.DataFrame(RESULTS).to_pickle(SCR / "res1.pkl")

if stage == "2":
    F["liqrank"] = F.groupby(["i", "m"])["adv20"].rank(ascending=False)
    ex3 = ["1510", "tp1", "vwap"]
    for m in [30]:
        L = F[(F.m == m) & (F.liqrank <= 100)]
        run(f"C0 대조: 유동1위 무조건 @{m}", pick(L[L.liqrank <= 1], "adv20"), EXITS["tp1"])
        run(f"C0 대조: 유동1위 무조건 @{m} 1510", pick(L[L.liqrank <= 1], "adv20"), EXITS["1510"])
        base = L[L.r_pc <= -0.03]
        filt = {
            "반등≥1%": base.bounce >= 0.01,
            "폭증없음 tratio≤2": base.tratio <= 2,
            "폭증 tratio≥3": base.tratio >= 3,
            "시장하락 mkt≤-1%": base.mkt <= -0.01,
            "시장평온 mkt≥-0.5%": base.mkt >= -0.005,
            "개별 r-mkt≤-3%": (base.r_pc - base.mkt) <= -0.03,
            "상승추세 ma60r>0": base.ma60r > 0,
            "일봉과매도 rsi2≤20": base.rsi2 <= 20,
            "시가대비 r_op≤-2%": base.r_op <= -0.02,
        }
        for fn, msk in filt.items():
            p = pick(base[msk], "adv20")
            for en in ex3:
                run(f"D -3%@{m} +{fn} | {en}", p, EXITS[en])
        p = pick(base, "r_pc", asc=True)
        for en in ex3:
            run(f"D -3%@{m} 최대낙폭 1위 | {en}", p, EXITS[en])
    print("설정 수", N_CFG[0])
    pd.DataFrame(RESULTS).to_pickle(SCR / "res2.pkl")

if stage == "3":
    F["liqrank"] = F.groupby(["i", "m"])["adv20"].rank(ascending=False)
    EXITS["tp1sl3"] = lambda i, j, m: {"tp": 0.01, "sl": 0.03}
    EXITS["tp05"] = lambda i, j, m: {"tp": 0.005}
    ex = ["1510", "tp05", "tp1", "tp1sl3", "vwap"]
    for m in [30, 60]:
        L = F[(F.m == m) & (F.liqrank <= 100)]
        base = L[L.r_pc <= -0.03]
        combos = {
            "반등≥1%+ma60r>0": (base.bounce >= 0.01) & (base.ma60r > 0),
            "시장≤-1%+반등≥1%": (base.mkt <= -0.01) & (base.bounce >= 0.01),
            "시장≤-1%": base.mkt <= -0.01,
            "시장≤-1%+ma60r>0": (base.mkt <= -0.01) & (base.ma60r > 0),
        }
        for fn, msk in combos.items():
            p = pick(base[msk], "adv20")
            for en in ex:
                run(f"E -3%@{m} {fn} | {en}", p, EXITS[en])
        # Nagel: 시장 하락일에 유동성 1위(과매도 조건 없음)
        L1 = L[(L.liqrank <= 1) & (L.mkt <= -0.01)]
        for en in ex:
            run(f"E 시장≤-1%@{m} 유동1위 | {en}", pick(L1, "adv20"), EXITS[en])
    print("설정 수", N_CFG[0])
    pd.DataFrame(RESULTS).to_pickle(SCR / "res3.pkl")

if stage == "4":   # 이웃 민감도: 중심(30분·시장≤-1%·종목≤-3%·반등≥1%)에서 축 하나씩
    F["liqrank"] = F.groupby(["i", "m"])["adv20"].rank(ascending=False)
    EXITS["tp05"] = lambda i, j, m: {"tp": 0.005}
    def cfg(m=30, mk=-0.01, X=0.03, bo=0.01, tag=""):
        L = F[(F.m == m) & (F.liqrank <= 100)]
        b = L[(L.r_pc <= -X) & (L.mkt <= mk) & (L.bounce >= bo)]
        p = pick(b, "adv20")
        for en in ["tp1", "tp05"]:
            run(f"N {tag} | {en}", p, EXITS[en])
    cfg(m=15, tag="m15"); cfg(m=10, tag="m10"); cfg(mk=-0.005, tag="시장≤-0.5%"); cfg(mk=-0.02, tag="시장≤-2%")
    cfg(X=0.05, tag="종목≤-5%"); cfg(X=0.02, tag="종목≤-2%"); cfg(bo=0.02, tag="반등≥2%")
    print("설정 수", N_CFG[0])
