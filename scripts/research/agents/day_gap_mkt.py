"""day_gap 시장 국면 — 일별 시장 평균 15:10 수익과 아침에 알 수 있는 시장 변수의 관계(개발 구간)."""
import numpy as np
from day_gap_study import D, lab, ND
np.seterr(all="ignore")
gap = D["o0"]/D["pc"]-1
base = np.isfinite(gap) & (D["to20"] > 5e9)
def mmean(a): return np.array([np.nanmean(a[n][base[n]]) for n in range(ND)])
for m in [5, 15, 30]:
    y = mmean(lab(m))
    X = {"mgap": mmean(gap), f"mret{m}": mmean(D[f"ret{m}"]), "mr1": mmean(D["r1"]), "mdr1": mmean(D["dr1"]),
         "mon1": mmean(D["on1"]), "mdr5": mmean(D["dr5"]), "mon5": mmean(D["on5"]), "mr5": mmean(D["r5"]),
         "breadth_m": np.array([np.nanmean(D[f"ret{m}"][n][base[n]]>0) for n in range(ND)])}
    print(f"--- m={m}: 시장평균 승(>0) {np.mean(y>0):.0%} 평균 {y.mean():+.3%}")
    for k, x in X.items():
        ok = np.isfinite(x)&np.isfinite(y)
        c = np.corrcoef(x[ok], y[ok])[0,1]
        q = np.quantile(x[ok], [1/3, 2/3])
        lo, hi = y[ok][x[ok]<=q[0]], y[ok][x[ok]>=q[1]]
        print(f"{k:<10} corr {c:+.3f} | 하위1/3 시장평균 {lo.mean():+.3%} (>0 {np.mean(lo>0):.0%}) 상위1/3 {hi.mean():+.3%} (>0 {np.mean(hi>0):.0%})")
