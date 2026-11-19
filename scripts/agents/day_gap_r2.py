"""day_gap 2라운드 — 갭 하락 회복(B) 변형 · 시장 국면 · 줄다리기 결합. 설정 수 이어서 센다(1라운드 28)."""
import numpy as np
from day_gap_study import D, lab, ND, pick, N_TESTED
np.seterr(all="ignore")
gap = D["o0"]/D["pc"]-1
liq = np.isfinite(gap) & (D["to20"] > 5e9)
def med(a): return np.array([np.nanmedian(a[n][liq[n]]) if liq[n].any() else np.nan for n in range(ND)])[:, None]
mgap = med(gap); mr1 = med(D["r1"])
for m in [30, 60]:
    r = D[f"ret{m}"]; vw = D[f"px{m}"] > D[f"vwap{m}"]
    mret = med(r)
    print(f"--- m={m}")
    pick(r, liq & (gap <= -0.02) & vw, m, f"G1 갭≤-2% · VWAP 위 · 회복 최대 m{m}")
    pick(r, liq & (gap <= -0.03) & vw, m, f"G2 갭≤-3% · VWAP 위 · 회복 최대 m{m}")
    pick(r, liq & (gap - mgap <= -0.02) & vw, m, f"G3 갭-시장갭≤-2% · VWAP 위 m{m}")
    pick(r, liq & (gap <= -0.01) & vw & (D[f"px{m}"] >= D["pc"]), m, f"G4 갭≤-1% · 전일종가 회복 m{m}")
    pick(r, liq & (gap <= -0.01) & vw & (mgap < 0), m, f"G5 B3 + 시장 갭<0 m{m}")
    pick(r, liq & (gap <= -0.01) & vw & (mgap >= 0), m, f"G6 B3 + 시장 갭≥0 m{m}")
    pick(r, liq & (gap <= -0.01) & vw & (mret > 0), m, f"G7 B3 + 시장 시가대비>0 m{m}")
    pick(r, liq & (gap <= -0.01) & vw & (D["on10"] < 0), m, f"G8 B3 + on10<0 m{m}")
    pick(r, liq & (gap <= -0.01) & vw & (D["dr5"] > 0), m, f"G9 B3 + dr5>0 m{m}")
    pick(-gap, liq & (gap <= -0.01) & vw, m, f"G10 갭≤-1% · VWAP 위 · 갭 최저 m{m}")
print("설정 수(누적)", 28 + N_TESTED[0])
