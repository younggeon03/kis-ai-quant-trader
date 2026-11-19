"""day_gap 1라운드 — 사전 지정 규칙(둥근 값). 15:10 매도. 설정 수는 pick() 호출 수."""
import numpy as np
from day_gap_study import D, lab, ND, pick, N_TESTED
np.seterr(all="ignore")
gap = D["o0"]/D["pc"]-1
liq = np.isfinite(gap) & (D["to20"] > 5e9)
def rk(a):  # 일별 백분위 순위
    out = np.full_like(a, np.nan)
    for n in range(ND):
        ok = np.isfinite(a[n]) & liq[n]
        if ok.sum() > 1:
            r = a[n][ok].argsort().argsort()
            out[n][ok] = r/(ok.sum()-1)
    return out
mgap = np.array([np.nanmedian(gap[n][liq[n]]) if liq[n].any() else np.nan for n in range(ND)])[:, None]
mr1 = np.array([np.nanmedian(D["r1"][n][liq[n]]) if liq[n].any() else np.nan for n in range(ND)])[:, None]
print("== A 줄다리기(낮 수익 지속·밤 수익 반대) ==")
for m in [5, 15]:
    pick(D["dr5"], liq, m, f"A1 dr5 최대 m{m}")
    pick(rk(D["dr10"]) - rk(D["on10"]), liq, m, f"A2 rank(dr10)-rank(on10) m{m}")
    pick(D["dr10"], liq & (D["on10"] < 0), m, f"A3 on10<0 중 dr10 최대 m{m}")
    pick(-D["on10"], liq, m, f"A4 on10 최소 m{m}")
print("== B 주의(갭 하락 뒤 회복) ==")
for m in [15, 30]:
    r = D[f"ret{m}"]
    pick(r, liq & (gap <= -0.02) & (r > 0), m, f"B1 갭≤-2% · 시가 대비 + 중 회복 최대 m{m}")
    pick(-gap, liq & (gap <= -0.02) & (r > 0), m, f"B2 갭≤-2% · 시가 대비 + 중 갭 최저 m{m}")
    pick(r, liq & (gap <= -0.01) & (D[f"px{m}"] > D[f"vwap{m}"]), m, f"B3 갭≤-1% · VWAP 위 중 회복 최대 m{m}")
print("== C 작은 갭 + 초반 강세(갭 큰 종목 피함) ==")
for m in [5, 15, 30]:
    r = D[f"ret{m}"]
    pick(r, liq & (np.abs(gap) <= 0.01), m, f"C1 |갭|≤1% 중 시가 대비 최대 m{m}")
    pick(r, liq & (gap <= 0.02) & (gap >= -0.02), m, f"C2 |갭|≤2% 중 시가 대비 최대 m{m}")
print("== D 갭 상승(주의) — 피해야 할 쪽 확인 ==")
pick(gap, liq, 5, "D1 갭 최대 m5")
pick(gap, liq & (D["ret5"] > 0), 5, "D2 갭 최대 · 시가 위 m5")
print("== E 저변동 · 줄다리기 ==")
pick(rk(D["dr10"]) - rk(D["on10"]) - rk(D["atr10"]), liq, 5, "E1 dr10-on10-atr10 m5")
pick(rk(D["ret15"]) - rk(gap) - rk(D["on10"]), liq, 15, "E2 ret15-gap-on10 m15")
pick(rk(D["ret30"]) - rk(gap) - rk(D["on10"]), liq, 30, "E3 ret30-gap-on10 m30")
print("== F 시장 국면 거르기(전일 시장 하락 & 오늘 시장 갭 ≥0) + E2 ==")
f = (mr1 < 0) & (mgap >= 0)
pick(rk(D["ret15"]) - rk(gap) - rk(D["on10"]), liq & f, 15, "F1 E2 + (mr1<0,mgap≥0)")
pick(rk(D["ret15"]) - rk(gap) - rk(D["on10"]), liq & (mgap >= 0), 15, "F2 E2 + mgap≥0")
pick(rk(D["ret15"]) - rk(gap) - rk(D["on10"]), liq & (mr1 < 0), 15, "F3 E2 + mr1<0")
print("설정 수", N_TESTED[0])
