"""day_gap 4라운드 — 익절 1~2% 기반 견고성(이웃 시각)과 소수 필터. 누적 67에서 이어 센다."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scripts.day_eval import evaluate
import day_gap as G
import warnings; warnings.filterwarnings("ignore")
n = 0
def run(name, fn):
    global n; n += 1
    return evaluate(fn, name)
print("== 이웃 시각 견고성 ==")
for m in [25, 35, 40]:
    for tp in [.01, .02]:
        run(f"m{m} tp{tp:.0%}", lambda c, m=m, tp=tp: G.gap_recover(c, m=m, rule={"tp": tp}))
for m in [25, 35, 40]:
    run(f"m{m} trail 2/1", lambda c, m=m: G.gap_recover(c, m=m, rule={"trail": (.02, .01)}))
print("== 필터(m30) ==")
def filt(c, lo=-0.05, full=False, rule=None, m=30):
    gap, ret, last, vwap = G._snap(c, m)
    mg = np.nanmedian(gap)
    if not (np.isfinite(mg) and mg >= 0): return None
    ok = np.isfinite(gap) & (gap <= -0.01) & (gap >= lo) & np.isfinite(vwap) & (last > vwap)
    if full: ok &= last >= np.asarray(c.prev_close, float)
    for j in np.argsort(-np.where(ok, ret, -np.inf))[: int(ok.sum())]:
        if G._liquid(c, int(j)): return int(j), m, dict(rule or {})
    return None
for tp in [None, .01, .02]:
    rr = {} if tp is None else {"tp": tp}
    run(f"m30 갭 -5~-1% tp{tp}", lambda c, rr=rr: filt(c, rule=rr))
    run(f"m30 전일종가 회복 tp{tp}", lambda c, rr=rr: filt(c, lo=-1, full=True, rule=rr))
print("설정 수(이번)", n, "누적", 67 + n)
