"""day_gap 3라운드 — 실평가기(evaluate)로 매수 시각 이웃·청산 변형. 설정 수 이어서 센다(누적 48)."""
import sys, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.research.day_eval import evaluate
from day_gap import gap_recover
import warnings; warnings.filterwarnings("ignore")
res = {}; n = 0
def run(name, **kw):
    global n; n += 1
    res[name] = evaluate(lambda c: gap_recover(c, **kw), name)
print("== 매수 시각 이웃(시장 갭≥0, 갭≤-1%, 15:10) ==")
for m in [20, 25, 30, 35, 40, 45]:
    run(f"m{m}", m=m)
print("== 시장 필터 없이 / 상대 갭 ==")
run("m30 시장필터 없음", m=30, need_mkt_up=False)
run("m30 상대갭≤-2% 필터없음", m=30, need_mkt_up=False, gap_max=-0.02, rel=True)
run("m30 갭≤-2%", m=30, gap_max=-0.02)
print("== 청산 변형(m30) ==")
for nm, r in [("tp1%", {"tp": .01}), ("tp2%", {"tp": .02}), ("tp3%", {"tp": .03}), ("tp5%", {"tp": .05}),
              ("sl2%", {"sl": .02}), ("sl3%", {"sl": .03}), ("tp2 sl3", {"tp": .02, "sl": .03}),
              ("tp3 sl3", {"tp": .03, "sl": .03}), ("trail 2%/1%", {"trail": (.02, .01)}),
              ("15:10 대신 13:00", {"t_exit": 240})]:
    run(f"m30 {nm}", m=30, rule=r)
print("설정 수(이번)", n, "누적", 48 + n)
pickle.dump(res, open(ROOT / "reports" / "agents" / "day_gap_r3.pkl", "wb"))
