"""채택 규칙(4분x3 + 전일↑ + VWAP 위 · dstop3 · 2자리x49%) 5일 창에서 목표 잠금·하한 잠금의 효과.

combo_study 의 main() 을 그대로 돌리되 dstop3 셀의 첫 호출에서 사건·종가를 가로채 중단하고,
같은 창 집합으로 잠금(lock)·하한(floor) 변형을 센다. 판정은 둘 다 종가 평가 한 번(장중은 안 본다, 보수적).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "C:/Users/young/Desktop/BanKis16")
import scripts.combo_study as cs  # noqa: E402

cs.COMP_BARS = 5
cs.MAX_GROSS = 0.98
BARS = 5
orig = cs.run_window_intraday
cap = {}


class Done(Exception):
    pass


def spy(events, arm, d0, d1, closes, cost, slots, weight, slide_cap=None, lock=0.0):
    if arm == "dstop3" and not cap:
        cap.update(events=events, closes=closes, cost=cost, slots=slots, weight=weight, first=d0)
        raise Done()
    return orig(events, arm, d0, d1, closes, cost, slots, weight, slide_cap, lock)


cs.run_window_intraday = spy
sys.argv = ["combo_study.py", "--ma-filters", "--open-window", "4", "--slots", "2",
            "--out", "C:/Users/young/AppData/Local/Temp/claude/b3e4fead543841f1/a61dbb29-0317-4fc7-a7d9-a89fec54432f/scratchpad/lock5_tmp.json"]
try:
    cs.main()
except Done:
    pass
assert cap, "dstop3 셀을 가로채지 못했다"

events, closes, cost = cap["events"], cap["closes"], cap["cost"]
slots, weight, first = cap["slots"], cap["weight"], cap["first"]
last = max(max(v) for v in closes.values())
starts = list(range(first, last - BARS + 2))
EOD = cs.EOD_MIN


def run(arm, d0, d1, lock=0.0, floor=0.0):
    """run_window_intraday 와 같은 규약 + 하한(floor<0): 종가 평가 수익률 ≤ floor 면 종가 전량 매도·창 끝까지 현금."""
    cash = 1.0
    held = []
    by_day = {}
    for ev in events:
        if d0 <= ev["d"] <= d1:
            by_day.setdefault(ev["d"], []).append(ev)

    def release(d, upto_min, kinds):
        nonlocal cash
        keep, sold = [], set()
        for h in held:
            if h["exit_d"] == d and h["when"] in kinds and h["exit_min"] <= upto_min:
                cash += h["alloc"] * (1.0 + h["ret"])
                sold.add(h["sym"])
            else:
                keep.append(h)
        held[:] = keep
        return sold

    for d in range(d0, d1 + 1):
        sold_today = release(d, EOD, ("open",))
        for ev in sorted(by_day.get(d, []), key=lambda e: e["detect_min"]):
            sold_today |= release(d, ev["detect_min"], ("intra",))
            if len(held) >= slots or ev["sym"] in sold_today:
                continue
            if any(h["sym"] == ev["sym"] for h in held):
                continue
            x = ev[arm]
            equity = cash + sum(h["alloc"] for h in held)
            alloc = min(equity * weight, cash)
            if alloc <= 0:
                continue
            cash -= alloc
            held.append({"alloc": alloc, "exit_d": x["exit_d"], "when": x["when"],
                         "exit_min": x["exit_min"], "ret": x["ret"], "sym": ev["sym"],
                         "entry_px": ev["entry_px"]})
        release(d, EOD, ("intra", "close"))
        if lock > 0 or floor < 0:
            mtm = cash
            for h in held:
                px = closes[h["sym"]].get(d)
                mtm += h["alloc"] * (px / h["entry_px"] * (1.0 - cost / 2.0)) if px and px > 0 else h["alloc"]
            r = mtm - 1.0
            if lock > 0 and r >= lock:
                return r, "lock"
            if floor < 0 and r <= floor:
                return r, "floor"
    for h in held:
        px = closes[h["sym"]].get(d1)
        cash += h["alloc"] * (px / h["entry_px"] * (1.0 - cost / 2.0)) if px and px > 0 else h["alloc"]
    return cash - 1.0, "end"


# 지금 위치: 증권사 평가 +22.38% · 시장가 평가 +24.55% · 2등 +21 · 3등 +18 · 4·5등 +16
T_MKT = 1.30 / 1.2455 - 1.0      # 시장가 기준 +30%까지 거리 (+4.4%)
T_KIS = 1.30 / 1.2238 - 1.0      # 증권사 평가 기준 (+6.2%)
LOSE_1ST = 1.21 / 1.2238 - 1.0   # 2등(정지)에게 1등을 내주는 선 (−1.1%)
OUT_TOP5 = 1.16 / 1.2238 - 1.0   # 5등(정지) 아래 (−5.2%)
F_A = 1.19 / 1.2238 - 1.0        # 하한 +19% (2등과 3등 사이)
F_B = 1.17 / 1.2238 - 1.0        # 하한 +17% (4·5등 바로 위)

configs = [
    ("그대로(잠금 없음)", 0.0, 0.0),
    (f"목표 잠금 +30%(시장가 거리 {T_MKT*100:+.1f}%)", T_MKT, 0.0),
    (f"목표 잠금 +30%(증권사 거리 {T_KIS*100:+.1f}%)", T_KIS, 0.0),
    (f"하한 잠금 +19%({F_A*100:+.1f}%)", 0.0, F_A),
    (f"하한 잠금 +17%({F_B*100:+.1f}%)", 0.0, F_B),
    (f"목표 +30%(시장가) & 하한 +19%", T_MKT, F_A),
    (f"목표 +30%(시장가) & 하한 +17%", T_MKT, F_B),
    (f"목표 +30%(증권사) & 하한 +19%", T_KIS, F_A),
]
rows = []
print()
print("=" * 118)
print(f" 채택 규칙 5일 창 {len(starts)}개 · 사건 {len(events):,} · 2자리x{weight*100:.0f}%  —  지금 +22.38%(증권사) / +24.55%(시장가)")
print("=" * 118)
print(f" {'설정':<34}{'평균':>7}{'중앙':>7}{'P(+30%도달)':>12}{'P(1등 뺏김)':>12}{'P(5위밖)':>9}{'P(≤-10%)':>9}{'잠금':>7}{'하한':>7}{'q05':>7}{'q95':>7}")
for name, lk, fl in configs:
    out = [run("dstop3", d0, d0 + BARS - 1, lk, fl) for d0 in starts]
    r = np.array([x[0] for x in out]); how = [x[1] for x in out]
    row = {"name": name, "lock": lk, "floor": fl, "mean": float(r.mean()), "median": float(np.median(r)),
           "reach30_mkt": float((r >= T_MKT).mean()), "reach30_kis": float((r >= T_KIS).mean()),
           "lose_1st": float((r <= LOSE_1ST).mean()), "out_top5": float((r <= OUT_TOP5).mean()),
           "loss10": float((r <= -0.10).mean()), "locked": how.count("lock") / len(how),
           "floored": how.count("floor") / len(how),
           "q05": float(np.percentile(r, 5)), "q95": float(np.percentile(r, 95))}
    rows.append(row)
    reach = row["reach30_kis"] if lk == T_KIS else row["reach30_mkt"]
    print(f" {name:<34}{row['mean']*100:+6.1f}%{row['median']*100:+6.1f}%{reach*100:11.1f}%{row['lose_1st']*100:11.1f}%"
          f"{row['out_top5']*100:8.1f}%{row['loss10']*100:8.1f}%{row['locked']*100:6.1f}%{row['floored']*100:6.1f}%"
          f"{row['q05']*100:+6.1f}%{row['q95']*100:+6.1f}%")
print("=" * 118)
print(f" 기준선: +30% 도달 거리 시장가 {T_MKT*100:+.2f}% · 증권사 {T_KIS*100:+.2f}% / 1등 뺏김 {LOSE_1ST*100:+.2f}% / 5위 밖 {OUT_TOP5*100:+.2f}%")
json.dump({"windows": len(starts), "events": len(events), "weight": weight,
           "thresholds": {"T_MKT": T_MKT, "T_KIS": T_KIS, "LOSE_1ST": LOSE_1ST, "OUT_TOP5": OUT_TOP5},
           "rows": rows}, open("C:/Users/young/AppData/Local/Temp/claude/b3e4fead543841f1/a61dbb29-0317-4fc7-a7d9-a89fec54432f/scratchpad/lock5.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(" 저장 → C:/Users/young/AppData/Local/Temp/claude/b3e4fead543841f1/a61dbb29-0317-4fc7-a7d9-a89fec54432f/scratchpad/lock5.json")
