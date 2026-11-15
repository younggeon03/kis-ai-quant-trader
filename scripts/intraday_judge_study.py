"""하루 여러 번 판정·갭 매도 측정 + "첫날 −3%~0% 보유"의 조건부 분포(11/15).

`combo_study.main()` 을 지금 서버 칸(5거래일 창 · 2자리x49%)으로 돌리고, dstop3 셀의 사건 목록을 가로채(중단하지 않고)
"첫날 종가가 매수가 −3%~0% 사이"인 사건만 골라 **다음 날 시가부터** 각 청산 규칙의 수익 분포를 센다 — 지금 보유 두 종목이 놓인 상태다.

    PYTHONIOENCODING=utf-8 UNIVERSE_FILE=config/universe_10b.json python scripts/intraday_judge_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.combo_study as cs  # noqa: E402

OUT = ROOT / "reports" / "combo_judge_index.json"
cs.COMP_BARS = 5
cs.MAX_GROSS = 0.98
stash = {}
orig = cs.run_window_intraday


def spy(events, arm, d0, d1, closes, cost, slots, weight, slide_cap=None, lock=0.0):
    if arm == "dstop3" and "events" not in stash:
        stash.update(events=events, cost=cost)
    return orig(events, arm, d0, d1, closes, cost, slots, weight, slide_cap, lock)


cs.run_window_intraday = spy
sys.argv = ["combo_study.py", "--ma-filters", "--open-window", "4", "--slots", "2", "--out", str(OUT)]
rc = cs.main()

ARMS = [("dstop3", "지금(아침 판정 → 익일 시가)"), ("gapopen3", "갭하락 시가 매도"), ("j1200", "아침+12:00"),
        ("j3h", "3시간 텀"), ("j2h", "2시간 텀"), ("j2h_1519", "2시간 텀+15:19"), ("j8", "8번 판정"), ("eod3", "15:19만")]
KEYS = ("mean", "median", "neg", "loss5", "loss10", "up10", "up20", "q05", "q95")
if OUT.exists():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    print()
    print(f" {'규칙(5일 창, 2자리x49%)':<46}{'평균':>7}{'중앙':>7}{'손실':>7}{'≤-5%':>7}{'≤-10%':>7}{'≥+10%':>7}{'≥+20%':>7}{'q05':>7}{'q95':>7}")
    for x in d["rows"]:
        n = x["rule"]
        if "채택진입] 청산" in n and any(k in n for k in ("−3% → 익일", "갭하락", "12:00", "시간 텀", "8번", "15:19 판정(")):
            print(f" {n[:46]:<46}{x['mean']*100:+6.1f}%{x['median']*100:+6.1f}%{x['neg']*100:6.1f}%{x['loss5']*100:6.1f}%"
                  f"{x['loss10']*100:6.1f}%{x['up10']*100:6.1f}%{x['up20']*100:6.1f}%{x['q05']*100:+6.1f}%{x['q95']*100:+6.1f}%")

ev = stash.get("events") or []
cost = float(stash.get("cost", 0.0))
sel = [e for e in ev if np.isfinite(e.get("d1_ret", np.nan)) and -0.03 < e["d1_ret"] <= 0.0 and e.get("o2", 0) > 0]
print()
print(f" 조건부 — 첫날 종가가 매수가 −3%~0% 사이였던 진입 {len(sel):,}건(전체 {len(ev):,}건) · **다음 날 시가부터** 각 규칙으로 들고 갈 때")
print(f" {'규칙':<28}{'평균':>8}{'중앙':>8}{'P(>0)':>8}{'≤-5%':>8}{'≤-10%':>8}{'≥+5%':>8}{'≥+10%':>8}")
cond = {}
for arm, label in ARMS:
    r = np.array([((e[arm]["ret"] + 1.0 + cost) * e["entry_px"]) / e["o2"] - 1.0 - cost / 2.0 for e in sel if arm in e])
    if not len(r):
        continue
    cond[arm] = {"n": int(len(r)), "mean": float(r.mean()), "median": float(np.median(r)), "p_pos": float((r > 0).mean()),
                 "loss5": float((r <= -0.05).mean()), "loss10": float((r <= -0.10).mean()),
                 "up5": float((r >= 0.05).mean()), "up10": float((r >= 0.10).mean())}
    c = cond[arm]
    print(f" {label:<28}{c['mean']*100:+7.2f}%{c['median']*100:+7.2f}%{c['p_pos']*100:7.1f}%{c['loss5']*100:7.1f}%"
          f"{c['loss10']*100:7.1f}%{c['up5']*100:7.1f}%{c['up10']*100:7.1f}%")
print(" (다음 날 시가에 바로 팔면 0.00% — 위 평균이 양수면 들고 가는 것이 기대값이 높다)")
if OUT.exists():
    d["conditional_d1_minus3_to_0"] = {"n": len(sel), "arms": cond}
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(rc)
