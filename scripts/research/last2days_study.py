"""마지막 2일 설계 측정(11/17) — "내일 잃었다"를 가정하고 11/19(목)·11/20(금) 두 번의 09:04 기회로 한 종목 풀매수(1자리 x 98%)를
할 때 "+g%를 맞출 확률"을 센다. 채택 진입(4분x3 · 전일↑ · VWAP 위 · 3일 상승 < 20%) 사건을 `combo_study.main()`에서 가로채
(첫 dstop3 호출에서 중단) 1일 창(당일 09:04 → 종가)과 2일 창(목 09:04 → 금 종가)을 실거래와 같은 분 순서 시뮬레이터로 돈다.

    PYTHONIOENCODING=utf-8 UNIVERSE_FILE=config/universe_10b.json python scripts/research/last2days_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.research.combo_study as cs  # noqa: E402

OUT = ROOT / "reports" / "last2days_index.json"
cs.COMP_BARS = 2
cs.MAX_GROSS = 0.98
cap = {}
orig = cs.run_window_intraday


class Done(Exception):
    pass


def spy(events, arm, d0, d1, closes, cost, slots, weight, slide_cap=None, lock=0.0):
    if arm == "dstop3" and not cap:
        cap.update(events=events, closes=closes, cost=cost, first=d0)
        raise Done()
    return orig(events, arm, d0, d1, closes, cost, slots, weight, slide_cap, lock)


cs.run_window_intraday = spy
sys.argv = ["combo_study.py", "--ma-filters", "--open-window", "4", "--slots", "2",
            "--out", str(ROOT / "reports" / "last2days_tmp.json")]
try:
    cs.main()
except Done:
    pass
assert cap, "dstop3 셀을 가로채지 못했다"

events_all, closes, cost, first = cap["events"], cap["closes"], cap["cost"], cap["first"]
# 실거래 규칙(11/16~)의 3일 상승 < 20% 필터 — 모르면 통과(엔진과 같다)
events = [e for e in events_all if not (np.isfinite(e["ma"].get("run3", np.nan)) and e["ma"]["run3"] >= 0.20)]
last = max(max(v) for v in closes.values())

ARMS = [("hold_only", "청산 없이 보유(창 끝 종가 평가)"),
        ("istop5_only", "하드스톱 −5%만"), ("istop7_only", "하드스톱 −7%만"), ("istop10_only", "하드스톱 −10%만"),
        ("eod_d1", "진입일 종가 매도 → 다음 날 09:04 다시(두 번 쏘기)"),
        ("d1fail_close", "진입일 종가 < 매수가면 종가 매도·다음 날 다시, 아니면 보유"),
        ("dstop3", "지금 규칙(종가 −3% · vol_alive → 익일 시가)"), ("istop7", "지금 규칙 + 하드스톱 −7%"),
        # 사용자 제안(11/17): 분봉 이평 데드크로스 매도(다음 분 시가, 아침 판정 없음)
        ("x5_20", "1분봉 5↓20 데드크로스"), ("x5_60", "1분봉 5↓60 데드크로스"), ("x20_60", "1분봉 20↓60 데드크로스"),
        ("x5_20p", "1분봉 5↓20 크로스 & 수익 중일 때만"), ("x5_60p", "1분봉 5↓60 크로스 & 수익 중"), ("x20_60p", "1분봉 20↓60 크로스 & 수익 중"),
        ("x5_20_h7", "1분봉 5↓20 크로스 + 하드스톱 −7%"), ("x20_60_h7", "1분봉 20↓60 크로스 + 하드스톱 −7%"),
        ("x25_100", "5분봉 5↓20 크로스(1분 환산 25↓100)"), ("px150", "30분봉 5이평(150분) 아래로 종가"),
        ("px150_h7", "30분봉 5이평 아래 + 하드스톱 −7%"),
        ("turn5p", "(참고) 1분봉 5이평 꺾임 & 수익>0 · 아침 vol_alive 포함"), ("turn20p", "(참고) 20이평 꺾임 & 수익>0 · 아침 vol_alive 포함")]
PKL = ROOT / "reports" / "last2days_events.pkl"
GAINS = (0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30)
LOSSES = (-0.05, -0.10, -0.15, -0.20)


def stats(r: np.ndarray) -> dict:
    return {"n": int(len(r)), "no_entry": float((r == 0.0).mean()), "mean": float(r.mean()), "median": float(np.median(r)),
            "q05": float(np.quantile(r, 0.05)), "q95": float(np.quantile(r, 0.95)),
            **{f"up{int(g*100)}": float((r >= g).mean()) for g in GAINS},
            **{f"dn{int(-l*100)}": float((r <= l).mean()) for l in LOSSES}}


import pickle  # noqa: E402
with open(PKL, "wb") as fh:                     # 변형을 더 재고 싶을 때 30분 재적재 없이 쓰려고 남긴다(git 제외)
    pickle.dump({"events": events_all, "closes": closes, "cost": cost, "first": first}, fh)
print(f"사건 저장: {PKL} ({PKL.stat().st_size / 1e6:.0f}MB)")

out = {"events": len(events), "events_before_run3": len(events_all), "cost": cost, "horizons": {}}
for H in (1, 2):
    starts = list(range(first, last - H + 2))
    rows = {}
    for slots, weight in ((1, 0.98), (2, 0.49)):
        for arm, label in ARMS:
            if slots == 2 and arm not in ("hold_only", "istop7_only", "dstop3", "istop7"):
                continue
            r = np.array([orig(events, arm, d0, d0 + H - 1, closes, cost, slots, weight) for d0 in starts])
            rows[f"{slots}자리x{weight:.0%} · {label}"] = stats(r)
    out["horizons"][str(H)] = {"windows": len(starts), "rows": rows}
    print()
    print(f"=== {H}일 창 ({len(starts):,}개 · 사건 {len(events):,}건 · 진입 당일 09:04 → {'그날' if H == 1 else '다음 날'} 종가 평가) ===")
    print(f" {'규칙':<58}{'미진입':>6}{'평균':>7}{'중앙':>7}" + "".join(f"{'≥+' + str(int(g*100)) + '%':>7}" for g in GAINS)
          + "".join(f"{'≤' + str(int(l*100)) + '%':>7}" for l in LOSSES) + f"{'q05':>7}{'q95':>7}")
    for name, s in rows.items():
        print(f" {name[:58]:<58}{s['no_entry']*100:5.1f}%{s['mean']*100:+6.1f}%{s['median']*100:+6.1f}%"
              + "".join(f"{s[f'up{int(g*100)}']*100:6.1f}%" for g in GAINS)
              + "".join(f"{s[f'dn{int(-l*100)}']*100:6.1f}%" for l in LOSSES)
              + f"{s['q05']*100:+6.1f}%{s['q95']*100:+6.1f}%")

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n저장: {OUT}")
