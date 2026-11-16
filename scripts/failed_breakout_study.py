"""실패 돌파 측정(11/16) — 09:04에 산 급등주가 장중에 매수가 아래로 되돌아올 때 "그때 바로 판다"가 나은지, 그리고
"그런 종목은 다음 날도 안 오른다"는 가설이 맞는지.

`combo_study.main()` 을 지금 서버 칸(5거래일 창 · 2자리x49%)으로 돌리고 dstop3 셀의 사건을 가로채(중단하지 않고),
진입일에 매수가 아래로 처음 내려온 분(`_touch`)을 기준으로 조건부 분포를 센다.

    PYTHONIOENCODING=utf-8 UNIVERSE_FILE=config/universe_10b.json python scripts/failed_breakout_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.combo_study as cs  # noqa: E402

OUT = ROOT / "reports" / "combo_failbreak_index.json"
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

KEYS = ("−3% → 익일", "실패 돌파", "매수가 −1% 아래", "매수가 −2% 아래", "본전 스톱", "하드스톱", "거래 식음 x0.3", "발동 분당")
if OUT.exists():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    print()
    print(f" {'규칙(5일 창, 2자리x49%)':<50}{'평균':>7}{'중앙':>7}{'손실':>7}{'≤-5%':>7}{'≤-10%':>7}{'≥+10%':>7}{'≥+20%':>7}")
    for x in d["rows"]:
        n = x["rule"]
        if "채택진입] 청산" in n and any(k in n for k in KEYS):
            print(f" {n[10:60]:<50}{x['mean']*100:+6.1f}%{x['median']*100:+6.1f}%{x['neg']*100:6.1f}%{x['loss5']*100:6.1f}%"
                  f"{x['loss10']*100:6.1f}%{x['up10']*100:6.1f}%{x['up20']*100:6.1f}%")

ev = stash.get("events") or []
cost = float(stash.get("cost", 0.0))
touched = [e for e in ev if e.get("_touch", {}).get("m", -1) >= 0 and np.isfinite(e["_touch"]["px"]) and e["_touch"]["px"] > 0]
print()
print(f" 진입일 장중에 매수가 아래로 내려온 사건: {len(touched):,} / {len(ev):,} ({len(touched) / max(1, len(ev)) * 100:.1f}%)")
if touched:
    mins = np.array([e["_touch"]["m"] for e in touched])
    print(f"   처음 내려온 분(09:00 기준): 중앙 {np.median(mins):.0f}분 · 10% {np.percentile(mins, 10):.0f} · 90% {np.percentile(mins, 90):.0f}")
    # 그때 팔면 0, 지금 규칙으로 들고 가면? (그 분 다음 분 시가 기준, 매도 비용 절반 반영)
    hold = np.array([((e["dstop3"]["ret"] + 1.0 + cost) * e["entry_px"]) / e["_touch"]["px"] - 1.0 - cost / 2.0 for e in touched])
    d1c = np.array([e["d1_ret"] for e in touched if np.isfinite(e.get("d1_ret", np.nan))])
    o2 = np.array([e["o2"] / e["_touch"]["px"] - 1.0 for e in touched if np.isfinite(e.get("o2", np.nan)) and e["o2"] > 0])
    print(f"   이탈 시점에 팔지 않고 지금 규칙으로 들고 갈 때(이탈가 대비): 평균 {hold.mean()*100:+.2f}% · 중앙 {np.median(hold)*100:+.2f}% · "
          f"P(>0) {(hold > 0).mean()*100:.1f}% · ≤−5% {(hold <= -0.05).mean()*100:.1f}% · ≥+5% {(hold >= 0.05).mean()*100:.1f}% · ≥+10% {(hold >= 0.10).mean()*100:.1f}%")
    print(f"   그 종목의 첫날 종가(매수가 대비): 평균 {d1c.mean()*100:+.2f}% · 종가 < 매수가 비율 {(d1c < 0).mean()*100:.1f}%")
    print(f"   다음 날 시가(이탈가 대비): 평균 {o2.mean()*100:+.2f}% · 중앙 {np.median(o2)*100:+.2f}% · P(>0) {(o2 > 0).mean()*100:.1f}%  ← '다음 날도 안 오른다' 검증")
    # 이탈하지 않은 사건과 비교
    kept = [e for e in ev if e not in touched]
    rk = np.array([e["dstop3"]["ret"] for e in kept]); rt = np.array([e["dstop3"]["ret"] for e in touched])
    print(f"   사건당 수익(지금 규칙, 매수가 대비): 이탈한 사건 평균 {rt.mean()*100:+.2f}% (중앙 {np.median(rt)*100:+.2f}%) · 이탈 안 한 사건 평균 {rk.mean()*100:+.2f}% (중앙 {np.median(rk)*100:+.2f}%)")
    if OUT.exists():
        d["touch_conditional"] = {"n_touch": len(touched), "n_all": len(ev), "hold_from_touch_mean": float(hold.mean()),
                                  "hold_from_touch_p_pos": float((hold > 0).mean()), "next_open_vs_touch_mean": float(o2.mean()),
                                  "next_open_vs_touch_p_pos": float((o2 > 0).mean()), "touched_event_ret": float(rt.mean()),
                                  "kept_event_ret": float(rk.mean())}
        OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(rc)
